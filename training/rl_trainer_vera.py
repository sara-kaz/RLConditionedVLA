"""
VERA Online RL Fine-Tuning Trainer
====================================
Phase 2: roll out the VERA policy in an environment and optimise using
REINFORCE with a value baseline + KL regularisation against the BC checkpoint.

The RL update extends the SFT loss with:
  L_total = -E[log π(a|s) · Â_t]          ← REINFORCE (policy gradient)
           + 0.5 · (V(s) - G_t)²           ← value baseline (MSE)
           - entropy_coef · H[π]           ← entropy bonus  (exploration)
           + kl_coef    · KL(π ∥ π_BC)    ← KL penalty     (no forgetting)

Key adaptation for VLLA:
  The rollout collector now maintains a prev_action_idx and prev_reward
  that are passed to model.forward(), feeding Feedback Channel A (semantic)
  at every rollout step.

Usage
-----
  python -m training.rl_trainer_vera --config configs/config.yaml
"""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from models.vera_model import VERAModel, RMSNorm


# ── Value head ────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Scalar state-value estimate from the CLS token representation.

    Uses RMSNorm (matching VERA's backbone) rather than LayerNorm to keep
    normalisation consistent with the policy model's internal activations.
    SiLU matches the SwiGLU gating used throughout the LLaMA fusion stack.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        """cls_features: (B, D) → values: (B,)"""
        return self.net(cls_features).squeeze(-1)


# ── Rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one batch of episode transitions for a policy-gradient update."""

    def __init__(self):
        self.frames:           List[torch.Tensor]          = []
        self.lang_tokens:      List[torch.Tensor]          = []
        self.action_hists:     List[torch.Tensor]          = []
        self.reward_hists:     List[torch.Tensor]          = []
        self.action_vec_hists: List[Optional[torch.Tensor]]= []  # (H, action_dim) or None
        self.prev_actions:     List[torch.Tensor]          = []
        self.prev_rewards_fb:  List[torch.Tensor]          = []
        self.state_deltas:     List[torch.Tensor]          = []
        self.actions:          List[int]                   = []
        self.rewards:          List[float]                 = []
        self.dones:            List[bool]                  = []

    def add(self, frame, lang_tok, act_hist, rew_hist, act_vec_hist,
            prev_a, prev_r, state_delta, action, reward, done):
        self.frames.append(frame)
        self.lang_tokens.append(lang_tok)
        self.action_hists.append(act_hist)
        self.reward_hists.append(rew_hist)
        self.action_vec_hists.append(act_vec_hist)   # may be None
        self.prev_actions.append(prev_a)
        self.prev_rewards_fb.append(prev_r)
        self.state_deltas.append(state_delta)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """Discounted returns with episode boundary resets, then standardised."""
        G, returns = 0.0, []
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0.0
            G = r + gamma * G
            returns.insert(0, G)
        ret = torch.tensor(returns, dtype=torch.float32)
        if ret.std() > 1e-8:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        return ret


# ── Rollout collector ─────────────────────────────────────────────────────────

def collect_rollout(
    model:            VERAModel,
    env,
    cfg:              dict,
    device:           str,
    tokenizer_cache:  dict,
) -> RolloutBuffer:
    """Execute the policy for one episode and store all transitions."""
    import torchvision.transforms as Tv
    from PIL import Image as PILImage

    buf            = RolloutBuffer()
    history_len    = cfg["model"]["history_len"]
    num_vis_frames = cfg["model"]["num_vis_frames"]
    num_actions    = cfg["model"]["num_actions"]
    img_size       = cfg["data"].get("img_size", 224)
    max_steps      = cfg["rl"].get("max_episode_steps", 50)

    transform = Tv.Compose([
        Tv.Resize((img_size, img_size)),
        Tv.ToTensor(),
        Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std= [0.26862954, 0.26130258, 0.27577711]),
    ])

    action_dim   = cfg["model"].get("action_dim", 4)
    null_vec     = np.zeros(action_dim, dtype=np.float32)  # padding vector at t=0

    print(f"[rollout] env.reset() ...", end=" ", flush=True)
    obs          = env.reset()
    print("ok", flush=True)
    frame_q      = deque(maxlen=num_vis_frames)
    action_q     = deque([num_actions] * history_len, maxlen=history_len)
    reward_q     = deque([0.0]         * history_len, maxlen=history_len)
    action_vec_q = deque([null_vec.copy() for _ in range(history_len)],
                         maxlen=history_len)   # (H, action_dim) rolling buffer
    prev_action  = num_actions
    prev_rew_fb  = 0.0

    # Tokenise instruction
    inst = obs["instruction"]
    if inst not in tokenizer_cache:
        tokenizer_cache[inst] = clip.tokenize([inst])[0]
    lang_tok = tokenizer_cache[inst]

    done, step = False, 0
    while not done and step < max_steps:
        # Build frame tensor
        frame_t  = transform(PILImage.fromarray(obs["frame"]))              # (3,H,W)
        frame_q.append(frame_t)
        pad      = num_vis_frames - len(frame_q)
        frames_t = torch.stack([torch.zeros_like(frame_t)] * pad + list(frame_q))

        frames_in   = frames_t.unsqueeze(0).to(device)
        lang_in     = lang_tok.unsqueeze(0).to(device)
        act_hist_in = torch.tensor(list(action_q),     dtype=torch.long).unsqueeze(0).to(device)
        rew_hist_in = torch.tensor(list(reward_q),     dtype=torch.float32).unsqueeze(0).to(device)
        prev_a_in   = torch.tensor([prev_action],      dtype=torch.long).to(device)
        prev_r_in   = torch.tensor([prev_rew_fb],      dtype=torch.float32).to(device)

        # Low-level action vector history — (1, H, action_dim)
        vec_hist_np  = np.stack(list(action_vec_q), axis=0)                # (H, action_dim)
        act_vec_in   = torch.tensor(vec_hist_np, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(frames_in, lang_in, act_hist_in, rew_hist_in,
                        prev_a_in, prev_r_in,
                        action_vec_hist=act_vec_in)
        # With chunk_size > 1 the model outputs num_actions*chunk_size logits;
        # for single-step rollout we only use the first chunk (first num_actions bins).
        _logits = out["logits"][:, :num_actions]
        action = torch.multinomial(F.softmax(_logits, dim=-1), 1).item()

        obs, reward, done, info = env.step(action)

        # Extract signed distance delta and the executed continuous action vector
        _delta      = info.get("dist_delta") if isinstance(info, dict) else None
        state_delta_t = torch.tensor(
            [_delta if _delta is not None else 0.0], dtype=torch.float32
        )

        # The env may expose the executed low-level vector (MetaWorld codebook entry).
        # Fall back to zeros if not available (BabyAI, dummy).
        executed_vec = info.get("action_vector", null_vec) if isinstance(info, dict) else null_vec
        executed_vec = np.asarray(executed_vec, dtype=np.float32).flatten()
        # Truncate or zero-pad to action_dim
        if len(executed_vec) >= action_dim:
            executed_vec = executed_vec[:action_dim]
        else:
            executed_vec = np.concatenate(
                [executed_vec, np.zeros(action_dim - len(executed_vec), dtype=np.float32)]
            )

        buf.add(
            frame        = frames_t.cpu(),
            lang_tok     = lang_tok,
            act_hist     = act_hist_in.squeeze(0).cpu(),
            rew_hist     = rew_hist_in.squeeze(0).cpu(),
            act_vec_hist = act_vec_in.squeeze(0).cpu(),     # (H, action_dim)
            prev_a       = prev_a_in.squeeze(0).cpu(),
            prev_r       = prev_r_in.squeeze(0).cpu(),
            state_delta  = state_delta_t.cpu(),
            action       = action,
            reward       = reward,
            done         = done,
        )

        action_q.append(action)
        reward_q.append(reward)
        action_vec_q.append(executed_vec)
        prev_action = action
        prev_rew_fb = reward
        step += 1

    return buf


# ── RL update step ─────────────────────────────────────────────────────────────

def rl_update(
    model:      VERAModel,
    value_head: ValueHead,
    buf:        RolloutBuffer,
    optimizer:  torch.optim.Optimizer,
    cfg:        dict,
    device:     str,
    bc_model:   Optional[VERAModel] = None,
) -> dict:
    """Single REINFORCE + value-baseline update over one rollout buffer."""
    model.train()
    value_head.train()

    returns = buf.compute_returns(gamma=cfg["rl"].get("gamma", 0.99)).to(device)

    frames       = torch.stack(buf.frames).to(device)           # (N, T, 3, H, W)
    lang_tokens  = torch.stack(buf.lang_tokens).to(device)      # (N, 77)
    act_hist     = torch.stack(buf.action_hists).to(device)     # (N, H)
    rew_hist     = torch.stack(buf.reward_hists).to(device)     # (N, H)
    prev_actions = torch.stack(buf.prev_actions).to(device)     # (N, 1) or (N,)
    prev_rewards = torch.stack(buf.prev_rewards_fb).to(device)  # (N, 1) or (N,)
    state_deltas = torch.stack(buf.state_deltas).to(device)     # (N, 1)
    actions      = torch.tensor(buf.actions, dtype=torch.long, device=device)

    # Stack low-level action vector history.
    # buf.action_vec_hists entries are (H, action_dim) tensors (never None at this
    # point — collect_rollout always stores a zero tensor when the env doesn't
    # expose a codebook vector).  We guard with a None-check anyway for safety.
    if any(v is None for v in buf.action_vec_hists):
        _action_dim  = cfg["model"].get("action_dim", 4)
        _history_len = act_hist.size(1)
        act_vec_hist = torch.zeros(
            len(buf.action_vec_hists), _history_len, _action_dim, device=device
        )
        for _i, _v in enumerate(buf.action_vec_hists):
            if _v is not None:
                act_vec_hist[_i] = _v.to(device)
    else:
        act_vec_hist = torch.stack(buf.action_vec_hists).to(device)  # (N, H, action_dim)

    # Flatten (N,1) → (N,)
    prev_actions = prev_actions.view(-1).long()
    prev_rewards = prev_rewards.view(-1).float()
    state_deltas = state_deltas.view(-1).float()

    # Forward pass (with grad)
    out = model(frames, lang_tokens, act_hist, rew_hist, prev_actions, prev_rewards,
                state_delta=state_deltas, action_vec_hist=act_vec_hist)
    _num_act   = cfg["model"]["num_actions"]
    logits     = out["logits"][:, :_num_act]                   # (N, A) — first chunk only
    cls_feat   = out["cls_features"]                           # (N, D)

    # Value estimate (no gradient back through policy for value loss)
    values    = value_head(cls_feat.detach())                   # (N,)
    advantage = returns - values.detach()                       # Â_t = G_t - V(s_t)

    # REINFORCE policy gradient
    log_probs   = F.log_softmax(logits, dim=-1)
    chosen_logp = log_probs[torch.arange(len(actions)), actions]
    policy_loss = -(chosen_logp * advantage).mean()

    # Value baseline MSE
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus
    probs   = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(-1).mean()

    # KL penalty against BC checkpoint (prevents catastrophic forgetting)
    kl_loss = torch.tensor(0.0, device=device)
    if bc_model is not None:
        bc_model.eval()
        with torch.no_grad():
            bc_out    = bc_model(frames, lang_tokens, act_hist, rew_hist,
                                 prev_actions, prev_rewards,
                                 state_delta=state_deltas,
                                 action_vec_hist=act_vec_hist)
            bc_probs  = F.softmax(bc_out["logits"][:, :_num_act], dim=-1)
        kl_loss = F.kl_div(log_probs, bc_probs, reduction="batchmean")

    vf_coef      = cfg["rl"].get("vf_coef", 0.5)
    entropy_coef = cfg["rl"].get("entropy_coef", 0.01)
    kl_coef      = cfg["rl"].get("kl_coef", 0.1)

    # Entropy floor: penalise collapse only when entropy drops below floor.
    # H_max = log(num_actions) = log(8) ≈ 2.08.  Floor = 0.7 * H_max ≈ 1.45.
    # Without this, entropy_coef=0 allows the policy to collapse to a single
    # action with no force to explore — causing return=-0.000 for all rollouts.
    entropy_floor = cfg["rl"].get("entropy_floor", 0.7)     # fraction of H_max
    H_max         = torch.log(torch.tensor(float(_num_act)))
    entropy_penalty = F.relu(entropy_floor * H_max - entropy)  # >0 only when below floor

    total_loss = (policy_loss
                  + vf_coef      * value_loss
                  - entropy_coef * entropy
                  + kl_coef      * kl_loss
                  + entropy_penalty)

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(
        list(p for p in model.parameters() if p.requires_grad)
        + list(value_head.parameters()),
        cfg["rl"].get("grad_clip", 1.0),
    )
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss":  value_loss.item(),
        "entropy":     entropy.item(),
        "kl_loss":     kl_loss.item(),
        "total_loss":  total_loss.item(),
        "mean_return": returns.mean().item(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def rl_train(cfg: dict):
    from envs.sim_env import make_env

    device  = cfg["training"].get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["training"]["output_dir"])
    print(f"[rl_vera] device = {device}")

    # YAML may deserialise scientific notation (e.g. 1e-4) as strings — cast all rl floats now
    cfg["rl"] = {k: float(v) if isinstance(v, str) else v for k, v in cfg["rl"].items()}

    vera_cfg = cfg.get("vera", {})

    # ── Detect chunk_size from checkpoint before building the model ───────────
    # The config yaml may say chunk_size=1 while the actual checkpoint was trained
    # with chunk_size=4 (or vice versa).  We read the LAST action_head weight
    # layer from the checkpoint and infer: chunk_size = out_dim / num_actions.
    _num_actions = cfg["model"]["num_actions"]
    _cfg_chunk   = cfg["model"].get("chunk_size", 1)
    bc_ckpt_path = out_dir / "best_sft_vera.pt"

    def _detect_chunk_size(ckpt_path: Path) -> int:
        if not ckpt_path.exists():
            return _cfg_chunk
        raw = torch.load(ckpt_path, map_location="cpu")
        sd  = raw.get("model_state", raw.get("model", raw))
        ah_layers = [(k, v) for k, v in sd.items()
                     if "action_head" in k and k.endswith(".weight") and "norm" not in k]
        if not ah_layers:
            return _cfg_chunk
        _last_k, _last_v = ah_layers[-1]
        if _last_v.shape[0] % _num_actions == 0:
            detected = _last_v.shape[0] // _num_actions
            if detected != _cfg_chunk:
                print(f"[rl_vera] chunk_size: config={_cfg_chunk} → detected={detected} "
                      f"from '{_last_k}' shape={list(_last_v.shape)}")
            return detected
        return _cfg_chunk

    sft_chunk_size = _detect_chunk_size(bc_ckpt_path)

    # ── TERA-RL architectural fix ─────────────────────────────────────────────
    # Three problems with using the SFT model directly for RL:
    #   1. chunk_size=4: action head predicts 32 logits jointly; RL slices first 8.
    #      The CLS token gradient is therefore wrong — optimising a 8-dim objective
    #      through a head designed as part of a 32-dim joint prediction.
    #   2. Consequence token always encodes "I made little progress" (reward≈0 in RL),
    #      adding a constant bias that hurts action diversity and exploration.
    #   3. Consequence token was trained on oracle verbalised outcomes from demos;
    #      the RL outcome verbalisations are out-of-distribution for that encoder.
    #
    # Fix: build RL model with chunk_size=1 + consequence token DISABLED.
    # Transfer backbone weights exactly; re-initialise ONLY the action head output
    # layer from the first-chunk slice of the SFT head (action_head.8.weight[:8, :]).
    # Freeze backbone; train only the 3-layer action head (~50K params → ~2K params).
    rl_chunk_size = 1   # single-step RL always uses chunk_size=1

    def make_model():
        return VERAModel(
            num_actions=cfg["model"]["num_actions"],
            history_len=cfg["model"]["history_len"],
            num_vis_frames=cfg["model"]["num_vis_frames"],
            fusion_layers=cfg["model"].get("fusion_layers", 6),
            fusion_heads=cfg["model"].get("fusion_heads", 8),
            d_model=cfg["model"].get("d_model", 256),
            d_ff_scale=cfg["model"].get("d_ff_scale", 4),
            dropout=cfg["model"].get("dropout", 0.1),
            vision_token_dropout=float(cfg["model"].get("vision_token_dropout", 0.0)),
            freeze_clip=cfg["model"].get("freeze_clip", True),
            use_lang_feedback=vera_cfg.get("use_lang_feedback", True),
            use_temporal_history=vera_cfg.get("use_temporal_history", True),
            use_reward_gate=vera_cfg.get("use_reward_gate", True),
            use_consequence_token=False,   # TERA-RL: disable — out-of-dist in RL rollouts
            action_dim=cfg["model"].get("action_dim", 4),
            action_vocab=vera_cfg.get("action_vocab"),
            chunk_size=rl_chunk_size,      # TERA-RL: always 1 for single-step RL
        )

    model = make_model().to(device)

    # Load BC checkpoint — transfer backbone exactly, re-init action head output layer
    bc_model = None
    if bc_ckpt_path.exists():
        ckpt    = torch.load(bc_ckpt_path, map_location=device)
        sft_sd  = ckpt.get("model_state", ckpt.get("model", ckpt))
        rl_sd   = model.state_dict()

        # Copy every key that exists in both and has matching shape (backbone).
        # The action head output layer (action_head.8.weight) will mismatch when
        # sft_chunk_size != rl_chunk_size — handle that separately below.
        transferred, skipped = 0, 0
        for k, v in sft_sd.items():
            if k not in rl_sd:
                skipped += 1; continue
            if v.shape != rl_sd[k].shape:
                skipped += 1; continue
            rl_sd[k] = v; transferred += 1

        # Re-initialise action head output layer from first-chunk slice of SFT weights.
        # SFT: action_head.8.weight shape = [num_actions*sft_chunk, d_model]
        # RL:  action_head.8.weight shape = [num_actions,            d_model]
        _ah_out_key = "action_head.8.weight"
        if _ah_out_key in sft_sd and sft_chunk_size > rl_chunk_size:
            sft_w = sft_sd[_ah_out_key]   # [32, 256] for chunk_size=4
            rl_sd[_ah_out_key] = sft_w[:_num_actions].clone()  # take first 8 rows
            print(f"[rl_vera] Action head: transferred first {_num_actions} rows "
                  f"from SFT head (shape {list(sft_w.shape)} → {list(rl_sd[_ah_out_key].shape)})")

        model.load_state_dict(rl_sd)
        print(f"[rl_vera] TERA-RL transfer: {transferred} layers copied, {skipped} reshaped/added")
        print(f"[rl_vera] chunk_size: SFT={sft_chunk_size} → RL={rl_chunk_size}  "
              f"consequence_token=OFF")
        # BC anchor: same TERA-RL architecture (chunk_size=1, no consequence token)
        bc_model = make_model().to(device)
        bc_model.load_state_dict(model.state_dict())   # copy already-transferred weights
        for p in bc_model.parameters():
            p.requires_grad = False
    else:
        print("[rl_vera] Warning: no BC checkpoint — training RL from scratch.")

    # ── Freeze backbone; train action_head + visual domain adapter (TERA-RL) ───
    # 1. Freeze everything first.
    # 2. Unfreeze action_head (task policy, ~265K params).
    # 3. Enable + unfreeze vis_adapter (sim-to-real visual correction, ~262K params).
    #    The adapter sits between vis_proj and the fusion transformer — fixing the
    #    visual feature distribution for simulation images benefits ALL downstream
    #    processing (transformer, cls_token, action head).
    for p in model.parameters():
        p.requires_grad = False
    for p in model.action_head.parameters():
        p.requires_grad = True
    model.enable_visual_adapter()   # zero-init residual adapter, now trainable

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[rl_vera] Trainable: {n_trainable:,} / {n_total:,} params "
          f"(action_head + vis_adapter, backbone frozen)")

    value_head = ValueHead(d_model=cfg["model"].get("d_model", 256)).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(value_head.parameters()),
        lr=cfg["rl"].get("lr", 1e-5),
        weight_decay=cfg["rl"].get("weight_decay", 1e-4),
    )

    env             = make_env(cfg)
    tokenizer_cache = {}
    rl_out_dir      = out_dir / "rl"
    rl_out_dir.mkdir(parents=True, exist_ok=True)

    log, best_return, best_sr = [], -float("inf"), -float("inf")
    cumulative_steps = 0          # total env steps taken — x-axis for sample efficiency curves
    max_ep_steps     = int(cfg["rl"].get("max_episode_steps", 50))
    num_rollouts     = int(cfg["rl"].get("num_rollouts", 4))

    # Early stopping: halt if SR does not improve for `sr_patience` epochs
    sr_patience      = int(cfg["rl"].get("sr_patience", 0))   # 0 = disabled
    epochs_no_sr_imp = 0   # counter

    for epoch in range(1, int(cfg["rl"]["epochs"]) + 1):
        print(f"\n── Epoch {epoch}/{int(cfg['rl']['epochs'])} "
              f"({num_rollouts} rollouts × {max_ep_steps} steps) ──", flush=True)
        epoch_returns, epoch_successes, epoch_lengths = [], [], []

        for ri in range(num_rollouts):
            print(f"  rollout {ri+1}/{num_rollouts} ...", end=" ", flush=True)
            buf     = collect_rollout(model, env, cfg, device, tokenizer_cache)
            metrics = rl_update(model, value_head, buf, optimizer, cfg, device, bc_model)

            # Track env steps for sample efficiency (steps = transitions in this rollout)
            ep_steps = len(buf.actions)
            cumulative_steps += ep_steps

            # Per-episode success: episode is successful if its total undiscounted
            # return exceeds the success threshold (mirrors evaluate_vera logic)
            success_thr = cfg.get("eval", {}).get("success_threshold", 1.0)
            ep_return   = sum(buf.rewards)
            epoch_returns.append(ep_return)
            epoch_successes.append(int(ep_return >= success_thr))
            epoch_lengths.append(ep_steps)
            print(f"steps={ep_steps} return={ep_return:.3f} "
                  f"{'✓' if ep_return >= success_thr else '✗'}", flush=True)

            buf.clear()

        mean_ret     = float(np.mean(epoch_returns))
        mean_success = float(np.mean(epoch_successes))
        mean_len     = float(np.mean(epoch_lengths))

        row = {
            "epoch":           epoch,
            "cumulative_steps": cumulative_steps,
            "mean_return":     round(mean_ret, 4),
            "success_rate":    round(mean_success, 4),
            "mean_ep_length":  round(mean_len, 2),
            **{k: round(v, 5) for k, v in metrics.items()},
        }
        log.append(row)

        print(f"RL Epoch {epoch:3d} | steps {cumulative_steps:7d} | "
              f"return {mean_ret:.4f} | success {mean_success*100:.1f}% | "
              f"policy {metrics['policy_loss']:.4f} | "
              f"entropy {metrics['entropy']:.4f} | "
              f"kl {metrics['kl_loss']:.4f}", flush=True)

        if mean_ret > best_return:
            best_return = mean_ret
            torch.save({
                "epoch":            epoch,
                "cumulative_steps": cumulative_steps,
                "model_state":      model.state_dict(),
            }, rl_out_dir / "best_rl_vera.pt")
            print(f"  ✓ best-return checkpoint saved (return={best_return:.4f} "
                  f"@ {cumulative_steps} steps)")

        # Save separately by task success rate — this is the metric that matters
        if mean_success > best_sr:
            best_sr = mean_success
            epochs_no_sr_imp = 0
            torch.save({
                "epoch":            epoch,
                "cumulative_steps": cumulative_steps,
                "success_rate":     best_sr,
                "model_state":      model.state_dict(),
            }, rl_out_dir / "best_sr_vera.pt")
            print(f"  ★ best-SR checkpoint saved   (SR={best_sr*100:.1f}% "
                  f"@ {cumulative_steps} steps)")
        else:
            epochs_no_sr_imp += 1

        # Early-stopping check
        if sr_patience > 0 and epochs_no_sr_imp >= sr_patience:
            print(f"\n[rl_vera] Early stopping: no SR improvement for "
                  f"{sr_patience} epochs (best SR={best_sr*100:.1f}%). Halting.")
            break

        if epoch % cfg["rl"].get("save_every", 20) == 0:
            torch.save({
                "epoch":            epoch,
                "cumulative_steps": cumulative_steps,
                "model_state":      model.state_dict(),
            }, rl_out_dir / f"rl_vera_epoch{epoch:04d}.pt")

        # Flush log every 10 epochs so results survive crashes
        if epoch % 10 == 0:
            with open(rl_out_dir / "rl_vera_log.json", "w") as f:
                json.dump(log, f, indent=2)

    with open(rl_out_dir / "rl_vera_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Also write a compact sample-efficiency CSV for plotting
    csv_lines = ["epoch,cumulative_steps,mean_return,success_rate"]
    for row in log:
        csv_lines.append(
            f"{row['epoch']},{row['cumulative_steps']},"
            f"{row['mean_return']},{row['success_rate']}"
        )
    (rl_out_dir / "sample_efficiency.csv").write_text("\n".join(csv_lines))

    print(f"[rl_vera] Done. Best mean return: {best_return:.4f} "
          f"| Total env steps: {cumulative_steps:,}")


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rl_train(cfg)
