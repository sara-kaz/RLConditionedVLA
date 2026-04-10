"""
VLLA Online RL Fine-Tuning Trainer
====================================
Phase 2: roll out the VLLA policy in an environment and optimise using
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
  python -m training.rl_trainer_vlla --config configs/config.yaml
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

from models.vlla_model import VLLAModel


# ── Value head ────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """Scalar state-value estimate from the CLS token representation."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        """cls_features: (B, D) → values: (B,)"""
        return self.net(cls_features).squeeze(-1)


# ── Rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one batch of episode transitions for a policy-gradient update."""

    def __init__(self):
        self.frames:          List[torch.Tensor] = []
        self.lang_tokens:     List[torch.Tensor] = []
        self.action_hists:    List[torch.Tensor] = []
        self.reward_hists:    List[torch.Tensor] = []
        self.prev_actions:    List[torch.Tensor] = []   # (1,) int64 — for semantic feedback
        self.prev_rewards_fb: List[torch.Tensor] = []   # (1,) float — for reward gate
        self.state_deltas:    List[torch.Tensor] = []   # (1,) float — signed dist-to-goal delta [NEW]
        self.actions:         List[int]           = []
        self.rewards:         List[float]         = []
        self.dones:           List[bool]          = []

    def add(self, frame, lang_tok, act_hist, rew_hist, prev_a, prev_r, state_delta, action, reward, done):
        self.frames.append(frame)
        self.lang_tokens.append(lang_tok)
        self.action_hists.append(act_hist)
        self.reward_hists.append(rew_hist)
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
    model:            VLLAModel,
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

    obs          = env.reset()
    frame_q      = deque(maxlen=num_vis_frames)
    action_q     = deque([num_actions] * history_len, maxlen=history_len)  # padded
    reward_q     = deque([0.0]         * history_len, maxlen=history_len)
    prev_action  = num_actions   # null action at start of episode
    prev_rew_fb  = 0.0

    # Tokenise instruction
    inst = obs["instruction"]
    if inst not in tokenizer_cache:
        tokenizer_cache[inst] = clip.tokenize([inst])[0]
    lang_tok = tokenizer_cache[inst]

    done, step = False, 0
    while not done and step < max_steps:
        # Build frame tensor
        frame_t = transform(PILImage.fromarray(obs["frame"]))   # (3, H, W)
        frame_q.append(frame_t)
        pad = num_vis_frames - len(frame_q)
        frames_t = torch.stack([torch.zeros_like(frame_t)] * pad + list(frame_q))  # (T,3,H,W)

        frames_in   = frames_t.unsqueeze(0).to(device)                              # (1,T,3,H,W)
        lang_in     = lang_tok.unsqueeze(0).to(device)                              # (1,77)
        act_hist_in = torch.tensor(list(action_q), dtype=torch.long).unsqueeze(0).to(device)
        rew_hist_in = torch.tensor(list(reward_q), dtype=torch.float32).unsqueeze(0).to(device)
        prev_a_in   = torch.tensor([prev_action], dtype=torch.long).to(device)      # (1,)
        prev_r_in   = torch.tensor([prev_rew_fb], dtype=torch.float32).to(device)   # (1,)

        model.eval()
        with torch.no_grad():
            out = model(frames_in, lang_in, act_hist_in, rew_hist_in, prev_a_in, prev_r_in)
        action = torch.multinomial(F.softmax(out["logits"], dim=-1), 1).item()

        obs, reward, done, info = env.step(action)

        # Extract signed distance delta if the env exposes it; else None → reward-only fallback
        _delta = info.get("dist_delta") if isinstance(info, dict) else None
        state_delta_t = torch.tensor(
            [_delta if _delta is not None else 0.0], dtype=torch.float32
        )

        buf.add(
            frame       = frames_t.cpu(),
            lang_tok    = lang_tok,
            act_hist    = act_hist_in.squeeze(0).cpu(),
            rew_hist    = rew_hist_in.squeeze(0).cpu(),
            prev_a      = prev_a_in.squeeze(0).cpu(),       # (1,) int
            prev_r      = prev_r_in.squeeze(0).cpu(),       # (1,) float
            state_delta = state_delta_t.cpu(),              # (1,) float [NEW]
            action      = action,
            reward      = reward,
            done        = done,
        )

        action_q.append(action)
        reward_q.append(reward)
        prev_action = action
        prev_rew_fb = reward
        step += 1

    return buf


# ── RL update step ─────────────────────────────────────────────────────────────

def rl_update(
    model:      VLLAModel,
    value_head: ValueHead,
    buf:        RolloutBuffer,
    optimizer:  torch.optim.Optimizer,
    cfg:        dict,
    device:     str,
    bc_model:   Optional[VLLAModel] = None,
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
    state_deltas = torch.stack(buf.state_deltas).to(device)     # (N, 1) [NEW]
    actions      = torch.tensor(buf.actions, dtype=torch.long, device=device)

    # Flatten (N,1) → (N,)
    prev_actions = prev_actions.view(-1).long()
    prev_rewards = prev_rewards.view(-1).float()
    state_deltas = state_deltas.view(-1).float()                # (N,) [NEW]

    # Forward pass (with grad)
    out = model(frames, lang_tokens, act_hist, rew_hist, prev_actions, prev_rewards,
                state_delta=state_deltas)
    logits     = out["logits"]                                  # (N, A)
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
                                 prev_actions, prev_rewards, state_delta=state_deltas)
            bc_probs  = F.softmax(bc_out["logits"], dim=-1)
        kl_loss = F.kl_div(log_probs, bc_probs, reduction="batchmean")

    vf_coef      = cfg["rl"].get("vf_coef", 0.5)
    entropy_coef = cfg["rl"].get("entropy_coef", 0.01)
    kl_coef      = cfg["rl"].get("kl_coef", 0.1)

    total_loss = (policy_loss
                  + vf_coef      * value_loss
                  - entropy_coef * entropy
                  + kl_coef      * kl_loss)

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
    from envs.sim_env import SimEnv

    device  = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["training"]["output_dir"])
    print(f"[rl_vlla] device = {device}")

    vlla_cfg = cfg.get("vlla", {})

    def make_model():
        return VLLAModel(
            num_actions=cfg["model"]["num_actions"],
            history_len=cfg["model"]["history_len"],
            num_vis_frames=cfg["model"]["num_vis_frames"],
            fusion_layers=cfg["model"].get("fusion_layers", 6),
            fusion_heads=cfg["model"].get("fusion_heads", 8),
            d_model=cfg["model"].get("d_model", 256),
            d_ff_scale=cfg["model"].get("d_ff_scale", 4),
            dropout=cfg["model"].get("dropout", 0.1),
            freeze_clip=cfg["model"].get("freeze_clip", True),
            use_lang_feedback=vlla_cfg.get("use_lang_feedback", True),
            use_temporal_history=vlla_cfg.get("use_temporal_history", True),
            use_reward_gate=vlla_cfg.get("use_reward_gate", True),
            use_consequence_token=vlla_cfg.get("use_consequence_token", True),
        )

    model = make_model().to(device)

    # Load BC checkpoint
    bc_ckpt_path = out_dir / "best_sft_vlla.pt"
    bc_model     = None
    if bc_ckpt_path.exists():
        ckpt = torch.load(bc_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[rl_vlla] Loaded BC checkpoint from {bc_ckpt_path}")
        bc_model = make_model().to(device)
        bc_model.load_state_dict(ckpt["model_state"])
        for p in bc_model.parameters():
            p.requires_grad = False
    else:
        print("[rl_vlla] Warning: no BC checkpoint — training RL from scratch.")

    value_head = ValueHead(d_model=cfg["model"].get("d_model", 256)).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(value_head.parameters()),
        lr=cfg["rl"].get("lr", 1e-5),
        weight_decay=cfg["rl"].get("weight_decay", 1e-4),
    )

    env             = SimEnv(cfg)
    tokenizer_cache = {}
    rl_out_dir      = out_dir / "rl"
    rl_out_dir.mkdir(parents=True, exist_ok=True)

    log, best_return = [], -float("inf")
    cumulative_steps = 0          # total env steps taken — x-axis for sample efficiency curves
    max_ep_steps     = cfg["rl"].get("max_episode_steps", 50)
    num_rollouts     = cfg["rl"].get("num_rollouts", 4)

    for epoch in range(1, cfg["rl"]["epochs"] + 1):
        epoch_returns, epoch_successes, epoch_lengths = [], [], []

        for _ in range(num_rollouts):
            buf     = collect_rollout(model, env, cfg, device, tokenizer_cache)
            metrics = rl_update(model, value_head, buf, optimizer, cfg, device, bc_model)

            # Track env steps for sample efficiency (steps = transitions in this rollout)
            ep_steps = len(buf.actions)
            cumulative_steps += ep_steps

            # Per-episode success: episode is successful if its total undiscounted
            # return exceeds the success threshold (mirrors evaluate_vlla logic)
            success_thr = cfg.get("eval", {}).get("success_threshold", 1.0)
            ep_return   = sum(buf.rewards)
            epoch_returns.append(ep_return)
            epoch_successes.append(int(ep_return >= success_thr))
            epoch_lengths.append(ep_steps)

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
              f"kl {metrics['kl_loss']:.4f}")

        if mean_ret > best_return:
            best_return = mean_ret
            torch.save({
                "epoch":            epoch,
                "cumulative_steps": cumulative_steps,
                "model_state":      model.state_dict(),
            }, rl_out_dir / "best_rl_vlla.pt")
            print(f"  ✓ best RL checkpoint saved (return={best_return:.4f} "
                  f"@ {cumulative_steps} steps)")

        if epoch % cfg["rl"].get("save_every", 20) == 0:
            torch.save({
                "epoch":            epoch,
                "cumulative_steps": cumulative_steps,
                "model_state":      model.state_dict(),
            }, rl_out_dir / f"rl_vlla_epoch{epoch:04d}.pt")

        # Flush log every 10 epochs so results survive crashes
        if epoch % 10 == 0:
            with open(rl_out_dir / "rl_vlla_log.json", "w") as f:
                json.dump(log, f, indent=2)

    with open(rl_out_dir / "rl_vlla_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Also write a compact sample-efficiency CSV for plotting
    csv_lines = ["epoch,cumulative_steps,mean_return,success_rate"]
    for row in log:
        csv_lines.append(
            f"{row['epoch']},{row['cumulative_steps']},"
            f"{row['mean_return']},{row['success_rate']}"
        )
    (rl_out_dir / "sample_efficiency.csv").write_text("\n".join(csv_lines))

    print(f"[rl_vlla] Done. Best mean return: {best_return:.4f} "
          f"| Total env steps: {cumulative_steps:,}")


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rl_train(cfg)
