"""
VERA Supervised Fine-Tuning (Behavioural Cloning) Trainer
==========================================================
Phase 1: Train the VERA model on expert demonstrations.

Loss = cross-entropy(logits, target)
     + alignment_loss_coef * contrastive_alignment_loss(instr_emb, action_lang_emb, reward)

The contrastive alignment loss teaches the model that successful actions
(positive reward) should be semantically close to the task instruction in
CLIP's shared embedding space.

Usage
-----
  python -m training.sft_trainer_vera --config configs/config.yaml

Optional EMA (``training.ema_decay`` in YAML): maintains a shadow weight average;
validation and ``best_sft_vera.pt`` can use those weights (``validate_with_ema``)
for better held-out accuracy without freezing CLIP.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models.vera_model import VERAModel
from data.trajectory_dataset import (
    TrajectoryDataset, load_episodes, make_random_episodes,
    load_language_table, load_calvin,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class ModelEMA:
    """
    Exponential moving average of floating-point tensors in ``model.state_dict()``.
    Updating after each train step smooths weights and often improves *validation*
    accuracy without freezing the backbone. Checkpoint saves can persist ``ema_state``.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._keys: list[str] = []
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self._keys.append(k)
                self.shadow[k] = v.detach().clone()
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {k: self.shadow[k].clone() for k in self._keys},
        }

    def load_state_dict(self, sd: Dict[str, Any], model: nn.Module) -> None:
        self.decay = float(sd["decay"])
        dev = next(model.parameters()).device
        for k, v in sd["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(dev))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        cur = model.state_dict()
        for k in self._keys:
            self.shadow[k].mul_(d).add_(cur[k].detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        sd = model.state_dict()
        self._backup = {k: sd[k].detach().clone() for k in self._keys}
        for k in self._keys:
            sd[k].copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            return
        sd = model.state_dict()
        for k in self._keys:
            sd[k].copy_(self._backup[k])
        self._backup = None


def resolve_device(cfg: dict) -> str:
    spec = cfg["training"].get("device", "auto")
    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return spec


def build_dataloaders(cfg: dict, device: str):
    data_cfg  = cfg["data"]
    ep_path   = data_cfg.get("episodes_path")
    dataset_type = data_cfg.get("dataset_type", "pkl").lower()  # "pkl"|"language_table"|"calvin"

    if ep_path and Path(ep_path).exists():
        if dataset_type == "language_table":
            episodes = load_language_table(ep_path)
            print(f"[data] Loaded {len(episodes)} Language-Table episodes from {ep_path}")
        elif dataset_type == "calvin":
            split = data_cfg.get("calvin_split", "training")
            episodes = load_calvin(ep_path, split=split)
            print(f"[data] Loaded {len(episodes)} CALVIN episodes from {ep_path} ({split})")
        else:
            episodes = load_episodes(ep_path)
            print(f"[data] Loaded {len(episodes)} episodes from {ep_path}")
    else:
        print("[data] No dataset found — generating synthetic episodes.")
        episodes = make_random_episodes(
            num_episodes=data_cfg.get("synthetic_episodes", 200),
            ep_len=data_cfg.get("ep_len", 30),
            num_actions=cfg["model"]["num_actions"],
            action_dim=cfg["model"].get("action_dim", 4),
        )

    # ── Episode-level train/val split (fixed seed = reproducible across runs) ──
    # Window-level random_split leaks data: neighbouring windows from the same
    # episode can end up in both splits, inflating val_acc when resuming from
    # a checkpoint (model already trained on adjacent windows).
    # Splitting episodes first and building two separate datasets fully prevents
    # cross-split episode leakage and gives consistent metrics across restarts.
    val_frac = cfg["training"].get("val_fraction", 0.1)
    t_tr = cfg["training"]
    # Episode shuffle seed: explicit split_seed if set; else training.seed so
    # multi-seed Colab runs do not all share the same val episode set.
    if t_tr.get("split_seed", None) is not None:
        split_seed = int(t_tr["split_seed"])
    else:
        split_seed = int(t_tr.get("seed", 42))
    rng = random.Random(split_seed)
    ep_indices = list(range(len(episodes)))
    rng.shuffle(ep_indices)
    n_val_ep  = max(1, int(len(episodes) * val_frac))
    val_idx   = ep_indices[:n_val_ep]
    train_idx = ep_indices[n_val_ep:]
    val_episodes   = [episodes[i] for i in sorted(val_idx)]
    train_episodes = [episodes[i] for i in sorted(train_idx)]
    print(f"[data] Episode split (seed={split_seed}): "
          f"{len(train_episodes)} train / {len(val_episodes)} val episodes")
    train_stride = max(1, int(cfg["data"].get("train_window_stride", 1)))
    if train_stride > 1:
        print(f"[data] train_window_stride={train_stride} (val uses stride 1)")

    ds_kwargs = dict(
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"]["num_vis_frames"],
        num_actions=cfg["model"]["num_actions"],
        action_dim=cfg["model"].get("action_dim", 4),
        img_size=cfg["data"].get("img_size", 224),
        device=device,
        chunk_size=cfg["model"].get("chunk_size", 1),
    )
    # Mild train-only augmentation reduces RGB memorisation; val stays deterministic.
    use_aug = bool(cfg["data"].get("augment_train", True))
    noise_std = float(cfg["data"].get("augment_noise_std", 0.0))
    train_ds = TrajectoryDataset(
        train_episodes,
        **ds_kwargs,
        augment_train=use_aug,
        window_stride=train_stride,
        augment_noise_std=noise_std if use_aug else 0.0,
    )
    val_ds = TrajectoryDataset(
        val_episodes,
        **ds_kwargs,
        augment_train=False,
        window_stride=1,
        augment_noise_std=0.0,
    )

    kw = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=(device != "cpu"),
    )
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
    )


def build_model(cfg: dict) -> VERAModel:
    vera_cfg = cfg.get("vera", {})
    vis_drop = float(cfg["model"].get("vision_token_dropout", 0.0))
    return VERAModel(
        num_actions=cfg["model"]["num_actions"],
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"]["num_vis_frames"],
        fusion_layers=cfg["model"].get("fusion_layers", 6),
        fusion_heads=cfg["model"].get("fusion_heads", 8),
        d_model=cfg["model"].get("d_model", 256),
        d_ff_scale=cfg["model"].get("d_ff_scale", 4),
        dropout=cfg["model"].get("dropout", 0.1),
        vision_token_dropout=float(vis_drop),
        freeze_clip=cfg["model"].get("freeze_clip", True),
        unfreeze_clip_vision=cfg["model"].get("unfreeze_clip_vision", False),
        use_lang_feedback=vera_cfg.get("use_lang_feedback", True),
        use_temporal_history=vera_cfg.get("use_temporal_history", True),
        use_reward_gate=vera_cfg.get("use_reward_gate", True),
        use_consequence_token=vera_cfg.get("use_consequence_token", True),
        action_dim=cfg["model"].get("action_dim", 4),   # 4=MetaWorld, 2=Language-Table, 7=CALVIN
        action_vocab=vera_cfg.get("action_vocab"),      # None → use built-in vocabulary
        chunk_size=cfg["model"].get("chunk_size", 1),   # K=1 → disabled; K=4 → π0-style
    )


# ── Training / validation loop ────────────────────────────────────────────────

def run_epoch(
    model:        VERAModel,
    loader:       DataLoader,
    optimizer:    Optional[torch.optim.Optimizer],
    criterion:    nn.Module,
    device:       str,
    is_train:     bool,
    grad_clip:    float = 1.0,
    align_coef:   float = 0.1,
    reg_coef:     float = 0.5,
    ema:          Optional[ModelEMA] = None,
) -> dict:
    """
    One full pass over *loader*.

    Losses
    ------
    ce       — cross-entropy on discrete action logits          (always)
               When chunk_size > 1, CE is averaged over all K chunk steps:
               ce = mean CE over {t, t+1, ..., t+K-1}.  This gives K×
               the supervised signal and enforces temporal consistency
               (inspired by π0 / GR-1 action chunking).
    align    — dual reward-weighted InfoNCE (experience + reasoning)
    reg      — MSE between predicted action_vec and target_vec  (only when
               the dataset provides continuous action targets, e.g. CALVIN)

    Diagnostics (logged but not optimised)
    ---------------------------------------
    cos_exp  — mean cosine(instr_emb, action_lang_emb)   ∈ [-1, 1]
    cos_rsn  — mean cosine(instr_emb, consequence_emb)   ∈ [-1, 1]

    If either alignment stream is disabled (ablation flags) the corresponding
    cosine value is reported as 0.0 with no contribution to cos_n.
    """
    model.train() if is_train else model.eval()

    total_loss = total_ce = total_align = total_reg = 0.0
    total_correct = total_samples = 0
    total_cos_exp = total_cos_rsn = 0.0
    cos_exp_n = cos_rsn_n = 0

    # Detect chunk_size from model (1 = disabled, no overhead)
    K = getattr(model, "chunk_size", 1)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            frames      = batch["frames"].to(device)        # (B, T, 3, H, W)
            lang_tokens = batch["lang_tokens"].to(device)   # (B, 77)
            action_hist = batch["action_hist"].to(device)   # (B, H)
            reward_hist = batch["reward_hist"].to(device)   # (B, H)
            target      = batch["target"].to(device)        # (B,)

            # Low-level action vector history — present for Language-Table / CALVIN,
            # None for dummy / discrete-only datasets (model degrades gracefully)
            avh = batch.get("action_vec_hist")
            action_vec_hist = avh.to(device) if isinstance(avh, torch.Tensor) else None

            # State delta for Stream 3b consequence verbalization.
            # Provided by Language-Table loader as a pseudo dist-to-goal change
            # derived from consecutive reward differences.  For legacy / synthetic
            # episodes the dataset returns 0.0 scalars, which map to the
            # "stationary" branch in verbalize_consequence (still informative).
            sd = batch.get("state_delta")
            state_delta = sd.to(device) if isinstance(sd, torch.Tensor) else None

            # ── Per-batch reward normalisation ────────────────────────────────
            # Normalise reward_hist to [0, 1] so that:
            #   (a) the reward gate MLP sees a useful dynamic range, and
            #   (b) the InfoNCE exponential weights exp(5·r) have full contrast.
            # We use a running-max normalisation anchored at the batch maximum
            # (not mean-std) so that zero-reward steps stay at 0.
            r_max = reward_hist.max().clamp(min=1e-6)
            reward_hist_norm = (reward_hist / r_max).clamp(0.0, 1.0)

            out = model(frames, lang_tokens, action_hist, reward_hist_norm,
                        state_delta=state_delta,
                        action_vec_hist=action_vec_hist)
            logits = out["logits"]                          # (B, A) — step t only

            # ── Cross-entropy loss (with optional action chunking) ────────────
            # K=1: standard single-step CE on logits (B, A) vs target (B,).
            # K>1: CE averaged over all K chunk steps — K× supervision signal.
            #   logits_chunk (B, K, A) vs target_chunk (B, K)
            #   Reshape to (B*K, A) / (B*K,) so nn.CrossEntropyLoss works.
            if K > 1 and "logits_chunk" in out:
                logits_chunk  = out["logits_chunk"]         # (B, K, A)
                target_chunk  = batch.get("target_chunk")
                if target_chunk is not None:
                    target_chunk = target_chunk.to(device)  # (B, K)
                    B_sz, A = logits.shape
                    ce = criterion(
                        logits_chunk.view(B_sz * K, A),     # (B*K, A)
                        target_chunk.view(B_sz * K),        # (B*K,)
                    )
                else:
                    # Fallback: dataset doesn't supply target_chunk yet
                    ce = criterion(logits, target)
            else:
                ce = criterion(logits, target)

            # ── Dual contrastive alignment loss ───────────────────────────────
            align = torch.tensor(0.0, device=device)
            if (out["instr_emb"] is not None
                    and align_coef > 0
                    and is_train):
                # Use normalised rewards so exp(5·r) gives full contrast [1, 148]
                prev_reward_norm = reward_hist_norm[:, -1]  # most recent (normalised)
                align = model.compute_alignment_loss(
                    out["instr_proj"],          # d_model projected — gradient flows here
                    out["action_lang_proj"],    # d_model projected — gradient flows here
                    prev_reward_norm,
                    out.get("consequence_proj"),  # d_model projected or None
                )

            # ── Continuous action regression loss ─────────────────────────────
            # Only activated when the dataset supplies expert continuous actions.
            # MSE on Tanh-bounded predictions → no explicit range clipping needed.
            reg = torch.tensor(0.0, device=device)
            tvec = batch.get("target_vec")
            if (isinstance(tvec, torch.Tensor)
                    and out.get("action_vec") is not None
                    and reg_coef > 0):
                reg = F.mse_loss(out["action_vec"], tvec.to(device))

            loss = ce + align_coef * align + reg_coef * reg

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            # ── Alignment cosine diagnostic (no gradient) ─────────────────────
            # Tracks whether the alignment loss is actually pulling embeddings
            # closer to the instruction over training.  Rising values = working.
            if out.get("alignment_score") is not None:
                total_cos_exp += out["alignment_score"].detach().mean().item()
                cos_exp_n += 1
            if out.get("consequence_score") is not None:
                total_cos_rsn += out["consequence_score"].detach().mean().item()
                cos_rsn_n += 1

            B = target.size(0)
            total_loss    += loss.item()  * B
            total_ce      += ce.item()    * B
            total_align   += align.item() * B
            total_reg     += reg.item()   * B
            total_correct += (logits.argmax(-1) == target).sum().item()
            total_samples += B

    n = max(total_samples, 1)
    return {
        "loss":       total_loss  / n,
        "ce_loss":    total_ce    / n,
        "align_loss": total_align / n,
        "reg_loss":   total_reg   / n,
        "accuracy":   total_correct / n,
        # Alignment cosine diagnostics — averaged over batches (not samples)
        # so they reflect mean per-batch similarity regardless of batch size.
        "cos_exp":    total_cos_exp / max(cos_exp_n, 1),
        "cos_rsn":    total_cos_rsn / max(cos_rsn_n, 1),
    }


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: dict, resume_from: Optional[str] = None):
    """
    Train VERA with optional warm-restart from a saved checkpoint.

    resume_from : path to a .pt file produced by this trainer.
                  If provided, loads model weights (+ optimizer/scheduler
                  states when available) and continues from the saved epoch.
                  The existing sft_vera_log.json is preserved and appended to.
    """
    device = resolve_device(cfg)
    print(f"[sft_vera] device = {device}")

    t_cfg = cfg["training"]
    run_seed = int(t_cfg.get("seed", 42))
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    train_loader, val_loader = build_dataloaders(cfg, device)
    model = build_model(cfg).to(device)
    print(f"[model] {model.param_summary()}")

    # Cast to float/int — YAML round-trips scientific notation (e.g. 1e-4) as
    # strings in PyYAML 6.x, which causes TypeError in torch.optim.AdamW.
    lr           = float(t_cfg["lr"])
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    total_epochs = int(t_cfg["epochs"])

    # ── Param-group-aware optimizer ──────────────────────────────────────────
    # When CLIP vision is unfrozen (unfreeze_clip_vision=True), use a much
    # smaller LR for those params to avoid destroying pretrained features.
    # clip_vision_lr defaults to 5% of the main LR if not specified.
    clip_vis_params = [p for p in model.clip_model.visual.parameters()
                       if p.requires_grad]
    if clip_vis_params:
        clip_vision_lr  = float(t_cfg.get("clip_vision_lr", lr * 0.05))
        clip_vis_ids    = {id(p) for p in clip_vis_params}
        non_clip_params = [p for p in model.parameters()
                           if p.requires_grad and id(p) not in clip_vis_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": non_clip_params, "lr": lr,            "weight_decay": weight_decay},
                {"params": clip_vis_params,  "lr": clip_vision_lr, "weight_decay": weight_decay * 0.1},
            ],
            betas=(0.9, 0.98),
        )
        print(f"[opt] 2 param groups — backbone {len(non_clip_params)} params "
              f"lr={lr:.2e}, clip-vision {len(clip_vis_params)} params "
              f"lr={clip_vision_lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),      # slightly higher β₂ for transformer training
        )

    # Warmup + cosine annealing: warmup for 5% of total steps, then cosine decay
    warmup_epochs = max(1, int(total_epochs * 0.05))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=float(t_cfg.get("lr_min", 1e-6)),
            ),
        ],
        milestones=[warmup_epochs],
    )

    criterion   = nn.CrossEntropyLoss(label_smoothing=float(t_cfg.get("label_smoothing", 0.05)))
    align_coef  = float(cfg.get("vera", {}).get("alignment_loss_coef", 0.1))
    reg_coef    = float(cfg.get("vera", {}).get("regression_loss_coef", 0.5))
    grad_clip   = float(t_cfg.get("grad_clip", 1.0))
    out_dir     = Path(t_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    patience         = int(t_cfg.get("early_stopping_patience", 10))
    patience_counter = 0
    log, best_val_acc = [], 0.0
    start_epoch = 0
    resume_ckpt: Optional[dict] = None

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if resume_from is not None:
        resume_ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(resume_ckpt["model_state"])
        start_epoch  = int(resume_ckpt.get("epoch", 0))
        best_val_acc = float(resume_ckpt.get("val_acc", 0.0))

        # Restore optimizer state (only available in checkpoints saved after
        # this update; older checkpoints will skip silently)
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            print(f"[resume] Optimizer state restored from checkpoint.")
        else:
            print(f"[resume] No optimizer state in checkpoint — "
                  f"fast-forwarding scheduler {start_epoch} steps.")

        # Restore / reconstruct scheduler
        if "scheduler_state" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        else:
            # Checkpoint pre-dates the resume feature — reconstruct the
            # LR schedule by stepping the scheduler start_epoch times
            # (no data required — purely mathematical).
            for _ in range(start_epoch):
                scheduler.step()

        # Load existing log so we append to it rather than overwrite
        log_path = out_dir / "sft_vera_log.json"
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            # Keep only entries up to start_epoch (discard any partial/corrupt tail)
            log = [r for r in log if r["epoch"] <= start_epoch]

        # Reconstruct patience counter from log tail
        for r in reversed(log):
            if r.get("val_acc", 0.0) < best_val_acc:
                patience_counter += 1
            else:
                break

        print(f"[resume] Continuing from epoch {start_epoch+1}/{total_epochs}  "
              f"best_val_acc={best_val_acc:.4f}  patience_counter={patience_counter}  "
              f"lr={scheduler.get_last_lr()[0]:.3e}")

    ema_decay = float(t_cfg.get("ema_decay", 0.0))
    ema: Optional[ModelEMA] = None
    if 0.0 < ema_decay < 1.0:
        ema = ModelEMA(model, ema_decay)
        if resume_ckpt is not None and resume_ckpt.get("ema_state") is not None:
            ema.load_state_dict(resume_ckpt["ema_state"], model)
            print(f"[ema] Restored shadow from checkpoint (decay={ema.decay}).")
        print(f"[ema] Enabled  decay={ema_decay}  validate_with_ema="
              f"{bool(t_cfg.get('validate_with_ema', True))}")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()

        train_m = run_epoch(
            model, train_loader, optimizer, criterion,
            device, is_train=True,
            grad_clip=grad_clip, align_coef=align_coef, reg_coef=reg_coef,
            ema=ema,
        )
        use_ema_val = ema is not None and bool(t_cfg.get("validate_with_ema", True))
        if use_ema_val:
            ema.apply_to(model)
        try:
            val_m = run_epoch(
                model, val_loader, None, criterion,
                device, is_train=False,
                align_coef=0.0, reg_coef=reg_coef,   # reg active at val for monitoring
            )
        finally:
            if use_ema_val:
                ema.restore(model)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch":           epoch,
            "train_loss":      round(train_m["loss"],      4),
            "train_acc":       round(train_m["accuracy"],  4),
            "train_align":     round(train_m["align_loss"],4),
            "train_reg":       round(train_m["reg_loss"],  4),
            "train_cos_exp":   round(train_m["cos_exp"],   4),
            "train_cos_rsn":   round(train_m["cos_rsn"],   4),
            "val_loss":        round(val_m["loss"],        4),
            "val_acc":         round(val_m["accuracy"],    4),
            "val_reg":         round(val_m["reg_loss"],    4),
            "val_cos_exp":     round(val_m["cos_exp"],     4),
            "val_cos_rsn":     round(val_m["cos_rsn"],     4),
            "lr":              round(scheduler.get_last_lr()[0], 8),
            "time_s":          round(elapsed, 1),
        }
        row["val_ema"] = bool(use_ema_val)
        log.append(row)
        val_tag = "val(EMA)" if use_ema_val else "val"
        print(f"Epoch {epoch:3d}/{total_epochs} | "
              f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} "
              f"align {row['train_align']:.4f} reg {row['train_reg']:.4f} | "
              f"cos_exp {row['train_cos_exp']:+.3f} cos_rsn {row['train_cos_rsn']:+.3f} | "
              f"{val_tag} loss {row['val_loss']:.4f} acc {row['val_acc']:.3f} | "
              f"lr {row['lr']:.2e} | {elapsed:.1f}s")

        # ── Incremental log write (survives disconnection) ────────────────────
        # Write after every epoch so no progress is lost if Colab disconnects.
        with open(out_dir / "sft_vera_log.json", "w") as f:
            json.dump(log, f, indent=2)

        # ── Save best checkpoint (with full resume state) ─────────────────────
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            patience_counter = 0
            sd_model = copy.deepcopy(model.state_dict())
            if use_ema_val and ema is not None:
                ema.apply_to(model)
                sd_model = copy.deepcopy(model.state_dict())
                ema.restore(model)
            best_payload = {
                "epoch":            epoch,
                "model_state":      sd_model,
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "val_acc":          best_val_acc,
                "cfg":              cfg,
            }
            if ema is not None:
                best_payload["ema_state"] = ema.state_dict()
            torch.save(best_payload, out_dir / "best_sft_vera.pt")
            print(f"  ✓ best checkpoint saved (val_acc={best_val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[sft_vera] Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs). "
                      f"Best val acc: {best_val_acc:.3f}")
                break

        # ── Periodic snapshot ─────────────────────────────────────────────────
        if epoch % cfg["training"].get("save_every", 10) == 0:
            snap = {
                "epoch":            epoch,
                "model_state":      model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "cfg":              cfg,
            }
            if ema is not None:
                snap["ema_state"] = ema.state_dict()
            torch.save(snap, out_dir / f"sft_vera_epoch{epoch:04d}.pt")

    print(f"\n[sft_vera] Done. Best val acc: {best_val_acc:.3f}")

    # ── Mark best checkpoint as training-complete ─────────────────────────────
    # The ablation runner reads this flag to distinguish a finished run (skip)
    # from an interrupted one (resume).  We patch it in-place so the best
    # model weights are preserved exactly as saved during training.
    best_ckpt_path = out_dir / "best_sft_vera.pt"
    if best_ckpt_path.exists():
        try:
            payload = torch.load(best_ckpt_path, map_location="cpu")
            payload["training_complete"] = True
            torch.save(payload, best_ckpt_path)
        except Exception as e:
            print(f"[sft_vera] Warning: could not mark checkpoint complete: {e}")

    return best_val_acc


# Alias used by run_ablations.py and run_calvin_ablations.py
sft_train = train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume_from", default=None,
                        help="Path to a best_sft_vera.pt checkpoint to resume from.")
    args = parser.parse_args()
    train(load_config(args.config), resume_from=args.resume_from)
