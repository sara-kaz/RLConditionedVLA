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
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

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

    dataset = TrajectoryDataset(
        episodes,
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"]["num_vis_frames"],
        num_actions=cfg["model"]["num_actions"],
        action_dim=cfg["model"].get("action_dim", 4),
        img_size=cfg["data"].get("img_size", 224),
        device=device,
    )

    val_frac = cfg["training"].get("val_fraction", 0.1)
    n_val    = max(1, int(len(dataset) * val_frac))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

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
    return VERAModel(
        num_actions=cfg["model"]["num_actions"],
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"]["num_vis_frames"],
        fusion_layers=cfg["model"].get("fusion_layers", 6),
        fusion_heads=cfg["model"].get("fusion_heads", 8),
        d_model=cfg["model"].get("d_model", 256),
        d_ff_scale=cfg["model"].get("d_ff_scale", 4),
        dropout=cfg["model"].get("dropout", 0.1),
        freeze_clip=cfg["model"].get("freeze_clip", True),
        use_lang_feedback=vera_cfg.get("use_lang_feedback", True),
        use_temporal_history=vera_cfg.get("use_temporal_history", True),
        use_reward_gate=vera_cfg.get("use_reward_gate", True),
        use_consequence_token=vera_cfg.get("use_consequence_token", True),
        action_dim=cfg["model"].get("action_dim", 4),   # 4=MetaWorld, 2=Language-Table, 7=CALVIN
        action_vocab=vera_cfg.get("action_vocab"),      # None → use built-in vocabulary
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
) -> dict:
    """
    One full pass over *loader*.

    Losses
    ------
    ce       — cross-entropy on discrete action logits          (always)
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

            out = model(frames, lang_tokens, action_hist, reward_hist,
                        action_vec_hist=action_vec_hist)
            logits = out["logits"]                          # (B, A)
            ce     = criterion(logits, target)

            # ── Dual contrastive alignment loss ───────────────────────────────
            align = torch.tensor(0.0, device=device)
            if (out["instr_emb"] is not None
                    and align_coef > 0
                    and is_train):
                prev_reward = reward_hist[:, -1]            # most recent reward
                align = model.compute_alignment_loss(
                    out["instr_emb"],
                    out["action_lang_emb"],
                    prev_reward,
                    out.get("consequence_emb"),             # None if stream disabled
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

def train(cfg: dict):
    device = resolve_device(cfg)
    print(f"[sft_vera] device = {device}")

    train_loader, val_loader = build_dataloaders(cfg, device)
    model = build_model(cfg).to(device)
    print(f"[model] {model.param_summary()}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
        betas=(0.9, 0.98),      # slightly higher β₂ for transformer training
    )

    # Warmup + cosine annealing: warmup for 5% of total steps, then cosine decay
    total_epochs = cfg["training"]["epochs"]
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
                eta_min=cfg["training"].get("lr_min", 1e-6),
            ),
        ],
        milestones=[warmup_epochs],
    )

    criterion   = nn.CrossEntropyLoss(label_smoothing=cfg["training"].get("label_smoothing", 0.05))
    align_coef  = cfg.get("vera", {}).get("alignment_loss_coef", 0.1)
    reg_coef    = cfg.get("vera", {}).get("regression_loss_coef", 0.5)
    grad_clip   = cfg["training"].get("grad_clip", 1.0)
    out_dir     = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    patience         = cfg["training"].get("early_stopping_patience", 10)
    patience_counter = 0
    log, best_val_acc = [], 0.0

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        train_m = run_epoch(
            model, train_loader, optimizer, criterion,
            device, is_train=True,
            grad_clip=grad_clip, align_coef=align_coef, reg_coef=reg_coef,
        )
        val_m = run_epoch(
            model, val_loader, None, criterion,
            device, is_train=False,
            align_coef=0.0, reg_coef=reg_coef,   # reg active at val for monitoring
        )
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
        log.append(row)
        print(f"Epoch {epoch:3d}/{total_epochs} | "
              f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} "
              f"align {row['train_align']:.4f} reg {row['train_reg']:.4f} | "
              f"cos_exp {row['train_cos_exp']:+.3f} cos_rsn {row['train_cos_rsn']:+.3f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.3f} | "
              f"lr {row['lr']:.2e} | {elapsed:.1f}s")

        # Save best checkpoint + early stopping
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     best_val_acc,
                "cfg":         cfg,
            }, out_dir / "best_sft_vera.pt")
            print(f"  ✓ best checkpoint saved (val_acc={best_val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[sft_vera] Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs). "
                      f"Best val acc: {best_val_acc:.3f}")
                break

        # Periodic snapshot
        if epoch % cfg["training"].get("save_every", 10) == 0:
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "cfg":         cfg,
            }, out_dir / f"sft_vera_epoch{epoch:04d}.pt")

    with open(out_dir / "sft_vera_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n[sft_vera] Done. Best val acc: {best_val_acc:.3f}")


# Alias used by run_ablations.py and run_calvin_ablations.py
sft_train = train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
