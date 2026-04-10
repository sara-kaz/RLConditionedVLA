"""
VLLA Supervised Fine-Tuning (Behavioural Cloning) Trainer
==========================================================
Phase 1: Train the VLLA model on expert demonstrations.

Loss = cross-entropy(logits, target)
     + alignment_loss_coef * contrastive_alignment_loss(instr_emb, action_lang_emb, reward)

The contrastive alignment loss teaches the model that successful actions
(positive reward) should be semantically close to the task instruction in
CLIP's shared embedding space.

Usage
-----
  python -m training.sft_trainer_vlla --config configs/config.yaml
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.vlla_model import VLLAModel
from data.trajectory_dataset import TrajectoryDataset, load_episodes, make_random_episodes


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
    ep_path = cfg["data"].get("episodes_path")
    if ep_path and Path(ep_path).exists():
        episodes = load_episodes(ep_path)
        print(f"[data] Loaded {len(episodes)} episodes from {ep_path}")
    else:
        print("[data] No dataset found — generating synthetic episodes.")
        episodes = make_random_episodes(
            num_episodes=cfg["data"].get("synthetic_episodes", 200),
            ep_len=cfg["data"].get("ep_len", 30),
            num_actions=cfg["model"]["num_actions"],
        )

    dataset = TrajectoryDataset(
        episodes,
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"]["num_vis_frames"],
        num_actions=cfg["model"]["num_actions"],
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


def build_model(cfg: dict) -> VLLAModel:
    vlla_cfg = cfg.get("vlla", {})
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


# ── Training / validation loop ────────────────────────────────────────────────

def run_epoch(
    model:        VLLAModel,
    loader:       DataLoader,
    optimizer:    Optional[torch.optim.Optimizer],
    criterion:    nn.Module,
    device:       str,
    is_train:     bool,
    grad_clip:    float = 1.0,
    align_coef:   float = 0.1,
) -> dict:
    model.train() if is_train else model.eval()

    total_loss = total_ce = total_align = total_correct = total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            frames      = batch["frames"].to(device)        # (B, T, 3, H, W)
            lang_tokens = batch["lang_tokens"].to(device)   # (B, 77)
            action_hist = batch["action_hist"].to(device)   # (B, H)
            reward_hist = batch["reward_hist"].to(device)   # (B, H)
            target      = batch["target"].to(device)        # (B,)

            out    = model(frames, lang_tokens, action_hist, reward_hist)
            logits = out["logits"]                          # (B, A)
            ce     = criterion(logits, target)

            # Dual contrastive alignment: action alignment + consequence alignment [UPDATED]
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

            loss = ce + align_coef * align

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                optimizer.step()

            B = target.size(0)
            total_loss    += loss.item() * B
            total_ce      += ce.item()   * B
            total_align   += align.item() * B
            total_correct += (logits.argmax(-1) == target).sum().item()
            total_samples += B

    n = max(total_samples, 1)
    return {
        "loss":       total_loss  / n,
        "ce_loss":    total_ce    / n,
        "align_loss": total_align / n,
        "accuracy":   total_correct / n,
    }


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: dict):
    device = resolve_device(cfg)
    print(f"[sft_vlla] device = {device}")

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
    align_coef  = cfg.get("vlla", {}).get("alignment_loss_coef", 0.1)
    grad_clip   = cfg["training"].get("grad_clip", 1.0)
    out_dir     = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log, best_val_acc = [], 0.0

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        train_m = run_epoch(
            model, train_loader, optimizer, criterion,
            device, is_train=True, grad_clip=grad_clip, align_coef=align_coef,
        )
        val_m = run_epoch(
            model, val_loader, None, criterion,
            device, is_train=False, align_coef=0.0,  # no aux loss at val
        )
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch":       epoch,
            "train_loss":  round(train_m["loss"],      4),
            "train_acc":   round(train_m["accuracy"],  4),
            "train_align": round(train_m["align_loss"],4),
            "val_loss":    round(val_m["loss"],        4),
            "val_acc":     round(val_m["accuracy"],    4),
            "lr":          round(scheduler.get_last_lr()[0], 8),
            "time_s":      round(elapsed, 1),
        }
        log.append(row)
        print(f"Epoch {epoch:3d}/{total_epochs} | "
              f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} "
              f"align {row['train_align']:.4f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.3f} | "
              f"lr {row['lr']:.2e} | {elapsed:.1f}s")

        # Save best checkpoint
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     best_val_acc,
                "cfg":         cfg,
            }, out_dir / "best_sft_vlla.pt")
            print(f"  ✓ best checkpoint saved (val_acc={best_val_acc:.3f})")

        # Periodic snapshot
        if epoch % cfg["training"].get("save_every", 10) == 0:
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "cfg":         cfg,
            }, out_dir / f"sft_vlla_epoch{epoch:04d}.pt")

    with open(out_dir / "sft_vlla_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n[sft_vlla] Done. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
