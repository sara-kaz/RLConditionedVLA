"""
VLLA Ablation Study Runner
==========================
Trains or evaluates all ablation variants of VLLA and produces:
  1. JSON results file with mean ± std per metric per ablation
  2. Markdown table (printed to stdout, suitable for paper)
  3. CSV for plotting

This script supports two modes:
  --mode eval   : load a single trained checkpoint and evaluate all ablations
                  by disabling modules at inference time (fast; ~10 min)
  --mode train  : fully re-train each ablation from scratch (slow; ~hours)
                  Required for a fully rigorous ablation — NeurIPS standard

Usage
-----
# Fast eval-only ablations (uses a pretrained full-model checkpoint)
python scripts/run_ablations.py \\
    --config configs/config.yaml \\
    --checkpoint checkpoints/rl/best_rl_vera.pt \\
    --mode eval --episodes 100 --seeds 5

# Full train ablations (each variant trained independently from scratch)
python scripts/run_ablations.py \\
    --config configs/config.yaml \\
    --mode train --seeds 3
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# ── Ablation definitions ──────────────────────────────────────────────────────
# Each entry: (display_name, vera_config_overrides)
# The "Full VLLA" row uses the config as-is (no overrides).

ABLATIONS = [
    # ── Core contribution ablations ──────────────────────────────────────────
    ("Full VLLA (ours)",
     {}),

    ("A — No language feedback (base VLA)",
     {"use_lang_feedback": False, "use_consequence_token": False}),

    ("B — No temporal history",
     {"use_temporal_history": False}),

    ("C — No reward gate on action token",
     {"use_reward_gate": False}),

    ("D — No contrastive alignment loss",
     {"alignment_loss_coef": 0.0}),

    # ── Consequence token ablations (new contribution) ────────────────────────
    ("F — Action token only (no consequence)",
     {"use_consequence_token": False}),

    ("G — Consequence token only (no action lang)",
     {"use_lang_feedback": False, "use_consequence_token": True}),

    # ── Minimal baseline ─────────────────────────────────────────────────────
    ("E — Minimal (no lang, no history)",
     {"use_lang_feedback": False,
      "use_consequence_token": False,
      "use_temporal_history": False}),
]


# ── Eval-mode ablation ────────────────────────────────────────────────────────

def run_eval_ablations(cfg: dict, checkpoint: str,
                        num_episodes: int, num_seeds: int) -> dict:
    """
    Load the same checkpoint once per ablation, disable components via
    config overrides, and evaluate. Fast but approximate — the shared
    weights were trained with all components enabled.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.evaluate_vera import build_vera_from_cfg, load_checkpoint, evaluate_once

    device  = "cuda"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" \
                 if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
                 else "cpu"
    except Exception:
        pass

    all_results = {}

    for name, overrides in ABLATIONS:
        print(f"\n{'='*60}")
        print(f"  Ablation: {name}")
        print(f"  Overrides: {overrides if overrides else '(none — full model)'}")
        print(f"{'='*60}")

        abl_cfg = copy.deepcopy(cfg)
        for k, v in overrides.items():
            abl_cfg["vera"][k] = v

        # training alignment_loss_coef has no effect at eval; skip model rebuild
        model = build_vera_from_cfg(abl_cfg, device)
        try:
            load_checkpoint(model, checkpoint, device)
        except RuntimeError:
            import torch
            ckpt  = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()

        seed_returns, seed_successes, seed_lengths, seed_entropies = [], [], [], []
        for s in range(num_seeds):
            t0 = time.time()
            res = evaluate_once(model, abl_cfg, num_episodes=num_episodes,
                                deterministic=True, seed=s * 17)
            elapsed = time.time() - t0
            seed_returns.append(res["mean_return"])
            seed_successes.append(res["success_rate"])
            seed_lengths.append(res["mean_length"])
            seed_entropies.append(res["mean_entropy"])
            print(f"  Seed {s+1}/{num_seeds} | return={res['mean_return']:.4f} "
                  f"success={res['success_rate']*100:.1f}% "
                  f"len={res['mean_length']:.1f} ({elapsed:.1f}s)")

        all_results[name] = {
            "overrides": overrides,
            "mean_return":  float(np.mean(seed_returns)),
            "std_return":   float(np.std(seed_returns)),
            "mean_success": float(np.mean(seed_successes)),
            "std_success":  float(np.std(seed_successes)),
            "mean_length":  float(np.mean(seed_lengths)),
            "std_length":   float(np.std(seed_lengths)),
            "mean_entropy": float(np.mean(seed_entropies)),
            "std_entropy":  float(np.std(seed_entropies)),
            "raw_returns":  seed_returns,
            "raw_successes": seed_successes,
        }

    return all_results


# ── Train-mode ablation ───────────────────────────────────────────────────────

def _run_one_seed(
    abl_cfg: dict,
    seed_out: Path,
    seed: int,
    sft_train_fn,
) -> float:
    """
    Train (or resume) a single ablation seed.

    Decision tree for ``seed_out/best_sft_vera.pt``:
      • File does not exist          → train from scratch.
      • File exists, training_complete=True  → skip; return stored val_acc.
      • File exists, training_complete=False → resume from that checkpoint.
    """
    ckpt_path = seed_out / "best_sft_vera.pt"

    if ckpt_path.exists():
        try:
            meta = torch.load(str(ckpt_path), map_location="cpu")
        except Exception as e:
            print(f"  [WARN] Could not read checkpoint ({e}). Retraining from scratch.")
            meta = {}

        if meta.get("training_complete", False):
            stored_acc = float(meta.get("val_acc", 0.0))
            print(f"  [SKIP] seed={seed}  already complete  best_val_acc={stored_acc:.4f}")
            return stored_acc

        stopped_epoch = int(meta.get("epoch", 0))
        stored_acc    = float(meta.get("val_acc", 0.0))
        print(f"  [RESUME] seed={seed}  from epoch {stopped_epoch}  "
              f"best_val_acc so far={stored_acc:.4f}")
        abl_cfg["training"]["seed"] = seed
        return sft_train_fn(abl_cfg, resume_from=str(ckpt_path))

    # ── Fresh run ─────────────────────────────────────────────────────────────
    print(f"  [TRAIN] seed={seed}  starting from scratch")
    torch.manual_seed(seed * 31)
    np.random.seed(seed * 31)
    abl_cfg["training"]["seed"] = seed
    return sft_train_fn(abl_cfg)


def run_train_ablations(cfg: dict, num_seeds: int, out_dir: Path) -> dict:
    """
    Re-train each ablation variant from scratch with ``num_seeds`` seeds.
    Interrupted seeds are **resumed** from their last best checkpoint;
    completed seeds are **skipped**.  This is the fully rigorous approach
    required for a CoRL / NeurIPS ablation study.
    Each variant gets its own output subdirectory.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.sft_trainer_vera import sft_train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results: dict = {}

    # Load partial results so we can append without losing earlier runs.
    json_path = out_dir / "ablation_results.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                all_results = json.load(f)
            print(f"[runner] Loaded partial results from {json_path} "
                  f"({len(all_results)} ablation(s) already stored).")
        except Exception as e:
            print(f"[runner] Could not load partial results ({e}). Starting fresh.")

    # Canonical seed list — fixed across all ablations for comparability.
    SEEDS = [42, 123, 456][:num_seeds]

    for name, overrides in ABLATIONS:
        print(f"\n{'='*60}")
        print(f"  Ablation: {name}")
        if overrides:
            print(f"  Overrides: {overrides}")
        print(f"{'='*60}")
        slug = name.split("—")[0].strip().replace(" ", "_").lower()

        # Check if this ablation is already fully done (all seeds complete).
        if name in all_results:
            stored = all_results[name]
            if stored.get("seeds_complete", 0) >= num_seeds:
                print(f"  [SKIP ABLATION] all {num_seeds} seeds already done.")
                continue

        seed_val_accs = list(all_results.get(name, {}).get("raw_val_accs", []))

        for s_idx, seed in enumerate(SEEDS):
            # Skip seeds already recorded.
            if s_idx < len(seed_val_accs):
                print(f"  [SKIP] seed={seed}  val_acc={seed_val_accs[s_idx]:.4f}  (already in results)")
                continue

            abl_cfg = copy.deepcopy(cfg)
            for k, v in overrides.items():
                abl_cfg["vera"][k] = v

            seed_out = out_dir / slug / f"seed{seed}"
            seed_out.mkdir(parents=True, exist_ok=True)
            abl_cfg["training"]["output_dir"] = str(seed_out)

            best_val_acc = _run_one_seed(abl_cfg, seed_out, seed, sft_train)
            seed_val_accs.append(best_val_acc)

            # Persist after every seed so Colab crashes lose at most one seed.
            all_results[name] = {
                "overrides":       overrides,
                "raw_val_accs":    seed_val_accs,
                "mean_val_acc":    float(np.mean(seed_val_accs)),
                "std_val_acc":     float(np.std(seed_val_accs)),
                "seeds_complete":  len(seed_val_accs),
                # Placeholders for eval metrics (filled in later if RL/eval is run)
                "mean_return":     0.0,
                "std_return":      0.0,
                "mean_success":    0.0,
                "std_success":     0.0,
            }
            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  ✓ seed={seed} done  val_acc={best_val_acc:.4f}  "
                  f"(saved to {json_path})")

        print(f"\n  ── {name}  ({len(seed_val_accs)} seed(s)):  "
              f"val_acc = {np.mean(seed_val_accs):.4f} ± {np.std(seed_val_accs):.4f}")

    return all_results


# ── Table printer ─────────────────────────────────────────────────────────────

def print_markdown_table(results: dict):
    """Print a CoRL-style ablation table in GitHub-flavoured markdown.

    Supports both SFT-only results (``mean_val_acc``) and full results that
    also include simulation return/success metrics (``mean_return``).
    """
    full_acc    = results.get("Full VLLA (ours)", {}).get("mean_val_acc", 0.0)
    has_sim     = any(v.get("mean_return", 0.0) != 0.0 for v in results.values())

    if has_sim:
        full_ret = results.get("Full VLLA (ours)", {}).get("mean_return", 0.0)
        full_suc = results.get("Full VLLA (ours)", {}).get("mean_success", 0.0)
        hdr = (f"| {'Method':<45} | {'Val Acc (mean±std)':>22} "
               f"| {'Return (mean±std)':>20} | {'ΔReturn':>9} |")
        sep = f"|{'-'*47}|{'-'*24}|{'-'*22}|{'-'*11}|"
        print("\n## Table 1 — VERA Ablation Study\n")
        print(hdr)
        print(sep)
        for name, vals in results.items():
            mu_a  = vals.get("mean_val_acc", 0.0)
            std_a = vals.get("std_val_acc", 0.0)
            mu_r  = vals.get("mean_return",  0.0)
            std_r = vals.get("std_return",   0.0)
            dr    = mu_r - full_ret if not name.startswith("Full") else 0.0
            marker = " †" if name.startswith("Full") else ""
            acc_str = f"{mu_a:.3f} ± {std_a:.3f}"
            ret_str = f"{mu_r:.3f} ± {std_r:.3f}"
            dr_str  = f"{dr:+.3f}" if not name.startswith("Full") else "—"
            print(f"| {name+marker:<45} | {acc_str:>22} | {ret_str:>20} | {dr_str:>9} |")
    else:
        hdr = (f"| {'Method':<45} | {'Seeds':>6} "
               f"| {'Val Acc (mean±std)':>22} | {'ΔAcc':>8} |")
        sep = f"|{'-'*47}|{'-'*8}|{'-'*24}|{'-'*10}|"
        print("\n## Table 1 — VERA Ablation Study (SFT, Language-Table)\n")
        print(hdr)
        print(sep)
        for name, vals in results.items():
            n_done = vals.get("seeds_complete", len(vals.get("raw_val_accs", [])))
            mu_a   = vals.get("mean_val_acc", 0.0)
            std_a  = vals.get("std_val_acc",  0.0)
            da     = mu_a - full_acc if not name.startswith("Full") else 0.0
            marker = " †" if name.startswith("Full") else ""
            acc_str = f"{mu_a:.3f} ± {std_a:.3f}"
            da_str  = f"{da:+.3f}" if not name.startswith("Full") else "—"
            print(f"| {name+marker:<45} | {n_done:>6} | {acc_str:>22} | {da_str:>8} |")

    print("\n† proposed method")
    print("Δ = difference from Full VERA (negative = ablation hurts performance)")


def save_csv(results: dict, path: Path):
    lines = ["method,seeds_complete,mean_val_acc,std_val_acc,mean_return,std_return,mean_success,std_success"]
    for name, vals in results.items():
        n = vals.get("seeds_complete", len(vals.get("raw_val_accs", [])))
        lines.append(
            f"\"{name}\",{n},"
            f"{vals.get('mean_val_acc',0.0):.4f},{vals.get('std_val_acc',0.0):.4f},"
            f"{vals.get('mean_return',0.0):.4f},{vals.get('std_return',0.0):.4f},"
            f"{vals.get('mean_success',0.0):.4f},{vals.get('std_success',0.0):.4f}"
        )
    path.write_text("\n".join(lines))
    print(f"CSV saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VERA ablation study runner")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Pretrained VLLA checkpoint (required for --mode eval)")
    parser.add_argument("--mode",       choices=["eval", "train"], default="eval",
                        help="eval=fast inference-time ablation; train=full retrain per variant")
    parser.add_argument("--episodes",   type=int, default=100,
                        help="Episodes per seed per ablation (eval mode only)")
    parser.add_argument("--seeds",      type=int, default=5,
                        help="Number of random seeds to average")
    parser.add_argument("--out",        default="results/ablations",
                        help="Output directory for results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --mode eval")
        results = run_eval_ablations(cfg, args.checkpoint,
                                     num_episodes=args.episodes,
                                     num_seeds=args.seeds)
    else:
        results = run_train_ablations(cfg, num_seeds=args.seeds, out_dir=out_dir)

    # Save JSON
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Print table
    print_markdown_table(results)

    # Save CSV
    save_csv(results, out_dir / "ablation_results.csv")


if __name__ == "__main__":
    main()
