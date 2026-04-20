"""
VERA — CALVIN Ablation Runner
==============================
Runs all 9 ablation variants × 3 seeds sequentially on CALVIN D→D.
Matches the Language-Table ablation protocol exactly for fair comparison.

Usage
-----
  # Set your CALVIN dataset path first:
  python scripts/run_calvin_ablations.py --calvin_path /data/calvin/task_D_D

  # Optional: run a single ablation to test before full run
  python scripts/run_calvin_ablations.py --calvin_path /path/to/task_D_D --dry_run

  # Resume from a specific ablation index (0-8) if a run was interrupted
  python scripts/run_calvin_ablations.py --calvin_path /path/to/task_D_D --start_from 3

Seeds used: 42, 123, 456  (same as partner's Language-Table runs)

Ablation variants (9 total):
  0  Full VERA           — all streams active
  1  BC/SFT baseline     — no language feedback, no history transformer
  2  No E_exp            — Stream 3b off (consequence token disabled)
  3  No E_act            — Stream 3a off (action narration disabled)
  4  No lang feedback    — both 3a and 3b off
  5  No history TF       — TemporalHistoryTransformer off
  6  No reward gate      — σ(MLP(r)) gate on action token off
  7  No dual head        — regression branch off (regression_loss_coef=0)
  8  Corrupted conseq    — consequence token receives random unrelated text
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Ablation definitions ─────────────────────────────────────────────────────
# (name, vera_overrides, training_overrides)
# Matches the 9 ablations agreed with partner for cross-environment comparison.

ABLATIONS = [
    (
        "full_vera",
        "Full VERA (all streams)",
        {},                                                     # vera overrides
        {},                                                     # training overrides
    ),
    (
        "bc_baseline",
        "BC/SFT baseline (no lang feedback, no temporal TF)",
        {"use_lang_feedback": False,
         "use_consequence_token": False,
         "use_temporal_history": False},
        {},
    ),
    (
        "no_exp",
        "No E_exp — Stream 3b off",
        {"use_consequence_token": False},
        {},
    ),
    (
        "no_act",
        "No E_act — Stream 3a off",
        {"use_lang_feedback": False},
        {},
    ),
    (
        "no_lang",
        "No language feedback — both 3a and 3b off",
        {"use_lang_feedback": False, "use_consequence_token": False},
        {},
    ),
    (
        "no_history_tf",
        "No temporal history transformer",
        {"use_temporal_history": False},
        {},
    ),
    (
        "no_reward_gate",
        "No reward gate on action token",
        {"use_reward_gate": False},
        {},
    ),
    (
        "no_dual_head",
        "No regression head (discrete only)",
        {"regression_loss_coef": 0.0},
        {},
    ),
    (
        "corrupted_conseq",
        "Corrupted consequence — random unrelated text",
        {},
        {"corrupt_consequence": True},   # handled in dataset / trainer
    ),
]

SEEDS = [42, 123, 456]


# ── Corrupt consequence helper ────────────────────────────────────────────────

def apply_corrupted_consequence(cfg: dict) -> dict:
    """
    Patch config so the consequence encoder receives random unrelated text
    instead of verbalize_consequence() output.
    Implemented by monkey-patching verbalize_consequence at runtime.
    """
    import models.vera_model as vm

    _RANDOM_PHRASES = [
        "The weather is sunny today",
        "A bicycle is parked outside",
        "The meeting starts at noon",
        "There are three apples on the table",
        "The train arrives at platform 4",
        "Music plays in the background",
        "A book is open on the desk",
        "The temperature dropped last night",
    ]
    rng = np.random.default_rng(seed=0)

    def _corrupted(*args, **kwargs):
        return rng.choice(_RANDOM_PHRASES)

    vm.verbalize_consequence = _corrupted
    print("  [corrupt] verbalize_consequence → random unrelated text")
    return cfg


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(calvin_path: str, base_cfg_path: str,
            out_root: str, start_from: int, dry_run: bool):

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.sft_trainer_vera import sft_train

    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    # Inject the CALVIN dataset path
    base_cfg["data"]["episodes_path"] = calvin_path

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}   # ablation_slug → {seed → best_val_acc}

    total_runs = len(ABLATIONS) * len(SEEDS)
    run_idx    = 0

    for abl_idx, (slug, display, vera_ov, train_ov) in enumerate(ABLATIONS):

        if abl_idx < start_from:
            print(f"[skip] {display}")
            run_idx += len(SEEDS)
            continue

        summary[slug] = {}

        for seed in SEEDS:
            run_idx += 1
            print(f"\n{'='*65}")
            print(f"  Run {run_idx}/{total_runs} | {display} | seed={seed}")
            print(f"{'='*65}")

            cfg = copy.deepcopy(base_cfg)

            # Apply vera overrides
            for k, v in vera_ov.items():
                cfg["vera"][k] = v

            # Apply training overrides (except corrupt_consequence — handled below)
            for k, v in train_ov.items():
                if k != "corrupt_consequence":
                    cfg["training"][k] = v

            # Set seed
            cfg["training"]["seed"] = seed

            # Output dir: checkpoints/calvin/<slug>/seed<seed>/
            run_out = out_root / slug / f"seed{seed}"
            run_out.mkdir(parents=True, exist_ok=True)
            cfg["training"]["output_dir"] = str(run_out)

            if dry_run:
                cfg["training"]["epochs"] = 2
                cfg["data"]["synthetic_episodes"] = 20
                print(f"  [dry_run] epochs=2, synthetic data")

            # Monkey-patch corrupted consequence if needed
            if train_ov.get("corrupt_consequence"):
                apply_corrupted_consequence(cfg)

            # Set random seeds
            torch.manual_seed(seed)
            np.random.seed(seed)

            t0 = time.time()
            sft_train(cfg)
            elapsed = time.time() - t0

            # Read best val acc from saved log
            log_path = run_out / "sft_vera_log.json"
            best_val = 0.0
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                if log:
                    best_val = max(row["val_acc"] for row in log)

            summary[slug][f"seed{seed}"] = best_val
            print(f"  Done in {elapsed/60:.1f} min | best val acc: {best_val:.4f}")

        # Print per-ablation summary across seeds so far
        vals  = list(summary[slug].values())
        mu    = float(np.mean(vals))
        std   = float(np.std(vals))
        print(f"\n  [{display}]  mean={mu:.4f}  std={std:.4f}  "
              f"({', '.join(f'{v:.4f}' for v in vals)})")

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  CALVIN ABLATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Ablation':<45} {'Mean':>7} {'Std':>7} {'Seeds'}")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*20}")
    for slug, display, _, _ in ABLATIONS:
        if slug not in summary:
            continue
        vals = list(summary[slug].values())
        mu   = float(np.mean(vals))
        std  = float(np.std(vals))
        seed_str = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  {display:<45} {mu:.4f}  {std:.4f}  {seed_str}")

    # Save summary JSON
    results_path = out_root / "calvin_ablation_summary.json"
    full_summary = {}
    for slug, display, vera_ov, train_ov in ABLATIONS:
        if slug not in summary:
            continue
        vals = list(summary[slug].values())
        full_summary[slug] = {
            "display":   display,
            "vera_overrides": vera_ov,
            "seeds":     summary[slug],
            "mean_val_acc": float(np.mean(vals)),
            "std_val_acc":  float(np.std(vals)),
        }
    with open(results_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\n  Results saved: {results_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VERA CALVIN ablation runner")
    parser.add_argument(
        "--calvin_path", required=True,
        help="Path to CALVIN task_D_D root (contains training/ and validation/ dirs)",
    )
    parser.add_argument(
        "--config", default="configs/calvin_config.yaml",
        help="CALVIN config file (default: configs/calvin_config.yaml)",
    )
    parser.add_argument(
        "--out", default="checkpoints/calvin",
        help="Root output directory for all runs",
    )
    parser.add_argument(
        "--start_from", type=int, default=0,
        help="Skip ablations before this index (0-8) — useful for resuming",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run 2 epochs on synthetic data to verify everything works before full run",
    )
    args = parser.parse_args()

    run_all(
        calvin_path  = args.calvin_path,
        base_cfg_path= args.config,
        out_root     = args.out,
        start_from   = args.start_from,
        dry_run      = args.dry_run,
    )


if __name__ == "__main__":
    main()
