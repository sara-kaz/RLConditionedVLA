"""
smoke_test_minigrid.py
======================
End-to-end smoke test for MiniGrid integration. Verifies:

  1. MiniGridEnv instantiates and produces correct observation shapes.
  2. A 3-episode random rollout completes without errors.
  3. A small episode list can be wrapped in TrajectoryDataset.
  4. The VLA model can do a forward pass on a batch from that dataset.

Run from the repo root:
  python scripts/smoke_test_minigrid.py

No GPU required. Exits 0 on success, 1 on any failure.
"""

import sys
import os

# Ensure repo root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

PASS = "[PASS]"
FAIL = "[FAIL]"


# ── 1. Environment ─────────────────────────────────────────────────────────────

print("\n=== 1. MiniGridEnv basic smoke test ===")
try:
    from envs.minigrid_env import MiniGridEnv

    cfg = {
        "env": {
            "env_id": "MiniGrid-Empty-5x5-v0",
            "instruction": "navigate to the green goal square",
        },
        "data": {"img_size": 64},
        "rl":   {"max_episode_steps": 30},
    }
    env = MiniGridEnv(cfg)
    episodes_raw = []

    for ep_idx in range(3):
        obs = env.reset()
        assert "frame" in obs,       "obs missing 'frame'"
        assert "instruction" in obs, "obs missing 'instruction'"
        assert obs["frame"].ndim == 3,         f"frame must be 3-D, got {obs['frame'].ndim}"
        assert obs["frame"].shape[2] == 3,     f"frame must be HxWx3, got {obs['frame'].shape}"
        assert obs["frame"].dtype == np.uint8, "frame must be uint8"

        frames, actions, rewards = [obs["frame"]], [], []
        done = False
        steps = 0
        while not done and steps < 30:
            action = np.random.randint(0, MiniGridEnv.NUM_ACTIONS)
            obs, reward, done, info = env.step(action)
            frames.append(obs["frame"])
            actions.append(action)
            rewards.append(reward)
            steps += 1

        episodes_raw.append({
            "frames":      np.stack(frames[:-1]),        # (T, H, W, 3)
            "instruction": obs["instruction"],
            "actions":     np.array(actions, dtype=np.int64),
            "rewards":     np.array(rewards, dtype=np.float32),
        })
        print(f"  Episode {ep_idx+1}: {steps} steps | "
              f"return={sum(rewards):.3f} | "
              f"frame shape={obs['frame'].shape}")

    env.close()
    print(f"{PASS} MiniGridEnv works correctly.\n")

except Exception as exc:
    print(f"{FAIL} MiniGridEnv: {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ── 2. TrajectoryDataset ───────────────────────────────────────────────────────

print("=== 2. TrajectoryDataset with MiniGrid episodes ===")
try:
    from data.trajectory_dataset import TrajectoryDataset

    # Pad episodes_raw with synthetic data so dataset is non-trivial
    from data.trajectory_dataset import make_random_episodes
    episodes = episodes_raw + make_random_episodes(
        num_episodes=5, ep_len=10, num_actions=7, img_size=64
    )

    ds = TrajectoryDataset(
        episodes,
        history_len=4,
        num_vis_frames=3,
        num_actions=7,
        img_size=224,   # CLIP ViT-B/32 requires 224x224; 64px causes 5 vs 50 token mismatch
        device="cpu",
    )
    print(f"  Dataset length: {len(ds)} samples")
    assert len(ds) > 0, "Dataset is empty"

    sample = ds[0]
    assert sample["frames"].shape == (3, 3, 224, 224), \
        f"Unexpected frames shape: {sample['frames'].shape}"
    assert sample["action_hist"].shape == (4,), \
        f"Unexpected action_hist shape: {sample['action_hist'].shape}"
    assert sample["target"].item() in range(7), \
        f"Target action out of range: {sample['target'].item()}"
    print(f"  Sample shapes — frames: {tuple(sample['frames'].shape)}, "
          f"action_hist: {tuple(sample['action_hist'].shape)}, "
          f"target: {sample['target'].item()}")
    print(f"{PASS} TrajectoryDataset builds correctly.\n")

except Exception as exc:
    print(f"{FAIL} TrajectoryDataset: {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ── 3. Model forward pass ──────────────────────────────────────────────────────

print("=== 3. VLA model forward pass (CPU, small batch) ===")
try:
    from torch.utils.data import DataLoader
    from models.vla_model import RLConditionedVLA

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = RLConditionedVLA(
        num_actions=7,
        history_len=4,
        fusion_layers=2,    # tiny for speed
        fusion_heads=4,
        dropout=0.0,
        freeze_clip=True,
    )
    model.eval()

    with torch.no_grad():
        logits = model(
            batch["frames"],       # (B, 3, 3, 64, 64)
            batch["lang_tokens"],  # (B, 77)
            batch["action_hist"],  # (B, 4)
            batch["reward_hist"],  # (B, 4)
        )

    assert logits.shape == (2, 7), f"Unexpected logits shape: {logits.shape}"
    print(f"  Logits shape: {tuple(logits.shape)}  (expected: (2, 7))")
    print(f"  Logits sample: {logits[0].tolist()}")
    print(f"{PASS} Model forward pass succeeded.\n")

except Exception as exc:
    print(f"{FAIL} Model forward pass: {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)


print("=" * 50)
print("All smoke tests passed. MiniGrid integration is working.")
print("=" * 50)
