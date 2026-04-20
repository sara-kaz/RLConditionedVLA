"""
TrajectoryDataset
=================
Loads robot trajectories stored as a list of episode dicts.

Episode format (minimum required):
  episode = {
      "frames"      : np.ndarray  (T_ep, H, W, 3)   uint8
      "instruction" : str
      "actions"     : np.ndarray  (T_ep,)            int64   discrete index
      "rewards"     : np.ndarray  (T_ep,)            float32
  }

Episode format (with low-level action vectors — recommended):
  episode = {
      ...same as above...
      "action_vectors": np.ndarray (T_ep, action_dim) float32
                        The actual continuous robot command executed at each step.
                        action_dim depends on dataset / robot:
                          MetaWorld      : 4   [Δx, Δy, Δz, gripper]
                          Language-Table : 2   [Δx, Δy]  (planar EE delta)
                          CALVIN         : 7   [x,y,z,roll,pitch,yaw,gripper]
                        If absent, action_vec_hist is returned as None and the
                        VERA model's history encoder falls back to discrete-only.
  }

Each __getitem__ returns one training window centred on timestep t:
  frames         : (num_vis_frames, 3, 224, 224)
  lang_tokens    : (77,)
  action_hist    : (history_len,)              int64   — discrete indices
  reward_hist    : (history_len,)              float32
  action_vec_hist: (history_len, action_dim)   float32  OR  None
  target         : int                         — discrete action label at t
  target_vec     : (action_dim,)               float32  — continuous action at t (OR None)

Dataset loaders
---------------
  load_episodes(path)                   — generic .pkl
  load_language_table(root)             — Language-Table (Lynch et al. 2023)
  load_calvin(root, split)              — CALVIN (Mees et al. 2022)
  make_random_episodes(...)             — synthetic, for unit tests
"""

import pickle
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import clip
from PIL import Image
import torchvision.transforms as T


# ── Synthetic episodes (unit testing) ─────────────────────────────────────────

def make_random_episodes(
    num_episodes: int  = 50,
    ep_len:       int  = 30,
    num_actions:  int  = 8,
    action_dim:   int  = 4,
    img_size:     int  = 64,
) -> List[Dict]:
    """
    Generate random synthetic episodes for pipeline testing.
    Includes action_vectors so the full VERA history encoder is exercised.
    """
    instructions = [
        "pick up the red cube",
        "move to the left side",
        "push the block forward",
        "grasp the cylinder",
    ]
    episodes = []
    for _ in range(num_episodes):
        T_ep = random.randint(ep_len // 2, ep_len)
        episodes.append({
            "frames":         np.random.randint(0, 255, (T_ep, img_size, img_size, 3), dtype=np.uint8),
            "instruction":    random.choice(instructions),
            "actions":        np.random.randint(0, num_actions, (T_ep,), dtype=np.int64),
            "rewards":        np.random.uniform(-1.0, 1.0, (T_ep,)).astype(np.float32),
            "action_vectors": np.random.uniform(-1.0, 1.0, (T_ep, action_dim)).astype(np.float32),
        })
    return episodes


# ── Language-Table loader (Lynch et al., 2023) ────────────────────────────────

def load_language_table(root: str) -> List[Dict]:
    """
    Load Language-Table episodes from the standard directory layout:

      root/
        episode_000/
          steps.pkl   — list of step dicts: {obs, action, reward, discount}
        episode_001/
          ...

    Action format: 2-DoF planar end-effector delta [Δx, Δy] ∈ [-1, 1]²
    (action_dim = 2).

    If the dataset uses the RLDS / TFRecord format, convert first with:
      pip install tensorflow tensorflow-datasets
      tfds build language_table --data_dir ./language_table_data

    Returns list of episode dicts compatible with TrajectoryDataset.
    """
    root = Path(root)
    episodes = []

    for ep_dir in sorted(root.glob("episode_*")):
        steps_path = ep_dir / "steps.pkl"
        if not steps_path.exists():
            continue
        with open(steps_path, "rb") as f:
            steps = pickle.load(f)   # list of step dicts

        frames      = []
        actions     = []
        rewards     = []
        action_vecs = []
        instruction = steps[0].get("instruction", "complete the task")

        for step in steps:
            obs = step.get("obs", {})
            # RGB frame — try common key names
            frame = obs.get("rgb", obs.get("image", obs.get("pixels", None)))
            if frame is None:
                continue

            action = step.get("action", np.zeros(2, dtype=np.float32))
            action = np.asarray(action, dtype=np.float32).flatten()

            # Discretise 2-DoF [Δx, Δy] continuous action into 8 directional bins:
            #   0=right, 1=up-right, 2=up, 3=up-left,
            #   4=left,  5=down-left, 6=down, 7=down-right
            # Uses angle quantisation so all 8 bins are reachable.
            if len(action) >= 2 and (action[0] != 0 or action[1] != 0):
                angle = np.arctan2(action[1], action[0])          # [-π, π]
                action_idx = int(round(angle / (np.pi / 4))) % 8  # 8 equal sectors
            else:
                action_idx = 0

            frames.append(np.asarray(frame, dtype=np.uint8))
            actions.append(action_idx)
            rewards.append(float(step.get("reward", 0.0)))
            action_vecs.append(action[:2])   # Language-Table: 2-DoF [Δx, Δy]

        if len(frames) < 2:
            continue

        episodes.append({
            "frames":         np.stack(frames),
            "instruction":    instruction,
            "actions":        np.array(actions, dtype=np.int64),
            "rewards":        np.array(rewards, dtype=np.float32),
            "action_vectors": np.stack(action_vecs).astype(np.float32),
        })

    print(f"[Language-Table] Loaded {len(episodes)} episodes from {root}")
    return episodes


# ── CALVIN loader (Mees et al., 2022) ─────────────────────────────────────────

def load_calvin(root: str, split: str = "training") -> List[Dict]:
    """
    Load CALVIN episodes from the standard layout:

      root/
        training/
          episode_0000000.npz
          episode_0000001.npz
          ...
          lang_annotations/
            auto_lang_ann.npy   — {language: {task: list[str]}, info: {indx: list}}
        validation/
          ...

    Each .npz contains keys: rgb_static, rgb_gripper, rel_actions, robot_obs,
    scene_obs, done.

    Action format: 7-DoF [x, y, z, roll, pitch, yaw, gripper] ∈ [-1, 1]⁷
    (action_dim = 7).

    Returns list of episode dicts compatible with TrajectoryDataset.
    """
    root      = Path(root) / split
    episodes  = []

    # Load language annotations if available
    lang_ann_path = root / "lang_annotations" / "auto_lang_ann.npy"
    lang_ann      = None
    if lang_ann_path.exists():
        lang_ann = np.load(lang_ann_path, allow_pickle=True).item()
        # lang_ann["language"]["task"] — list of task strings per indexed segment
        # lang_ann["info"]["indx"]     — list of (start, end) episode indices

    episode_files = sorted(root.glob("episode_*.npz"))
    if not episode_files:
        print(f"[CALVIN] No .npz files found in {root}. Check dataset path.")
        return episodes

    # Group files into episodes using language annotation boundaries if available
    if lang_ann is not None:
        indx      = lang_ann["info"]["indx"]       # [(start, end), ...]
        tasks     = lang_ann["language"]["task"]   # [str, ...]

        for (start, end), task_str in zip(indx, tasks):
            ep_files = episode_files[start:end + 1]
            if len(ep_files) < 2:
                continue

            frames      = []
            actions_idx = []
            rewards     = []
            action_vecs = []

            for npz_path in ep_files:
                data = np.load(npz_path, allow_pickle=True)

                # Use static camera RGB
                frame = data.get("rgb_static", None)
                if frame is None:
                    continue
                frames.append(np.asarray(frame, dtype=np.uint8))

                # 7-DoF relative action vector
                rel_action = data.get("rel_actions", np.zeros(7, dtype=np.float32))
                rel_action = np.asarray(rel_action, dtype=np.float32).flatten()[:7]

                # Discretise into 14 direction-aware bins:
                #   0-11 : dominant translational/rotational axis × direction
                #          axis 0=x, 1=y, 2=z, 3=roll, 4=pitch, 5=yaw
                #          even = positive, odd = negative
                #   12   : gripper open  (rel_action[6] > 0.5)
                #   13   : gripper close (rel_action[6] < -0.5)
                if rel_action[6] > 0.5:
                    action_idx = 12   # gripper open
                elif rel_action[6] < -0.5:
                    action_idx = 13   # gripper close
                else:
                    dom = int(np.argmax(np.abs(rel_action[:6])))
                    action_idx = dom * 2 + (0 if rel_action[dom] >= 0 else 1)

                actions_idx.append(action_idx)
                rewards.append(float(data.get("done", 0)))   # reward = 1 on success
                action_vecs.append(rel_action)

            if len(frames) < 2:
                continue

            episodes.append({
                "frames":         np.stack(frames),
                "instruction":    task_str,
                "actions":        np.array(actions_idx, dtype=np.int64),
                "rewards":        np.array(rewards, dtype=np.float32),
                "action_vectors": np.stack(action_vecs).astype(np.float32),
            })
    else:
        # No annotations: treat each file as a single-step episode (limited)
        print("[CALVIN] No lang_annotations found — loading raw episodes without instructions.")
        frames, actions_idx, rewards, action_vecs = [], [], [], []
        for npz_path in episode_files:
            data  = np.load(npz_path, allow_pickle=True)
            frame = data.get("rgb_static", None)
            if frame is None:
                continue
            rel_action = np.asarray(
                data.get("rel_actions", np.zeros(7, dtype=np.float32)), dtype=np.float32
            ).flatten()[:7]
            if rel_action[6] > 0.5:
                action_idx = 12
            elif rel_action[6] < -0.5:
                action_idx = 13
            else:
                dom = int(np.argmax(np.abs(rel_action[:6])))
                action_idx = dom * 2 + (0 if rel_action[dom] >= 0 else 1)
            frames.append(np.asarray(frame, dtype=np.uint8))
            actions_idx.append(action_idx)
            rewards.append(float(data.get("done", 0)))
            action_vecs.append(rel_action)

        if len(frames) >= 2:
            episodes.append({
                "frames":         np.stack(frames),
                "instruction":    "complete the manipulation task",
                "actions":        np.array(actions_idx, dtype=np.int64),
                "rewards":        np.array(rewards, dtype=np.float32),
                "action_vectors": np.stack(action_vecs).astype(np.float32),
            })

    print(f"[CALVIN] Loaded {len(episodes)} annotated episodes from {root}")
    return episodes


# ── Generic loader ─────────────────────────────────────────────────────────────

def load_episodes(path: str) -> List[Dict]:
    """Load episodes from a .pkl file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_episodes(episodes: List[Dict], path: str) -> None:
    """Save episodes to a .pkl file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episodes, f)


# ── Dataset ────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Sliding-window dataset over robot episodes.

    Handles episodes with or without 'action_vectors'. When present,
    action_vec_hist is returned as a (history_len, action_dim) tensor and
    passed to VERA's history encoder for richer low-level conditioning.
    When absent, action_vec_hist is None and the encoder falls back to
    discrete-only history (fully backward compatible).
    """

    def __init__(
        self,
        episodes:        List[Dict],
        history_len:     int  = 4,
        num_vis_frames:  int  = 3,
        num_actions:     int  = 8,
        action_dim:      int  = 4,
        img_size:        int  = 224,
        clip_model_name: str  = "ViT-B/32",
        device:          str  = "cpu",
    ):
        self.history_len    = history_len
        self.num_vis_frames = num_vis_frames
        self.num_actions    = num_actions
        self.action_dim     = action_dim
        self.pad_action     = num_actions   # "no history" discrete padding index

        _, self.preprocess = clip.load(clip_model_name, device=device)

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                        std= [0.26862954, 0.26130258, 0.27577711]),
        ])

        # Check whether any episode has action_vectors
        self.has_action_vecs = any("action_vectors" in ep for ep in episodes)

        # Validate action_dim consistency
        if self.has_action_vecs:
            for ep in episodes:
                if "action_vectors" in ep:
                    ep_dim = ep["action_vectors"].shape[1]
                    if ep_dim != action_dim:
                        print(
                            f"[TrajectoryDataset] Warning: episode action_dim={ep_dim} "
                            f"but dataset action_dim={action_dim}. "
                            f"Truncating/padding to {action_dim}."
                        )
                    break

        # Build flat (episode_idx, timestep) index.
        # Require each episode to have at least history_len + 1 steps so that
        # __getitem__ never produces an out-of-bounds action_hist window.
        min_len = history_len + 1
        self.index: List[tuple] = []
        self.episodes = episodes
        skipped = 0
        for ep_i, ep in enumerate(episodes):
            T_ep = len(ep["actions"])
            if T_ep < min_len:
                skipped += 1
                continue
            for t in range(1, T_ep):
                self.index.append((ep_i, t))
        if skipped:
            print(f"[TrajectoryDataset] Skipped {skipped} episodes shorter than "
                  f"history_len+1={min_len} steps.")

        # Pre-tokenize all unique instructions
        unique_inst  = list({ep["instruction"] for ep in episodes})
        tokens       = clip.tokenize(unique_inst)
        self._token_cache = {inst: tokens[i] for i, inst in enumerate(unique_inst)}

        print(
            f"[TrajectoryDataset] {len(episodes)} episodes, "
            f"{len(self.index)} windows, "
            f"action_vectors={'yes' if self.has_action_vecs else 'no'}, "
            f"action_dim={action_dim}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        ep_i, t = self.index[idx]
        ep      = self.episodes[ep_i]

        # ── Language tokens ───────────────────────────────────────────────────
        lang_tokens = self._token_cache[ep["instruction"]]   # (77,)

        # ── Visual frames: last num_vis_frames up to t-1 ──────────────────────
        start_vis  = max(0, t - self.num_vis_frames)
        raw_frames = ep["frames"][start_vis:t]
        pad_needed = self.num_vis_frames - len(raw_frames)
        if pad_needed > 0:
            pad = np.zeros_like(raw_frames[0:1]).repeat(pad_needed, axis=0)
            raw_frames = np.concatenate([pad, raw_frames], axis=0)
        frames = torch.stack([
            self.transform(Image.fromarray(f)) for f in raw_frames
        ])   # (num_vis_frames, 3, H, W)

        # ── Discrete action + reward history ──────────────────────────────────
        start_h      = max(0, t - self.history_len)
        hist_actions = ep["actions"][start_h:t].tolist()
        hist_rewards = ep["rewards"][start_h:t].tolist()
        pad_h        = self.history_len - len(hist_actions)
        hist_actions = [self.pad_action] * pad_h + hist_actions
        hist_rewards = [0.0]            * pad_h + hist_rewards

        action_hist = torch.tensor(hist_actions, dtype=torch.long)
        reward_hist = torch.tensor(hist_rewards, dtype=torch.float32)

        # ── Low-level continuous action vectors ───────────────────────────────
        action_vec_hist = None
        if "action_vectors" in ep:
            raw_vecs = ep["action_vectors"][start_h:t]   # (<= H, action_dim)
            ep_dim   = raw_vecs.shape[1] if len(raw_vecs) > 0 else self.action_dim

            # Truncate or zero-pad to self.action_dim
            if ep_dim > self.action_dim:
                raw_vecs = raw_vecs[:, :self.action_dim]
            elif ep_dim < self.action_dim:
                pad_cols = np.zeros((len(raw_vecs), self.action_dim - ep_dim), dtype=np.float32)
                raw_vecs = np.concatenate([raw_vecs, pad_cols], axis=1)

            # Left-pad timestep dimension with zero vectors
            if pad_h > 0:
                zero_pad = np.zeros((pad_h, self.action_dim), dtype=np.float32)
                raw_vecs = np.concatenate([zero_pad, raw_vecs], axis=0)

            action_vec_hist = torch.tensor(raw_vecs, dtype=torch.float32)
            # shape: (history_len, action_dim)

        # ── Target action (label for discrete classifier) ─────────────────────
        target = int(ep["actions"][t])

        # ── Target continuous action vector (label for regression head) ────────
        # This is the expert's executed action at timestep t — the same step
        # whose discrete index is in `target`.  Truncated/padded to action_dim.
        target_vec = None
        if "action_vectors" in ep:
            tv = ep["action_vectors"][t].astype(np.float32)   # (ep_dim,)
            if len(tv) > self.action_dim:
                tv = tv[:self.action_dim]
            elif len(tv) < self.action_dim:
                tv = np.concatenate(
                    [tv, np.zeros(self.action_dim - len(tv), dtype=np.float32)]
                )
            target_vec = torch.tensor(tv, dtype=torch.float32)   # (action_dim,)

        return {
            "frames":          frames,           # (num_vis_frames, 3, H, W)
            "lang_tokens":     lang_tokens,      # (77,)
            "action_hist":     action_hist,      # (history_len,)
            "reward_hist":     reward_hist,      # (history_len,)
            "action_vec_hist": action_vec_hist,  # (history_len, action_dim) or None
            "target":          torch.tensor(target, dtype=torch.long),
            "target_vec":      target_vec,       # (action_dim,) or None
        }
