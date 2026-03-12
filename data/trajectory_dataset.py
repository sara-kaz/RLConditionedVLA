"""
TrajectoryDataset
=================
Loads robot trajectories stored as a list of episode dicts:

  episode = {
      "frames"      : np.ndarray  (T_ep, H, W, 3)  uint8
      "instruction" : str
      "actions"     : np.ndarray  (T_ep,)           int64
      "rewards"     : np.ndarray  (T_ep,)           float32
  }

Each __getitem__ returns a training sample built by sliding a window of
length (history_len + 1) over an episode:

  - frames      : (num_frames, 3, 224, 224)  — last `num_frames` obs up to t
  - lang_tokens : (77,)                       — CLIP-tokenized instruction
  - action_hist : (history_len,)              — actions at t-H … t-1
  - reward_hist : (history_len,)              — rewards at t-H … t-1
  - target      : int                         — action at t  (label)

Storage format
--------------
  Save a list of episode dicts as a .pkl file, or generate synthetic ones
  via `make_random_episodes()` for quick testing.
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


def make_random_episodes(
    num_episodes: int = 50,
    ep_len: int = 30,
    num_actions: int = 8,
    img_size: int = 64,
) -> List[Dict]:
    """Generate random synthetic episodes for unit-testing the pipeline."""
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
            "frames":      np.random.randint(0, 255, (T_ep, img_size, img_size, 3), dtype=np.uint8),
            "instruction": random.choice(instructions),
            "actions":     np.random.randint(0, num_actions, (T_ep,), dtype=np.int64),
            "rewards":     np.random.uniform(-1.0, 1.0, (T_ep,)).astype(np.float32),
        })
    return episodes


class TrajectoryDataset(Dataset):
    """Sliding-window dataset over a list of robot episodes."""

    def __init__(
        self,
        episodes: List[Dict],
        history_len: int = 4,
        num_vis_frames: int = 3,
        num_actions: int = 8,
        img_size: int = 224,
        clip_model_name: str = "ViT-B/32",
        device: str = "cpu",
    ):
        self.history_len    = history_len
        self.num_vis_frames = num_vis_frames
        self.num_actions    = num_actions
        self.pad_action     = num_actions          # index used for "no history" padding

        # CLIP tokenizer (text only — image preprocessing done per-sample)
        _, self.preprocess = clip.load(clip_model_name, device=device)

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                        std= [0.26862954, 0.26130258, 0.27577711]),
        ])

        # Build flat index: list of (episode_idx, timestep_t)
        self.index: List[tuple] = []
        self.episodes = episodes
        for ep_i, ep in enumerate(episodes):
            T_ep = len(ep["actions"])
            # Need at least 1 step to predict (t >= 1 so there's a target)
            for t in range(1, T_ep):
                self.index.append((ep_i, t))

        # Pre-tokenize all unique instructions
        unique_inst = list({ep["instruction"] for ep in episodes})
        tokens = clip.tokenize(unique_inst)     # (N_unique, 77)
        self._token_cache = {inst: tokens[i] for i, inst in enumerate(unique_inst)}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        ep_i, t = self.index[idx]
        ep = self.episodes[ep_i]

        # ── Language tokens ───────────────────────────────────────────────
        lang_tokens = self._token_cache[ep["instruction"]]  # (77,)

        # ── Visual frames: last `num_vis_frames` frames up to t-1 ─────────
        start_vis = max(0, t - self.num_vis_frames)
        raw_frames = ep["frames"][start_vis:t]              # (<= num_vis_frames, H, W, 3)
        # Pad on the left if not enough history
        pad_needed = self.num_vis_frames - len(raw_frames)
        if pad_needed > 0:
            pad_frame = np.zeros_like(raw_frames[0:1]).repeat(pad_needed, axis=0)
            raw_frames = np.concatenate([pad_frame, raw_frames], axis=0)

        frames = torch.stack([
            self.transform(Image.fromarray(f)) for f in raw_frames
        ])  # (num_vis_frames, 3, H, W)

        # ── Action / reward history: steps t-H … t-1 ─────────────────────
        start_h = max(0, t - self.history_len)
        hist_actions = ep["actions"][start_h:t].tolist()    # list of ints
        hist_rewards = ep["rewards"][start_h:t].tolist()    # list of floats
        # Left-pad with "no action" / 0 reward
        pad_h = self.history_len - len(hist_actions)
        hist_actions = [self.pad_action] * pad_h + hist_actions
        hist_rewards = [0.0]            * pad_h + hist_rewards

        action_hist = torch.tensor(hist_actions, dtype=torch.long)
        reward_hist = torch.tensor(hist_rewards, dtype=torch.float32)

        # ── Target action ─────────────────────────────────────────────────
        target = int(ep["actions"][t])

        return {
            "frames":      frames,           # (num_vis_frames, 3, H, W)
            "lang_tokens": lang_tokens,      # (77,)
            "action_hist": action_hist,      # (history_len,)
            "reward_hist": reward_hist,      # (history_len,)
            "target":      torch.tensor(target, dtype=torch.long),
        }


def load_episodes(path: str) -> List[Dict]:
    """Load episodes from a .pkl file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_episodes(episodes: List[Dict], path: str) -> None:
    """Save episodes to a .pkl file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episodes, f)
