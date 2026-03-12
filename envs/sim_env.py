"""
SimEnv — Gym-compatible simulation environment wrapper
======================================================
Wraps any OpenAI Gym / Gymnasium environment and exposes the
VLA-friendly interface:

    obs  = env.reset()   -> {"frame": np.ndarray (H,W,3), "instruction": str}
    obs, reward, done, info = env.step(action_idx)

Supports:
  - Any Gym env that returns pixel observations (render_mode="rgb_array")
  - FrankaKitchen, MiniGrid, Meta-World, and custom envs via the adapter pattern
  - Domain randomization hooks for sim-to-real transfer

For real robot use, swap `SimEnv` for `RealEnv` defined at the bottom of this file.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


# ── Base interface ─────────────────────────────────────────────────────────────

class BaseEnv:
    """Minimal interface every environment must implement."""

    def reset(self) -> Dict[str, Any]:
        """Returns {"frame": np.ndarray (H,W,3) uint8, "instruction": str}"""
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Returns (obs, reward, done, info)."""
        raise NotImplementedError

    def close(self):
        pass


# ── Synthetic dummy env (no Gym required) ─────────────────────────────────────

class RandomDummyEnv(BaseEnv):
    """
    Minimal dummy environment for unit-testing the training pipeline
    without installing Gym/MuJoCo.

    Actions: 0..num_actions-1
    Reward:  +1 if action matches a hidden target, else 0
    Done:    after max_steps
    """

    INSTRUCTIONS = [
        "pick up the red cube",
        "move to the left side",
        "push the block forward",
        "grasp the cylinder and place it on the shelf",
    ]

    def __init__(self, num_actions: int = 8, max_steps: int = 30, img_size: int = 64):
        self.num_actions = num_actions
        self.max_steps   = max_steps
        self.img_size    = img_size
        self._step       = 0
        self._target     = 0
        self._instruction = ""

    def reset(self) -> Dict[str, Any]:
        self._step        = 0
        self._target      = np.random.randint(0, self.num_actions)
        self._instruction = np.random.choice(self.INSTRUCTIONS)
        return {
            "frame":       np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8),
            "instruction": self._instruction,
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self._step += 1
        reward = 1.0 if action == self._target else -0.1
        done   = (self._step >= self.max_steps) or (action == self._target)
        obs    = {
            "frame":       np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8),
            "instruction": self._instruction,
        }
        return obs, reward, done, {"target": self._target}


# ── Gym wrapper ────────────────────────────────────────────────────────────────

class SimEnv(BaseEnv):
    """
    Wraps a Gym environment.  The env must support render_mode="rgb_array".

    If the Gym env cannot be imported (not installed), falls back to
    RandomDummyEnv so the rest of the pipeline can still be tested.

    Supported envs (examples):
      "MiniGrid-Empty-5x5-v0"       — needs pip install minigrid
      "FrankaKitchen-v1"            — needs pip install gym-robotics
      "FetchReach-v2"               — needs pip install gym-robotics
      "CartPole-v1"                 — needs pip install gymnasium

    Set env_id: "dummy" to always use RandomDummyEnv.
    """

    # Map action counts to Gym discrete action spaces
    ACTION_MAPS: Dict[str, List[int]] = {}

    def __init__(self, cfg: dict):
        env_cfg     = cfg.get("env", {})
        env_id      = env_cfg.get("env_id", "dummy")
        num_actions = cfg["model"]["num_actions"]
        img_size    = cfg["data"].get("img_size", 64)

        self._gym_env = None
        self._instruction = env_cfg.get("instruction", "complete the task")
        self._img_size    = img_size

        if env_id == "dummy":
            self._dummy = RandomDummyEnv(
                num_actions=num_actions,
                max_steps=cfg["rl"].get("max_episode_steps", 50),
                img_size=img_size,
            )
            return

        try:
            import gymnasium as gym
            self._gym_env = gym.make(env_id, render_mode="rgb_array")
            print(f"[SimEnv] Loaded Gym env: {env_id}")
        except Exception as e:
            print(f"[SimEnv] Could not load '{env_id}': {e}. Falling back to dummy env.")
            self._dummy = RandomDummyEnv(
                num_actions=num_actions,
                max_steps=cfg["rl"].get("max_episode_steps", 50),
                img_size=img_size,
            )

        # Domain randomization settings
        self._domain_rand = env_cfg.get("domain_randomization", False)
        self._noise_std   = env_cfg.get("obs_noise_std", 5.0)

    def reset(self) -> Dict[str, Any]:
        if self._gym_env is None:
            return self._dummy.reset()
        self._gym_env.reset()
        frame = self._render_frame()
        return {"frame": frame, "instruction": self._instruction}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        if self._gym_env is None:
            return self._dummy.step(action)
        _, reward, terminated, truncated, info = self._gym_env.step(action)
        done  = terminated or truncated
        frame = self._render_frame()
        return {"frame": frame, "instruction": self._instruction}, float(reward), done, info

    def _render_frame(self) -> np.ndarray:
        frame = self._gym_env.render()                              # (H, W, 3)
        if self._domain_rand:
            noise = np.random.normal(0, self._noise_std, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if frame.shape[:2] != (self._img_size, self._img_size):
            from PIL import Image
            frame = np.array(Image.fromarray(frame).resize((self._img_size, self._img_size)))
        return frame

    def close(self):
        if self._gym_env is not None:
            self._gym_env.close()


# ── Real robot environment stub ────────────────────────────────────────────────

class RealEnv(BaseEnv):
    """
    Stub for a real robot interface.
    Replace the body of each method with your robot SDK calls.

    Expected hardware interface:
      - camera: returns BGR frame from cv2 (converted to RGB here)
      - robot:  accepts discrete action index, returns done flag + reward signal
    """

    def __init__(self, cfg: dict):
        self._instruction = cfg.get("env", {}).get("instruction", "complete the task")
        self._img_size    = cfg["data"].get("img_size", 224)
        # TODO: initialize camera, robot arm SDK, reward sensor here

    def reset(self) -> Dict[str, Any]:
        # TODO: move robot to home position, reset sensors
        frame = self._capture_frame()
        return {"frame": frame, "instruction": self._instruction}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        # TODO: send `action` to robot hardware
        # reward = read from force/torque sensor or task completion logic
        reward = 0.0
        done   = False
        frame  = self._capture_frame()
        return {"frame": frame, "instruction": self._instruction}, reward, done, {}

    def _capture_frame(self) -> np.ndarray:
        # TODO: replace with actual camera capture
        # import cv2
        # ret, frame = self._cap.read()
        # frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self._img_size, self._img_size))
        return np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)
