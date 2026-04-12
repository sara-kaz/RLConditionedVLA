"""
MiniGridEnv
===========
Thin wrapper around any MiniGrid gymnasium environment that implements
the SimEnv interface expected by training/rl_trainer.py and evaluation/evaluate.py:

  obs = env.reset()                    -> {"frame": np.ndarray (H,W,3) uint8,
                                           "instruction": str}
  obs, reward, done, info = env.step(action)
  env.close()

MiniGrid action space (7 discrete actions):
  0 = turn left
  1 = turn right
  2 = move forward
  3 = pick up object
  4 = drop object
  5 = toggle / interact
  6 = done

The rendered RGB image (full grid view, render_mode="rgb_array") is used as
the observation frame so CLIP sees the full grid layout.

Usage:
  from envs.minigrid_env import MiniGridEnv
  cfg = {"env": {"env_id": "MiniGrid-Empty-5x5-v0",
                 "instruction": "navigate to the green goal square"},
         "data": {"img_size": 64}}
  env = MiniGridEnv(cfg)
"""

import numpy as np

try:
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
except ImportError as e:
    raise ImportError(
        "MiniGrid dependencies not installed. Run:\n"
        "  pip install gymnasium minigrid"
    ) from e


# Human-readable action descriptions used as per-step language feedback
MINIGRID_ACTION_DESCRIPTIONS = {
    0: "I turned left",
    1: "I turned right",
    2: "I moved forward",
    3: "I picked up an object",
    4: "I dropped an object",
    5: "I toggled or interacted",
    6: "I signalled done",
}

# Default instruction per well-known env (overridden by cfg["env"]["instruction"])
MINIGRID_DEFAULT_INSTRUCTIONS = {
    "MiniGrid-Empty-5x5-v0":       "navigate to the green goal square",
    "MiniGrid-Empty-8x8-v0":       "navigate to the green goal square",
    "MiniGrid-Empty-16x16-v0":     "navigate to the green goal square",
    "MiniGrid-FourRooms-v0":       "navigate to the green goal square",
    "MiniGrid-DoorKey-5x5-v0":     "pick up the key, open the door, reach the goal",
    "MiniGrid-Fetch-5x5-N2-v0":    "pick up the matching object",
}

NUM_ACTIONS = 7  # All MiniGrid envs share the same 7-action discrete space


class MiniGridEnv:
    """
    MiniGrid environment wrapper implementing the project's SimEnv interface.

    Parameters
    ----------
    cfg : dict
        Project config dict (see configs/minigrid_config.yaml).
        Reads:
          cfg["env"]["env_id"]       — gymnasium env id (default: MiniGrid-Empty-5x5-v0)
          cfg["env"]["instruction"]  — task instruction string (optional, has sane default)
          cfg["data"]["img_size"]    — unused for frame capture but stored for reference
          cfg["rl"]["max_episode_steps"] — soft step limit (handled by caller)
    """

    NUM_ACTIONS = NUM_ACTIONS

    def __init__(self, cfg: dict):
        env_cfg = cfg.get("env", {})
        self.env_id = env_cfg.get("env_id", "MiniGrid-Empty-5x5-v0")

        # Instruction: explicit override > per-env default > generic fallback
        self.instruction = (
            env_cfg.get("instruction")
            or MINIGRID_DEFAULT_INSTRUCTIONS.get(self.env_id)
            or "complete the navigation task"
        )

        # Create the gymnasium environment with RGB rendering
        self._env = gym.make(self.env_id, render_mode="rgb_array")
        self._last_frame: np.ndarray | None = None

    # ------------------------------------------------------------------
    # SimEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset environment, return initial observation dict."""
        _obs, _info = self._env.reset()
        frame = self._env.render()          # (H, W, 3) uint8 — full grid view
        self._last_frame = frame
        return {"frame": frame, "instruction": self.instruction}

    def step(self, action: int) -> tuple:
        """
        Take one step.

        Parameters
        ----------
        action : int  in [0, NUM_ACTIONS)

        Returns
        -------
        obs  : dict   {"frame": np.ndarray (H,W,3), "instruction": str}
        reward : float
        done   : bool
        info   : dict  (includes "action_description" for language feedback)
        """
        obs_raw, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        frame = self._env.render()          # (H, W, 3) uint8
        self._last_frame = frame

        info["action_description"] = MINIGRID_ACTION_DESCRIPTIONS.get(action, "I took an action")
        info["terminated"] = terminated
        info["truncated"] = truncated

        obs = {"frame": frame, "instruction": self.instruction}
        return obs, float(reward), done, info

    def close(self) -> None:
        """Release environment resources."""
        self._env.close()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_actions(self) -> int:
        return self.NUM_ACTIONS

    def action_description(self, action: int) -> str:
        """Human-readable label for a discrete action index."""
        return MINIGRID_ACTION_DESCRIPTIONS.get(action, f"action {action}")

    def __repr__(self) -> str:
        return f"MiniGridEnv(env_id={self.env_id!r}, instruction={self.instruction!r})"
