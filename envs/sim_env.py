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


# ── MetaWorld wrapper ──────────────────────────────────────────────────────────

class MetaWorldEnv(BaseEnv):
    """
    Wraps a Meta-World ML1 / MT50 task and exposes the VLA interface.

    Key additions over the plain Gym wrapper:
      • `info["dist_delta"]` — signed change in end-effector–to–goal Euclidean
        distance between consecutive steps. Positive = moved away, negative = closer.
        This populates Stream 3b (consequence language encoder) with a real signal.
      • Action discretisation — MetaWorld has a continuous 4-DoF action space.
        We project it to `num_actions` discrete bins via a fixed codebook built
        from a uniform grid over [-1, 1]^4.
      • Language instruction — generated from the task name automatically.

    Install: pip install metaworld

    Example config:
      env:
        env_id: metaworld-reach-v2    # any metaworld task name
        num_actions: 8                # discrete bins
    """

    # Task name → human-readable instruction
    _TASK_INSTRUCTIONS: Dict[str, str] = {
        "reach-v2":          "move the robot arm to reach the target position",
        "push-v2":           "push the puck to the target location",
        "pick-place-v2":     "pick up the object and place it at the target",
        "door-open-v2":      "grasp the door handle and open the door",
        "drawer-close-v2":   "push the drawer closed",
        "drawer-open-v2":    "pull the drawer open",
        "button-press-v2":   "press the button down",
        "peg-insert-side-v2":"insert the peg into the hole from the side",
        "window-open-v2":    "slide the window open",
        "window-close-v2":   "slide the window closed",
    }

    def __init__(self, cfg: dict):
        import metaworld
        env_cfg      = cfg.get("env", {})
        task_name    = env_cfg.get("env_id", "reach-v2").replace("metaworld-", "")
        num_actions  = cfg["model"]["num_actions"]
        self._img    = cfg["data"].get("img_size", 224)

        ml1 = metaworld.ML1(task_name)
        self._env = ml1.train_classes[task_name]()
        task      = np.random.choice(ml1.train_tasks)
        self._env.set_task(task)

        self._task_name  = task_name
        self._num_actions = num_actions
        self._instruction = self._TASK_INSTRUCTIONS.get(
            task_name, f"complete the {task_name.replace('-', ' ')} task"
        )

        # Build discrete action codebook: num_actions centroids in [-1,1]^4
        self._action_dim  = 4    # MetaWorld: xyz delta + gripper
        self._codebook    = self._build_codebook(num_actions)
        self._prev_dist   = None

        print(f"[MetaWorldEnv] Task: {task_name} | "
              f"Actions: {num_actions} discrete bins | "
              f"Instruction: \"{self._instruction}\"")

    def _build_codebook(self, num_actions: int) -> np.ndarray:
        """
        Simple codebook: evenly space `num_actions` points across the
        action space principal directions. Each action moves along a
        different axis or combination.
        """
        rng     = np.linspace(-1, 1, max(2, int(np.ceil(num_actions ** 0.25))))
        grid    = np.array(np.meshgrid(rng, rng, rng, rng)).T.reshape(-1, 4)
        idx     = np.linspace(0, len(grid) - 1, num_actions, dtype=int)
        return grid[idx].astype(np.float32)

    def _dist_to_goal(self) -> float:
        obs  = self._env._get_obs()
        hand = obs[:3]
        goal = self._env._get_pos_goal()
        return float(np.linalg.norm(hand - goal))

    def reset(self) -> Dict[str, Any]:
        self._env.reset()
        self._prev_dist = self._dist_to_goal()
        frame = self._render(self._img)
        return {"frame": frame, "instruction": self._instruction}

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        cont_action = self._codebook[action_idx % self._num_actions]
        _, reward, done, info = self._env.step(cont_action)
        frame = self._render(self._img)

        curr_dist  = self._dist_to_goal()
        dist_delta = float(curr_dist - self._prev_dist)   # negative = closer
        self._prev_dist = curr_dist

        # Expose the executed codebook vector so the RL rollout collector can
        # feed it into Stream 4 (ActionRewardHistoryEncoder) at the next step.
        # Shape: (action_dim,) = (4,) for MetaWorld [Δx, Δy, Δz, gripper].
        info["dist_delta"]    = dist_delta
        info["action_vector"] = cont_action.copy()   # numpy array (4,)
        obs = {"frame": frame, "instruction": self._instruction}
        return obs, float(reward), bool(done), info

    def _render(self, size: int) -> np.ndarray:
        frame = self._env.render(offscreen=True)
        if frame.shape[:2] != (size, size):
            from PIL import Image
            frame = np.array(Image.fromarray(frame).resize((size, size)))
        return frame

    def close(self):
        self._env.close()


# ── BabyAI / MiniGrid wrapper ─────────────────────────────────────────────────

class BabyAIEnv(BaseEnv):
    """
    Wraps a BabyAI / MiniGrid environment for cheap language-grounded RL.

    Why BabyAI for VLLA?
      • Procedurally generated language instructions ("go to the red ball")
      • Discrete action space (7 actions) — maps cleanly to verbalization
      • Dense enough episodes (H~100 steps) to exercise the history encoder
      • dist_delta = L1 distance change to goal object, exposable per step
      • Fast to run: 1000 episodes in <1min on CPU

    Install: pip install minigrid

    Config:
      env:
        env_id: babyai-GoToLocal-v0
        num_actions: 7

    Exposed info keys:
      dist_delta : signed change in Manhattan distance to target object
                   (negative = closer, positive = farther)
      mission    : the full language instruction string
    """

    # 7 MiniGrid primitive actions (matches minigrid.core.actions.Actions)
    ACTION_NAMES = [
        "turn left",
        "turn right",
        "move forward",
        "pick up the object",
        "drop the object",
        "toggle the door or switch",
        "done",
    ]

    def __init__(self, cfg: dict):
        env_cfg    = cfg.get("env", {})
        env_id     = env_cfg.get("env_id", "BabyAI-GoToLocal-v0")
        self._img  = cfg["data"].get("img_size", 224)

        try:
            import gymnasium as gym
            self._env = gym.make(env_id, render_mode="rgb_array")
            print(f"[BabyAIEnv] Loaded: {env_id}")
        except Exception as e:
            raise RuntimeError(f"BabyAI env '{env_id}' could not be loaded: {e}") from e

        self._instruction  = ""
        self._target_pos   = None
        self._prev_dist    = None

    def reset(self) -> Dict[str, Any]:
        obs, _ = self._env.reset()
        self._instruction = obs.get("mission", "complete the task")
        self._target_pos  = self._find_target()
        self._prev_dist   = self._agent_dist_to_target()
        frame = self._render()
        return {"frame": frame, "instruction": self._instruction}

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        obs, reward, terminated, truncated, info = self._env.step(action_idx)
        done = terminated or truncated

        self._instruction = obs.get("mission", self._instruction)
        self._target_pos  = self._find_target()
        curr_dist         = self._agent_dist_to_target()
        dist_delta        = float(curr_dist - self._prev_dist)
        self._prev_dist   = curr_dist

        info["dist_delta"] = dist_delta
        frame = self._render()
        return {"frame": frame, "instruction": self._instruction}, float(reward), done, info

    def _find_target(self) -> Optional[Tuple[int, int]]:
        """Return grid position of the mission target object if findable."""
        try:
            grid = self._env.unwrapped.grid
            for i in range(grid.width):
                for j in range(grid.height):
                    cell = grid.get(i, j)
                    if cell is not None and cell.type not in ("wall", "floor", "door"):
                        return (i, j)
        except Exception:
            pass
        return None

    def _agent_dist_to_target(self) -> float:
        if self._target_pos is None:
            return 0.0
        agent_pos = self._env.unwrapped.agent_pos
        return float(abs(agent_pos[0] - self._target_pos[0])
                   + abs(agent_pos[1] - self._target_pos[1]))

    def _render(self) -> np.ndarray:
        frame = self._env.render()   # (H, W, 3)
        if frame.shape[:2] != (self._img, self._img):
            from PIL import Image
            frame = np.array(Image.fromarray(frame).resize((self._img, self._img)))
        return frame

    def close(self):
        self._env.close()


# ── Language-Table wrapper ─────────────────────────────────────────────────────

class LanguageTableEnv(BaseEnv):
    """
    Wraps Google Research Language-Table environments for RL fine-tuning.

    Language-Table (Lynch et al., 2023) is a tabletop block-pushing benchmark
    where the robot arm must push coloured blocks to goal positions specified
    by natural language instructions ("push the red block to the blue block").

    Action mapping: num_actions discrete directional bins → 2D end-effector
    velocity [vx, vy], clipped to the LT action space ([-0.03, 0.03] m/s).

    Observation: {'frame': (H, W, 3) uint8, 'instruction': str}
    Reward:      1.0 on task success (binary), 0 otherwise.
    Done:        True when task succeeds OR max_episode_steps reached.

    Requires:
        pip install --no-deps git+https://github.com/google-research/language-table.git
        pip install pybullet
        # Colab headless: apt-get install -y xvfb && Xvfb :99 -ac &
    """

    # 8-bin directional codebook matching config.yaml action_vocab
    # Bin 0 = East (right), progressing counter-clockwise
    _CODEBOOK = np.array([
        [ 1.0,   0.0  ],   # 0: right         "I pushed the object to the right"
        [ 0.707,  0.707],  # 1: up-right (NE)  "I pushed the object up and to the right"
        [ 0.0,   1.0  ],   # 2: up             "I pushed the object upward"
        [-0.707,  0.707],  # 3: up-left (NW)   "I pushed the object up and to the left"
        [-1.0,   0.0  ],   # 4: left           "I pushed the object to the left"
        [-0.707, -0.707],  # 5: down-left (SW) "I pushed the object down and to the left"
        [ 0.0,  -1.0  ],   # 6: down           "I pushed the object downward"
        [ 0.707, -0.707],  # 7: down-right (SE)"I pushed the object down and to the right"
    ], dtype=np.float32)

    def __init__(self, cfg: dict):
        import importlib

        env_cfg           = cfg.get("env", {})
        self._img_size    = cfg["data"].get("img_size", 224)
        self._num_actions = cfg["model"]["num_actions"]
        self._vel_scale      = float(env_cfg.get("lt_velocity_scale", 0.03))
        # Dense reward shaping: add ±(delta_dist × scale) each step so
        # REINFORCE gets a gradient even when binary success reward = 0.
        # Scale kept small (default 0.1) so shaped_total << 1.0 and
        # success_threshold = 1.0 still correctly identifies task completion.
        self._shaping_scale  = float(env_cfg.get("lt_shaping_scale", 0.1))
        self._prev_block_dist: Optional[float] = None
        self._instruction    = "push the block to the target location"
        self._dm_env         = False   # True when env uses dm_env TimeStep API

        print("[LanguageTableEnv] Loading Language-Table environment …")
        try:
            from language_table.environments import language_table as lt_module

            # ── Get block_mode from LanguageTableBlockVariants enum ──────────
            # Confirmed API (from diagnostic): block_mode must be a member of
            # language_table.environments.blocks.LanguageTableBlockVariants.
            # FIXED_4 = 4-block setup (simplest for RL); fall back to first member.
            from language_table.environments import blocks as _blocks_mod
            bm_enum = getattr(_blocks_mod, "LanguageTableBlockVariants", None)
            if bm_enum is None:
                raise RuntimeError(
                    "LanguageTableBlockVariants not found in "
                    "language_table.environments.blocks — "
                    f"available: {[a for a in dir(_blocks_mod) if not a.startswith('_')]}"
                )
            # Confirmed member names from live Colab diagnostic (May 2026):
            #   BLOCK_1, BLOCK_4, BLOCK_8, BLOCK_4_WPOLE, BLOCK_8_WPOLE, N_CHOOSE_K
            #
            # IMPORTANT: BLOCK_1 = only 1 block on table.  BlockToBlockReward needs
            # to sample 2 blocks (start + target) without replacement, so BLOCK_1
            # causes ValueError at reset().  Use BLOCK_4 (4 blocks) as the minimum.
            block_mode = None
            for preferred in ("BLOCK_4", "BLOCK_8",
                              "BLOCK_4_WPOLE", "BLOCK_8_WPOLE",
                              "BLOCK_1", "N_CHOOSE_K",
                              "FIXED_4", "TRAIN", "TRAIN_COMBINATIONS", "FIXED_8"):
                val = getattr(bm_enum, preferred, None)
                if val is not None:
                    block_mode = val
                    break
            if block_mode is None:
                block_mode = next(iter(bm_enum))   # first enum member as last resort
            print(f"  block_mode: LanguageTableBlockVariants.{block_mode.name}")

            # ── Probe for reward_factory ──────────────────────────────────────
            reward_factory = None
            for rf_search in [
                ("language_table.environments.rewards.block2block",    "BlockToBlockReward"),
                ("language_table.environments.rewards.block2location",  "BlockToLocationReward"),
                ("language_table.environments.rewards",                 "BlockToBlockReward"),
            ]:
                try:
                    mod = importlib.import_module(rf_search[0])
                    reward_factory = getattr(mod, rf_search[1])
                    print(f"  reward_factory: {rf_search[1]}")
                    break
                except Exception:
                    continue

            seed = int(cfg["training"].get("seed", 42))

            # ── Try constructor signatures from most to least specific ────────
            # Build candidate kwargs dicts using local variables (not module attrs)
            # to avoid eager AttributeError.
            candidates = []
            if block_mode is not None and reward_factory is not None:
                candidates.append({"block_mode": block_mode,
                                    "reward_factory": reward_factory, "seed": seed})
            if block_mode is not None:
                candidates.append({"block_mode": block_mode, "seed": seed})
            candidates += [{"seed": seed}, {}]

            last_err = None
            for kw in candidates:
                try:
                    self._env = lt_module.LanguageTable(**kw)
                    print(f"  constructor kwargs: {list(kw.keys())}")
                    break
                except Exception as e:
                    last_err = e
                    continue
            else:
                raise RuntimeError(
                    f"All constructor signatures failed. Last error: {last_err}"
                )

            # ── Detect dm_env vs gym API ──────────────────────────────────────
            # language-table is built on dm_env; reset() returns a TimeStep object
            # (not a tuple), and TimeStep.step_type indicates episode boundaries.
            try:
                import dm_env as _dm
                self._dm_env = isinstance(self._env, _dm.Environment)
            except ImportError:
                # Detect by duck-typing: dm_env TimeStep has .observation attribute
                self._dm_env = hasattr(self._env, "observation_spec")

            print(f"[LanguageTableEnv] Ready — API={'dm_env' if self._dm_env else 'gym'}, "
                  f"{self._num_actions} bins × {self._vel_scale} m/s")

        except Exception as e:
            raise RuntimeError(
                f"[LanguageTableEnv] Failed: {e}\n"
                "Install: pip install --no-deps "
                "git+https://github.com/google-research/language-table.git\n"
                "         pip install dm-env\n"
                "Headless: apt-get install -y xvfb && Xvfb :99 -ac &"
            ) from e

    def _cache_pybullet_block_ids(self, effector_xy=None):
        """Scan pybullet bodies and cache IDs of objects that look like blocks.

        Blocks sit on the table surface at z ≈ 0.03–0.08 m.  We collect every
        body in that z band and exclude the body closest to the effector (that is
        the end-effector tip, not a block).  Called once per episode on reset so
        body IDs are refreshed if the environment recreates the scene.
        """
        try:
            import pybullet as pb
            candidates = []
            for body_id in range(pb.getNumBodies()):
                try:
                    pos, _ = pb.getBasePositionAndOrientation(body_id)
                    z  = pos[2]
                    xy = np.array(pos[:2], dtype=np.float32)
                    if 0.01 <= z <= 0.12:
                        candidates.append((body_id, xy))
                except Exception:
                    continue

            if effector_xy is not None and len(candidates) > 1:
                # Remove the body that coincides with the effector position
                candidates = sorted(candidates,
                    key=lambda t: float(np.linalg.norm(t[1] - effector_xy)))
                candidates = candidates[1:]   # drop closest = effector tip

            self._pb_block_ids = [bid for bid, _ in candidates]
        except Exception:
            self._pb_block_ids = []

    def _get_block_dist(self, obs) -> Optional[float]:
        """Shaping distance metric — block-to-block distance via obs dict or pybullet.

        Priority 1: any non-effector *_translation keys in the obs dict.
          LT may use colour-prefixed names ('red_translation', 'blue_block_translation',
          etc.).  We accept any key that has 'translation' but not 'effector'.

        Priority 2: pybullet direct query using cached body IDs.
          Language-Table IS a pybullet simulation.  After reset we scan all bodies
          at table-surface height (z ≈ 0.01–0.12 m), cache their body IDs, and
          query positions each step.  This gives real block-to-block distance even
          when the obs dict does not expose block positions.

        No oracle-target fallback: ‖effector − effector_target‖ ≈ 0 at reset
        (oracle target is initialised to the current effector position) and
        telescopes to ≈ 0 over the episode, giving zero gradient.
        """
        # ── Priority 1: obs dict (broadened — any non-effector *_translation) ─
        if isinstance(obs, dict):
            non_eff = [(k, v) for k, v in obs.items()
                       if "translation" in k.lower() and "effector" not in k.lower()]
            if len(non_eff) >= 2:
                p1 = np.asarray(non_eff[0][1], dtype=np.float32).ravel()[:2]
                p2 = np.asarray(non_eff[1][1], dtype=np.float32).ravel()[:2]
                return float(np.linalg.norm(p1 - p2))

        # ── Priority 2: pybullet direct query ─────────────────────────────────
        block_ids = getattr(self, "_pb_block_ids", None)
        if block_ids is None:
            # Cache not populated yet — build it now (should have run at reset)
            eff_xy = None
            if isinstance(obs, dict) and "effector_translation" in obs:
                eff_xy = np.asarray(obs["effector_translation"],
                                    dtype=np.float32).ravel()[:2]
            self._cache_pybullet_block_ids(eff_xy)
            block_ids = self._pb_block_ids

        if block_ids and len(block_ids) >= 2:
            try:
                import pybullet as pb
                positions = []
                for bid in block_ids:
                    pos, _ = pb.getBasePositionAndOrientation(bid)
                    positions.append(np.array(pos[:2], dtype=np.float32))
                if len(positions) >= 2:
                    return float(np.linalg.norm(positions[0] - positions[1]))
            except Exception:
                self._pb_block_ids = None   # stale IDs — will re-scan next call

        return None   # shaping unavailable this step

    def _unpack_timestep(self, timestep):
        """Extract (obs_dict, reward, done) from a dm_env TimeStep or gym tuple."""
        # dm_env TimeStep: named tuple with .step_type / .reward / .observation
        if hasattr(timestep, "observation"):
            obs    = timestep.observation
            reward = float(timestep.reward or 0.0)
            # LAST step_type means episode ended
            try:
                import dm_env as _dm
                done = (timestep.step_type == _dm.StepType.LAST)
            except Exception:
                done = getattr(timestep, "last", lambda: False)()
            return obs, reward, bool(done)

        # Gym tuple: (obs, reward, done[, truncated][, info])
        if isinstance(timestep, (tuple, list)):
            if len(timestep) >= 5:
                obs, reward, terminated, truncated = timestep[:4]
                return obs, float(reward), bool(terminated or truncated)
            if len(timestep) == 4:
                obs, reward, done, _ = timestep
                return obs, float(reward), bool(done)
            obs, reward, done = timestep[:3]
            return obs, float(reward), bool(done)

        raise ValueError(f"Unexpected step/reset return type: {type(timestep)}")

    def reset(self) -> Dict[str, Any]:
        result = self._env.reset()
        if hasattr(result, "observation"):
            obs, _, _ = self._unpack_timestep(result)
        elif isinstance(result, (tuple, list)):
            obs = result[0]
        else:
            obs = result

        # ── Refresh pybullet block-ID cache every episode ─────────────────────
        # Body IDs may change if the env recreates the scene on reset.
        self._pb_block_ids = None   # force re-scan in _get_block_dist / _cache
        eff_xy = None
        if isinstance(obs, dict) and "effector_translation" in obs:
            eff_xy = np.asarray(obs["effector_translation"],
                                dtype=np.float32).ravel()[:2]
        self._cache_pybullet_block_ids(eff_xy)

        # ── One-time diagnostic: print shaping source ─────────────────────────
        if not getattr(self, "_obs_keys_printed", False):
            self._obs_keys_printed = True
            if isinstance(obs, dict):
                _keys = list(obs.keys())
                _non_eff_t = [k for k in _keys
                              if "translation" in k.lower() and "effector" not in k.lower()]
                print(f"[LT obs keys] {_keys}")
                print(f"[LT non-effector translation keys] {_non_eff_t}")
                d0 = self._get_block_dist(obs)
                n_pb = len(getattr(self, "_pb_block_ids", []))
                src  = ("obs-dict" if _non_eff_t
                        else f"pybullet({n_pb} bodies)" if d0 is not None
                        else "UNAVAILABLE — shaping disabled")
                print(f"[LT shaping] block_dist={d0:.4f}  scale={self._shaping_scale}"
                      f"  source={src}")
            else:
                print(f"[LT obs] type={type(obs).__name__} — not a dict, shaping disabled")

        # Initialise shaped-reward baseline for this episode
        self._prev_block_dist = self._get_block_dist(obs)
        return {"frame": self._extract_frame(obs),
                "instruction": self._extract_instruction(obs)}

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        idx = int(action_idx) % self._num_actions
        vel = (self._CODEBOOK[idx % len(self._CODEBOOK)] * self._vel_scale).astype(np.float32)

        result = self._env.step(vel)
        obs, reward, done = self._unpack_timestep(result)

        # ── Dense reward shaping ─────────────────────────────────────────────
        # Add (prev_dist − cur_dist) × scale each step.
        # Positive when robot moves blocks closer → gives REINFORCE a gradient
        # even when the sparse terminal reward is 0.  Scale = 0.1 keeps the
        # total shaped reward well below 1.0 so success_threshold = 1.0 still
        # correctly identifies task-completion episodes.
        if self._shaping_scale > 0:
            dist_now = self._get_block_dist(obs)
            if dist_now is not None and self._prev_block_dist is not None:
                reward += (self._prev_block_dist - dist_now) * self._shaping_scale
            self._prev_block_dist = dist_now

        return (
            {"frame": self._extract_frame(obs),
             "instruction": self._extract_instruction(obs)},
            reward, done, {},
        )

    def _extract_frame(self, obs) -> np.ndarray:
        frame = None
        if isinstance(obs, dict):
            for key in ("rgb", "image", "pixels", "obs"):
                if key in obs:
                    frame = np.asarray(obs[key])
                    break
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            frame = obs

        if frame is None or frame.size == 0:
            return np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)

        if frame.dtype != np.uint8:
            if frame.max() <= 1.0 + 1e-5:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)

        if frame.shape[:2] != (self._img_size, self._img_size):
            from PIL import Image
            frame = np.array(
                Image.fromarray(frame).resize(
                    (self._img_size, self._img_size), Image.BILINEAR)
            )
        return frame

    def _extract_instruction(self, obs) -> str:
        if isinstance(obs, dict):
            for key in ("instruction_str", "instruction", "task", "mission"):
                val = obs.get(key)
                if val is None:
                    continue
                if isinstance(val, (bytes, np.bytes_)):
                    val = bytes(val).decode("utf-8")
                elif isinstance(val, np.ndarray):
                    val = val.item() if val.ndim == 0 else bytes(val).decode("utf-8")
                val = str(val).strip()
                if val:
                    self._instruction = val
                    return val
        return self._instruction

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ── factory ────────────────────────────────────────────────────────────────────

def make_env(cfg: dict) -> BaseEnv:
    """
    Build the correct environment from config.

    env_id routing:
      "dummy"                           → RandomDummyEnv   (no install required)
      "language_table" / "lt" / "LT"   → LanguageTableEnv (pip install language-table)
      "babyai-*" / "BabyAI-*"          → BabyAIEnv        (pip install minigrid)
      "metaworld-*"                     → MetaWorldEnv     (pip install metaworld)
      anything else                     → SimEnv (Gymnasium wrapper)
    """
    env_id = cfg.get("env", {}).get("env_id", "dummy")
    env_id_lower = env_id.lower().replace("-", "_")

    if env_id == "dummy":
        return SimEnv(cfg)   # already defaults to RandomDummyEnv
    if env_id_lower in ("language_table", "lt", "languagetable", "language_table_env"):
        return LanguageTableEnv(cfg)
    if env_id_lower.startswith(("babyai", "minigrid")):
        return BabyAIEnv(cfg)
    if env_id_lower.startswith("metaworld"):
        return MetaWorldEnv(cfg)
    return SimEnv(cfg)


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
