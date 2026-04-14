"""
collect_minigrid_demos.py
=========================
Collect MiniGrid demonstration trajectories and save them in the episode
format expected by TrajectoryDataset / sft_trainer.py:

  episode = {
      "frames":      np.ndarray  (T, H, W, 3) uint8
      "instruction": str
      "actions":     np.ndarray  (T,) int64
      "rewards":     np.ndarray  (T,) float32
  }

Policy options (--policy):
  random     — uniformly random actions (baseline; many episodes won't succeed)
  forward    — biased toward 'move forward' (action 2) to reach goal faster

For MiniGrid-Empty-5x5-v0 the forward-biased policy succeeds ~30% of the time
within 50 steps, giving a reasonable mix of success and failure demonstrations.

Usage:
  # Collect 200 episodes, save to data/minigrid_demos.pkl
  python scripts/collect_minigrid_demos.py

  # Custom settings
  python scripts/collect_minigrid_demos.py \
      --env MiniGrid-Empty-5x5-v0 \
      --episodes 500 \
      --max-steps 100 \
      --policy forward \
      --output data/minigrid_demos.pkl \
      --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.minigrid_env import MiniGridEnv
from data.trajectory_dataset import save_episodes


# ── Policies ───────────────────────────────────────────────────────────────────

def random_policy(obs, num_actions: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, num_actions))


def forward_biased_policy(obs, num_actions: int, rng: np.random.Generator) -> int:
    """
    Weights: forward (2) = 50%, turn left (0) = 20%, turn right (1) = 20%,
             remaining = 10% spread across actions 3-6.
    This gives more purposeful movement than pure random.
    """
    weights = [0.20, 0.20, 0.50, 0.03, 0.03, 0.02, 0.02]
    weights = weights[:num_actions]
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()
    return int(rng.choice(num_actions, p=weights))


def expert_policy(obs, num_actions: int, rng: np.random.Generator, env: MiniGridEnv) -> int:
    """
    Simple state-conditioned expert for navigation in MiniGrid-Empty-5x5-v0.
    Uses the current agent pose and goal location to choose among:
      0 = turn left
      1 = turn right
      2 = move forward
    """
    base_env = env._env.unwrapped
    agent_x, agent_y = base_env.agent_pos
    agent_dir = base_env.agent_dir

    goal_pos = None
    for x in range(base_env.grid.width):
        for y in range(base_env.grid.height):
            cell = base_env.grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "goal":
                goal_pos = (x, y)
                break
        if goal_pos is not None:
            break

    if goal_pos is None:
        return 2 if num_actions > 2 else 0

    goal_x, goal_y = goal_pos
    dx = goal_x - agent_x
    dy = goal_y - agent_y

    if abs(dx) >= abs(dy):
        desired_dir = 0 if dx > 0 else 2 if dx < 0 else (1 if dy > 0 else 3)
    else:
        desired_dir = 1 if dy > 0 else 3

    if (agent_x, agent_y) == goal_pos:
        return 2 if num_actions > 2 else 0

    if agent_dir == desired_dir:
        return 2

    turn_right_steps = (desired_dir - agent_dir) % 4
    turn_left_steps = (agent_dir - desired_dir) % 4
    return 1 if turn_right_steps < turn_left_steps else 0


def expert_noisy_policy(obs, num_actions: int, rng: np.random.Generator, env: MiniGridEnv) -> int:
    if rng.random() < 0.10:
        return int(rng.choice([0, 1, 2]))
    return expert_policy(obs, num_actions, rng, env)


def expert_recovery_policy(obs, num_actions: int, rng: np.random.Generator, env: MiniGridEnv) -> int:
    burst_steps = getattr(expert_recovery_policy, "_burst_steps", 0)
    if burst_steps > 0:
        expert_recovery_policy._burst_steps = burst_steps - 1
        return int(rng.choice([0, 1, 2]))

    if rng.random() < 0.10:
        expert_recovery_policy._burst_steps = int(rng.integers(1, 4)) - 1
        return int(rng.choice([0, 1, 2]))

    return expert_policy(obs, num_actions, rng, env)


POLICIES = {
    "random":  random_policy,
    "forward": forward_biased_policy,
    "expert":  expert_policy,
    "expert_noisy": expert_noisy_policy,
    "expert_recovery": expert_recovery_policy,
}


# ── Collection loop ────────────────────────────────────────────────────────────

def collect(
    env_id: str,
    num_episodes: int,
    max_steps: int,
    policy_name: str,
    output_path: str,
    seed: int,
    save_every: int,
) -> None:
    rng = np.random.default_rng(seed)
    policy_fn = POLICIES[policy_name]

    cfg = {
        "env": {
            "env_id":      env_id,
            "instruction": None,   # use per-env default
        },
        "data": {"img_size": 64},   # collect at 64px; dataset resizes to 224 on load
        "rl":  {"max_episode_steps": max_steps},
    }
    env = MiniGridEnv(cfg)
    num_actions = MiniGridEnv.NUM_ACTIONS

    print(f"Collecting {num_episodes} episodes from {env_id}")
    print(f"  Policy:    {policy_name}")
    print(f"  Max steps: {max_steps}")
    print(f"  Output:    {output_path}")
    print(f"  Seed:      {seed}")
    print()

    episodes = []
    successes = 0

    try:
        for ep_idx in range(num_episodes):
            obs = env.reset()
            instruction = obs["instruction"]
            if policy_name == "expert_recovery":
                expert_recovery_policy._burst_steps = 0

            frames, actions, rewards = [], [], []
            done = False
            steps = 0

            while not done and steps < max_steps:
                frames.append(obs["frame"].copy())  # (H, W, 3)
                if policy_name in {"expert", "expert_noisy", "expert_recovery"}:
                    action = policy_fn(obs, num_actions, rng, env)
                else:
                    action = policy_fn(obs, num_actions, rng)
                obs, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                steps += 1

            # Append the final frame so frames and actions are same length
            frames.append(obs["frame"].copy())

            ep_return = sum(rewards)
            if ep_return > 0:
                successes += 1

            episode = {
                "frames":      np.stack(frames[:-1], axis=0),   # (T, H, W, 3)
                "instruction": instruction,
                "actions":     np.array(actions, dtype=np.int64),
                "rewards":     np.array(rewards, dtype=np.float32),
            }
            episodes.append(episode)

            if (ep_idx + 1) % 50 == 0 or ep_idx == 0:
                print(f"  [{ep_idx+1:4d}/{num_episodes}] "
                      f"steps={steps:3d}  return={ep_return:.3f}  "
                      f"success_rate={successes/(ep_idx+1)*100:.1f}%")

            if save_every > 0 and (ep_idx + 1) % save_every == 0:
                save_episodes(episodes, output_path)
                print(f"  [save] partial save: {len(episodes)} episodes -> {output_path}")
    except KeyboardInterrupt:
        save_episodes(episodes, output_path)
        print(f"\n[interrupt] KeyboardInterrupt caught. Saved {len(episodes)} episodes to {output_path}")
        env.close()
        return

    env.close()

    save_episodes(episodes, output_path)
    print(f"\nSaved {len(episodes)} episodes to {output_path}")
    print(f"Success rate: {successes}/{num_episodes} = {successes/num_episodes*100:.1f}%")
    total_steps = sum(len(ep["actions"]) for ep in episodes)
    print(f"Total timesteps: {total_steps:,}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Collect MiniGrid demo trajectories")
    p.add_argument("--env",      default="MiniGrid-Empty-5x5-v0",
                   help="MiniGrid env id (default: MiniGrid-Empty-5x5-v0)")
    p.add_argument("--episodes", type=int, default=200,
                   help="Number of episodes to collect (default: 200)")
    p.add_argument("--max-steps", type=int, default=100,
                   help="Max steps per episode (default: 100)")
    p.add_argument("--policy",   default="forward", choices=list(POLICIES),
                   help="Collection policy: random | forward | expert | expert_noisy | expert_recovery (default: forward)")
    p.add_argument("--output",   default="data/minigrid_demos.pkl",
                   help="Output .pkl path (default: data/minigrid_demos.pkl)")
    p.add_argument("--seed",     type=int, default=0,
                   help="Random seed (default: 0)")
    p.add_argument("--save-every", type=int, default=25,
                   help="Save partial progress every N completed episodes (default: 25)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(
        env_id=args.env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        policy_name=args.policy,
        output_path=args.output,
        seed=args.seed,
        save_every=args.save_every,
    )
