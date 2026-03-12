# VLA-Robot-Learning

**Vision-Language-Action model with RL feedback for robot next-action prediction.**

Predicts the logical next discrete action given:
- **Video** — recent camera frames (CLIP ViT-B/32 encoder)
- **Language** — natural-language task instruction (CLIP text encoder)
- **Action-Reward history** — previous `(action, reward)` pairs as RL feedback

---

## Architecture

```
[Video frames]      → CLIP ViT-B/32 → vis_tokens   (B, T, 256)
[Language instr.]   → CLIP text enc  → lang_token   (B, 1, 256)
[(action, reward)×H]→ AR encoder     → hist_tokens  (B, H, 256)
                                              ↓
                           [CLS | lang | vis | hist]
                                              ↓
                           Causal Transformer (6 layers, pre-norm)
                                              ↓
                               CLS token → Action head
                                              ↓
                             Logits → N discrete actions
```

---

## Two-Phase Training

### Phase 1 — Behavioral Cloning (SFT)
Learn from expert demonstrations via cross-entropy loss.

```bash
python -m training.sft_trainer --config configs/config.yaml
```

### Phase 2 — Online RL Fine-Tuning (REINFORCE + baseline)
Roll out in the environment, collect `(s, a, r)` trajectories,
update with REINFORCE + value baseline + KL penalty against BC checkpoint.

```bash
python -m training.rl_trainer --config configs/config.yaml
```

---

## Evaluation

```bash
# Evaluate RL checkpoint (deterministic greedy)
python -m evaluation.evaluate \
    --config configs/config.yaml \
    --checkpoint checkpoints/rl/best_rl.pt \
    --episodes 50

# Stochastic (sample from policy)
python -m evaluation.evaluate \
    --config configs/config.yaml \
    --checkpoint checkpoints/rl/best_rl.pt \
    --stochastic
```

---

## Project Structure

```
VLA-Robot-Learning/
├── models/
│   └── vla_model.py              # RLConditionedVLA + ActionRewardHistoryEncoder
├── training/
│   ├── sft_trainer.py            # Phase 1: behavioral cloning
│   └── rl_trainer.py             # Phase 2: REINFORCE + value baseline
├── data/
│   └── trajectory_dataset.py     # Episode dataset with sliding-window sampling
├── envs/
│   └── sim_env.py                # SimEnv (Gym wrapper) + RealEnv stub
├── evaluation/
│   └── evaluate.py               # Episode rollout + metrics
├── configs/
│   └── config.yaml               # All hyperparameters
├── checkpoints/                  # Saved model weights
└── logs/                         # Training logs
```

---

## Installation

```bash
pip install torch torchvision Pillow PyYAML numpy matplotlib
pip install git+https://github.com/openai/CLIP.git

# Optional: simulation environment
pip install gymnasium
pip install gym-robotics   # for FrankaKitchen, FetchReach
pip install minigrid       # for MiniGrid navigation tasks
```

---

## Using Your Own Data

Save your robot episodes as a Python list of dicts and pass to the dataset:

```python
episodes = [
    {
        "frames":      np.ndarray,  # (T, H, W, 3) uint8
        "instruction": str,          # natural-language goal
        "actions":     np.ndarray,  # (T,) int64 — discrete action indices
        "rewards":     np.ndarray,  # (T,) float32
    },
    ...
]

from data.trajectory_dataset import save_episodes
save_episodes(episodes, "data/my_episodes.pkl")
```

Then set `data.episodes_path: data/my_episodes.pkl` in `configs/config.yaml`.

---

## Connecting a Real Robot

Edit `envs/sim_env.py → RealEnv` to hook in your robot SDK,
then change the import in `training/rl_trainer.py`:

```python
# from envs.sim_env import SimEnv
from envs.sim_env import RealEnv as SimEnv
```
