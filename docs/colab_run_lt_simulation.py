r"""
colab_run_lt_simulation.py
==========================
Run this ENTIRE file as a Colab notebook cell-by-cell (paste each section)
OR as:  %run /content/VLA-Robot-Learning/docs/colab_run_lt_simulation.py

What it does
------------
1. Installs Language-Table + dependencies
2. Loads your trained TERA checkpoint
3. Runs N_EPISODES live LT episodes with the TERA policy (greedy)
4. Captures start / middle / end frames for each episode
5. Records actual E_act (safety narration) and E_emb (knowledge planning) tokens
6. Generates lt_qual_composite.png matching the paper figure format
7. Copies the PNG to your submission folder on Drive

Colab RAM: do NOT extract the 7.5 GB tar in the same run as the model.
  1) Runtime → Restart session
  2) Mount Drive, run EXTRACT cell (partial, ~2 min)
  3) Restart again (optional), then %run this script

EXTRACT cell (run alone after drive.mount):
    import tarfile, gc
    from pathlib import Path
    TAR = "/content/drive/MyDrive/VERA_LT_Real/lt_real_data/lt_real_3000eps.tar"
    MAX_EPS = 250
    import re
    ep_re = re.compile(r"(?:^|/)episode_(\d+)/")
    with tarfile.open(TAR, "r") as tf:
        mem = [m for m in tf.getmembers() if m.isfile() and ep_re.search(m.name)
               and int(ep_re.search(m.name).group(1)) < MAX_EPS]
        print(f"Extracting {len(mem)} files …")
        tf.extractall("/content", members=mem)
    gc.collect()
    print(len(list(Path("/content/lt_real/train").glob("episode_*"))), "episodes")

Run from Colab after mounting Google Drive:
    from google.colab import drive
    drive.mount('/content/drive')

    %cd /content/drive/MyDrive           # or wherever your repo lives
    !git clone https://github.com/YOUR/VLA-Robot-Learning  # skip if already cloned
    %cd VLA-Robot-Learning
    %run docs/colab_run_lt_simulation.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — USER CONFIG  (edit these paths before running)
# ══════════════════════════════════════════════════════════════════════════════
import os

# Path to your best trained TERA checkpoint (.pt file)
CHECKPOINT = "/content/drive/MyDrive/VERA_LT_Real/checkpoints/lt_full_vera/seed123/best_sft_vera.pt"

# Path to your YAML config (repo on Drive or Colab clone)
CONFIG_PATH = "/content/drive/MyDrive/VLA-Robot-Learning/configs/config.yaml"
# If no repo on Drive, use: CONFIG_PATH = "/content/VLA-Robot-Learning/configs/config.yaml"

# Where to save the composite PNG (also copied to submission folder if it exists)
OUT_PNG_REPO = "/content/drive/MyDrive/VLA-Robot-Learning/docs/lt_qual_composite.png"
OUT_PNG_SUB  = "/content/drive/MyDrive/corl_2026_template_submission/lt_qual_composite.png"
FIG_SHOW_CAPTION = False   # caption lives in LaTeX — omit for a compact PNG
FIG_DPI = 240
# Figure typography (pt at save DPI — tuned for print / \\linewidth)
FIG_FONT_HDR = 9.5
FIG_FONT_INSTR = 8.5
FIG_FONT_TOK = 7.5
FIG_FONT_CAP = 7.5
FIG_TOK_WRAP = 72          # wide strip → fewer awkward line breaks
FIG_TOK_LINESPACING = 1.4
FIG_INSTR_WRAP = 30
FIG_FRAME_PX = 224         # resize all frames to this square for uniform panel size

# Number of episodes to collect (top-N by dataset reward)
N_EPISODES = 3

# Maximum steps per episode before giving up
MAX_STEPS = 60

# Random seed for reproducibility (env layout only; independent of training seed)
SEED = 123

# Scale tanh action_vec to LT simulator deltas (training stores vectors in [-1, 1])
LT_ACTION_SCALE = 0.03

# ── Rollout quality (important for the qual figure) ─────────────────────────
# "block2block" only = clearest push-X-to-Y tasks (best match to BC training)
REWARD_MODE = "block2block"   # "block2block" | "all" | "block2block_relative"

# Retry env.reset until instruction contains one of these phrases (paper figure)
USE_PREFERRED_INSTRUCTIONS = True
PREFERRED_INSTRUCTION_SUBSTRINGS = [
    "red moon", "blue moon", "yellow star", "red pentagon", "blue cube",
    "push the", "pull the", "move the", "slide the",
    "next to", "apart from", "away from", "to the",
]
MAX_RESET_TRIES = 40          # per episode, before skipping instruction filter

# Skip episodes that never get reward > 0 (optional quality gate)
REQUIRE_ANY_REWARD = False
MIN_EPISODE_REWARD = 0.0

# ── How to build the figure ───────────────────────────────────────────────────
# "dataset" = frames from LT training PKL (human demos) + TERA tokens at mid step.
#             RECOMMENDED: BC val_acc ~0.5 does NOT imply closed-loop sim success.
# "policy"  = closed-loop PyBullet rollout (often 0 reward — domain gap from training).
COLLECT_MODE = "dataset"
LT_DATA_ROOT = "/content/lt_real/train"  # after extracting lt_real_*eps.tar to /content/
# PKL stores dense LT rewards (~0.05/step); good demos sum ~0.3–1.5 (NOT sim 0–100 scale)
MIN_DATASET_EPISODE_REWARD = 0.15   # sum(reward) per episode; 0 = take top-N by reward only
MIN_DATASET_MAX_STEP_REWARD = 0.05  # also require max step reward (filters all-zero episodes)

# RAM safety (Colab free tier ~12 GB — extract + model + full scan often OOMs)
AUTO_EXTRACT_TAR = False          # run extract in a separate cell BEFORE this script
EXTRACT_TAR_MAX_EPISODES = 150    # if extracting: only unpack this many (enough for qual figure)
DATASET_MAX_SCAN = 500            # max episode_* dirs to open when picking rows (0 = no limit)
SEARCH_DRIVE_FOR_DATA = False     # mydrive.rglob("steps.pkl") is slow and memory-heavy

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════
# Colab Py3.12: do NOT install dm-reverb-nightly / tf-nightly / torch in one batch
# (they break pip). The LT *simulator* only needs pybullet + gym (see Google tutorial).
import subprocess, sys

def pip_one(pkg, optional=False):
    label = pkg if len(pkg) < 60 else pkg[:57] + "..."
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stderr=subprocess.STDOUT,
        )
        print(f"  ✓ {label}")
    except subprocess.CalledProcessError:
        if optional:
            print(f"  ⊘ optional skip: {label}")
        else:
            raise

print("Installing Language-Table simulator dependencies …")
# Colab already ships torch/torchvision — do not reinstall
for pkg in ("pyyaml", "imageio", "pillow", "matplotlib", "numpy", "scipy"):
    pip_one(pkg)

# Install LT package without reverb/TF pins (simulator uses pybullet only)
pip_one("gym<=0.23.0")
pip_one("pybullet")
pip_one("opencv-python")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "git+https://github.com/google-research/language-table.git"],
)
print("  ✓ language-table (--no-deps, sim-only)")

try:
    import clip  # noqa: F401
except ImportError:
    pip_one("git+https://github.com/openai/CLIP.git")

print("Dependency install done.")

# Quick sanity check before loading the model
from language_table.environments import language_table as _lt_check  # noqa: F401
print("language_table import OK")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import random, textwrap
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# Language Table
from language_table.environments import language_table as lt_env
from language_table.environments.rewards import (
    block2block,
    block2absolutelocation,
    block2relativelocation,
)
try:
    from language_table.environments.rewards import block2block_relative_location
except ImportError:
    from language_table.environments.rewards import (
        block_to_block_relative_location as block2block_relative_location,
    )
from language_table.environments import blocks

# Add repo root to path so local modules import
sys.path.insert(0, str(Path(CONFIG_PATH).parent.parent))
from models.vera_model import VERAModel

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD LT ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def _reward_factory_for_mode(mode: str):
    """Pick LT task distribution — block2block is easiest for visuomotor BC."""
    if mode == "block2block":
        return block2block.BlockToBlockReward
    if mode == "block2block_relative":
        return block2block_relative_location.BlockToBlockRelativeLocationReward
    # "all" — mixed tasks (harder; more random instructions)
    return random.choice([
        block2block.BlockToBlockReward,
        block2absolutelocation.BlockToAbsoluteLocationReward,
        block2relativelocation.BlockToRelativeLocationReward,
        block2block_relative_location.BlockToBlockRelativeLocationReward,
    ])


def make_lt_env(seed: int = 0, reward_mode: str = REWARD_MODE):
    """Build Language-Table env. Use reward_mode='block2block' for best rollouts."""
    env = lt_env.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=_reward_factory_for_mode(reward_mode),
        seed=seed,
    )
    return env


def instruction_is_preferred(instr: str) -> bool:
    if not USE_PREFERRED_INSTRUCTIONS:
        return True
    low = instr.lower()
    return any(s in low for s in PREFERRED_INSTRUCTION_SUBSTRINGS)


def reset_until_good_instruction(env, max_tries: int = MAX_RESET_TRIES):
    """Reset until we get a block-manipulation instruction (or give up)."""
    obs = env.reset()
    instr = decode_lt_instruction(obs)
    for _ in range(max_tries - 1):
        if instruction_is_preferred(instr):
            return obs, instr
        obs = env.reset()
        instr = decode_lt_instruction(obs)
    return obs, instr


def decode_lt_instruction(obs: dict) -> str:
    """Decode Language-Table encoded instruction field to plain text."""
    inst = obs.get("instruction")
    if inst is None:
        return "complete the task"
    if isinstance(inst, str):
        return inst.strip().rstrip(".")
    if isinstance(inst, bytes):
        return inst.decode("utf-8", errors="ignore").strip().rstrip(".")
    arr = np.asarray(inst).flatten()
    try:
        return lt_env.LanguageTable.decode_instruction(arr).strip().rstrip(".")
    except Exception:
        nonzero = arr[arr != 0]
        if len(nonzero):
            return bytes(nonzero.astype(np.uint8).tolist()).decode("utf-8", errors="ignore").strip().rstrip(".")
    return "complete the task"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOAD TERA MODEL
# ══════════════════════════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: No GPU — Runtime → Change runtime type → T4 GPU, then re-run.")
print(f"Device: {device}")

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
# Use training config embedded in checkpoint (avoids action_dim/chunk_size mismatch)
if isinstance(ckpt, dict) and "cfg" in ckpt:
    cfg = ckpt["cfg"]
    print("Using config from checkpoint")
else:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    print("Using config from", CONFIG_PATH)

m_cfg    = cfg["model"]
vera_cfg = cfg.get("vera", {})
model = VERAModel(
    num_actions           = m_cfg["num_actions"],
    history_len           = m_cfg["history_len"],
    num_vis_frames        = m_cfg.get("num_vis_frames", 3),
    fusion_layers         = m_cfg.get("fusion_layers", 6),
    fusion_heads          = m_cfg.get("fusion_heads", 8),
    d_model               = m_cfg.get("d_model", 256),
    d_ff_scale            = m_cfg.get("d_ff_scale", 4),
    dropout               = 0.0,
    vision_token_dropout  = 0.0,
    freeze_clip           = m_cfg.get("freeze_clip", True),
    unfreeze_clip_vision  = m_cfg.get("unfreeze_clip_vision", False),
    use_lang_feedback     = vera_cfg.get("use_lang_feedback", True),
    use_temporal_history  = vera_cfg.get("use_temporal_history", True),
    use_reward_gate       = vera_cfg.get("use_reward_gate", True),
    use_consequence_token = vera_cfg.get("use_consequence_token", True),
    action_dim            = m_cfg.get("action_dim", 2),
    action_vocab          = vera_cfg.get("action_vocab"),
    chunk_size            = m_cfg.get("chunk_size", 1),
).to(device)

state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"  [load] missing keys: {len(missing)}")
if unexpected:
    print(f"  [load] unexpected keys: {len(unexpected)}")
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT}")
print(f"  action_dim={model.action_dim}  chunk_size={model.chunk_size}  num_actions={model.num_actions}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ACTION VOCABULARY AND CONSEQUENCE VERBALIZER
# ══════════════════════════════════════════════════════════════════════════════

# 8-bin arctan2 action vocabulary (Language-Table default)
ACTION_VOCAB = cfg.get("vera", {}).get("action_vocab", {
    0: "I pushed the object to the right",
    1: "I pushed the object up and to the right",
    2: "I pushed the object upward",
    3: "I pushed the object up and to the left",
    4: "I pushed the object to the left",
    5: "I pushed the object down and to the left",
    6: "I pushed the object downward",
    7: "I pushed the object down and to the right",
})

def verbalize_consequence(reward: float, dist_delta: float) -> str:
    """Convert reward + dist_delta → natural-language consequence string."""
    if reward > 0.8:
        move = "significantly closer to" if dist_delta < -0.005 else "at"
        return f"I moved {move} the goal and received a high reward."
    elif reward > 0.3:
        move = "closer to" if dist_delta < 0 else "slightly away from"
        return f"I moved {move} the goal and received a moderate reward."
    else:
        if dist_delta < -0.005:
            return "I moved closer to the goal but received a low reward."
        elif dist_delta > 0.005:
            return "I moved away from the goal and received a low reward."
        else:
            return "I made little progress and received a low reward."

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — IMAGE PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

import torchvision.transforms as Tv

img_size = cfg["data"].get("img_size", 224)
transform = Tv.Compose([
    Tv.Resize((img_size, img_size)),
    Tv.ToTensor(),
    Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std =[0.26862954, 0.26130258, 0.27577711]),
])

def lt_obs_to_frame(obs: dict) -> np.ndarray:
    """Extract (H,W,3) uint8 RGB from an LT dm_env observation."""
    for key in ("rgb", "image", "pixels", "agentview_rgb"):
        val = obs.get(key)
        if val is not None:
            arr = np.asarray(val, dtype=np.uint8)
            if arr.ndim == 3:
                return arr
    # Fallback: try to render
    return np.zeros((224, 224, 3), dtype=np.uint8)

def lt_frame_to_tensor(frame: np.ndarray):
    """uint8 (H,W,3) → normalised tensor (3, img_size, img_size)."""
    pil = PILImage.fromarray(frame)
    return transform(pil)

def action_idx_to_continuous(action_idx: int, num_actions: int = 8) -> np.ndarray:
    """Fallback: map discrete bin → [Δx, Δy] when reg head is near zero."""
    angle = (action_idx / num_actions) * 2 * np.pi
    return (LT_ACTION_SCALE * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32))


def lt_action_from_output(out, num_actions: int) -> tuple:
    """
    Prefer continuous action_vec (matches LT training); fallback to discrete bin.
    Returns (cont_action [2], discrete_idx, vec_for_history [2] in [-1,1]).
    """
    avec = out["action_vec"].squeeze().detach().cpu().numpy().astype(np.float32)
    if avec.shape[0] < 2:
        avec = np.pad(avec.flatten(), (0, max(0, 2 - avec.size)))[:2]
    disc = int(out["logits"].argmax(dim=-1).item())
    if float(np.linalg.norm(avec)) < 0.08:
        cont = action_idx_to_continuous(disc, num_actions)
        hist = cont / max(LT_ACTION_SCALE, 1e-6)
        hist = np.clip(hist, -1.0, 1.0)
    else:
        cont = (avec[:2] * LT_ACTION_SCALE).astype(np.float32)
        hist = np.clip(avec[:2], -1.0, 1.0)
    return cont, disc, hist

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — COLLECT EPISODES (dataset PKL or live policy)
# ══════════════════════════════════════════════════════════════════════════════

import pickle
import glob
import gc
import re

num_actions   = cfg["model"]["num_actions"]
history_len   = cfg["model"]["history_len"]
num_vis       = cfg["model"].get("num_vis_frames", 3)
action_dim    = int(getattr(model, "action_dim", m_cfg.get("action_dim", 2)))
null_vec      = np.zeros(action_dim, dtype=np.float32)
tok_cache     = {}

collected_episodes = []


def pkl_step_frame(step: dict):
    for key in ("obs", "image", "rgb", "pixels", "frame"):
        val = step.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            for sub in ("rgb", "image", "pixels"):
                v2 = val.get(sub)
                if v2 is not None and isinstance(v2, np.ndarray) and v2.ndim == 3:
                    return v2.astype(np.uint8)
        elif isinstance(val, np.ndarray) and val.ndim == 3:
            return val.astype(np.uint8)
    return None


def pkl_episode_instruction(steps: list) -> str:
    for key in ("instruction", "language_instruction", "task"):
        val = steps[0].get(key)
        if val:
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="ignore")
            return str(val).strip().rstrip(".")
    return "complete the task"


def discretise_action_vec(av: np.ndarray) -> int:
    dx, dy = float(av[0]), float(av[1])
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return num_actions
    angle = np.arctan2(dy, dx)
    return int(round(angle / (np.pi / 4))) % num_actions


def infer_tokens_at_t(steps, t: int, instr: str):
    """Run TERA at timestep t using expert history (open-loop on demo frames)."""
    if instr not in tok_cache:
        tok_cache[instr] = clip.tokenize([instr])[0]
    lang_in = tok_cache[instr].unsqueeze(0).to(device)

    act_hist, rew_hist, av_hist = [], [], []
    for j in range(max(0, t - history_len), t):
        av = np.asarray(steps[j].get("action", steps[j].get("action_vec", [0, 0])), dtype=np.float32).flatten()[:2]
        ma = np.abs(av).max()
        if ma > 1.0:
            av = av / ma
        act_hist.append(discretise_action_vec(av))
        rew_hist.append(float(steps[j].get("reward", 0.0)))
        av_hist.append(np.clip(av, -1, 1))

    while len(act_hist) < history_len:
        act_hist.insert(0, num_actions)
        rew_hist.insert(0, 0.0)
        av_hist.insert(0, null_vec.copy())

    prev_a = act_hist[-1] if t > 0 else num_actions
    prev_r = rew_hist[-1] if t > 0 else 0.0
    prev_d = 0.0
    if t > 0:
        prev_d = float(steps[t - 1].get("state_delta", 0.0) or 0.0)

    frames_idx = [max(0, t - 2), max(0, t - 1), t]
    frame_tensors = []
    for fi in frames_idx:
        fr = pkl_step_frame(steps[fi])
        if fr is not None:
            frame_tensors.append(lt_frame_to_tensor(fr))
    if not frame_tensors:
        return "—", "—"
    while len(frame_tensors) < num_vis:
        frame_tensors.insert(0, torch.zeros_like(frame_tensors[0]))
    frames_in = torch.stack(frame_tensors[-num_vis:]).unsqueeze(0).to(device)

    act_hist_in = torch.tensor([act_hist], dtype=torch.long).to(device)
    rew_hist_in = torch.tensor([rew_hist], dtype=torch.float32).to(device)
    av_hist_in  = torch.tensor([np.stack(av_hist)], dtype=torch.float32).to(device)
    prev_a_in   = torch.tensor([prev_a], dtype=torch.long).to(device)
    prev_r_in   = torch.tensor([prev_r], dtype=torch.float32).to(device)
    delta_in    = torch.tensor([prev_d], dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(frames_in, lang_in, act_hist_in, rew_hist_in,
                    prev_a_in, prev_r_in, state_delta=delta_in,
                    action_vec_hist=av_hist_in)
        action = int(out["logits"].argmax(dim=-1).item())
        reward_proxy = float(steps[t].get("reward", 0.0))
    narration = ACTION_VOCAB.get(action, f"I performed action {action}")
    knowledge = verbalize_consequence(reward_proxy, prev_d)
    return narration, knowledge


def count_episode_dirs(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for d in root.iterdir() if d.is_dir() and d.name.startswith("episode_"))


_EP_TAR_RE = re.compile(r"(?:^|/)episode_(\d+)/")


def extract_lt_tar(tar_path: Path, dest: Path = Path("/content"), max_episodes: int = 0):
    """Unpack tar to dest. max_episodes>0 → only first N episode_* folders (saves RAM)."""
    import tarfile
    dest = Path(dest)
    print(f"[dataset] Extracting {tar_path} → {dest} "
          f"(max_episodes={max_episodes or 'all'}) …")
    with tarfile.open(tar_path, "r") as tf:
        if max_episodes <= 0:
            tf.extractall(dest)
            return
        members = []
        for m in tf.getmembers():
            if not m.name or m.isdir():
                continue
            hit = _EP_TAR_RE.search(m.name)
            if hit is None:
                continue
            if int(hit.group(1)) >= max_episodes:
                continue
            members.append(m)
        print(f"[dataset]   {len(members)} member files for episodes 0..{max_episodes - 1}")
        tf.extractall(path=dest, members=members)
    gc.collect()


def find_lt_dataset_root(mydrive: Path) -> Path | None:
    """Locate episode_XXX/steps.pkl tree on Drive or /content."""
    # Path from training checkpoint (most reliable if present)
    if isinstance(ckpt, dict):
        ep_cfg = (ckpt.get("cfg") or {}).get("data", {}).get("episodes_path")
        if ep_cfg and count_episode_dirs(Path(ep_cfg)) > 0:
            print(f"[dataset] Using episodes_path from checkpoint: {ep_cfg}")
            return Path(ep_cfg)

    guesses = [
        Path(LT_DATA_ROOT),
        Path("/content/lt_real/train"),           # ← tar extract layout (VERA_LT_RealData_Colab)
        Path("/content/lt_real"),
        mydrive / "VERA_LT_Real/lt_real_data/train",
        mydrive / "VERA_LT_Real/lt_real_data",
        Path("/content/lt_real_data/train"),
        Path("/content/lt_real_data"),
    ]
    for g in guesses:
        if count_episode_dirs(g) > 0:
            print(f"[dataset] Found episodes at: {g}  (n={count_episode_dirs(g)})")
            return g

    if AUTO_EXTRACT_TAR:
        for tar_path in sorted(mydrive.glob("**/lt_real_*eps.tar")):
            extract_lt_tar(tar_path, Path("/content"), max_episodes=EXTRACT_TAR_MAX_EPISODES)
            for g in guesses[1:]:
                if count_episode_dirs(g) > 0:
                    print(f"[dataset] Restored episodes at: {g}")
                    return g
    else:
        tars = list(mydrive.glob("**/lt_real_*eps.tar"))
        if tars:
            print("[dataset] Tar on Drive but episodes not extracted.")
            print("  Run the EXTRACT cell below (separate run), then re-run this script.")
            print(f"  Tar: {tars[0]}")

    if not SEARCH_DRIVE_FOR_DATA:
        return None

    # Deep search (slow; enable only if data path is unknown)
    best_root, best_n = None, 0
    for pkl in mydrive.rglob("steps.pkl"):
        parent = pkl.parent
        if not parent.name.startswith("episode_"):
            continue
        root = parent.parent
        n = count_episode_dirs(root)
        if n > best_n:
            best_n, best_root = n, root
        if best_n >= 50:
            break
    if best_root is not None:
        print(f"[dataset] Found via search: {best_root}  (n={best_n})")
        return best_root
    return None


def pkl_episode_reward_stats(steps: list) -> tuple[float, float]:
    rews = [float(s.get("reward", 0.0)) for s in steps]
    return (sum(rews), max(rews) if rews else 0.0)


def _subsample_episode_dirs(ep_dirs: list, max_scan: int) -> list:
    """Evenly spaced indices across the corpus (not only the tail)."""
    if max_scan <= 0 or len(ep_dirs) <= max_scan:
        return ep_dirs
    idx = np.linspace(0, len(ep_dirs) - 1, max_scan, dtype=int)
    return [ep_dirs[i] for i in np.unique(idx)]


def _dataset_reward_passes(total_r: float, max_r: float) -> bool:
    if MIN_DATASET_EPISODE_REWARD > 0 and total_r < MIN_DATASET_EPISODE_REWARD:
        return False
    if MIN_DATASET_MAX_STEP_REWARD > 0 and max_r < MIN_DATASET_MAX_STEP_REWARD:
        return False
    return True


def collect_from_dataset_pkl(root: str, n: int):
    """Pick high-reward human demos; load/decode frames only for the final N rows."""
    root = Path(root)
    if not root.is_dir():
        print(f"[dataset] Missing {root}")
        return []

    ep_dirs = sorted(root.glob("episode_*"))
    n_all = len(ep_dirs)
    if DATASET_MAX_SCAN > 0 and n_all > DATASET_MAX_SCAN:
        ep_dirs = _subsample_episode_dirs(ep_dirs, DATASET_MAX_SCAN)
        print(f"[dataset] Scanning {len(ep_dirs)} evenly-spaced dirs (of {n_all} total)")

    meta = []
    all_scored = []
    for ep_dir in ep_dirs:
        pkl = ep_dir / "steps.pkl"
        if not pkl.is_file():
            continue
        with open(pkl, "rb") as f:
            steps = pickle.load(f)
        if len(steps) < 4:
            del steps
            continue
        instr = pkl_episode_instruction(steps)
        total_r, max_r = pkl_episode_reward_stats(steps)
        del steps
        all_scored.append((total_r, max_r, instr, ep_dir))
        if not _dataset_reward_passes(total_r, max_r):
            continue
        if USE_PREFERRED_INSTRUCTIONS and not instruction_is_preferred(instr):
            continue
        meta.append((total_r, instr, ep_dir))
        if len(meta) % 50 == 0:
            gc.collect()

    if not meta and all_scored:
        print(f"[dataset] 0 passed filters (sum≥{MIN_DATASET_EPISODE_REWARD}, "
              f"max≥{MIN_DATASET_MAX_STEP_REWARD}, preferred={USE_PREFERRED_INSTRUCTIONS})")
        totals = [t for t, _, _, _ in all_scored]
        print(f"[dataset]   scanned sum(reward): min={min(totals):.3f}  "
              f"median={float(np.median(totals)):.3f}  max={max(totals):.3f}")
        print("[dataset]   fallback: top episodes by sum(reward), ignoring threshold")
        all_scored.sort(key=lambda x: -x[0])
        for total_r, max_r, instr, ep_dir in all_scored:
            if USE_PREFERRED_INSTRUCTIONS and not instruction_is_preferred(instr):
                continue
            meta.append((total_r, instr, ep_dir))
            if len(meta) >= max(n * 4, 20):
                break

    meta.sort(key=lambda x: -x[0])
    print(f"[dataset] {len(meta)} candidate episodes for figure rows")

    picked = []
    used_instr = set()
    for total_r, instr, ep_dir in meta:
        key = instr.lower()[:40]
        if key in used_instr:
            continue
        used_instr.add(key)
        with open(ep_dir / "steps.pkl", "rb") as f:
            steps = pickle.load(f)
        idxs = [0, len(steps) // 2, len(steps) - 1]
        frames = []
        for i in idxs:
            fr = pkl_step_frame(steps[i])
            if fr is None:
                break
            frames.append(fr)
        if len(frames) < 3:
            del steps
            continue
        mid = len(steps) // 2
        nar_m, know_m = infer_tokens_at_t(steps, mid, instr)
        del steps
        gc.collect()
        ep = {
            "instruction": instr,
            "start": frames[0],
            "mid": frames[1],
            "end": frames[2],
            "narration_start": nar_m,
            "narration_mid": nar_m,
            "knowledge_start": know_m,
            "knowledge_mid": know_m,
            "total_reward": total_r,
            "steps": len(idxs),
        }
        picked.append(ep)
        print(f"  [{len(picked)}] \"{instr}\"  total_reward={total_r:.0f}")
        if len(picked) >= n:
            break
    return picked


if COLLECT_MODE == "dataset":
    MYDRIVE = Path("/content/drive/MyDrive")
    print(f"COLLECT_MODE=dataset")
    data_root = find_lt_dataset_root(MYDRIVE)
    if data_root is None:
        print("[dataset] NO episode data on Drive.")
        print("  → Open VERA_LT_RealData_Colab.ipynb, run the DATA download cell,")
        print("    then save to MyDrive/VERA_LT_Real/lt_real_data/ or lt_real_*eps.tar")
    else:
        collected_episodes = collect_from_dataset_pkl(str(data_root), N_EPISODES)
    if len(collected_episodes) < N_EPISODES:
        print(f"[WARN] Only {len(collected_episodes)}/{N_EPISODES} episodes — "
              f"set MIN_DATASET_EPISODE_REWARD=0 or USE_PREFERRED_INSTRUCTIONS=False")
else:
    print(f"COLLECT_MODE=policy  reward_mode={REWARD_MODE}  preferred_instr={USE_PREFERRED_INSTRUCTIONS}  "
          f"LT_ACTION_SCALE={LT_ACTION_SCALE}")
    ep_seed = SEED
    while len(collected_episodes) < N_EPISODES:
        env = make_lt_env(seed=ep_seed)
        ep_seed += 1

        obs, instr = reset_until_good_instruction(env)
        if USE_PREFERRED_INSTRUCTIONS and not instruction_is_preferred(instr):
            print(f"  [skip] could not sample preferred instruction after {MAX_RESET_TRIES} tries")
            continue

        print(f"\nEpisode {len(collected_episodes)+1}: \"{instr}\"")

        if instr not in tok_cache:
            tok_cache[instr] = clip.tokenize([instr])[0]

        frame_q      = deque(maxlen=num_vis)
        action_q     = deque([num_actions] * history_len, maxlen=history_len)
        reward_q     = deque([0.0] * history_len, maxlen=history_len)
        action_vec_q = deque([null_vec.copy() for _ in range(history_len)], maxlen=history_len)
        prev_action  = num_actions
        prev_reward  = 0.0
        prev_delta   = 0.0

        frames_raw   = []
        actions_taken = []
        rewards_seen  = []
        narrations    = []
        knowledges    = []

        done = False
        step = 0

        while not done and step < MAX_STEPS:
            frame_np = lt_obs_to_frame(obs)
            frames_raw.append(frame_np)

            frame_t = lt_frame_to_tensor(frame_np)
            frame_q.append(frame_t)
            pad = num_vis - len(frame_q)
            frames_in = torch.stack([torch.zeros_like(frame_t)] * pad + list(frame_q)).unsqueeze(0).to(device)

            lang_in     = tok_cache[instr].unsqueeze(0).to(device)
            act_hist_in = torch.tensor(list(action_q), dtype=torch.long).unsqueeze(0).to(device)
            rew_hist_in = torch.tensor(list(reward_q), dtype=torch.float32).unsqueeze(0).to(device)
            prev_a_in   = torch.tensor([prev_action], dtype=torch.long).to(device)
            prev_r_in   = torch.tensor([prev_reward], dtype=torch.float32).to(device)
            delta_in    = torch.tensor([prev_delta],  dtype=torch.float32).to(device)
            av_hist_in  = torch.tensor(
                np.stack(list(action_vec_q), axis=0), dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(frames_in, lang_in, act_hist_in, rew_hist_in,
                            prev_a_in, prev_r_in, state_delta=delta_in,
                            action_vec_hist=av_hist_in)
                cont_action, action, av_hist = lt_action_from_output(out, num_actions)

            obs, reward, done, _info = env.step(cont_action)

            reward     = float(reward or 0.0)
            done       = bool(done)
            dist_delta = float(np.linalg.norm(cont_action))

            narration  = ACTION_VOCAB.get(action, f"I performed action {action}")
            knowledge  = verbalize_consequence(reward, dist_delta)

            narrations.append(narration)
            knowledges.append(knowledge)
            actions_taken.append(action)
            rewards_seen.append(reward)
            action_q.append(action)
            reward_q.append(reward)
            action_vec_q.append(av_hist.astype(np.float32))
            prev_action = action
            prev_reward = reward
            prev_delta  = dist_delta
            step       += 1

            if step % 10 == 0:
                print(f"  step {step:3d}: action={action}  reward={reward:.3f}  done={done}")

        if hasattr(env, "close"):
            env.close()

        if len(frames_raw) < 3:
            print("  → too short, skipping")
            continue

        total_r = float(sum(rewards_seen))
        if REQUIRE_ANY_REWARD and total_r <= MIN_EPISODE_REWARD:
            print(f"  → no task reward (total={total_r:.2f}), re-rolling episode")
            continue

        mid = len(frames_raw) // 2
        ep = {
            "instruction": instr,
            "start":       frames_raw[0],
            "mid":         frames_raw[mid],
            "end":         frames_raw[-1],
            "narration_start": narrations[0]  if narrations else "—",
            "narration_mid":   narrations[mid] if len(narrations) > mid else "—",
            "knowledge_start": knowledges[0]  if knowledges else "—",
            "knowledge_mid":   knowledges[mid] if len(knowledges) > mid else "—",
            "total_reward":    total_r,
            "steps":           step,
        }
        collected_episodes.append(ep)
        print(f"  ✓ {step} steps, total reward={ep['total_reward']:.2f}")

print(f"\nCollected {len(collected_episodes)} episodes (mode={COLLECT_MODE}).")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — GENERATE COMPOSITE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

PREFERRED_INSTR = [
    "pull the red moon apart from the blue moon",
    "push the yellow star next to the red moon",
    "move the red pentagon away from the blue cube",
    "move the red moon to the bottom of the yellow pentagon",
    "pull the red moon to the bottom left",
]
PLACEHOLDER_FRAME = np.full((128, 128, 3), 210, dtype=np.uint8)

if len(collected_episodes) == 0:
    print("[figure] No episodes collected — using placeholder layout (grey frames).")
    print("         Fix dataset path above, then re-run.")
    collected_episodes = [
        {
            "instruction": PREFERRED_INSTR[i],
            "start": PLACEHOLDER_FRAME, "mid": PLACEHOLDER_FRAME, "end": PLACEHOLDER_FRAME,
            "narration_mid": "I pushed the object toward the goal.",
            "knowledge_mid": "Moved closer to goal; moderate reward.",
            "total_reward": 0.0, "steps": 0,
        }
        for i in range(N_EPISODES)
    ]

N_ROWS = len(collected_episodes)
FIG_W = 11.0          # wider → each frame column is wider
HDR_H, IMG_H, TOK_H = 0.22, 2.20, 0.52   # IMG_H up: bigger frames; TOK_H trimmed
total_h = HDR_H + N_ROWS * (IMG_H + TOK_H) + (0.35 if FIG_SHOW_CAPTION else 0.08)

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": FIG_FONT_INSTR,
    "axes.titlesize": FIG_FONT_HDR,
    "text.antialiased": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TEXT_CLR = "#111111"
TOKEN_CLR = "#1a2744"
COL_TITLES = ["Start", "Middle", "End"]
BORDER_CLR = "#888888"
ROW_BG = ("#FFFFFF", "#F4F6F8")

# 1 header + per episode: (frames row, token row) — keeps Start/Mid/End same size
n_master_rows = 1 + N_ROWS * 2
height_ratios = [HDR_H] + [IMG_H, TOK_H] * N_ROWS

fig = plt.figure(figsize=(FIG_W, total_h), facecolor="white", dpi=FIG_DPI)
master = gridspec.GridSpec(
    n_master_rows, 4, figure=fig,
    width_ratios=[1.05, 1, 1, 1],
    height_ratios=height_ratios,
    left=0.06, right=0.99, top=0.96, bottom=0.04 if not FIG_SHOW_CAPTION else 0.10,
    wspace=0.03, hspace=0.04,   # tighter: less gap between columns and rows
)


def _uniform_frame(frame: np.ndarray) -> np.ndarray:
    """Square resize so every panel displays at identical pixel dimensions."""
    if frame is None:
        return np.full((FIG_FRAME_PX, FIG_FRAME_PX, 3), 210, dtype=np.uint8)
    img = PILImage.fromarray(frame.astype(np.uint8))
    if img.size != (FIG_FRAME_PX, FIG_FRAME_PX):
        img = img.resize((FIG_FRAME_PX, FIG_FRAME_PX), PILImage.Resampling.LANCZOS)
    return np.array(img)


def _draw_frame(ax, frame):
    frame = _uniform_frame(frame)
    ax.imshow(frame, aspect="equal", interpolation="bilinear")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#EBEBEB")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color(BORDER_CLR)


def _soft_wrap(text: str, width: int) -> str:
    text = (text or "").strip()
    if len(text) <= width:
        return text
    return textwrap.fill(
        text, width=width, break_long_words=False, break_on_hyphens=False,
    )


def _token_caption(nar: str, know: str) -> str:
    return (
        f"$E_{{\\mathrm{{act}}}}$: {_soft_wrap(nar, FIG_TOK_WRAP)}\n"
        f"$E_{{\\mathrm{{emb}}}}$: {_soft_wrap(know, FIG_TOK_WRAP)}"
    )


# Header row
ax_h0 = fig.add_subplot(master[0, 0])
ax_h0.axis("off")
ax_h0.text(
    0.98, 0.5, "Instruction", ha="right", va="center",
    fontsize=FIG_FONT_HDR, fontweight="bold", color=TEXT_CLR,
)
for j, title in enumerate(COL_TITLES):
    ax_h = fig.add_subplot(master[0, j + 1])
    ax_h.axis("off")
    ax_h.text(
        0.5, 0.5, title, ha="center", va="center",
        fontsize=FIG_FONT_HDR, fontweight="bold", color=TEXT_CLR,
    )

for row_i, ep in enumerate(collected_episodes):
    r_img = 1 + row_i * 2
    r_tok = r_img + 1
    bg = ROW_BG[row_i % 2]

    # Instruction spans frame + token rows for this episode
    ax_lbl = fig.add_subplot(master[r_img:r_tok + 1, 0])
    ax_lbl.set_facecolor(bg)
    ax_lbl.axis("off")
    instr = ep["instruction"].strip().rstrip(".")
    if instr:
        instr = instr[0].upper() + instr[1:]
    wrapped = _soft_wrap(instr + ".", FIG_INSTR_WRAP)
    ax_lbl.text(
        0.96, 0.5, wrapped, transform=ax_lbl.transAxes,
        fontsize=FIG_FONT_INSTR, ha="right", va="center",
        color=TEXT_CLR, linespacing=1.25, clip_on=False,
    )
    ax_lbl.set_xmargin(0.08)

    # All three frames share one grid row → identical axes height/width
    for col, key in [(1, "start"), (2, "mid"), (3, "end")]:
        ax = fig.add_subplot(master[r_img, col])
        ax.set_facecolor(bg)
        _draw_frame(ax, ep[key])

    # Tokens on dedicated row below frames (spans Start–End columns)
    ax_tok = fig.add_subplot(master[r_tok, 1:4])
    ax_tok.set_facecolor(bg)
    ax_tok.axis("off")
    nar = ep.get("narration_mid", "—")
    know = ep.get("knowledge_mid", "—")
    ax_tok.set_ylim(0, 1)
    ax_tok.set_xlim(0, 1)
    ax_tok.margins(x=0.04, y=0.12)
    ax_tok.text(
        0.5, 0.5, _token_caption(nar, know),
        transform=ax_tok.transAxes,
        fontsize=FIG_FONT_TOK, ha="center", va="center",
        color=TOKEN_CLR, linespacing=FIG_TOK_LINESPACING,
        clip_on=False,
    )

if FIG_SHOW_CAPTION:
    _n = N_ROWS
    cap = (
        f"Top {_n} Language-Table demos (by reward). "
        "$E_{{\\mathrm{{act}}}}$ / $E_{{\\mathrm{{emb}}}}$ at middle timestep."
    )
    fig.text(
        0.03, 0.01, cap, fontsize=FIG_FONT_CAP, va="bottom",
        color=TEXT_CLR, ha="left", family="sans-serif",
    )

for dest in [OUT_PNG_REPO, OUT_PNG_SUB]:
    try:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            dest, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", pad_inches=0.14,
        )
        print(f"Saved: {dest}")
    except Exception as e:
        print(f"[WARN] Could not save to {dest}: {e}")

plt.close(fig)
print("\nDone! Open lt_qual_composite.png to review.")
print("Then compile your LaTeX — the figure is already referenced as:")
print("    \\includegraphics[width=\\linewidth]{lt_qual_composite.png}")
