r"""
colab_run_calvin_visualization.py
==================================
Run this ENTIRE file as a Colab cell-by-cell (paste each section)
OR as:  %run /content/VLA-Robot-Learning/docs/colab_run_calvin_visualization.py

What it does
------------
1. Installs CALVIN dependencies
2. Loads your trained TERA-CALVIN checkpoint (.pt)
3. Picks N_EPISODES high-quality demos from the CALVIN dataset (.npz)
4. Captures start / middle / end frames for each episode
5. Records actual E_act (safety narration) and E_emb (knowledge planning) tokens
6. Generates calvin_qual_composite.png matching the paper figure format

Dataset layout expected (CALVIN task_D_D):
  <CALVIN_DATA_ROOT>/
    training/
      episode_0000000.npz
      episode_0000001.npz
      ...
      lang_annotations/
        auto_lang_ann.npy

Run from Colab after mounting Google Drive:
    from google.colab import drive
    drive.mount('/content/drive')

    %cd /content/drive/MyDrive           # or wherever your repo lives
    !git clone https://github.com/sara-kaz/RLConditionedVLA  # skip if already cloned
    %cd RLConditionedVLA
    %run docs/colab_run_calvin_visualization.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — USER CONFIG  (edit these paths before running)
# ══════════════════════════════════════════════════════════════════════════════
import os

# Path to your best trained TERA-CALVIN checkpoint (.pt file)
CHECKPOINT = "/content/drive/MyDrive/VERA_CALVIN/checkpoints/calvin_vera/seed42/best_sft_vera.pt"

# Path to your CALVIN config YAML
CONFIG_PATH = "/content/drive/MyDrive/RLConditionedVLA/configs/calvin_config.yaml"
# If running from cloned repo: CONFIG_PATH = "/content/RLConditionedVLA/configs/calvin_config.yaml"

# Root of CALVIN dataset (dir containing training/ and validation/)
CALVIN_DATA_ROOT = "/content/drive/MyDrive/VERA_CALVIN/task_D_D"
CALVIN_SPLIT     = "training"   # "training" or "validation"

# Where to save the composite PNG
OUT_PNG_REPO = "/content/drive/MyDrive/RLConditionedVLA/docs/calvin_qual_composite.png"
OUT_PNG_SUB  = "/content/drive/MyDrive/corl_2026_template_submission/calvin_qual_composite.png"

FIG_SHOW_CAPTION = False   # caption lives in LaTeX — omit for a compact PNG
FIG_DPI = 240

# ── Figure typography ─────────────────────────────────────────────────────────
FIG_FONT_HDR   = 11.5
FIG_FONT_INSTR = 10.5
FIG_FONT_TOK   = 9.5
FIG_FONT_CAP   = 8.5
FIG_TOK_WRAP   = 38          # each token gets ~half the row width → ~38 chars
FIG_TOK_LINESPACING = 1.35
FIG_INSTR_WRAP = 26
FIG_FRAME_PX   = 224         # resize all frames to this square for uniform panel size

# Number of episodes to collect (top-N by task success / reward)
N_EPISODES = 3

# Max episode dirs to scan when picking rows (0 = no limit)
DATASET_MAX_SCAN = 600

# Seed for reproducibility (shuffling order only)
SEED = 42

# ── Preferred instructions (partial match; leave empty to take any) ────────────
USE_PREFERRED_INSTRUCTIONS = True
PREFERRED_INSTRUCTION_SUBSTRINGS = [
    "rotate", "push", "pick up", "lift", "place", "stack",
    "move", "slide", "open", "close", "put",
]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════
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

print("Installing CALVIN visualization dependencies …")
for pkg in ("pyyaml", "imageio", "pillow", "matplotlib", "numpy", "scipy"):
    pip_one(pkg)

try:
    import clip  # noqa: F401
except ImportError:
    pip_one("git+https://github.com/openai/CLIP.git")

print("Dependency install done.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import random, textwrap, gc
from pathlib import Path

import numpy as np
import torch
import yaml
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# Add repo root to sys.path so local modules import
sys.path.insert(0, str(Path(CONFIG_PATH).parent.parent))
from models.vera_model import VERAModel

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD TERA-CALVIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: No GPU — Runtime → Change runtime type → T4 GPU, then re-run.")
print(f"Device: {device}")

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
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
    num_actions           = m_cfg["num_actions"],       # 14 for CALVIN
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
    action_dim            = m_cfg.get("action_dim", 7),   # 7-DoF for CALVIN
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
print(f"  action_dim={model.action_dim}  num_actions={model.num_actions}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ACTION VOCABULARY AND CONSEQUENCE VERBALIZER
# ══════════════════════════════════════════════════════════════════════════════

# 14-bin direction-aware CALVIN action vocabulary (matches calvin_config.yaml)
_DEFAULT_CALVIN_VOCAB = {
    0:  "I moved the end-effector to the right",
    1:  "I moved the end-effector to the left",
    2:  "I moved the end-effector forward",
    3:  "I moved the end-effector backward",
    4:  "I moved the end-effector upward",
    5:  "I moved the end-effector downward",
    6:  "I rotated the wrist clockwise",
    7:  "I rotated the wrist counterclockwise",
    8:  "I pitched the end-effector forward",
    9:  "I pitched the end-effector backward",
    10: "I yawed the end-effector to the left",
    11: "I yawed the end-effector to the right",
    12: "I opened the gripper",
    13: "I closed the gripper",
}
ACTION_VOCAB = vera_cfg.get("action_vocab") or _DEFAULT_CALVIN_VOCAB
# Ensure int keys (YAML loads them as ints, but just in case)
ACTION_VOCAB = {int(k): v for k, v in ACTION_VOCAB.items()}

num_actions = m_cfg["num_actions"]   # 14
history_len = m_cfg["history_len"]   # 4
num_vis     = m_cfg.get("num_vis_frames", 3)
action_dim  = int(getattr(model, "action_dim", m_cfg.get("action_dim", 7)))
null_vec    = np.zeros(action_dim, dtype=np.float32)


def verbalize_consequence(done: float, action_magnitude: float) -> str:
    """Convert CALVIN done-flag + action magnitude → consequence string."""
    if done >= 1.0:
        return "I completed the sub-task and received a success signal."
    elif action_magnitude > 0.3:
        return "I made a large movement but the task is not yet complete."
    elif action_magnitude > 0.05:
        return "I made progress toward the goal but have not finished."
    else:
        return "I made a small adjustment with no task completion yet."

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — IMAGE PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

import torchvision.transforms as Tv

img_size = cfg.get("data", {}).get("img_size", 224)
transform = Tv.Compose([
    Tv.Resize((img_size, img_size)),
    Tv.ToTensor(),
    Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std =[0.26862954, 0.26130258, 0.27577711]),
])

def frame_to_tensor(frame: np.ndarray):
    """uint8 (H,W,3) → normalised tensor (3, img_size, img_size)."""
    pil = PILImage.fromarray(frame.astype(np.uint8))
    return transform(pil)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CALVIN DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _discretise_calvin(rel_action: np.ndarray) -> int:
    """14-bin direction-aware discretisation of 7-DoF CALVIN action."""
    if rel_action[6] > 0.5:
        return 12   # gripper open
    elif rel_action[6] < -0.5:
        return 13   # gripper close
    dom = int(np.argmax(np.abs(rel_action[:6])))
    return dom * 2 + (0 if rel_action[dom] >= 0 else 1)


def load_calvin_episodes(root: str, split: str = "training", max_scan: int = 0):
    """
    Load CALVIN episodes from root/<split>/*.npz + lang_annotations/.
    Returns list of dicts: {frames (T,H,W,3), instruction, actions (T,),
                             rewards (T,), action_vectors (T,7)}.
    """
    root = Path(root) / split
    if not root.is_dir():
        print(f"[CALVIN] Directory not found: {root}")
        return []

    # Language annotations
    lang_ann_path = root / "lang_annotations" / "auto_lang_ann.npy"
    lang_ann = None
    if lang_ann_path.exists():
        lang_ann = np.load(lang_ann_path, allow_pickle=True).item()

    episode_files = sorted(root.glob("episode_*.npz"))
    if not episode_files:
        print(f"[CALVIN] No .npz files in {root}")
        return []
    print(f"[CALVIN] Found {len(episode_files)} .npz files in {root}")

    available = {int(f.stem.split("_")[1]): f for f in episode_files}

    def _load_ep(frame_indices, task_str):
        frames, act_idx, rews, avecs = [], [], [], []
        for idx in frame_indices:
            if idx not in available:
                return None
            try:
                data = np.load(available[idx], allow_pickle=True)
            except Exception:
                return None
            frame = data.get("rgb_static", None)
            if frame is None:
                return None
            rel_action = np.asarray(
                data.get("rel_actions", np.zeros(7, dtype=np.float32)),
                dtype=np.float32).flatten()[:7]
            frames.append(np.asarray(frame, dtype=np.uint8))
            act_idx.append(_discretise_calvin(rel_action))
            rews.append(float(data.get("done", 0)))
            avecs.append(rel_action)
        if len(frames) < 2:
            return None
        return {
            "frames":         np.stack(frames),
            "instruction":    task_str,
            "actions":        np.array(act_idx, dtype=np.int64),
            "rewards":        np.array(rews,    dtype=np.float32),
            "action_vectors": np.stack(avecs).astype(np.float32),
        }

    episodes = []
    ep_se_path = root / "ep_start_end_ids.npy"

    if lang_ann is not None:
        indx  = lang_ann["info"]["indx"]
        tasks = lang_ann["language"]["task"]
        print(f"[CALVIN] lang_annotations: {len(indx)} episodes")
    elif ep_se_path.exists():
        ep_se = np.load(ep_se_path)
        indx  = [(int(s), int(e)) for s, e in ep_se]
        tasks = ["complete the manipulation task"] * len(indx)
        print(f"[CALVIN] ep_start_end_ids: {len(indx)} episodes (generic instruction)")
    else:
        indx, tasks = [], []

    n_scan = len(indx) if max_scan <= 0 else min(len(indx), max_scan)
    skipped = 0
    for (start, end), task_str in zip(indx[:n_scan], tasks[:n_scan]):
        ep = _load_ep(list(range(start, end + 1)), task_str)
        if ep is None:
            skipped += 1
        else:
            episodes.append(ep)
        if len(episodes) % 200 == 0 and len(episodes) > 0:
            gc.collect()

    if skipped:
        print(f"[CALVIN] Skipped {skipped} episodes (missing frames)")
    print(f"[CALVIN] Loaded {len(episodes)} usable episodes")
    return episodes


def instruction_ok(instr: str) -> bool:
    if not USE_PREFERRED_INSTRUCTIONS:
        return True
    low = instr.lower()
    return any(s in low for s in PREFERRED_INSTRUCTION_SUBSTRINGS)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — INFERENCE TOKENS AT MID-STEP
# ══════════════════════════════════════════════════════════════════════════════

tok_cache = {}

def infer_tokens_at_t(ep: dict, t: int) -> tuple[str, str]:
    """Run TERA at timestep t using expert history from episode dict."""
    instr = ep["instruction"]
    if instr not in tok_cache:
        tok_cache[instr] = clip.tokenize([instr])[0]
    lang_in = tok_cache[instr].unsqueeze(0).to(device)

    # Build history (act, rew, action_vec) up to t
    act_hist, rew_hist, av_hist = [], [], []
    for j in range(max(0, t - history_len), t):
        act_hist.append(int(ep["actions"][j]))
        rew_hist.append(float(ep["rewards"][j]))
        av = ep["action_vectors"][j][:action_dim]
        av_hist.append(np.clip(av, -1, 1))

    # Pad to history_len
    while len(act_hist) < history_len:
        act_hist.insert(0, num_actions)   # num_actions = "no-op" token
        rew_hist.insert(0, 0.0)
        av_hist.insert(0, null_vec.copy())

    prev_a = act_hist[-1] if t > 0 else num_actions
    prev_r = rew_hist[-1] if t > 0 else 0.0

    # Build visual context: 3 frames ending at t
    frames_idx = [max(0, t - 2), max(0, t - 1), t]
    frame_tensors = []
    for fi in frames_idx:
        frame_tensors.append(frame_to_tensor(ep["frames"][fi]))
    while len(frame_tensors) < num_vis:
        frame_tensors.insert(0, torch.zeros_like(frame_tensors[0]))
    frames_in = torch.stack(frame_tensors[-num_vis:]).unsqueeze(0).to(device)

    act_hist_in = torch.tensor([act_hist], dtype=torch.long).to(device)
    rew_hist_in = torch.tensor([rew_hist], dtype=torch.float32).to(device)
    av_hist_in  = torch.tensor([np.stack(av_hist)], dtype=torch.float32).to(device)
    prev_a_in   = torch.tensor([prev_a], dtype=torch.long).to(device)
    prev_r_in   = torch.tensor([prev_r], dtype=torch.float32).to(device)
    delta_in    = torch.tensor([0.0],    dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(frames_in, lang_in, act_hist_in, rew_hist_in,
                    prev_a_in, prev_r_in, state_delta=delta_in,
                    action_vec_hist=av_hist_in)
        action = int(out["logits"].argmax(dim=-1).item())

    done_val        = float(ep["rewards"][t])
    action_mag      = float(np.linalg.norm(ep["action_vectors"][t]))
    narration       = ACTION_VOCAB.get(action, f"I performed action {action}")
    knowledge       = verbalize_consequence(done_val, action_mag)
    return narration, knowledge

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PICK TOP-N EPISODES
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading CALVIN dataset …")
all_episodes = load_calvin_episodes(CALVIN_DATA_ROOT, CALVIN_SPLIT, DATASET_MAX_SCAN)

# Score: prefer episodes where done=1 at any step (task success)
def episode_score(ep):
    return (float(ep["rewards"].max()), len(ep["frames"]))

all_episodes.sort(key=episode_score, reverse=True)

collected_episodes = []
used_instr = set()

for ep in all_episodes:
    instr = ep["instruction"].strip()
    key   = instr.lower()[:40]
    if key in used_instr:
        continue
    if not instruction_ok(instr):
        continue
    used_instr.add(key)

    T      = len(ep["frames"])
    idxs   = [0, T // 2, T - 1]
    frames = [ep["frames"][i] for i in idxs]

    mid             = T // 2
    nar_m, know_m   = infer_tokens_at_t(ep, mid)
    gc.collect()

    collected_episodes.append({
        "instruction":    instr,
        "start":          frames[0],
        "mid":            frames[1],
        "end":            frames[2],
        "narration_mid":  nar_m,
        "knowledge_mid":  know_m,
        "total_reward":   float(ep["rewards"].sum()),
    })
    print(f"  [{len(collected_episodes)}] \"{instr}\"  max_done={ep['rewards'].max():.0f}")
    if len(collected_episodes) >= N_EPISODES:
        break

if not collected_episodes:
    print("[WARN] No episodes collected — check CALVIN_DATA_ROOT and CALVIN_SPLIT paths.")
elif len(collected_episodes) < N_EPISODES:
    print(f"[WARN] Only {len(collected_episodes)}/{N_EPISODES} episodes collected.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — BUILD COMPOSITE FIGURE  (identical layout to LT figure)
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib.gridspec as gridspec

N_ROWS   = len(collected_episodes)
FIG_W    = 11.0
HDR_H, IMG_H, TOK_H = 0.22, 2.20, 0.52
total_h  = HDR_H + N_ROWS * (IMG_H + TOK_H) + (0.35 if FIG_SHOW_CAPTION else 0.08)

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      FIG_FONT_INSTR,
    "axes.titlesize": FIG_FONT_HDR,
    "figure.dpi":     FIG_DPI,
})

ROW_BG   = ["#F5F7FA", "#FFFFFF"]
TEXT_CLR = "#1a2744"
TOKEN_CLR = "#1a2744"
COL_TITLES = ["Start", "Middle", "End"]

height_ratios = [HDR_H] + [IMG_H, TOK_H] * N_ROWS
n_master_rows = 1 + N_ROWS * 2

fig = plt.figure(figsize=(FIG_W, total_h), facecolor="white", dpi=FIG_DPI)
master = gridspec.GridSpec(
    n_master_rows, 4, figure=fig,
    width_ratios=[1.05, 1, 1, 1],
    height_ratios=height_ratios,
    left=0.06, right=0.99, top=0.96, bottom=0.04 if not FIG_SHOW_CAPTION else 0.10,
    wspace=0.03, hspace=0.04,
)


def _thumb(frame: np.ndarray) -> np.ndarray:
    """Resize frame to FIG_FRAME_PX square; return placeholder if None."""
    if frame is None:
        return np.full((FIG_FRAME_PX, FIG_FRAME_PX, 3), 210, dtype=np.uint8)
    img = PILImage.fromarray(frame.astype(np.uint8))
    if img.size != (FIG_FRAME_PX, FIG_FRAME_PX):
        img = img.resize((FIG_FRAME_PX, FIG_FRAME_PX), PILImage.Resampling.LANCZOS)
    return np.asarray(img)


def _draw_frame(ax, frame):
    ax.imshow(_thumb(frame))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor("#CCCCCC")


def _soft_wrap(text: str, width: int) -> str:
    text = (text or "").strip()
    if len(text) <= width:
        return text
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _token_left(nar: str) -> str:
    return f"$E_{{\\mathrm{{act}}}}$: {_soft_wrap(nar, FIG_TOK_WRAP)}"


def _token_right(know: str) -> str:
    return f"$E_{{\\mathrm{{emb}}}}$: {_soft_wrap(know, FIG_TOK_WRAP)}"


# ── Header row ────────────────────────────────────────────────────────────────
ax_h0 = fig.add_subplot(master[0, 0])
ax_h0.axis("off")
ax_h0.text(0.98, 0.5, "Instruction", ha="right", va="center",
           fontsize=FIG_FONT_HDR, fontweight="bold", color=TEXT_CLR)
for j, title in enumerate(COL_TITLES):
    ax_h = fig.add_subplot(master[0, j + 1])
    ax_h.axis("off")
    ax_h.text(0.5, 0.5, title, ha="center", va="center",
              fontsize=FIG_FONT_HDR, fontweight="bold", color=TEXT_CLR)

# ── Episode rows ──────────────────────────────────────────────────────────────
for row_i, ep in enumerate(collected_episodes):
    r_img = 1 + row_i * 2
    r_tok = r_img + 1
    bg    = ROW_BG[row_i % 2]

    # Instruction label (spans img + tok rows)
    ax_lbl = fig.add_subplot(master[r_img:r_tok + 1, 0])
    ax_lbl.set_facecolor(bg)
    ax_lbl.axis("off")
    instr = ep["instruction"].strip().rstrip(".")
    if instr:
        instr = instr[0].upper() + instr[1:]
    ax_lbl.text(
        0.96, 0.5, _soft_wrap(instr + ".", FIG_INSTR_WRAP),
        transform=ax_lbl.transAxes,
        fontsize=FIG_FONT_INSTR, ha="right", va="center",
        color=TEXT_CLR, linespacing=1.25, clip_on=False,
    )

    # Frames
    for col, key in [(1, "start"), (2, "mid"), (3, "end")]:
        ax = fig.add_subplot(master[r_img, col])
        ax.set_facecolor(bg)
        _draw_frame(ax, ep[key])

    # Token row — E_act LEFT | E_emb RIGHT
    ax_tok = fig.add_subplot(master[r_tok, 1:4])
    ax_tok.set_facecolor(bg)
    ax_tok.axis("off")
    ax_tok.set_ylim(0, 1)
    ax_tok.set_xlim(0, 1)
    ax_tok.margins(x=0.02, y=0.10)
    ax_tok.text(
        0.01, 0.5, _token_left(ep.get("narration_mid", "—")),
        transform=ax_tok.transAxes,
        fontsize=FIG_FONT_TOK, ha="left", va="center",
        color=TOKEN_CLR, linespacing=FIG_TOK_LINESPACING, clip_on=False,
    )
    ax_tok.axvline(0.50, color="#cccccc", linewidth=0.8, clip_on=False)
    ax_tok.text(
        0.52, 0.5, _token_right(ep.get("knowledge_mid", "—")),
        transform=ax_tok.transAxes,
        fontsize=FIG_FONT_TOK, ha="left", va="center",
        color=TOKEN_CLR, linespacing=FIG_TOK_LINESPACING, clip_on=False,
    )

if FIG_SHOW_CAPTION:
    fig.text(
        0.03, 0.01,
        f"Top {N_ROWS} CALVIN demos (by task success). "
        "$E_{\\mathrm{act}}$ / $E_{\\mathrm{emb}}$ at middle timestep.",
        fontsize=FIG_FONT_CAP, va="bottom", color=TEXT_CLR,
        ha="left", family="sans-serif",
    )

# ── Save ──────────────────────────────────────────────────────────────────────
for dest in [OUT_PNG_REPO, OUT_PNG_SUB]:
    try:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved: {dest}")
    except Exception as e:
        print(f"[WARN] Could not save to {dest}: {e}")

plt.show()
print("\nDone. Upload calvin_qual_composite.png to your submission folder.")
