r"""
colab_combined_lt_calvin_figure.py
====================================
Generates a single composite figure with:
  Row 1 — LT demo 1
  Row 2 — LT demo 2
  Row 3 — CALVIN demo 1
  Row 4 — CALVIN demo 2

Each row: Instruction | Start | Middle | End  +  E_act / E_emb token strip.
A subtle section label ("Language-Table" / "CALVIN") appears in the left margin.

HOW TO RUN (Colab):
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/RLConditionedVLA   # or wherever your repo lives
    %run docs/colab_combined_lt_calvin_figure.py

Output: combined_lt_calvin_composite.png
        (also copied to your submission folder)
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — USER CONFIG
# ══════════════════════════════════════════════════════════════════════════════
import os

# ── LT checkpoint + dataset ───────────────────────────────────────────────────
LT_CHECKPOINT = "/content/drive/MyDrive/VERA_LT_Real/checkpoints/lt_full_vera/seed123/best_sft_vera.pt"
LT_CONFIG     = "/content/drive/MyDrive/RLConditionedVLA/configs/config.yaml"
LT_DATA_ROOT  = "/content/lt_real/train"       # after extracting tar
LT_N_ROWS     = 2                              # how many LT rows in the figure

# ── CALVIN checkpoint + dataset ───────────────────────────────────────────────
CAL_CHECKPOINT = "/content/drive/MyDrive/VERA_CALVIN/checkpoints/calvin_vera/seed42/best_sft_vera.pt"
CAL_CONFIG     = "/content/drive/MyDrive/RLConditionedVLA/configs/calvin_config.yaml"
CAL_DATA_ROOT  = "/content/drive/MyDrive/VERA_CALVIN/task_D_D"
CAL_SPLIT      = "training"
CAL_N_ROWS     = 2

# ── Output ────────────────────────────────────────────────────────────────────
OUT_PNG_REPO = "/content/drive/MyDrive/RLConditionedVLA/docs/combined_lt_calvin_composite.png"
OUT_PNG_SUB  = "/content/drive/MyDrive/corl_2026_template_submission/combined_lt_calvin_composite.png"

FIG_SHOW_CAPTION = False
FIG_DPI          = 240

# ── Figure typography ─────────────────────────────────────────────────────────
FIG_FONT_HDR    = 11.5
FIG_FONT_INSTR  = 10.5
FIG_FONT_TOK    = 9.5
FIG_FONT_SECT   = 9.0    # "Language-Table" / "CALVIN" section labels
FIG_TOK_WRAP    = 38
FIG_TOK_LINESPACING = 1.35
FIG_INSTR_WRAP  = 26
FIG_FRAME_PX    = 224

SEED = 42

# ── Preferred instructions ────────────────────────────────────────────────────
LT_PREFERRED_SUBSTRINGS = [
    "red moon", "blue moon", "yellow star", "red pentagon", "blue cube",
    "push the", "pull the", "move the", "slide the",
    "next to", "apart from", "away from", "to the",
]
CAL_PREFERRED_SUBSTRINGS = [
    "rotate", "push", "pick up", "lift", "place",
    "stack", "move", "slide", "open", "close", "put",
]
DATASET_MAX_SCAN = 600

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INSTALLS
# ══════════════════════════════════════════════════════════════════════════════
import subprocess, sys

def pip_one(pkg, optional=False):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg],
                              stderr=subprocess.STDOUT)
        print(f"  ✓ {pkg[:60]}")
    except subprocess.CalledProcessError:
        if optional: print(f"  ⊘ optional: {pkg[:60]}")
        else: raise

print("Installing dependencies …")
for pkg in ("pyyaml", "imageio", "pillow", "matplotlib", "numpy", "scipy"):
    pip_one(pkg)
pip_one("gym<=0.23.0")
pip_one("pybullet", optional=True)
try:
    import clip  # noqa
except ImportError:
    pip_one("git+https://github.com/openai/CLIP.git")
print("Done.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import random, textwrap, gc, pickle, glob
from pathlib import Path
from collections import deque

import numpy as np
import torch
import yaml
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage
import torchvision.transforms as Tv

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Add repo root to sys.path (try both LT and CALVIN config paths)
for _cp in (LT_CONFIG, CAL_CONFIG):
    _repo = str(Path(_cp).parent.parent)
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
from models.vera_model import VERAModel

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _load_cfg(ckpt_path: str, config_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg = ckpt["cfg"]
        print(f"  config from checkpoint ({ckpt_path})")
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"  config from {config_path}")
    return ckpt, cfg


def _build_model(ckpt, cfg, device):
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
    missing, unexp = model.load_state_dict(state, strict=False)
    if missing: print(f"  [load] missing: {len(missing)}")
    if unexp:   print(f"  [load] unexpected: {len(unexp)}")
    model.eval()
    return model


def _make_transform(img_size=224):
    return Tv.Compose([
        Tv.Resize((img_size, img_size)),
        Tv.ToTensor(),
        Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std =[0.26862954, 0.26130258, 0.27577711]),
    ])


def _frame_to_tensor(frame: np.ndarray, transform):
    return transform(PILImage.fromarray(frame.astype(np.uint8)))


def _thumb(frame, px=FIG_FRAME_PX):
    if frame is None:
        return np.full((px, px, 3), 210, dtype=np.uint8)
    img = PILImage.fromarray(np.asarray(frame, dtype=np.uint8))
    if img.size != (px, px):
        img = img.resize((px, px), PILImage.Resampling.LANCZOS)
    return np.asarray(img)


def _soft_wrap(text, width):
    text = (text or "").strip()
    return text if len(text) <= width else textwrap.fill(
        text, width=width, break_long_words=False, break_on_hyphens=False)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LT EPISODE COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Loading LT model ──────────────────────────────────")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

lt_ckpt, lt_cfg = _load_cfg(LT_CHECKPOINT, LT_CONFIG)
lt_model        = _build_model(lt_ckpt, lt_cfg, device)
lt_transform    = _make_transform(lt_cfg.get("data", {}).get("img_size", 224))

_lt_m = lt_cfg["model"]
_lt_v = lt_cfg.get("vera", {})
LT_NUM_ACTIONS = _lt_m["num_actions"]
LT_HISTORY_LEN = _lt_m["history_len"]
LT_NUM_VIS     = _lt_m.get("num_vis_frames", 3)
LT_ACTION_DIM  = _lt_m.get("action_dim", 2)
LT_NULL_VEC    = np.zeros(LT_ACTION_DIM, dtype=np.float32)

LT_ACTION_VOCAB = {
    0: "I pushed the object to the right",
    1: "I pushed the object up and to the right",
    2: "I pushed the object upward",
    3: "I pushed the object up and to the left",
    4: "I pushed the object to the left",
    5: "I pushed the object down and to the left",
    6: "I pushed the object downward",
    7: "I pushed the object down and to the right",
}
if _lt_v.get("action_vocab"):
    LT_ACTION_VOCAB = {int(k): v for k, v in _lt_v["action_vocab"].items()}


def lt_verbalize(reward: float, dist_delta: float) -> str:
    if reward > 0.8:
        return "I moved significantly closer to the goal and received a high reward."
    elif reward > 0.3:
        return "I moved closer to the goal and received a moderate reward."
    else:
        if dist_delta < -0.005:
            return "I moved closer to the goal but received a low reward."
        elif dist_delta > 0.005:
            return "I moved away from the goal and received a low reward."
        return "I made little progress and received a low reward."


def _lt_discretise(av):
    dx, dy = float(av[0]), float(av[1])
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return LT_NUM_ACTIONS
    return int(round(np.arctan2(dy, dx) / (np.pi / 4))) % LT_NUM_ACTIONS


_lt_tok_cache = {}

def _lt_infer_tokens(steps, t, instr):
    if instr not in _lt_tok_cache:
        _lt_tok_cache[instr] = clip.tokenize([instr])[0]
    lang_in = _lt_tok_cache[instr].unsqueeze(0).to(device)

    act_h, rew_h, av_h = [], [], []
    for j in range(max(0, t - LT_HISTORY_LEN), t):
        av = np.asarray(steps[j].get("action", steps[j].get("action_vec", [0, 0])),
                        dtype=np.float32).flatten()[:2]
        if np.abs(av).max() > 1.0:
            av = av / np.abs(av).max()
        act_h.append(_lt_discretise(av))
        rew_h.append(float(steps[j].get("reward", 0.0)))
        av_h.append(np.clip(av, -1, 1))
    while len(act_h) < LT_HISTORY_LEN:
        act_h.insert(0, LT_NUM_ACTIONS)
        rew_h.insert(0, 0.0)
        av_h.insert(0, LT_NULL_VEC.copy())

    prev_a = act_h[-1] if t > 0 else LT_NUM_ACTIONS
    prev_r = rew_h[-1] if t > 0 else 0.0
    prev_d = float(steps[t - 1].get("state_delta", 0.0) or 0.0) if t > 0 else 0.0

    def _get_frame(step):
        for k in ("obs", "image", "rgb", "pixels", "frame"):
            v = step.get(k)
            if v is None: continue
            if isinstance(v, np.ndarray) and v.ndim == 3: return v.astype(np.uint8)
            if isinstance(v, dict):
                for sk in ("rgb", "image", "pixels"):
                    v2 = v.get(sk)
                    if v2 is not None and isinstance(v2, np.ndarray) and v2.ndim == 3:
                        return v2.astype(np.uint8)
        return None

    frame_tensors = []
    for fi in [max(0, t - 2), max(0, t - 1), t]:
        fr = _get_frame(steps[fi])
        if fr is not None:
            frame_tensors.append(_frame_to_tensor(fr, lt_transform))
    if not frame_tensors: return "—", "—"
    while len(frame_tensors) < LT_NUM_VIS:
        frame_tensors.insert(0, torch.zeros_like(frame_tensors[0]))

    frames_in   = torch.stack(frame_tensors[-LT_NUM_VIS:]).unsqueeze(0).to(device)
    act_h_in    = torch.tensor([act_h], dtype=torch.long).to(device)
    rew_h_in    = torch.tensor([rew_h], dtype=torch.float32).to(device)
    av_h_in     = torch.tensor([np.stack(av_h)], dtype=torch.float32).to(device)
    prev_a_in   = torch.tensor([prev_a], dtype=torch.long).to(device)
    prev_r_in   = torch.tensor([prev_r], dtype=torch.float32).to(device)
    delta_in    = torch.tensor([prev_d], dtype=torch.float32).to(device)

    with torch.no_grad():
        out    = lt_model(frames_in, lang_in, act_h_in, rew_h_in,
                          prev_a_in, prev_r_in, state_delta=delta_in,
                          action_vec_hist=av_h_in)
        action = int(out["logits"].argmax(dim=-1).item())
    rew_val = float(steps[t].get("reward", 0.0))
    narr    = LT_ACTION_VOCAB.get(action, f"I performed action {action}")
    know    = lt_verbalize(rew_val, prev_d)
    return narr, know


def _lt_pkl_frame(step):
    for k in ("obs", "image", "rgb", "pixels", "frame"):
        v = step.get(k)
        if v is None: continue
        if isinstance(v, dict):
            for sk in ("rgb", "image", "pixels"):
                v2 = v.get(sk)
                if v2 is not None and isinstance(v2, np.ndarray) and v2.ndim == 3:
                    return v2.astype(np.uint8)
        elif isinstance(v, np.ndarray) and v.ndim == 3:
            return v.astype(np.uint8)
    return None


def collect_lt_episodes(root, n):
    root = Path(root)
    if not root.is_dir():
        print(f"[LT] Not found: {root}")
        return []
    ep_dirs = sorted(root.glob("episode_*"))
    meta = []
    for ep_dir in ep_dirs:
        pkl = ep_dir / "steps.pkl"
        if not pkl.is_file(): continue
        with open(pkl, "rb") as f:
            steps = pickle.load(f)
        if len(steps) < 4: continue
        instr_raw = None
        for k in ("instruction", "language_instruction", "task"):
            v = steps[0].get(k)
            if v:
                instr_raw = v.decode() if isinstance(v, bytes) else str(v)
                break
        instr = (instr_raw or "complete the task").strip().rstrip(".")
        if not any(s in instr.lower() for s in LT_PREFERRED_SUBSTRINGS):
            continue
        total_r = sum(float(s.get("reward", 0)) for s in steps)
        meta.append((total_r, instr, ep_dir))
        del steps
    meta.sort(key=lambda x: -x[0])

    picked, used = [], set()
    for total_r, instr, ep_dir in meta:
        key = instr.lower()[:40]
        if key in used: continue
        used.add(key)
        with open(ep_dir / "steps.pkl", "rb") as f:
            steps = pickle.load(f)
        T      = len(steps)
        frames = [_lt_pkl_frame(steps[i]) for i in [0, T // 2, T - 1]]
        if any(f is None for f in frames):
            continue
        mid = T // 2
        nar_m, know_m = _lt_infer_tokens(steps, mid, instr)
        del steps; gc.collect()
        picked.append({
            "dataset":        "Language-Table",
            "instruction":    instr,
            "start":          frames[0],
            "mid":            frames[1],
            "end":            frames[2],
            "narration_mid":  nar_m,
            "knowledge_mid":  know_m,
        })
        print(f"  [LT {len(picked)}] \"{instr}\"  reward={total_r:.2f}")
        if len(picked) >= n: break
    return picked

print("\nCollecting LT episodes …")
lt_episodes = collect_lt_episodes(LT_DATA_ROOT, LT_N_ROWS)
if len(lt_episodes) < LT_N_ROWS:
    print(f"[WARN] Only {len(lt_episodes)}/{LT_N_ROWS} LT episodes found.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CALVIN EPISODE COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Loading CALVIN model ──────────────────────────────")
cal_ckpt, cal_cfg = _load_cfg(CAL_CHECKPOINT, CAL_CONFIG)
cal_model         = _build_model(cal_ckpt, cal_cfg, device)
cal_transform     = _make_transform(cal_cfg.get("data", {}).get("img_size", 224))

_cal_m = cal_cfg["model"]
_cal_v = cal_cfg.get("vera", {})
CAL_NUM_ACTIONS = _cal_m["num_actions"]    # 14
CAL_HISTORY_LEN = _cal_m["history_len"]
CAL_NUM_VIS     = _cal_m.get("num_vis_frames", 3)
CAL_ACTION_DIM  = _cal_m.get("action_dim", 7)
CAL_NULL_VEC    = np.zeros(CAL_ACTION_DIM, dtype=np.float32)

_DEFAULT_CAL_VOCAB = {
    0: "I moved the end-effector to the right",
    1: "I moved the end-effector to the left",
    2: "I moved the end-effector forward",
    3: "I moved the end-effector backward",
    4: "I moved the end-effector upward",
    5: "I moved the end-effector downward",
    6: "I rotated the wrist clockwise",
    7: "I rotated the wrist counterclockwise",
    8: "I pitched the end-effector forward",
    9: "I pitched the end-effector backward",
    10: "I yawed the end-effector to the left",
    11: "I yawed the end-effector to the right",
    12: "I opened the gripper",
    13: "I closed the gripper",
}
CAL_ACTION_VOCAB = {int(k): v for k, v in
                    (_cal_v.get("action_vocab") or _DEFAULT_CAL_VOCAB).items()}


def cal_verbalize(done: float, action_magnitude: float) -> str:
    if done >= 1.0:
        return "I completed the sub-task and received a success signal."
    elif action_magnitude > 0.3:
        return "I made a large movement but the task is not yet complete."
    elif action_magnitude > 0.05:
        return "I made progress toward the goal but have not finished."
    return "I made a small adjustment with no task completion yet."


def _cal_discretise(rel_action):
    if rel_action[6] > 0.5:  return 12
    if rel_action[6] < -0.5: return 13
    dom = int(np.argmax(np.abs(rel_action[:6])))
    return dom * 2 + (0 if rel_action[dom] >= 0 else 1)


_cal_tok_cache = {}

def _cal_infer_tokens(ep_frames, ep_actions, ep_rewards, ep_avecs, t, instr):
    if instr not in _cal_tok_cache:
        _cal_tok_cache[instr] = clip.tokenize([instr])[0]
    lang_in = _cal_tok_cache[instr].unsqueeze(0).to(device)

    act_h, rew_h, av_h = [], [], []
    for j in range(max(0, t - CAL_HISTORY_LEN), t):
        act_h.append(int(ep_actions[j]))
        rew_h.append(float(ep_rewards[j]))
        av_h.append(np.clip(ep_avecs[j][:CAL_ACTION_DIM], -1, 1))
    while len(act_h) < CAL_HISTORY_LEN:
        act_h.insert(0, CAL_NUM_ACTIONS)
        rew_h.insert(0, 0.0)
        av_h.insert(0, CAL_NULL_VEC.copy())

    prev_a = act_h[-1] if t > 0 else CAL_NUM_ACTIONS
    prev_r = rew_h[-1] if t > 0 else 0.0

    frame_tensors = [_frame_to_tensor(ep_frames[max(0, t - 2)], cal_transform),
                     _frame_to_tensor(ep_frames[max(0, t - 1)], cal_transform),
                     _frame_to_tensor(ep_frames[t],              cal_transform)]
    while len(frame_tensors) < CAL_NUM_VIS:
        frame_tensors.insert(0, torch.zeros_like(frame_tensors[0]))

    frames_in = torch.stack(frame_tensors[-CAL_NUM_VIS:]).unsqueeze(0).to(device)
    act_h_in  = torch.tensor([act_h], dtype=torch.long).to(device)
    rew_h_in  = torch.tensor([rew_h], dtype=torch.float32).to(device)
    av_h_in   = torch.tensor([np.stack(av_h)], dtype=torch.float32).to(device)
    prev_a_in = torch.tensor([prev_a], dtype=torch.long).to(device)
    prev_r_in = torch.tensor([prev_r], dtype=torch.float32).to(device)
    delta_in  = torch.tensor([0.0],    dtype=torch.float32).to(device)

    with torch.no_grad():
        out    = cal_model(frames_in, lang_in, act_h_in, rew_h_in,
                           prev_a_in, prev_r_in, state_delta=delta_in,
                           action_vec_hist=av_h_in)
        action = int(out["logits"].argmax(dim=-1).item())

    done_val   = float(ep_rewards[t])
    action_mag = float(np.linalg.norm(ep_avecs[t]))
    narr       = CAL_ACTION_VOCAB.get(action, f"I performed action {action}")
    know       = cal_verbalize(done_val, action_mag)
    return narr, know


def collect_calvin_episodes(root, split, n):
    root = Path(root) / split
    if not root.is_dir():
        print(f"[CALVIN] Not found: {root}")
        return []

    lang_ann_path = root / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_ann_path.exists():
        print(f"[CALVIN] No lang_annotations at {lang_ann_path}")
        return []

    lang_ann = np.load(lang_ann_path, allow_pickle=True).item()
    indx  = lang_ann["info"]["indx"]
    tasks = lang_ann["language"]["task"]

    ep_files = sorted(root.glob("episode_*.npz"))
    available = {int(f.stem.split("_")[1]): f for f in ep_files}
    print(f"[CALVIN] {len(ep_files)} .npz files, {len(indx)} annotated episodes")

    # Score: episodes with done=1 rank highest
    scored = []
    for (start, end), task_str in zip(indx[:DATASET_MAX_SCAN], tasks[:DATASET_MAX_SCAN]):
        indices = list(range(start, end + 1))
        if not all(i in available for i in indices): continue
        dones = [float(np.load(available[i], allow_pickle=True).get("done", 0))
                 for i in indices[-3:]]   # fast: only check last few frames
        scored.append((max(dones), task_str, start, end))
    scored.sort(key=lambda x: -x[0])

    picked, used = [], set()
    for score, task_str, start, end in scored:
        key = task_str.lower()[:40]
        if key in used: continue
        if not any(s in task_str.lower() for s in CAL_PREFERRED_SUBSTRINGS): continue
        used.add(key)

        indices = list(range(start, end + 1))
        frames, act_idx, rews, avecs = [], [], [], []
        ok = True
        for idx in indices:
            try:
                data = np.load(available[idx], allow_pickle=True)
            except Exception:
                ok = False; break
            fr = data.get("rgb_static", None)
            if fr is None: ok = False; break
            rel_act = np.asarray(data.get("rel_actions", np.zeros(7, dtype=np.float32)),
                                 dtype=np.float32).flatten()[:7]
            frames.append(np.asarray(fr, dtype=np.uint8))
            act_idx.append(_cal_discretise(rel_act))
            rews.append(float(data.get("done", 0)))
            avecs.append(rel_act)
        if not ok or len(frames) < 3: continue

        T     = len(frames)
        mid   = T // 2
        nar_m, know_m = _cal_infer_tokens(
            frames,
            np.array(act_idx, dtype=np.int64),
            np.array(rews,    dtype=np.float32),
            np.stack(avecs).astype(np.float32),
            mid, task_str,
        )
        gc.collect()
        picked.append({
            "dataset":        "CALVIN",
            "instruction":    task_str,
            "start":          frames[0],
            "mid":            frames[mid],
            "end":            frames[-1],
            "narration_mid":  nar_m,
            "knowledge_mid":  know_m,
        })
        print(f"  [CAL {len(picked)}] \"{task_str}\"  done={score:.0f}")
        if len(picked) >= n: break

    return picked


print("\nCollecting CALVIN episodes …")
cal_episodes = collect_calvin_episodes(CAL_DATA_ROOT, CAL_SPLIT, CAL_N_ROWS)
if len(cal_episodes) < CAL_N_ROWS:
    print(f"[WARN] Only {len(cal_episodes)}/{CAL_N_ROWS} CALVIN episodes found.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BUILD COMBINED FIGURE
# ══════════════════════════════════════════════════════════════════════════════

all_episodes = lt_episodes + cal_episodes
N_ROWS = len(all_episodes)
if N_ROWS == 0:
    print("[ERROR] No episodes collected from either dataset.")
    raise SystemExit(1)

FIG_W    = 11.0
HDR_H    = 0.22
IMG_H    = 2.20
TOK_H    = 0.52
total_h  = HDR_H + N_ROWS * (IMG_H + TOK_H) + (0.35 if FIG_SHOW_CAPTION else 0.08)

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      FIG_FONT_INSTR,
    "axes.titlesize": FIG_FONT_HDR,
    "figure.dpi":     FIG_DPI,
})

# Alternating background within each dataset group, reset at group boundary
LT_BG   = ["#EEF3FA", "#F7FAFF"]   # cool blue tint for LT rows
CAL_BG  = ["#F5F2EE", "#FBF8F5"]   # warm beige tint for CALVIN rows
TEXT_CLR  = "#1a2744"
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


def _draw_frame(ax, frame, bg):
    ax.set_facecolor(bg)
    ax.imshow(_thumb(frame))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor("#CCCCCC")


# Header
ax_h0 = fig.add_subplot(master[0, 0])
ax_h0.axis("off")
ax_h0.text(0.98, 0.5, "Instruction", ha="right", va="center",
           fontsize=FIG_FONT_HDR, fontweight="bold", color=TEXT_CLR)
for j, title in enumerate(COL_TITLES):
    ax_h = fig.add_subplot(master[0, j + 1])
    ax_h.axis("off")
    ax_h.text(0.5, 0.5, title, ha="center", va="center",
              fontsize=FIG_FONT_HDR, fontweight="bold", color=TEXT_CLR)

# Dataset divider positions (row index in figure coords)
lt_row_count  = len(lt_episodes)
cal_row_count = len(cal_episodes)

for row_i, ep in enumerate(all_episodes):
    r_img = 1 + row_i * 2
    r_tok = r_img + 1
    is_cal = ep["dataset"] == "CALVIN"

    # Choose background palette
    local_i = row_i if not is_cal else row_i - lt_row_count
    bg = (CAL_BG if is_cal else LT_BG)[local_i % 2]

    # Instruction label
    ax_lbl = fig.add_subplot(master[r_img:r_tok + 1, 0])
    ax_lbl.set_facecolor(bg)
    ax_lbl.axis("off")

    # Section label on very first row of each dataset (top of cell)
    if row_i == 0 or (is_cal and row_i == lt_row_count):
        sect_label = "CALVIN" if is_cal else "Language-Table"
        ax_lbl.text(0.98, 0.95, sect_label,
                    transform=ax_lbl.transAxes,
                    fontsize=FIG_FONT_SECT, ha="right", va="top",
                    color="#888888", style="italic", clip_on=False)

    instr = ep["instruction"].strip().rstrip(".")
    if instr: instr = instr[0].upper() + instr[1:]
    ax_lbl.text(
        0.96, 0.5, _soft_wrap(instr + ".", FIG_INSTR_WRAP),
        transform=ax_lbl.transAxes,
        fontsize=FIG_FONT_INSTR, ha="right", va="center",
        color=TEXT_CLR, linespacing=1.25, clip_on=False,
    )

    # Frames
    for col, key in [(1, "start"), (2, "mid"), (3, "end")]:
        ax = fig.add_subplot(master[r_img, col])
        _draw_frame(ax, ep[key], bg)

    # Token strip — E_act | E_emb side by side
    ax_tok = fig.add_subplot(master[r_tok, 1:4])
    ax_tok.set_facecolor(bg)
    ax_tok.axis("off")
    ax_tok.set_ylim(0, 1); ax_tok.set_xlim(0, 1)
    ax_tok.margins(x=0.02, y=0.10)
    ax_tok.text(
        0.01, 0.5,
        f"$E_{{\\mathrm{{act}}}}$: {_soft_wrap(ep.get('narration_mid', '—'), FIG_TOK_WRAP)}",
        transform=ax_tok.transAxes,
        fontsize=FIG_FONT_TOK, ha="left", va="center",
        color=TOKEN_CLR, linespacing=FIG_TOK_LINESPACING, clip_on=False,
    )
    ax_tok.axvline(0.50, color="#cccccc", linewidth=0.8, clip_on=False)
    ax_tok.text(
        0.52, 0.5,
        f"$E_{{\\mathrm{{emb}}}}$: {_soft_wrap(ep.get('knowledge_mid', '—'), FIG_TOK_WRAP)}",
        transform=ax_tok.transAxes,
        fontsize=FIG_FONT_TOK, ha="left", va="center",
        color=TOKEN_CLR, linespacing=FIG_TOK_LINESPACING, clip_on=False,
    )

    # Thin horizontal divider between LT and CALVIN sections
    if row_i == lt_row_count - 1 and cal_row_count > 0:
        ax_tok.axhline(0.0, color="#aaaaaa", linewidth=1.2,
                       xmin=0, xmax=1, clip_on=False, zorder=10)

if FIG_SHOW_CAPTION:
    fig.text(
        0.03, 0.01,
        f"Language-Table (rows 1–{lt_row_count}) and CALVIN (rows {lt_row_count+1}–{N_ROWS}) "
        "demonstrations. $E_{\\mathrm{act}}$ and $E_{\\mathrm{emb}}$ tokens at the middle timestep.",
        fontsize=8.5, va="bottom", color=TEXT_CLR, ha="left",
    )

# ── Save ──────────────────────────────────────────────────────────────────────
for dest in [OUT_PNG_REPO, OUT_PNG_SUB]:
    try:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved: {dest}")
    except Exception as e:
        print(f"[WARN] {e}")

plt.show()
print("\nDone — combined_lt_calvin_composite.png ready.")
