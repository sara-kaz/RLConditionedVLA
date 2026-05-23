r"""
eval_lt_task_success.py
=======================
Closed-loop Language-Table task-success evaluation for TERA and all 6 ablation
conditions.  Run cell-by-cell in Colab (GPU runtime recommended).

UPLOAD INSTRUCTIONS
-------------------
1. Upload your entire "Test1" checkpoints folder to Google Drive:
      MyDrive/VERA_LT_Checkpoints/
        lt_full_vera/seed42/best_sft_vera.pt
        lt_full_vera/seed123/best_sft_vera.pt
        lt_full_vera/seed456/best_sft_vera.pt
        lt_bc_baseline/seed42/... (etc.)
        lt_no_lang/...
        lt_no_act/...
        lt_no_exp/...
        lt_no_hist_tf/...

   NOTE: The local checkpoint directories use "checkpoints N" naming.
   Re-organise them into the flat structure above before uploading,
   OR set CKPT_ROOT below to point at a different layout and adjust
   CHECKPOINT_MAP accordingly.

2. Mount Drive and run:
      from google.colab import drive
      drive.mount('/content/drive')
      %run /content/drive/MyDrive/VLA-Robot-Learning/docs/eval_lt_task_success.py

Results are saved to:
    MyDrive/VERA_LT_Checkpoints/lt_task_success_results.json
    MyDrive/VERA_LT_Checkpoints/lt_task_success_results.txt
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — USER CONFIG
# ══════════════════════════════════════════════════════════════════════════════
import os

MYDRIVE = "/content/drive/MyDrive"

# Root directory where you placed the organised checkpoints
CKPT_ROOT = f"{MYDRIVE}/VERA_LT_Checkpoints"

# Path to config.yaml (from your repo clone on Drive or Colab)
CONFIG_PATH = f"{MYDRIVE}/VLA-Robot-Learning/configs/config.yaml"

# Episodes per condition per seed (50 × 3 seeds = 150 total per condition)
# Increase to 100 for camera-ready; 50 runs in ~25 min on T4 GPU.
N_EPISODES = 50

# Seeds used during training — must match checkpoint subdirectory names
SEEDS = [42, 123, 456]

# Max steps before episode is declared a failure
MAX_STEPS = 60

# LT reward mode (block2block gives clearest push-X-to-Y task signal)
REWARD_MODE = "block2block"   # "block2block" | "all"

# Scale tanh action_vec → LT continuous action [Δx, Δy]
LT_ACTION_SCALE = 0.03

# Success criterion: episode is a SUCCESS if `done=True` OR
# (fallback) max per-step reward >= SUCCESS_REWARD_THR.
# LT block2block gives ~0.15-0.20 per step near goal; done=True on arrival.
SUCCESS_REWARD_THR = 0.15   # fallback; only used if done never fires

# Output paths
OUT_JSON = f"{CKPT_ROOT}/lt_task_success_results.json"
OUT_TXT  = f"{CKPT_ROOT}/lt_task_success_results.txt"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CHECKPOINT MAP
# Each entry: condition_key → (display_name, {vera_flag_overrides}, suppress_action_tok)
# suppress_action_tok=True: pass null prev_action_idx (simulates no-E_act token)
# ══════════════════════════════════════════════════════════════════════════════

# These must match the directory names you used when organising checkpoints.
ABLATION_CONDITIONS = [
    # (dir_name, display_name, vera_flags_override, suppress_action_token)
    (
        "lt_full_vera",
        "Full TERA ★ (all 5 streams)",
        {},           # use all default True flags from checkpoint cfg
        False,
    ),
    (
        "lt_bc_baseline",
        "BC/SFT baseline",
        {"use_lang_feedback": False, "use_temporal_history": False},
        False,
    ),
    (
        "lt_no_lang",
        "No lang. feedback (1,2,4)",
        {"use_lang_feedback": False},
        False,
    ),
    (
        "lt_no_exp",
        "No E_emb — narration only (1,2,3a,4)",
        {"use_consequence_token": False},
        False,
    ),
    (
        "lt_no_act",
        "No E_act — emb. know. only (1,2,3b,4)",
        {},           # model flags unchanged; null action_idx suppresses E_act
        True,         # ← suppress_action_token
    ),
    (
        "lt_no_hist_tf",
        "No hist. TF (1,2,3a,3b)",
        {"use_temporal_history": False},
        False,
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INSTALL DEPENDENCIES
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

print("Installing dependencies …")
for pkg in ("pyyaml", "imageio", "pillow", "numpy"):
    pip_one(pkg)
pip_one("gym<=0.23.0")
pip_one("pybullet")
pip_one("opencv-python")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "git+https://github.com/google-research/language-table.git"],
)
print("  ✓ language-table (sim-only)")

try:
    import clip  # noqa
except ImportError:
    pip_one("git+https://github.com/openai/CLIP.git")

print("Dependencies done.\n")

# Sanity check
from language_table.environments import language_table as lt_env_module  # noqa
print("language_table import OK")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import copy, json, random, time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import clip
import torchvision.transforms as Tv
from PIL import Image as PILImage

from language_table.environments import language_table as lt_env_module
from language_table.environments.rewards import block2block
try:
    from language_table.environments.rewards import block2block_relative_location
except ImportError:
    from language_table.environments.rewards import (
        block_to_block_relative_location as block2block_relative_location,
    )
from language_table.environments.rewards import (
    block2absolutelocation,
    block2relativelocation,
)
from language_table.environments import blocks

# Make repo importable
sys.path.insert(0, str(Path(CONFIG_PATH).parent.parent))
from models.vera_model import VERAModel

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: No GPU — Runtime → Change runtime type → T4 GPU for speed.")
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HELPERS: LT ENV, IMAGE PROCESSING, ACTION VOCAB
# ══════════════════════════════════════════════════════════════════════════════

def make_lt_env(seed: int = 0):
    reward_factory = block2block.BlockToBlockReward if REWARD_MODE == "block2block" else (
        lambda: random.choice([
            block2block.BlockToBlockReward,
            block2absolutelocation.BlockToAbsoluteLocationReward,
            block2relativelocation.BlockToRelativeLocationReward,
        ])()
    )
    if REWARD_MODE == "block2block":
        rf = block2block.BlockToBlockReward
    else:
        rf = random.choice([
            block2block.BlockToBlockReward,
            block2absolutelocation.BlockToAbsoluteLocationReward,
        ])
    return lt_env_module.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=rf,
        seed=seed,
    )


def decode_lt_instruction(obs: dict) -> str:
    inst = obs.get("instruction")
    if inst is None:
        return "complete the task"
    if isinstance(inst, str):
        return inst.strip().rstrip(".")
    if isinstance(inst, bytes):
        return inst.decode("utf-8", errors="ignore").strip().rstrip(".")
    arr = np.asarray(inst).flatten()
    try:
        return lt_env_module.LanguageTable.decode_instruction(arr).strip().rstrip(".")
    except Exception:
        nonzero = arr[arr != 0]
        if len(nonzero):
            return bytes(nonzero.astype(np.uint8).tolist()).decode("utf-8", errors="ignore").strip()
    return "complete the task"


def lt_obs_to_frame(obs: dict) -> np.ndarray:
    for key in ("rgb", "image", "pixels", "agentview_rgb"):
        val = obs.get(key)
        if val is not None:
            arr = np.asarray(val, dtype=np.uint8)
            if arr.ndim == 3:
                return arr
    return np.zeros((224, 224, 3), dtype=np.uint8)


def build_transform(img_size: int):
    return Tv.Compose([
        Tv.Resize((img_size, img_size)),
        Tv.ToTensor(),
        Tv.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                     std =[0.26862954, 0.26130258, 0.27577711]),
    ])


def action_idx_to_continuous(action_idx: int, num_actions: int = 8) -> np.ndarray:
    angle = (action_idx / num_actions) * 2 * np.pi
    return LT_ACTION_SCALE * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


def lt_action_from_output(out, num_actions: int) -> tuple:
    """Return (cont_action [2], discrete_idx, hist_vec [2])."""
    avec = out["action_vec"].squeeze().detach().cpu().numpy().astype(np.float32)
    if avec.shape[0] < 2:
        avec = np.pad(avec.flatten(), (0, max(0, 2 - avec.size)))[:2]
    disc = int(out["logits"].argmax(dim=-1).item())
    if float(np.linalg.norm(avec)) < 0.08:
        cont = action_idx_to_continuous(disc, num_actions)
        hist = np.clip(cont / max(LT_ACTION_SCALE, 1e-6), -1.0, 1.0)
    else:
        cont = (avec[:2] * LT_ACTION_SCALE).astype(np.float32)
        hist = np.clip(avec[:2], -1.0, 1.0)
    return cont, disc, hist


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_vera(ckpt_path: str, flag_overrides: dict, cfg_base: dict) -> VERAModel:
    """Load a TERA checkpoint, applying vera flag overrides for ablation."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefer config embedded in checkpoint (matches training exactly)
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg = copy.deepcopy(ckpt["cfg"])
    else:
        cfg = copy.deepcopy(cfg_base)

    m_cfg    = cfg["model"]
    vera_cfg = cfg.get("vera", {})

    # Apply ablation overrides
    for k, v in flag_overrides.items():
        vera_cfg[k] = v

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
        unfreeze_clip_vision  = m_cfg.get("unfreeze_clip_vision", True),
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
        print(f"    [load] {len(missing)} missing keys (expected for ablations)")
    model.eval()
    return model, cfg


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SINGLE-EPISODE ROLLOUT
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    model:                VERAModel,
    cfg:                  dict,
    env_seed:             int,
    transform,
    tok_cache:            dict,
    suppress_action_tok:  bool = False,
) -> dict:
    """
    Run one closed-loop episode.
    Returns dict with: success (bool), total_reward, steps, instruction.
    """
    m_cfg       = cfg["model"]
    num_actions = m_cfg["num_actions"]
    history_len = m_cfg["history_len"]
    num_vis     = m_cfg.get("num_vis_frames", 3)
    action_dim  = m_cfg.get("action_dim", 2)
    null_vec    = np.zeros(action_dim, dtype=np.float32)

    env  = make_lt_env(seed=env_seed)
    obs  = env.reset()
    instr = decode_lt_instruction(obs)

    if instr not in tok_cache:
        tok_cache[instr] = clip.tokenize([instr])[0]

    frame_q      = deque(maxlen=num_vis)
    action_q     = deque([num_actions] * history_len, maxlen=history_len)
    reward_q     = deque([0.0]         * history_len, maxlen=history_len)
    action_vec_q = deque([null_vec.copy() for _ in range(history_len)], maxlen=history_len)
    prev_action  = num_actions   # null sentinel
    prev_reward  = 0.0
    prev_delta   = 0.0

    total_reward = 0.0
    max_step_reward = 0.0
    success = False
    step    = 0
    done    = False

    while not done and step < MAX_STEPS:
        frame_np = lt_obs_to_frame(obs)
        frame_t  = transform(PILImage.fromarray(frame_np))
        frame_q.append(frame_t)
        pad       = num_vis - len(frame_q)
        frames_in = torch.stack(
            [torch.zeros_like(frame_t)] * pad + list(frame_q)
        ).unsqueeze(0).to(device)

        lang_in = tok_cache[instr].unsqueeze(0).to(device)

        # suppress_action_tok: pass null idx to zero out E_act narration
        eff_prev_action = num_actions if suppress_action_tok else prev_action

        act_hist_in = torch.tensor(list(action_q),  dtype=torch.long).unsqueeze(0).to(device)
        rew_hist_in = torch.tensor(list(reward_q),  dtype=torch.float32).unsqueeze(0).to(device)
        av_hist_in  = torch.tensor(
            np.stack(list(action_vec_q)), dtype=torch.float32
        ).unsqueeze(0).to(device)
        prev_a_in   = torch.tensor([eff_prev_action], dtype=torch.long).to(device)
        prev_r_in   = torch.tensor([prev_reward],     dtype=torch.float32).to(device)
        delta_in    = torch.tensor([prev_delta],      dtype=torch.float32).to(device)

        with torch.no_grad():
            out = model(
                frames_in, lang_in, act_hist_in, rew_hist_in,
                prev_a_in, prev_r_in,
                state_delta=delta_in,
                action_vec_hist=av_hist_in,
            )
            cont_action, disc_action, av_hist = lt_action_from_output(out, num_actions)

        obs, reward, done, _info = env.step(cont_action)
        reward = float(reward or 0.0)
        done   = bool(done)

        total_reward    += reward
        max_step_reward = max(max_step_reward, reward)

        action_q.append(disc_action)
        reward_q.append(reward)
        action_vec_q.append(av_hist.astype(np.float32))
        prev_action = disc_action
        prev_reward = reward
        prev_delta  = float(np.linalg.norm(cont_action))
        step       += 1

        if done:
            success = True

    env.close()

    # Fallback: count as success if max step reward exceeds threshold
    if not success and max_step_reward >= SUCCESS_REWARD_THR:
        success = True

    return {
        "success":      success,
        "total_reward": total_reward,
        "steps":        step,
        "instruction":  instr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("Loading base config …")
with open(CONFIG_PATH) as f:
    cfg_base = yaml.safe_load(f)

img_size  = cfg_base["data"].get("img_size", 224)
transform = build_transform(img_size)
tok_cache = {}

all_results = {}   # condition → list of per-seed dicts

print(f"\n{'='*70}")
print(f"Language-Table closed-loop task-success evaluation")
print(f"  {len(ABLATION_CONDITIONS)} conditions × {len(SEEDS)} seeds × {N_EPISODES} episodes")
print(f"  Max steps per episode: {MAX_STEPS}  |  Reward mode: {REWARD_MODE}")
print(f"{'='*70}\n")

for dir_name, display_name, flag_overrides, suppress_action_tok in ABLATION_CONDITIONS:
    print(f"\n── {display_name} ──")
    condition_results = []

    for seed in SEEDS:
        ckpt_path = Path(CKPT_ROOT) / dir_name / f"seed{seed}" / "best_sft_vera.pt"
        if not ckpt_path.is_file():
            print(f"  [WARN] checkpoint not found: {ckpt_path}")
            print(f"         Skipping seed {seed}.")
            continue

        print(f"  Seed {seed}: loading {ckpt_path.name} …", end=" ", flush=True)
        t0 = time.time()
        model, cfg_ckpt = load_vera(str(ckpt_path), flag_overrides, cfg_base)

        successes, rewards, steps_list = [], [], []
        for ep_i in range(N_EPISODES):
            ep_seed = seed * 1000 + ep_i
            result  = run_episode(
                model, cfg_ckpt, ep_seed, transform, tok_cache, suppress_action_tok,
            )
            successes.append(int(result["success"]))
            rewards.append(result["total_reward"])
            steps_list.append(result["steps"])

        success_rate = float(np.mean(successes)) * 100.0
        mean_reward  = float(np.mean(rewards))
        mean_steps   = float(np.mean(steps_list))
        elapsed      = time.time() - t0

        print(f"success={success_rate:.1f}%  reward={mean_reward:.3f}  "
              f"steps={mean_steps:.1f}  ({elapsed:.0f}s)")

        condition_results.append({
            "seed":         seed,
            "success_rate": success_rate,
            "mean_reward":  mean_reward,
            "mean_steps":   mean_steps,
            "successes":    successes,
            "rewards":      rewards,
        })

        # Free GPU memory between conditions
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if condition_results:
        seed_rates = [r["success_rate"] for r in condition_results]
        seed_rwds  = [r["mean_reward"]  for r in condition_results]
        all_results[dir_name] = {
            "display_name":  display_name,
            "per_seed":      condition_results,
            "success_mean":  float(np.mean(seed_rates)),
            "success_std":   float(np.std(seed_rates)),
            "reward_mean":   float(np.mean(seed_rwds)),
            "reward_std":    float(np.std(seed_rwds)),
        }
        print(f"  → MEAN SUCCESS: {np.mean(seed_rates):.1f}% ± {np.std(seed_rates):.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PRINT RESULTS TABLE + SAVE
# ══════════════════════════════════════════════════════════════════════════════

# Table header
header = f"\n{'='*70}\n"
header += f"{'Method':<42} {'Success (%)':>14} {'Mean Reward':>13}\n"
header += f"{'-'*42} {'-'*14} {'-'*13}\n"

rows = []
order = [c[0] for c in ABLATION_CONDITIONS]   # preserve paper order
for dir_name in order:
    if dir_name not in all_results:
        rows.append(f"  {dir_name:<40} {'N/A':>14} {'N/A':>13}")
        continue
    res   = all_results[dir_name]
    name  = res["display_name"]
    sr    = res["success_mean"]
    sr_s  = res["success_std"]
    rr    = res["reward_mean"]
    rr_s  = res["reward_std"]
    star  = " ★" if dir_name == "lt_full_vera" else "  "
    rows.append(f"  {name+star:<42} {sr:6.1f} ± {sr_s:4.1f}%  {rr:5.3f} ± {rr_s:.3f}")

print(header + "\n".join(rows) + f"\n{'='*70}")
print(f"\nN_EPISODES={N_EPISODES} per seed × {len(SEEDS)} seeds.  "
      f"Success = done=True OR max-step-reward ≥ {SUCCESS_REWARD_THR}.")

# Save JSON
Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved JSON: {OUT_JSON}")

# Save text summary
txt_lines = [
    "Language-Table Closed-Loop Task-Success Evaluation",
    f"N={N_EPISODES} episodes × {len(SEEDS)} seeds | MAX_STEPS={MAX_STEPS} | "
    f"REWARD_MODE={REWARD_MODE} | SUCCESS_THR={SUCCESS_REWARD_THR}",
    "",
    header.strip(),
] + rows + [f"{'='*70}"]
with open(OUT_TXT, "w") as f:
    f.write("\n".join(txt_lines) + "\n")
print(f"Saved TXT:  {OUT_TXT}")

# ── Paper-ready LaTeX snippet ─────────────────────────────────────────────────
print("\n── LaTeX snippet for ablation table (CALVIN column) ──")
for dir_name in order:
    if dir_name not in all_results:
        continue
    res = all_results[dir_name]
    sr  = res["success_mean"]
    std = res["success_std"]
    bold_open  = r"\mathbf{" if dir_name == "lt_full_vera" else ""
    bold_close = "}"         if dir_name == "lt_full_vera" else ""
    print(f"  % {res['display_name']}")
    print(f"  & ${bold_open}{sr:.1f}\\pm{std:.1f}{bold_close}\\%$ \\\\")

print("\nDone! Copy the LaTeX snippet into tab:ablations (CALVIN D→D column).")
print("Update the paper with real task-success numbers.")
