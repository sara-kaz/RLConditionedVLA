r"""
eval_full_tera_lt.py
====================
Run Full TERA on Language-Table (3 seeds × 50 episodes) and report task success rate.

Run in Colab (T4 GPU):
    from google.colab import drive; drive.mount('/content/drive')
    %run /content/drive/MyDrive/VLA-Robot-Learning/docs/eval_full_tera_lt.py

Upload the organised checkpoints folder to Drive first:
    MyDrive/VERA_LT_Checkpoints/lt_full_vera/seed42/best_sft_vera.pt
    MyDrive/VERA_LT_Checkpoints/lt_full_vera/seed123/best_sft_vera.pt
    MyDrive/VERA_LT_Checkpoints/lt_full_vera/seed456/best_sft_vera.pt
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────
MYDRIVE    = "/content/drive/MyDrive"
CKPT_DIR   = f"{MYDRIVE}/VERA_LT_Checkpoints/lt_full_vera"
REPO_URL   = "https://github.com/sara-kaz/RLConditionedVLA.git"
REPO_PATH  = "/content/repo"          # where the repo is cloned
SEEDS      = [42, 123, 456]
N_EPISODES = 50      # per seed  →  150 total rollouts
MAX_STEPS  = 60      # steps before declaring failure
LT_SCALE   = 0.03   # tanh action_vec → LT continuous delta

# ── INSTALL ───────────────────────────────────────────────────────────────────
import subprocess, sys, os
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg],
                          stderr=subprocess.STDOUT)

# Clone repo if not already present
if not os.path.isdir(REPO_PATH):
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, REPO_PATH])
    print(f"Cloned repo → {REPO_PATH}")
else:
    print(f"Repo already at {REPO_PATH}")

CONFIG = f"{REPO_PATH}/configs/config.yaml"

for pkg in ["pyyaml", "pillow", "numpy", "gym<=0.23.0", "pybullet"]:
    _pip(pkg)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    "git+https://github.com/google-research/language-table.git"])
try:
    import clip
except ImportError:
    _pip("git+https://github.com/openai/CLIP.git")
    import clip

from language_table.environments import language_table as _lt
from language_table.environments.rewards import block2block
from language_table.environments import blocks
print("language_table OK\n")

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import copy, sys, time, yaml, json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as Tv
from PIL import Image as PILImage

sys.path.insert(0, REPO_PATH)
from models.vera_model import VERAModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
transform = Tv.Compose([
    Tv.Resize((224, 224)),
    Tv.ToTensor(),
    Tv.Normalize([0.48145466, 0.4578275, 0.40821073],
                 [0.26862954, 0.26130258, 0.27577711]),
])

def make_env(seed):
    return _lt.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        seed=seed,
    )

def decode_instr(obs):
    v = obs.get("instruction")
    if isinstance(v, str):   return v.strip().rstrip(".")
    if isinstance(v, bytes): return v.decode().strip().rstrip(".")
    arr = np.asarray(v).flatten()
    try:    return _lt.LanguageTable.decode_instruction(arr).strip().rstrip(".")
    except: return "complete the task"

def get_frame(obs):
    for k in ("rgb", "image", "pixels"):
        v = obs.get(k)
        if v is not None:
            a = np.asarray(v, dtype=np.uint8)
            if a.ndim == 3: return a
    return np.zeros((224, 224, 3), dtype=np.uint8)

def load_model(ckpt_path, cfg_base):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = copy.deepcopy(ckpt["cfg"]) if isinstance(ckpt, dict) and "cfg" in ckpt else copy.deepcopy(cfg_base)
    m    = cfg["model"]; vcfg = cfg.get("vera", {})

    d_model     = m.get("d_model", 256)
    num_actions = m["num_actions"]
    chunk_size  = m.get("chunk_size", 1)

    # ── Peek at checkpoint BEFORE building model ──────────────────────────────
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    ah1   = state.get("action_head.1.weight")
    # Old head (no expand-compress): action_head.1 is Linear(D→D) → shape [D,D]
    # New head (expand-compress):    action_head.1 is Linear(D→2D) → shape [2D,D]
    old_head = (ah1 is not None and tuple(ah1.shape) == (d_model, d_model))

    model = VERAModel(
        num_actions=num_actions, history_len=m["history_len"],
        num_vis_frames=m.get("num_vis_frames", 3),
        fusion_layers=m.get("fusion_layers", 6), fusion_heads=m.get("fusion_heads", 8),
        d_model=d_model, d_ff_scale=m.get("d_ff_scale", 4),
        dropout=0.0, vision_token_dropout=0.0,
        freeze_clip=m.get("freeze_clip", True),
        unfreeze_clip_vision=m.get("unfreeze_clip_vision", True),
        use_lang_feedback=vcfg.get("use_lang_feedback", True),
        use_temporal_history=vcfg.get("use_temporal_history", True),
        use_reward_gate=vcfg.get("use_reward_gate", True),
        use_consequence_token=vcfg.get("use_consequence_token", True),
        action_dim=m.get("action_dim", 2),
        action_vocab=vcfg.get("action_vocab"),
        chunk_size=chunk_size,
    ).to(device)

    if old_head:
        # Rebuild action head to match checkpoint's simpler 3-layer MLP:
        #   RMSNorm(D) → Linear(D→D) → SiLU → Dropout →
        #   Linear(D→D) → SiLU → Dropout → Linear(D→A)
        # indices:  0               1          2     3
        #                           4          5     6     7
        rmsnorm0 = model.action_head[0]          # reuse already-built RMSNorm[D]
        model.action_head = nn.Sequential(
            rmsnorm0,                                                         # 0
            nn.Linear(d_model, d_model, bias=False),                         # 1
            nn.SiLU(),                                                        # 2
            nn.Dropout(0.0),                                                  # 3
            nn.Linear(d_model, d_model, bias=False),                         # 4
            nn.SiLU(),                                                        # 5
            nn.Dropout(0.0),                                                  # 6
            nn.Linear(d_model, num_actions * chunk_size, bias=False),        # 7
        ).to(device)
        print("  [load_model] old action head detected → rebuilt 3-layer MLP to match checkpoint")

    # ── Load weights (all keys should now match) ──────────────────────────────
    cur = model.state_dict()
    compatible = {k: val for k, val in state.items()
                  if k in cur and val.shape == cur[k].shape}
    skipped = [k for k in state if k not in compatible]
    if skipped:
        print(f"  [load_model] still-skipped {len(skipped)} key(s): "
              f"{skipped[:6]}{'...' if len(skipped)>6 else ''}")
    else:
        print("  [load_model] all checkpoint keys loaded successfully ✓")
    model.load_state_dict(compatible, strict=False)
    model.eval()
    return model, cfg

def run_episode(model, cfg, ep_seed, tok_cache):
    m       = cfg["model"]
    N_ACT   = m["num_actions"]
    H       = m["history_len"]
    N_VIS   = m.get("num_vis_frames", 3)
    ADIM    = m.get("action_dim", 2)
    null_v  = np.zeros(ADIM, dtype=np.float32)

    env     = make_env(ep_seed)
    obs     = env.reset()
    instr   = decode_instr(obs)
    if instr not in tok_cache:
        tok_cache[instr] = clip.tokenize([instr])[0]

    fq  = deque(maxlen=N_VIS)
    aq  = deque([N_ACT] * H, maxlen=H)
    rq  = deque([0.0]   * H, maxlen=H)
    avq = deque([null_v.copy() for _ in range(H)], maxlen=H)
    pa, pr, pd = N_ACT, 0.0, 0.0

    total_r, max_r, done, step = 0.0, 0.0, False, 0
    while not done and step < MAX_STEPS:
        fr = get_frame(obs)
        ft = transform(PILImage.fromarray(fr)); fq.append(ft)
        pad = N_VIS - len(fq)
        fi  = torch.stack([torch.zeros_like(ft)] * pad + list(fq)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(
                fi,
                tok_cache[instr].unsqueeze(0).to(device),
                torch.tensor(list(aq),  dtype=torch.long).unsqueeze(0).to(device),
                torch.tensor(list(rq),  dtype=torch.float32).unsqueeze(0).to(device),
                torch.tensor([pa],  dtype=torch.long).to(device),
                torch.tensor([pr],  dtype=torch.float32).to(device),
                state_delta=torch.tensor([pd], dtype=torch.float32).to(device),
                action_vec_hist=torch.tensor(np.stack(list(avq)), dtype=torch.float32).unsqueeze(0).to(device),
            )
        # Action: prefer continuous vec, fallback to discrete
        avec = out["action_vec"].squeeze().detach().cpu().numpy().astype(np.float32)[:ADIM]
        disc = int(out["logits"].argmax(-1).item())
        if np.linalg.norm(avec) < 0.08:
            ang  = (disc / N_ACT) * 2 * np.pi
            cont = LT_SCALE * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            hist = np.clip(cont / LT_SCALE, -1, 1)
        else:
            cont = (avec[:2] * LT_SCALE).astype(np.float32)
            hist = np.clip(avec[:2], -1, 1)

        obs, reward, done, _ = env.step(cont)
        reward = float(reward or 0.0); done = bool(done)
        total_r += reward; max_r = max(max_r, reward)
        aq.append(disc); rq.append(reward); avq.append(hist.astype(np.float32))
        pa, pr, pd = disc, reward, float(np.linalg.norm(cont))
        step += 1

    env.close()
    success = done or (max_r >= 0.15)   # done=True is primary; high-reward step as fallback
    return {"success": success, "total_reward": total_r, "steps": step}

# ── MAIN ──────────────────────────────────────────────────────────────────────
with open(CONFIG) as f:
    cfg_base = yaml.safe_load(f)

tok_cache     = {}
seed_rates    = []
seed_rewards  = []

print(f"Full TERA — Language-Table task success  ({N_EPISODES} ep × {len(SEEDS)} seeds)\n")

for seed in SEEDS:
    ckpt = Path(CKPT_DIR) / f"seed{seed}" / "best_sft_vera.pt"
    if not ckpt.is_file():
        print(f"  [SKIP] seed {seed}: {ckpt} not found"); continue

    print(f"  Seed {seed}: ", end="", flush=True)
    t0 = time.time()
    model, cfg_ckpt = load_model(str(ckpt), cfg_base)

    results = [run_episode(model, cfg_ckpt, seed * 1000 + i, tok_cache)
               for i in range(N_EPISODES)]

    sr = 100.0 * np.mean([r["success"]      for r in results])
    mr =         np.mean([r["total_reward"]  for r in results])
    ms =         np.mean([r["steps"]         for r in results])
    print(f"success={sr:.1f}%  reward={mr:.3f}  steps={ms:.1f}  ({time.time()-t0:.0f}s)")
    seed_rates.append(sr); seed_rewards.append(mr)

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print(f"\n{'='*50}")
print(f"  Full TERA — LT Task Success")
print(f"  {np.mean(seed_rates):.1f}% ± {np.std(seed_rates):.1f}%  "
      f"(seeds: {', '.join(str(s) for s in SEEDS)})")
print(f"  Mean episode reward: {np.mean(seed_rewards):.3f}")
print(f"{'='*50}")
print(f"\nLaTeX: $\\mathbf{{{np.mean(seed_rates):.1f}\\pm{np.std(seed_rates):.1f}\\%}}$")
