"""
make_lt_qual_figure.py — Language-Table rollout visualization for TERA paper.

THREE WAYS TO RUN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. COMPOSITE FIGURE (default) — one PNG with all 3 episodes:
   python3 docs/make_lt_qual_figure.py /path/to/lt_data
   → saves lt_qual_composite.png

2. INDIVIDUAL FRAMES — 9 separate PNGs so you can compose in any tool:
   python3 docs/make_lt_qual_figure.py /path/to/lt_data --save-individual
   → saves individual_frames/ep0_start.png, ep0_mid.png, ep0_end.png, ...
      + individual_frames/episode_info.txt (instructions + token strings)

3. PLACEHOLDER (no data — layout preview only):
   python3 docs/make_lt_qual_figure.py
   → grey-box composite so you can check LaTeX layout

Output (composite):  docs/lt_qual_composite.png
                     ~/Downloads/corl_2026_template_submission/lt_qual_composite.png
"""

import sys, os, pickle, glob, random, textwrap, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("lt_root", nargs="?", default=None,
                    help="Path to folder containing episode_*/steps.pkl")
parser.add_argument("--save-individual", action="store_true",
                    help="Export each frame as its own PNG instead of composite")
parser.add_argument("--dpi",  type=int,   default=200,
                    help="DPI for composite PNG (default 200)")
parser.add_argument("--out",  default=None,
                    help="Override output path for composite PNG")
args = parser.parse_args()

LT_ROOT         = args.lt_root
SAVE_INDIVIDUAL = args.save_individual
DPI             = args.dpi

# ── ╔══════════════════════════════════════════╗ ──────────────────────────────
# ── ║  EASY CUSTOMISATION — edit here freely  ║ ──────────────────────────────
# ── ╚══════════════════════════════════════════╝ ──────────────────────────────

FIG_W           = 9.5   # figure width in inches
ROW_H           = 1.90  # height of each image row in inches
TOK_H           = 0.40  # height of token-annotation row in inches
SEP_H           = 0.06  # thin gap between rows
LABEL_W_FRAC    = 0.22  # width fraction for "Instruction" column
FRAME_BORDER    = "#AAAAAA"
FONT_HEADER     = 9     # "Instruction / Start / Middle / End" bold header size
FONT_LABEL      = 8     # instruction text italic size
FONT_TOKEN      = 7.2   # E_act / E_emb annotation size
FONT_CAPTION    = 6.5   # in-figure caption size (set 0 to suppress caption)
INCLUDE_CAPTION = True  # set False to omit the caption text inside the PNG
INDIVIDUAL_DPI  = 300   # DPI for individual frame exports
INDIVIDUAL_DIR  = None  # None → auto: docs/individual_frames/ next to script

# ── Token strings (one pair per episode row) ──────────────────────────────────
# Edit these to match your actual TERA output for the chosen episodes.
TERA_TOKENS = [
    # (E_act string,                           E_emb string)
    ("I pushed the object to the left.",
     "I made little progress and received a low reward."),
    ("I pushed the object upward.",
     "I made little progress and received a low reward."),
    ("I pushed the object up and to the left.",
     "I made little progress and received a low reward."),
]

# ── Preferred episode instructions (matched against stored episode text) ───────
PREFERRED = [
    "move the red star into the yellow hexagon towards the bottom center",
    "move the yellow heart to the bottom right corner",
    "place the blue cube to the top of the red circle",
]

# ── Output paths (composite) ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
if args.out:
    OUT_PATHS = [Path(args.out)]
else:
    OUT_PATHS = [
        SCRIPT_DIR / "lt_qual_composite.png",
        Path("/Users/HP/Downloads/corl_2026_template_submission/lt_qual_composite.png"),
    ]

# ── Frame extraction helpers ──────────────────────────────────────────────────
def _extract_frame(step: dict):
    for key in ("obs", "image", "rgb", "pixels", "frame"):
        val = step.get(key)
        if val is None:
            continue
        if isinstance(val, np.ndarray) and val.ndim == 3:
            return val.astype(np.uint8)
        if isinstance(val, dict):
            for sub in ("rgb", "image", "pixels", "agentview_rgb"):
                v2 = val.get(sub)
                if v2 is not None and isinstance(v2, np.ndarray) and v2.ndim == 3:
                    return v2.astype(np.uint8)
    return None

def _get_instruction(steps):
    for key in ("instruction", "task", "lang", "language_instruction"):
        val = steps[0].get(key)
        if val:
            if isinstance(val, bytes):
                val = val.decode()
            return str(val).lower().strip().rstrip(".")
    return ""

def load_episodes(lt_root, n=3, seed=42):
    if not lt_root or not os.path.isdir(lt_root):
        return []
    ep_dirs = sorted(glob.glob(os.path.join(lt_root, "episode_*")))
    random.seed(seed)
    random.shuffle(ep_dirs)
    buckets = {p: None for p in PREFERRED}
    extras  = []
    for ep_dir in ep_dirs:
        pkl = os.path.join(ep_dir, "steps.pkl")
        if not os.path.isfile(pkl):
            continue
        with open(pkl, "rb") as f:
            steps = pickle.load(f)
        if len(steps) < 4:
            continue
        instr  = _get_instruction(steps)
        frames = [_extract_frame(s) for s in steps]
        frames = [f for f in frames if f is not None]
        if len(frames) < 3:
            continue
        mid = len(frames) // 2
        ep  = {"instruction": instr,
               "start": frames[0], "mid": frames[mid], "end": frames[-1]}
        matched = False
        for pref in PREFERRED:
            if pref in instr and buckets[pref] is None:
                buckets[pref] = ep
                matched = True
                break
        if not matched:
            extras.append(ep)
        if all(v is not None for v in buckets.values()):
            break
    result = []
    for pref in PREFERRED:
        result.append(buckets[pref] if buckets[pref] is not None
                      else (extras.pop(0) if extras else None))
    return [e for e in result if e is not None][:n]

# ── Load episodes ─────────────────────────────────────────────────────────────
PLACEHOLDER = np.full((128, 128, 3), 210, dtype=np.uint8)
episodes    = load_episodes(LT_ROOT, n=3)
using_real  = len(episodes) > 0

for i in range(len(episodes), 3):
    episodes.append({
        "instruction": PREFERRED[i],
        "start": PLACEHOLDER,
        "mid":   PLACEHOLDER,
        "end":   PLACEHOLDER,
    })

if not using_real:
    print("[WARN] No LT data found — using grey placeholder frames.")
    print("       Pass the LT data root as the first argument.")

# ═══════════════════════════════════════════════════════════════════════════════
# MODE A — save individual frames (9 PNGs + info text)
# ═══════════════════════════════════════════════════════════════════════════════
if SAVE_INDIVIDUAL:
    out_dir = Path(INDIVIDUAL_DIR) if INDIVIDUAL_DIR else SCRIPT_DIR / "individual_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    info_lines = ["# TERA Language-Table — individual frames",
                  "# Use these to compose your own figure in any tool.\n"]

    for ep_i, ep in enumerate(episodes):
        instr_str = ep["instruction"].capitalize().rstrip(".") + "."
        sn, ek    = TERA_TOKENS[ep_i]
        info_lines.append(f"## Episode {ep_i}")
        info_lines.append(f"Instruction : {instr_str}")
        info_lines.append(f"E_act       : {sn}")
        info_lines.append(f"E_emb       : {ek}\n")

        for slot, frame in [("start", ep["start"]),
                             ("mid",   ep["mid"]),
                             ("end",   ep["end"])]:
            fig_f, ax_f = plt.subplots(1, 1, figsize=(3, 3),
                                       facecolor="white")
            ax_f.imshow(frame)
            ax_f.set_xticks([]); ax_f.set_yticks([])
            for sp in ax_f.spines.values():
                sp.set_linewidth(0.8)
                sp.set_color(FRAME_BORDER)
            fig_f.tight_layout(pad=0.1)
            dest = out_dir / f"ep{ep_i}_{slot}.png"
            fig_f.savefig(str(dest), dpi=INDIVIDUAL_DPI,
                          bbox_inches="tight", facecolor="white")
            plt.close(fig_f)
            print(f"Saved frame: {dest}")

    info_path = out_dir / "episode_info.txt"
    info_path.write_text("\n".join(info_lines))
    print(f"Saved info : {info_path}")
    print(f"\nAll 9 frames + info in: {out_dir}")
    print("Import into Illustrator / Figma / Keynote / PowerPoint to compose.")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════════════
# MODE B — composite figure
# ═══════════════════════════════════════════════════════════════════════════════
N_ROWS  = 3
total_h = 0.45 + N_ROWS * (ROW_H + TOK_H + SEP_H) + 0.40
fig     = plt.figure(figsize=(FIG_W, total_h), facecolor="white")

outer = gridspec.GridSpec(
    N_ROWS * 2, 1,
    figure=fig,
    hspace=0.0,
    left=0.01, right=0.99,
    top=0.96,
    bottom=(0.12 if INCLUDE_CAPTION else 0.04),
)

frame_axes = []

for row_i in range(N_ROWS):
    img_gs = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=outer[row_i * 2],
        width_ratios=[LABEL_W_FRAC, 1, 1, 1],
        wspace=0.04,
    )
    ax_lbl = fig.add_subplot(img_gs[0])
    ax_s   = fig.add_subplot(img_gs[1])
    ax_m   = fig.add_subplot(img_gs[2])
    ax_e   = fig.add_subplot(img_gs[3])
    frame_axes.append((ax_lbl, ax_s, ax_m, ax_e))

    tok_gs  = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[row_i * 2 + 1])
    ax_tok  = fig.add_subplot(tok_gs[0])
    ax_tok.axis("off")
    sn, ek  = TERA_TOKENS[row_i]
    ax_tok.text(
        0.50, 0.80,
        f"$\\mathit{{E_{{\\mathrm{{act}}}}}}$: \"{sn}\"    "
        f"$\\mathit{{E_{{\\mathrm{{emb}}}}}}$: \"{ek}\"",
        transform=ax_tok.transAxes,
        fontsize=FONT_TOKEN, color="#2C3E50",
        va="top", ha="center", style="italic",
    )

COL_TITLES = ["Start", "Middle", "End"]

for row_i, ep in enumerate(episodes):
    ax_lbl, ax_s, ax_m, ax_e = frame_axes[row_i]
    ax_lbl.axis("off")
    wrapped = textwrap.fill(
        ep["instruction"].capitalize().rstrip(".") + ".", width=22)
    ax_lbl.text(0.95, 0.5, wrapped,
                transform=ax_lbl.transAxes,
                fontsize=FONT_LABEL, ha="right", va="center", style="italic")

    for ax_img, frame, col_lbl in [
        (ax_s, ep["start"], COL_TITLES[0]),
        (ax_m, ep["mid"],   COL_TITLES[1]),
        (ax_e, ep["end"],   COL_TITLES[2]),
    ]:
        ax_img.imshow(frame)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_linewidth(0.6); sp.set_color(FRAME_BORDER)
        if row_i == 0:
            ax_img.set_title(col_lbl, fontsize=FONT_HEADER,
                             fontweight="bold", pad=3)

frame_axes[0][0].set_title("Instruction", fontsize=FONT_HEADER,
                            fontweight="bold", pad=3)

if INCLUDE_CAPTION and FONT_CAPTION > 0:
    cap = (
        "Figure 2. Three Language-Table demonstrations where TERA successfully "
        "follows the text instruction (start → middle → end frame). The per-step "
        "Trustworthy Safety Narration token ($E_{\\mathrm{act}}$) and Embodied "
        "Knowledge Planning token ($E_{\\mathrm{emb}}$) are shown below each row, "
        "illustrating closed-loop language feedback driving progressive trajectory correction."
    )
    fig.text(0.01, 0.01, cap, fontsize=FONT_CAPTION,
             va="bottom", color="#333333", ha="left",
             wrap=True)

for dest in OUT_PATHS:
    try:
        fig.savefig(str(dest), dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved composite: {dest}")
    except Exception as e:
        print(f"[WARN] Could not save to {dest}: {e}")

plt.close(fig)
