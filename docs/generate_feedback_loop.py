"""
VLLA Feedback Loop Diagram — standalone, focused, clean.
Shows exactly what feeds back from one timestep to the next.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

VIS  = "#1A6FA8"; INS  = "#6C3483"; ACT  = "#1A7A40"
CON  = "#B9610E"; HIS  = "#8B1A1A"; FUSE = "#0B6B58"
ENV  = "#2C3E50"; HEAD = "#7D4608"; DARK = "#1C2833"
BG   = "#FFFFFF"; LGRAY= "#F4F6F7"; RED  = "#C0392B"
GOLD = "#B7950B"

W, H = 28, 20
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# ── helpers ───────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, fc, txt, fs=12, tc="white", bold=False,
        ec=DARK, lw=1.8, ls="-", alpha=1.0):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.18,rounding_size=0.25",
                       fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center", linespacing=1.55)

def arr(ax, x0, y0, x1, y1, col, lw=2.2, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=col,
                                lw=lw, mutation_scale=20,
                                shrinkA=4, shrinkB=4), zorder=6)

def label(x, y, txt, col=DARK, fs=11, ha="center", bold=False):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=fs, color=col,
            fontweight="bold" if bold else "normal", zorder=7,
            multialignment="center", linespacing=1.4)

def badge(cx, cy, txt, fc=RED):
    p = FancyBboxPatch((cx-0.52, cy-0.25), 1.04, 0.50,
                       boxstyle="round,pad=0.07", fc=fc, ec="none", zorder=9)
    ax.add_patch(p)
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=9, color="white", fontweight="bold", zorder=10)

# ════════════════════════════════════════════════════════════════════════════
#  TITLE
# ════════════════════════════════════════════════════════════════════════════
ax.text(W/2, 19.5,
        "VLLA Feedback Loop  —  How the Previous Step Feeds Into the Next",
        ha="center", va="center", fontsize=18, fontweight="bold", color=DARK)
ax.text(W/2, 19.0,
        "At every timestep the model receives THREE streams of feedback from what just happened",
        ha="center", va="center", fontsize=12, color="#555", style="italic")

# ════════════════════════════════════════════════════════════════════════════
#  CENTRAL BOXES:  MODEL  |  ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════════
# Model box (left-centre)
MX, MY = 7.5, 10.5
box(MX, MY, 8.5, 6.5, FUSE,
    "VLLA\nModel\n\n"
    "Sees all 5 streams\n"
    "→ produces action logits\n"
    "→ samples action  aₜ",
    fs=13, bold=True)

# Environment box (right-centre)
EX, EY = 20.5, 10.5
box(EX, EY, 8.5, 6.5, ENV,
    "Environment\n\n"
    "Receives  aₜ\n"
    "Executes it\n"
    "→ returns  oₜ₊₁,  rₜ,  done\n"
    "→ info['dist_delta']  (opt.)",
    fs=13, bold=True)

# ════════════════════════════════════════════════════════════════════════════
#  FORWARD ARROW:  action from model to env
# ════════════════════════════════════════════════════════════════════════════
arr(ax, MX+4.25, MY+1.0, EX-4.25, EY+1.0, GOLD, lw=3.5)
label(14.0, MY+1.7,
      "action  aₜ\n(argmax or sample from logits)",
      col=GOLD, fs=12, bold=True)

# ════════════════════════════════════════════════════════════════════════════
#  THREE FEEDBACK ARROWS  (env → model, going back left)
# ════════════════════════════════════════════════════════════════════════════

# ── Feedback A:  Action token  (Stream 3a) ───────────────────────────────
# Arrow path: env → top arc → model
arr(ax, EX-4.25, EY+0.3, MX+4.25, MY+0.3, ACT, lw=2.8)
label(14.0, MY+0.85,
      "prev_action_idx  aₜ\n→  vocab lookup  →  \"I moved forward\"\n→  CLIP text encoder  →  L_action token  (type = 1)",
      col=ACT, fs=11.5)

# ── Feedback B:  Consequence token  (Stream 3b) ──────────────────────────
arr(ax, EX-4.25, EY-0.5, MX+4.25, MY-0.5, CON, lw=2.8)
label(14.0, MY-0.55,
      "rₜ  +  dist_delta\n→  verbalize_consequence( )  →  \"I got closer to the goal\"\n→  CLIP text encoder  →  L_consequence token  (type = 4)",
      col=CON, fs=11.5)
badge(18.2, MY-0.10, "NEW")

# ── Feedback C:  Numerical history  (Stream 4) ───────────────────────────
arr(ax, EX-4.25, EY-1.3, MX+4.25, MY-1.3, HIS, lw=2.8)
label(14.0, MY-1.75,
      "aₜ, rₜ  appended to rolling window\n→  action_hist  (B, H)  +  reward_hist  (B, H)\n→  History Encoder  →  H history tokens  (type = 3)",
      col=HIS, fs=11.5)

# ════════════════════════════════════════════════════════════════════════════
#  TIMESTEP LABELS
# ════════════════════════════════════════════════════════════════════════════
box(MX, 7.0, 5.5, 0.70, DARK,
    "Timestep  t", fs=13, bold=True)
box(EX, 7.0, 5.5, 0.70, DARK,
    "Timestep  t", fs=13, bold=True)
box((MX+EX)/2, 5.8, 12.0, 0.70, "#7F8C8D",
    "→  feeds into  →  Timestep  t + 1",
    fs=13, bold=True)

# ════════════════════════════════════════════════════════════════════════════
#  WHAT STAYS THE SAME EVERY STEP (top box)
# ════════════════════════════════════════════════════════════════════════════
box(MX, 17.5, 8.5, 1.50, INS,
    "Instruction  (FIXED for whole episode)\n"
    "\"pick up the red cube\"  →  L_instr token  (type = 0)",
    fs=12)
label(MX, 16.6, "stays the same every timestep", col="#888", fs=10.5)

box(MX, 15.6, 8.5, 1.50, VIS,
    "Vision frames  (UPDATED every timestep)\n"
    "current  +  previous frames  →  V₁ V₂ V₃ tokens  (type = 2)",
    fs=12)

# dashed arrow: instruction flows into model every step
arr(ax, MX, 16.75, MX, MY+3.25, INS, lw=1.8)
arr(ax, MX, 14.85, MX, MY+3.25, VIS, lw=1.8)

# ════════════════════════════════════════════════════════════════════════════
#  BOTTOM SUMMARY
# ════════════════════════════════════════════════════════════════════════════
ax.plot([0.5, W-0.5], [4.65, 4.65], color="#CCC", lw=1.5)

ax.text(W/2, 4.30,
        "The feedback loop in three lines:",
        ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)

rows = [
    (ACT,  "Stream 3a  ACTION TOKEN",
           "Model's own last action (aₜ) verbalized → "
           "\"I moved forward\" → CLIP → 1 token"),
    (CON,  "Stream 3b  CONSEQUENCE TOKEN  [NEW]",
           "Reward + distance change verbalized → "
           "\"I got closer to the goal\" → CLIP → 1 token"),
    (HIS,  "Stream 4   HISTORY TOKENS",
           "Rolling window of last H (action, reward) pairs → "
           "History Encoder → H tokens"),
]
for i, (col, heading, detail) in enumerate(rows):
    iy = 3.60 - i * 1.10
    p = FancyBboxPatch((0.8, iy-0.38), 1.4, 0.76,
                       boxstyle="round,pad=0.07", fc=col, ec="none", zorder=5)
    ax.add_patch(p)
    ax.text(1.50, iy, heading, ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", zorder=6)
    ax.text(2.50, iy, detail, ha="left", va="center",
            fontsize=12, color=DARK, zorder=6)

badge(20.5, 3.62, "NEW  in this version")

fig.savefig("docs/vlla_feedback_loop.png", dpi=160,
            bbox_inches="tight", facecolor=BG, edgecolor="none")
print("Saved → docs/vlla_feedback_loop.png")
plt.close()
