"""
VLLA — 3D Isometric Architecture Diagram
==========================================
Draws the VLLA pipeline as an isometric 3D perspective:
  • Five streams rise from the bottom as vertical columns
  • Each processing stage is a floating 3D slab at a different height
  • Arrows flow upward through the stages
  • The five streams converge into the token sequence, then the LLaMA
    Transformer, then the action head at the top

Saved to: docs/vlla_architecture_3d.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Colour palette ─────────────────────────────────────────────────────────────
VIS  = "#1A6FA8"   # blue   — Stream 1: Vision
INS  = "#6C3483"   # purple — Stream 2: Instruction
ACT  = "#1A7A40"   # green  — Stream 3a: Action Language
CON  = "#B9610E"   # orange — Stream 3b: Consequence [NEW]
HIS  = "#8B1A1A"   # red    — Stream 4: History
CLIP = "#4A4A4A"   # dark grey — CLIP backbone
FUSE = "#0B6B58"   # teal   — LLaMA Fusion
HEAD = "#7D4608"   # brown  — Action Head
BG   = "#FAFAFA"
DARK = "#1C2833"
NEW  = "#E67E22"   # orange highlight for NEW badge

STREAM_COLS = [VIS, INS, ACT, CON, HIS]
STREAM_NAMES = [
    "S1\nVISION",
    "S2\nINSTR",
    "S3a\nACTION",
    "S3b\nCONSEQ",
    "S4\nHISTORY",
]


# ══════════════════════════════════════════════════════════════════════════════
# Isometric helper functions
# ══════════════════════════════════════════════════════════════════════════════

def iso(x, y, z):
    """
    Convert (x, y, z) world coordinates to 2D isometric screen coordinates.
    Isometric projection: x-axis goes right-down, y-axis goes right-up, z-axis goes up.
    """
    sx = (x - y) * np.cos(np.radians(30))
    sy = (x + y) * np.sin(np.radians(30)) + z
    return sx, sy


def draw_box_3d(ax, cx, cy, cz, w, d, h, fc, ec=DARK, alpha=0.92, lw=1.5):
    """
    Draw a 3D rectangular box centred at (cx,cy,cz) with dimensions (w,d,h).
    Uses isometric projection to compute 8 corners, then draws three visible faces:
      - Top face    (z = cz + h/2)
      - Front face  (y = cy - d/2)
      - Right face  (x = cx + w/2)
    """
    x0, x1 = cx - w/2, cx + w/2
    y0, y1 = cy - d/2, cy + d/2
    z0, z1 = cz - h/2, cz + h/2

    # 8 corners in world space
    corners = {
        "LBBot": (x0, y0, z0), "RBBot": (x1, y0, z0),
        "RFBot": (x1, y1, z0), "LFBot": (x0, y1, z0),
        "LBTop": (x0, y0, z1), "RBTop": (x1, y0, z1),
        "RFTop": (x1, y1, z1), "LFTop": (x0, y1, z1),
    }

    def p(name):
        return iso(*corners[name])

    # Lighten colour for top face, darken for side faces
    import matplotlib.colors as mc

    def lighten(col, factor=0.35):
        c = np.array(mc.to_rgb(col))
        return mc.to_hex(c + (1 - c) * factor)

    def darken(col, factor=0.30):
        c = np.array(mc.to_rgb(col))
        return mc.to_hex(c * (1 - factor))

    # Top face (lightest)
    top = [p("LBTop"), p("RBTop"), p("RFTop"), p("LFTop")]
    poly_top = plt.Polygon(top, closed=True,
                           fc=lighten(fc), ec=ec, lw=lw, alpha=alpha, zorder=3)
    ax.add_patch(poly_top)

    # Front face (medium — y = y1)
    front = [p("LFBot"), p("RFBot"), p("RFTop"), p("LFTop")]
    poly_front = plt.Polygon(front, closed=True,
                             fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=3)
    ax.add_patch(poly_front)

    # Right face (darkest — x = x1)
    right = [p("RBBot"), p("RFBot"), p("RFTop"), p("RBTop")]
    poly_right = plt.Polygon(right, closed=True,
                              fc=darken(fc), ec=ec, lw=lw, alpha=alpha, zorder=3)
    ax.add_patch(poly_right)

    # Return the 2D screen centre of the top face (for text placement)
    tx = np.mean([p("LBTop")[0], p("RBTop")[0], p("RFTop")[0], p("LFTop")[0]])
    ty = np.mean([p("LBTop")[1], p("RBTop")[1], p("RFTop")[1], p("LFTop")[1]])
    return tx, ty


def draw_arrow_3d(ax, x0, y0, z0, x1, y1, z1, col=DARK, lw=2.0):
    """Draw an isometric arrow from world (x0,y0,z0) to (x1,y1,z1)."""
    sx0, sy0 = iso(x0, y0, z0)
    sx1, sy1 = iso(x1, y1, z1)
    ax.annotate(
        "", xy=(sx1, sy1), xytext=(sx0, sy0),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=16),
        zorder=6,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layout constants
# ══════════════════════════════════════════════════════════════════════════════

# Stream x-positions (0..4 from left to right)
SX = [0, 3, 6, 9, 12]
# Stream y-position (all at same depth — front of diagram)
SY = 2.0
# Stage z-levels (height in world units)
Z_INPUT  =  0.0
Z_CLIP   =  3.5
Z_PROJ   =  7.0
Z_MERGE  = 10.5
Z_SEQ    = 13.5
Z_FUSE   = 17.0
Z_HEAD   = 21.0
Z_OUT    = 24.5

BOX_W  = 2.2   # box width (x-dir)
BOX_D  = 1.4   # box depth (y-dir)
BOX_H  = 1.2   # box height (z-dir)

# CLIP/Fusion blocks are wide
CLIP_W = 14.0
FUSE_W = 14.0

# Centre of all streams (for wide blocks)
CX_ALL = (SX[0] + SX[-1]) / 2   # = 6.0
CY_ALL = SY


# ══════════════════════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(32, 52))
ax.set_aspect("equal")
ax.axis("off")
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)

# Compute screen extent — must include all boxes including INPUT at bottom
sx_min_raw, _  = iso(SX[0]  - BOX_W - 5, SY + BOX_D + 1, Z_INPUT - BOX_H - 1)
sx_max_raw, _  = iso(SX[-1] + BOX_W + 8, SY - BOX_D,     Z_OUT   + BOX_H)
_, sy_min_raw  = iso(CX_ALL, SY + BOX_D + 2, Z_INPUT - BOX_H - 2)
_, sy_max_raw  = iso(CX_ALL, SY - BOX_D,     Z_OUT   + BOX_H + 4)

ax.set_xlim(sx_min_raw - 1, sx_max_raw + 1)
ax.set_ylim(sy_min_raw - 1, sy_max_raw + 1)


# ── TITLE ─────────────────────────────────────────────────────────────────────
_, title_y = iso(CX_ALL, CY_ALL, Z_OUT + 3.5)
ax.text(iso(CX_ALL, CY_ALL, 0)[0], title_y,
        "VLLA — Isometric Architecture View",
        ha="center", va="center", fontsize=28, fontweight="bold", color=DARK,
        zorder=10)
_, sub_y = iso(CX_ALL, CY_ALL, Z_OUT + 2.8)
ax.text(iso(CX_ALL, CY_ALL, 0)[0], sub_y,
        "Five streams  ▶  CLIP  ▶  Project  ▶  Token Sequence  ▶  LLaMA  ▶  Action",
        ha="center", va="center", fontsize=15, color="#555", zorder=10)

# ── STAGE LABELS (left margin) ────────────────────────────────────────────────
stages = [
    (Z_INPUT,  "① INPUT"),
    (Z_CLIP,   "② CLIP ENCODE"),
    (Z_PROJ,   "③ PROJECT"),
    (Z_MERGE,  "④ MERGE"),
    (Z_SEQ,    "⑤ TOKEN SEQ"),
    (Z_FUSE,   "⑥ LLAMA FUSION"),
    (Z_HEAD,   "⑦ ACTION HEAD"),
]
for z, lbl in stages:
    sx, sy_ = iso(SX[0] - 2.8, SY - 0.3, z)
    ax.text(sx, sy_, lbl, ha="right", va="center", fontsize=13,
            color="#666", fontweight="bold", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# ① INPUT — five raw input boxes
# ══════════════════════════════════════════════════════════════════════════════
for i, (x, col, nm) in enumerate(zip(SX, STREAM_COLS, STREAM_NAMES)):
    tx, ty = draw_box_3d(ax, x, SY, Z_INPUT, BOX_W, BOX_D, BOX_H, col)
    ax.text(tx, ty, nm, ha="center", va="center",
            fontsize=10.5, color="white", fontweight="bold", zorder=8)

# NEW badge on consequence column
bx, by = iso(SX[3] + 1.3, SY - BOX_D/2, Z_INPUT + BOX_H/2 + 0.4)
badge = FancyBboxPatch((bx - 0.5, by - 0.18), 1.0, 0.36,
                        boxstyle="round,pad=0.05", fc=NEW, ec="none", zorder=12)
ax.add_patch(badge)
ax.text(bx, by, "NEW", ha="center", va="center",
        fontsize=9, color="white", fontweight="bold", zorder=13)


# ══════════════════════════════════════════════════════════════════════════════
# Arrows: INPUT → CLIP
# ══════════════════════════════════════════════════════════════════════════════
for x, col in zip(SX, STREAM_COLS):
    draw_arrow_3d(ax, x, SY, Z_INPUT + BOX_H/2 + 0.05,
                      x, SY, Z_CLIP  - BOX_H/2 - 0.05, col=col, lw=2.5)


# ══════════════════════════════════════════════════════════════════════════════
# ② CLIP ENCODE — one wide slab for streams 1-4, separate slab for history
# ══════════════════════════════════════════════════════════════════════════════
# CLIP panel covers streams 0-3
cx_clip = (SX[0] + SX[3]) / 2   # = 4.5
clip_w  = SX[3] - SX[0] + BOX_W + 0.4  # = 11.6
tx, ty = draw_box_3d(ax, cx_clip, SY, Z_CLIP, clip_w, BOX_D + 0.2, BOX_H + 0.2,
                     CLIP, alpha=0.88)
ax.text(tx, ty,
        "Frozen CLIP ViT-B/32\n(image enc / text enc  ×3)\n→ 512-dim",
        ha="center", va="center", fontsize=11, color="white", zorder=8)

# History encoder (independent)
tx2, ty2 = draw_box_3d(ax, SX[4], SY, Z_CLIP, BOX_W + 0.4, BOX_D + 0.2,
                        BOX_H + 0.2, HIS, alpha=0.90)
ax.text(tx2, ty2, "History\nEncoder\n→ 256-dim", ha="center", va="center",
        fontsize=10, color="white", fontweight="bold", zorder=8)


# ══════════════════════════════════════════════════════════════════════════════
# Arrows: CLIP → PROJECT
# ══════════════════════════════════════════════════════════════════════════════
for x, col in zip(SX, STREAM_COLS):
    draw_arrow_3d(ax, x, SY, Z_CLIP + BOX_H/2 + 0.05 + 0.1,
                      x, SY, Z_PROJ - BOX_H/2 - 0.05, col=col, lw=2.5)


# ══════════════════════════════════════════════════════════════════════════════
# ③ PROJECT — five independent projection boxes (now 256-dim each)
# ══════════════════════════════════════════════════════════════════════════════
proj_labels = [
    "vis_proj\nLinear\n512→256",
    "lang_proj\nLinear\n512→256",
    "act_proj\n+RMSNorm\n+RewardGate",
    "con_proj\n+RMSNorm\n(own wts)",
    "Pass-thru\n(already\n256-dim)",
]
for i, (x, col, lbl) in enumerate(zip(SX, STREAM_COLS, proj_labels)):
    h = BOX_H + (0.4 if i in (2, 3) else 0)   # taller for new streams
    tx, ty = draw_box_3d(ax, x, SY, Z_PROJ, BOX_W, BOX_D, h, col)
    ax.text(tx, ty, lbl, ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold", zorder=8)


# ══════════════════════════════════════════════════════════════════════════════
# Arrows: PROJECT → MERGE (all converging to centre)
# ══════════════════════════════════════════════════════════════════════════════
for x, col in zip(SX, STREAM_COLS):
    draw_arrow_3d(ax, x, SY, Z_PROJ + BOX_H/2 + 0.05,
                      CX_ALL, CY_ALL, Z_MERGE - BOX_H/2 - 0.05, col=col, lw=2.5)


# ══════════════════════════════════════════════════════════════════════════════
# ④ MERGE — ViLT modality-type embeddings added
# ══════════════════════════════════════════════════════════════════════════════
tx, ty = draw_box_3d(ax, CX_ALL, CY_ALL, Z_MERGE,
                     FUSE_W, BOX_D + 0.3, BOX_H + 0.3, FUSE, alpha=0.90)
ax.text(tx, ty,
        "ViLT Modality-Type Embeddings\n(types 0,1,2,3,4 — one per stream)\nadded to each token",
        ha="center", va="center", fontsize=11, color="white", zorder=8)


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ TOKEN SEQUENCE — horizontal token bar
# ══════════════════════════════════════════════════════════════════════════════
draw_arrow_3d(ax, CX_ALL, CY_ALL, Z_MERGE + BOX_H/2 + 0.1 + 0.15,
                  CX_ALL, CY_ALL, Z_SEQ   - BOX_H/2 - 0.05, col=FUSE, lw=3)

# Each token as a narrow box
token_defs = [
    ("L_instr\nt=0",   INS, 1.4),
    ("L_act\nt=1",     ACT, 1.4),
    ("L_con\nt=4",     CON, 1.6),
    ("V₁V₂V₃\nt=2",   VIS, 2.2),
    ("H₁..H₄\nt=3",   HIS, 2.2),
    ("CLS",            FUSE, 1.2),
]
token_widths = [w for _, _, w in token_defs]
GAP = 0.25
total_tok_w = sum(token_widths) + GAP * (len(token_widths) - 1)
tok_start   = CX_ALL - total_tok_w / 2

cur = tok_start
for lbl, col, tw in token_defs:
    tcx = cur + tw / 2
    draw_box_3d(ax, tcx, CY_ALL, Z_SEQ, tw, BOX_D, BOX_H + 0.1, col)
    tsx, tsy = iso(tcx, CY_ALL, Z_SEQ + (BOX_H + 0.1)/2 + 0.4)
    ax.text(tsx, tsy, lbl, ha="center", va="bottom",
            fontsize=9.5, color=col, fontweight="bold", zorder=8)
    cur += tw + GAP

# Token count label
tsx_mid, _ = iso(CX_ALL, CY_ALL, Z_SEQ)
_, tsy_bot  = iso(CX_ALL, CY_ALL + BOX_D/2 + 0.1, Z_SEQ - BOX_H/2 - 0.55)
ax.text(tsx_mid, tsy_bot,
        "S = 1+1+1+3+4+1 = 11 tokens  →  (B, 11, 256)",
        ha="center", va="top", fontsize=12, color="#444",
        style="italic", zorder=8)


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ LLAMA FUSION TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════
draw_arrow_3d(ax, CX_ALL, CY_ALL, Z_SEQ  + BOX_H/2 + 0.05,
                  CX_ALL, CY_ALL, Z_FUSE - BOX_H/2 - 0.05, col=FUSE, lw=3.5)

tx, ty = draw_box_3d(ax, CX_ALL, CY_ALL, Z_FUSE,
                     FUSE_W, BOX_D + 0.5, BOX_H * 2.5, FUSE, alpha=0.92)
ax.text(tx, ty,
        "LLaMA Fusion Transformer  ×6\n\n"
        "RMSNorm  +  RoPE Attention  +  SwiGLU\n"
        "Causal mask  —  CLS attends to all 10 prior tokens\n\n"
        "8 heads  ·  d_model = 256",
        ha="center", va="center", fontsize=12, color="white", zorder=8)


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ ACTION HEAD
# ══════════════════════════════════════════════════════════════════════════════
draw_arrow_3d(ax, CX_ALL, CY_ALL, Z_FUSE + BOX_H * 2.5/2 + 0.05,
                  CX_ALL, CY_ALL, Z_HEAD - BOX_H/2 - 0.05, col=HEAD, lw=3.5)

tx, ty = draw_box_3d(ax, CX_ALL, CY_ALL, Z_HEAD,
                     FUSE_W - 1.5, BOX_D + 0.3, BOX_H + 0.5, HEAD, alpha=0.93)
ax.text(tx, ty,
        "Action Head  (3-layer MLP)\nCLS  →  RMSNorm  →  256  →  128  →  N logits",
        ha="center", va="center", fontsize=12, color="white", zorder=8)


# ── OUTPUT ────────────────────────────────────────────────────────────────────
draw_arrow_3d(ax, CX_ALL, CY_ALL, Z_HEAD + BOX_H/2 + 0.25 + 0.05,
                  CX_ALL, CY_ALL, Z_OUT  - 0.2, col=DARK, lw=4)

out_sx, out_sy = iso(CX_ALL, CY_ALL, Z_OUT)
out_box = FancyBboxPatch((out_sx - 3.8, out_sy - 0.5), 7.6, 1.0,
                          boxstyle="round,pad=0.15", fc=DARK, ec="none",
                          alpha=0.9, zorder=9)
ax.add_patch(out_box)
ax.text(out_sx, out_sy, "Action logits  â_t  ∈  ℝᴺ",
        ha="center", va="center", fontsize=15, color="white",
        fontweight="bold", zorder=10)


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK LOOP ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════
# Orange curved bracket on the right side showing S3a, S3b, S4 feed back

# Anchor points in isometric screen space
fb_top_sx, fb_top_sy = iso(SX[-1] + 1.8, CY_ALL, Z_PROJ + BOX_H/2 + 0.3)
fb_bot_sx, fb_bot_sy = iso(SX[2]  + 1.0, CY_ALL, Z_INPUT - BOX_H/2 - 0.2)

# Draw a vertical orange arrow on the right side going DOWN (feedback direction)
ax.annotate(
    "",
    xy=(fb_bot_sx + 0.4, fb_bot_sy),
    xytext=(fb_top_sx + 0.4, fb_top_sy),
    arrowprops=dict(
        arrowstyle="-|>", color=NEW, lw=3.0, mutation_scale=20,
        connectionstyle="arc3,rad=-0.25",
    ),
    zorder=12,
)

# Label
fb_mid_sx = (fb_top_sx + fb_bot_sx) / 2 + 2.5
fb_mid_sy = (fb_top_sy + fb_bot_sy) / 2
fb_box = FancyBboxPatch((fb_mid_sx - 2.0, fb_mid_sy - 0.75), 4.0, 1.5,
                          boxstyle="round,pad=0.1", fc=NEW, ec="none",
                          alpha=0.9, zorder=12)
ax.add_patch(fb_box)
ax.text(fb_mid_sx, fb_mid_sy,
        "FEEDBACK\nLOOP\n(t → t+1)",
        ha="center", va="center", fontsize=11,
        color="white", fontweight="bold", zorder=13)


# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
leg_items = [
    (VIS,  "Stream 1 — Vision (CLIP image)"),
    (INS,  "Stream 2 — Instruction (CLIP text)"),
    (ACT,  "Stream 3a — Action Language (reward-gated)"),
    (CON,  "Stream 3b — Consequence Language  [NEW]"),
    (HIS,  "Stream 4 — Action-Reward History"),
    (CLIP, "Frozen CLIP ViT-B/32 (shared backbone)"),
    (FUSE, "LLaMA Fusion Transformer (RMSNorm+RoPE+SwiGLU)"),
    (HEAD, "Action Head (3-layer MLP)"),
]

# Place legend to the right of the diagram, at a fixed screen position
# anchored near the midpoint of the diagram height
leg_anchor_sx, leg_anchor_sy = iso(SX[-1] + 4.5, CY_ALL - 1.0, Z_FUSE - 2)
ax.text(leg_anchor_sx, leg_anchor_sy, "LEGEND",
        ha="left", va="bottom", fontsize=15, fontweight="bold", color=DARK, zorder=12)
for i, (col, lbl) in enumerate(leg_items):
    ly = leg_anchor_sy - 1.05 * (i + 1)
    rect = FancyBboxPatch((leg_anchor_sx, ly - 0.30), 0.8, 0.60,
                           boxstyle="round,pad=0.05", fc=col, ec="none", zorder=12)
    ax.add_patch(rect)
    ax.text(leg_anchor_sx + 1.05, ly, lbl, ha="left", va="center",
            fontsize=11.5, color=DARK, zorder=12)


# ── Save ──────────────────────────────────────────────────────────────────────
fig.tight_layout()
fig.savefig("docs/vlla_architecture_3d.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print("Saved: docs/vlla_architecture_3d.png")
plt.close(fig)
