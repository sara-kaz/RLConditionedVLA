"""
VERA Architecture Diagram — v2 (correct: 6-layer fusion, λ_align=0.10)
Run: python3 docs/make_vera_diagram.py
Outputs: docs/VERA.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Canvas ────────────────────────────────────────────────────────────────────
W, H = 28, 16
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
BG = "#F4F6FB"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Palette ───────────────────────────────────────────────────────────────────
C_VIS   = "#1565C0"   # Stream 1 Vision
C_INS   = "#2E7D32"   # Stream 2 Instruction
C_ACT   = "#E65100"   # Stream 3a Action Narration
C_EMB   = "#AD1457"   # Stream 3b Consequence  [NEW]
C_HIS   = "#6A1B9A"   # Stream 4 History
C_CLIP  = "#37474F"   # CLIP backbone (frozen)
C_SUB   = "#4527A0"   # Causal sub-transformer
C_FUSE  = "#1A237E"   # Bidirectional fusion
C_DIS   = "#BF360C"   # Discrete head
C_REG   = "#1B5E20"   # Regression head
C_LOSS  = "#E65100"   # Alignment loss
C_NEW   = "#B71C1C"   # NEW badge
WHITE   = "white"
DARK    = "#1C1C1C"
MID     = "#555555"

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec=WHITE, lw=1.8, alpha=1.0, z=3):
    p = FancyBboxPatch(
        (x + 0.06, y + 0.06), w - 0.12, h - 0.12,
        boxstyle="round,pad=0.06",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z,
    )
    ax.add_patch(p)

def txt(x, y, s, fs=8.5, c=WHITE, ha="center", va="center",
        bold=False, italic=False, z=6):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, zorder=z,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal")

def arr(x1, y1, x2, y2, c=MID, lw=1.8, rad=0.0, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=c, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=5,
    )

def new_badge(x, y):
    ax.text(x, y, " NEW ", ha="center", va="center", fontsize=6.5,
            color=WHITE, fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.18", fc=C_NEW, ec="none"))

def frozen_tag(cx, y, fs=7):
    ax.text(cx, y, "❄  frozen", ha="center", va="center",
            fontsize=fs, color="#90A4AE", zorder=6)

# ── Stream geometry ───────────────────────────────────────────────────────────
# Each row: (y_centre, row_height)
ROWS = {
    "vis": (13.4, 1.6),
    "ins": (11.1, 1.1),
    "act": (9.1,  1.1),
    "emb": (7.1,  1.1),
    "his": (4.6,  2.2),
}

def yb(key):          # bottom y of row box
    yc, h = ROWS[key]
    return yc - h / 2

def yh(key):          # height of row box
    return ROWS[key][1]

def yc(key):          # centre y
    return ROWS[key][0]

# Column x positions
X_LBL  = 0.10
W_LBL  = 1.45
X_IN   = 1.65
W_IN   = 2.40
X_ENC  = 4.20
W_ENC  = 2.80
X_TOK  = 7.15
W_TOK  = 1.50
X_FUSE = 9.00
W_FUSE = 4.30
X_OUT  = 13.55
W_OUT  = 5.50
X_LEG  = 19.40

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
txt(W / 2, 15.55,
    "VERA  —  Vision · Experience · Reasoning · Action",
    fs=16, c=DARK, bold=True)
txt(W / 2, 15.05,
    "5-stream closed-loop robot policy  ·  "
    "6-layer bidirectional LLaMA Fusion Transformer  ·  "
    "K = 4 action chunking  ·  λ_align = 0.10",
    fs=9, c=MID)

# ══════════════════════════════════════════════════════════════════════════════
# STREAM LABELS  (left column, lightly tinted)
# ══════════════════════════════════════════════════════════════════════════════
labels = [
    ("vis", C_VIS,  "Stream 1\nVision"),
    ("ins", C_INS,  "Stream 2\nInstruction"),
    ("act", C_ACT,  "Stream 3a\nAction\nNarration"),
    ("emb", C_EMB,  "Stream 3b\nEmbodied\nConsequence"),
    ("his", C_HIS,  "Stream 4\nHistory"),
]
for key, col, lbl in labels:
    box(X_LBL, yb(key), W_LBL, yh(key), fc=col, alpha=0.18, ec=col, lw=2, z=2)
    txt(X_LBL + W_LBL / 2, yc(key), lbl, fs=7.5, c=col, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# STREAM 1 — VISION
# ══════════════════════════════════════════════════════════════════════════════
k = "vis"
box(X_IN, yb(k), W_IN, yh(k), C_VIS)
txt(X_IN + W_IN / 2, yc(k) + 0.25, "3 RGB Frames", fs=9, bold=True)
txt(X_IN + W_IN / 2, yc(k) - 0.25, "224 × 224 × 3", fs=8, c="#BBDEFB")

arr(X_IN + W_IN, yc(k), X_ENC, yc(k), c=C_VIS)

box(X_ENC, yb(k), W_ENC, yh(k), C_CLIP)
txt(X_ENC + W_ENC / 2, yc(k) + 0.28, "CLIP ViT-B/32", fs=9, bold=True)
frozen_tag(X_ENC + W_ENC / 2, yc(k))
txt(X_ENC + W_ENC / 2, yc(k) - 0.30, "197 patch tokens  →  d = 512", fs=7.5, c="#90A4AE")

arr(X_ENC + W_ENC, yc(k), X_TOK, yc(k), c=C_VIS)

box(X_TOK, yb(k), W_TOK, yh(k), C_VIS)
txt(X_TOK + W_TOK / 2, yc(k) + 0.22, "V", fs=13, bold=True)
txt(X_TOK + W_TOK / 2, yc(k) - 0.25, "197 tokens", fs=7.5)

arr(X_TOK + W_TOK, yc(k), X_FUSE, yc(k), c=C_VIS, lw=2.2)

# ══════════════════════════════════════════════════════════════════════════════
# STREAM 2 — INSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
k = "ins"
box(X_IN, yb(k), W_IN, yh(k), C_INS)
txt(X_IN + W_IN / 2, yc(k) + 0.18, "Task Instruction", fs=9, bold=True)
txt(X_IN + W_IN / 2, yc(k) - 0.18, '"push block left"', fs=8, c="#C8E6C9", italic=True)

arr(X_IN + W_IN, yc(k), X_ENC, yc(k), c=C_INS)

box(X_ENC, yb(k), W_ENC, yh(k), C_CLIP)
txt(X_ENC + W_ENC / 2, yc(k) + 0.18, "CLIP Text Encoder", fs=9, bold=True)
frozen_tag(X_ENC + W_ENC / 2, yc(k) - 0.18)

arr(X_ENC + W_ENC, yc(k), X_TOK, yc(k), c=C_INS)

box(X_TOK, yb(k), W_TOK, yh(k), C_INS)
txt(X_TOK + W_TOK / 2, yc(k) + 0.16, "t_ins", fs=9, bold=True)
txt(X_TOK + W_TOK / 2, yc(k) - 0.16, "1 token", fs=7.5)

arr(X_TOK + W_TOK, yc(k), X_FUSE, yc(k), c=C_INS, lw=2.2)

# ══════════════════════════════════════════════════════════════════════════════
# STREAM 3a — ACTION NARRATION
# ══════════════════════════════════════════════════════════════════════════════
k = "act"
box(X_IN, yb(k), W_IN, yh(k), C_ACT)
txt(X_IN + W_IN / 2, yc(k) + 0.18, "Prior Action  a_{t−1}", fs=9, bold=True)
txt(X_IN + W_IN / 2, yc(k) - 0.18, "discrete bin index", fs=7.5, c="#FFE0B2")

arr(X_IN + W_IN, yc(k), X_ENC, yc(k), c=C_ACT)

# Split encoder: verbalize | CLIP
v_w = W_ENC * 0.42
c_w = W_ENC * 0.58
box(X_ENC, yb(k), v_w, yh(k), fc=C_ACT, alpha=0.75)
txt(X_ENC + v_w / 2, yc(k) + 0.16, "verbalize()", fs=8, bold=True)
txt(X_ENC + v_w / 2, yc(k) - 0.16, "→ text", fs=7.5, c="#FFE0B2")

arr(X_ENC + v_w, yc(k), X_ENC + v_w + 0.05, yc(k), c=C_ACT, lw=1.2)

box(X_ENC + v_w, yb(k), c_w, yh(k), C_CLIP)
txt(X_ENC + v_w + c_w / 2, yc(k) + 0.16, "CLIP Text", fs=8, bold=True)
frozen_tag(X_ENC + v_w + c_w / 2, yc(k) - 0.16)

arr(X_ENC + W_ENC, yc(k), X_TOK, yc(k), c=C_ACT)

box(X_TOK, yb(k), W_TOK, yh(k), C_ACT)
txt(X_TOK + W_TOK / 2, yc(k) + 0.16, "t_act", fs=9, bold=True)
txt(X_TOK + W_TOK / 2, yc(k) - 0.16, "E_act  1 tok", fs=7)

arr(X_TOK + W_TOK, yc(k), X_FUSE, yc(k), c=C_ACT, lw=2.2)

# ══════════════════════════════════════════════════════════════════════════════
# STREAM 3b — EMBODIED CONSEQUENCE  [NEW]
# ══════════════════════════════════════════════════════════════════════════════
k = "emb"
box(X_IN, yb(k), W_IN, yh(k), C_EMB)
txt(X_IN + W_IN / 2, yc(k) + 0.18, "r_{t−1}   +   Δd_{t−1}", fs=9, bold=True)
txt(X_IN + W_IN / 2, yc(k) - 0.18, "reward  ·  state delta", fs=7.5, c="#F8BBD0")
new_badge(X_IN + W_IN - 0.35, yb(k) + yh(k) - 0.22)

arr(X_IN + W_IN, yc(k), X_ENC, yc(k), c=C_EMB)

vc_w = W_ENC * 0.50
cc_w = W_ENC * 0.50
box(X_ENC, yb(k), vc_w, yh(k), fc=C_EMB, alpha=0.75)
txt(X_ENC + vc_w / 2, yc(k) + 0.16, "verbalize_consequence()", fs=7, bold=True)
txt(X_ENC + vc_w / 2, yc(k) - 0.16, "→ outcome text", fs=7, c="#F8BBD0")

arr(X_ENC + vc_w, yc(k), X_ENC + vc_w + 0.05, yc(k), c=C_EMB, lw=1.2)

box(X_ENC + vc_w, yb(k), cc_w, yh(k), C_CLIP)
txt(X_ENC + vc_w + cc_w / 2, yc(k) + 0.16, "CLIP Text", fs=8, bold=True)
frozen_tag(X_ENC + vc_w + cc_w / 2, yc(k) - 0.16)

arr(X_ENC + W_ENC, yc(k), X_TOK, yc(k), c=C_EMB)

box(X_TOK, yb(k), W_TOK, yh(k), C_EMB)
txt(X_TOK + W_TOK / 2, yc(k) + 0.16, "t_emb", fs=9, bold=True)
txt(X_TOK + W_TOK / 2, yc(k) - 0.16, "E_emb  1 tok", fs=7)
new_badge(X_TOK + W_TOK - 0.28, yb(k) + yh(k) - 0.20)

arr(X_TOK + W_TOK, yc(k), X_FUSE, yc(k), c=C_EMB, lw=2.2)

# ══════════════════════════════════════════════════════════════════════════════
# STREAM 4 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
k = "his"
box(X_IN, yb(k), W_IN, yh(k), C_HIS)
txt(X_IN + W_IN / 2, yc(k) + 0.50, "History Window  H = 4", fs=9, bold=True)
txt(X_IN + W_IN / 2, yc(k) + 0.08, "(action, reward) pairs", fs=8, c="#E1BEE7")
txt(X_IN + W_IN / 2, yc(k) - 0.30, "reward gate  σ(MLP(r))", fs=7.5, c="#CE93D8")
txt(X_IN + W_IN / 2, yc(k) - 0.60, "→ gated context vectors", fs=7, c="#CE93D8")

arr(X_IN + W_IN, yc(k), X_ENC, yc(k), c=C_HIS)

h1w = W_ENC * 0.45
h2w = W_ENC * 0.45
hgap = W_ENC * 0.10

box(X_ENC, yb(k), h1w, yh(k), fc=C_HIS, alpha=0.75)
txt(X_ENC + h1w / 2, yc(k) + 0.38, "History Encoder", fs=8, bold=True)
txt(X_ENC + h1w / 2, yc(k) + 0.04, "gate(a, r) → d=256", fs=7.5, c="#E1BEE7")
txt(X_ENC + h1w / 2, yc(k) - 0.30, "reward-conditioned", fs=7, c="#CE93D8")
txt(X_ENC + h1w / 2, yc(k) - 0.58, "gating per step", fs=7, c="#CE93D8")

arr(X_ENC + h1w, yc(k), X_ENC + h1w + hgap, yc(k), c=C_HIS, lw=1.4)

box(X_ENC + h1w + hgap, yb(k), h2w, yh(k), C_SUB)
txt(X_ENC + h1w + hgap + h2w / 2, yc(k) + 0.30, "2-Layer Causal", fs=8, bold=True)
txt(X_ENC + h1w + hgap + h2w / 2, yc(k) + 0.00, "LLaMA Sub-TF", fs=8, bold=True)
txt(X_ENC + h1w + hgap + h2w / 2, yc(k) - 0.32, "temporal encoding", fs=7, c="#B39DDB")
txt(X_ENC + h1w + hgap + h2w / 2, yc(k) - 0.58, "(causal mask ✓)", fs=7, c="#9575CD")

arr(X_ENC + W_ENC, yc(k), X_TOK, yc(k), c=C_HIS)

box(X_TOK, yb(k), W_TOK, yh(k), C_HIS)
txt(X_TOK + W_TOK / 2, yc(k) + 0.38, "h₁ … h₄", fs=9, bold=True)
txt(X_TOK + W_TOK / 2, yc(k) + 0.00, "4 tokens", fs=8)
txt(X_TOK + W_TOK / 2, yc(k) - 0.35, "d = 256", fs=7.5, c="#D1C4E9")

arr(X_TOK + W_TOK, yc(k), X_FUSE, yc(k), c=C_HIS, lw=2.2)

# ── Projection annotation ─────────────────────────────────────────────────────
proj_x = (X_TOK + W_TOK + X_FUSE) / 2
ax.text(proj_x, 2.35,
        "Independent RMSNorm + Linear → d = 256\n"
        "+ ViLT modality-type embedding per stream",
        ha="center", va="bottom", fontsize=7.5, color="#004D40", zorder=6,
        bbox=dict(boxstyle="round,pad=0.28", fc="#E0F2F1", ec="#00695C", lw=1.2))

# ══════════════════════════════════════════════════════════════════════════════
# LLAMA FUSION TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════
F_YB = 2.70
F_H  = 12.10

# Outer body
box(X_FUSE, F_YB, W_FUSE, F_H, fc=C_FUSE, ec="#7986CB", lw=2.5, z=3)

# Header bar
box(X_FUSE, F_YB + F_H - 1.35, W_FUSE, 1.35, fc="#283593", ec="none", z=4)
txt(X_FUSE + W_FUSE / 2, F_YB + F_H - 0.62,
    "LLaMA Fusion Transformer", fs=11, bold=True)
txt(X_FUSE + W_FUSE / 2, F_YB + F_H - 1.15,
    "◈  BIDIRECTIONAL  —  No Causal Mask  ◈",
    fs=8.5, c="#FFD54F", bold=True)

# Specs
sx = X_FUSE + W_FUSE / 2
txt(sx, F_YB + F_H - 2.00, "6 Layers  ·  8 Heads",        fs=10, c="#C5CAE9", bold=True)
txt(sx, F_YB + F_H - 2.55, "d_model = 256  ·  d_ff = 1024", fs=9,  c="#9FA8DA")
txt(sx, F_YB + F_H - 3.00, "RMSNorm  ·  RoPE  ·  SwiGLU",  fs=9,  c="#9FA8DA")

# Layer stack
layer_x  = X_FUSE + 0.45
layer_w  = W_FUSE - 0.90
layer_h  = 0.55
layer_gap = 0.18
stack_y  = F_YB + 1.20   # start of first layer (bottom)

for i in range(6):
    ly  = stack_y + i * (layer_h + layer_gap)
    fc_l = "#1565C0" if i % 2 == 0 else "#1976D2"
    box(layer_x, ly, layer_w, layer_h, fc=fc_l, ec="#90CAF9", lw=0.9, z=5)
    txt(layer_x + layer_w / 2, ly + layer_h / 2,
        f"Layer {i + 1}  ·  MHA (RoPE)  +  SwiGLU (RMSNorm)",
        fs=7.5, c="#E3F2FD")

# Token sequence label (between layers and specs)
seq_y = F_YB + F_H - 4.00
txt(sx, seq_y + 0.22,
    "Input token sequence:",
    fs=8, c="#7986CB")
txt(sx, seq_y - 0.20,
    "[ t_ins | t_act | t_emb | V_patches | h₁…h₄ | CLS ]",
    fs=8, c="#C5CAE9", bold=True)

# CLS output bar at bottom of fusion
box(X_FUSE + 0.30, F_YB + 0.10, W_FUSE - 0.60, 0.90,
    fc="#0D47A1", ec="#FFD54F", lw=2.2, z=5)
txt(sx, F_YB + 0.55, "CLS  →  Policy Head Input", fs=9, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT HEADS
# ══════════════════════════════════════════════════════════════════════════════
# Discrete Action Head
DIS_YB = 12.30
DIS_H  = 2.35
box(X_OUT, DIS_YB, W_OUT, DIS_H, C_DIS, ec=WHITE, lw=2, z=4)
txt(X_OUT + W_OUT / 2, DIS_YB + DIS_H - 0.50, "Discrete Action Head", fs=10, bold=True)
txt(X_OUT + W_OUT / 2, DIS_YB + DIS_H / 2 - 0.02,
    "Expand-Compress Bottleneck → FC", fs=8.5, c="#FFCCBC")
txt(X_OUT + W_OUT / 2, DIS_YB + 0.55, "8-bin logits   ×   K = 4 chunks", fs=8.5, bold=True)
txt(X_OUT + W_OUT / 2, DIS_YB + 0.20, "(π₀ / GR-1 style action chunking)", fs=7.5, c="#FFCCBC")

arr(X_FUSE + W_FUSE, F_YB + F_H - 0.85, X_OUT, DIS_YB + DIS_H / 2, c=C_DIS, lw=2.2)

# Continuous Regression Head
REG_YB = 8.80
REG_H  = 2.80
box(X_OUT, REG_YB, W_OUT, REG_H, C_REG, ec=WHITE, lw=2, z=4)
txt(X_OUT + W_OUT / 2, REG_YB + REG_H - 0.50, "Continuous Regression Head", fs=10, bold=True)
txt(X_OUT + W_OUT / 2, REG_YB + REG_H / 2 + 0.08,
    "RMSNorm  +  Tanh activation", fs=8.5, c="#C8E6C9")
txt(X_OUT + W_OUT / 2, REG_YB + REG_H / 2 - 0.38,
    "Continuous action vector  [Δx,  Δy]", fs=8.5, bold=True)
txt(X_OUT + W_OUT / 2, REG_YB + 0.55,
    "action_dim = 2  (Language-Table pushes)", fs=8, c="#C8E6C9")
txt(X_OUT + W_OUT / 2, REG_YB + 0.20,
    "λ_reg = 0.5  ·  MSE vs. expert trajectory", fs=7.5, c="#A5D6A7")

arr(X_FUSE + W_FUSE, F_YB + F_H / 2, X_OUT, REG_YB + REG_H / 2, c=C_REG, lw=2.2)

# InfoNCE Alignment Loss
LOS_YB = 4.30
LOS_H  = 3.70
box(X_OUT, LOS_YB, W_OUT, LOS_H, fc="#F57F17", ec=WHITE, lw=2, z=4)
txt(X_OUT + W_OUT / 2, LOS_YB + LOS_H - 0.52,
    "InfoNCE Alignment Loss", fs=10, bold=True)
txt(X_OUT + W_OUT / 2, LOS_YB + LOS_H - 1.05,
    "(training only)", fs=8, c="#FFF8E1", italic=True)
txt(X_OUT + W_OUT / 2, LOS_YB + LOS_H / 2 + 0.28,
    "InfoNCE ( t_emb,  t_act  ‖  t_ins )", fs=9, c="#FFF8E1", bold=True)
txt(X_OUT + W_OUT / 2, LOS_YB + LOS_H / 2 - 0.18,
    "reward-weighted  ·  exp(5 · r̃)", fs=8.5, c="#FFF8E1")
txt(X_OUT + W_OUT / 2, LOS_YB + LOS_H / 2 - 0.60,
    "λ_align = 0.10", fs=9, c="#FFF3E0", bold=True)
txt(X_OUT + W_OUT / 2, LOS_YB + 0.68,
    "aligns consequence + action tokens", fs=8, c="#FFF8E1")
txt(X_OUT + W_OUT / 2, LOS_YB + 0.30,
    "toward the task instruction embedding", fs=8, c="#FFF8E1")

arr(X_FUSE + W_FUSE, F_YB + 1.00, X_OUT, LOS_YB + LOS_H / 2, c="#F57F17", lw=2.2)

# ══════════════════════════════════════════════════════════════════════════════
# CLOSED-LOOP FEEDBACK ARROW
# ══════════════════════════════════════════════════════════════════════════════
# From bottom of output side back to history/action inputs on left
ax.annotate(
    "", xy=(X_IN + W_IN / 2, yb("his") - 0.18),
    xytext=(X_OUT + W_OUT / 2, LOS_YB - 0.12),
    arrowprops=dict(
        arrowstyle="->", color="#6D4C41", lw=2.0,
        connectionstyle="arc3,rad=0.28",
    ),
    zorder=4,
)
ax.text(
    8.2, 1.75,
    "⟲  Closed-loop: predicted action + observed consequence "
    "re-injected as stream inputs at t+1",
    ha="center", va="center", fontsize=8, color="#4E342E", zorder=6,
    bbox=dict(boxstyle="round,pad=0.28", fc="#EFEBE9", ec="#795548", lw=1.2),
)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
leg_items = [
    (C_VIS,  "Stream 1 — Vision"),
    (C_INS,  "Stream 2 — Instruction"),
    (C_ACT,  "Stream 3a — Action Narration"),
    (C_EMB,  "Stream 3b — Embodied Consequence  [NEW]"),
    (C_HIS,  "Stream 4 — History"),
    (C_CLIP, "CLIP Encoder  (❄ frozen)"),
    (C_SUB,  "2-Layer Causal Sub-Transformer"),
    (C_FUSE, "6-Layer Bidirectional Fusion TF"),
    (C_DIS,  "Discrete Head  (8-bin × K=4 chunks)"),
    (C_REG,  "Regression Head  (continuous Δx, Δy)"),
    ("#F57F17", "InfoNCE Alignment Loss  (λ=0.10)"),
]

LY_TOP = 14.60
LY_STEP = 0.90
LX = X_LEG

box(LX - 0.20, LY_TOP - len(leg_items) * LY_STEP - 0.35,
    8.30, len(leg_items) * LY_STEP + 0.90,
    fc=WHITE, ec="#BDBDBD", lw=1.5, alpha=0.96, z=6)
txt(LX + 3.9, LY_TOP - 0.20, "Legend", fs=10, c=DARK, bold=True, z=7)

for i, (col, lbl) in enumerate(leg_items):
    ly = LY_TOP - 0.65 - i * LY_STEP
    box(LX, ly - 0.20, 0.55, 0.45, fc=col, ec="none", z=7)
    ax.text(LX + 0.70, ly + 0.02, lbl, ha="left", va="center",
            fontsize=8, color=DARK, zorder=7)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
import os
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "VERA.png")

plt.tight_layout(pad=0.4)
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out_path}")
