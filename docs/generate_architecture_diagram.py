"""
VLLA Architecture Diagram — Two-page clean version.
Page 1: Main pipeline (top to bottom, well spaced, one box per step).
Page 2: Legend, token key, numbers.
Saved as two PNGs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

VIS="#1A6FA8"; INS="#6C3483"; ACT="#1A7A40"; CON="#B9610E"
HIS="#8B1A1A"; CLIP="#4A4A4A"; FUSE="#0B6B58"; HEAD="#7D4608"
DARK="#1C2833"; BG="#FFFFFF"; LGRAY="#F4F6F7"; RED="#C0392B"
COLS=[VIS,INS,ACT,CON,HIS]

# ─── shared helpers ──────────────────────────────────────────────────────────
def make_fig(w, h):
    f, a = plt.subplots(figsize=(w, h))
    a.set_xlim(0, w); a.set_ylim(0, h); a.axis("off")
    f.patch.set_facecolor(BG); a.set_facecolor(BG)
    return f, a

def B(ax, cx, cy, w, h, fc, txt, fs=13, tc="white",
      bold=False, ec=DARK, lw=1.8, ls="-", alpha=1.0):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.2,rounding_size=0.3",
                       fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center", linespacing=1.5)

def Arr(ax, x0, y0, x1, y1, col=DARK, lw=2.8):
    ax.annotate("", xy=(x1, y1+0.06), xytext=(x0, y0-0.06),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=22), zorder=5)

def HAr(ax, x0, y, x1, col=DARK, lw=2.2):
    ax.annotate("", xy=(x1-0.06, y), xytext=(x0+0.06, y),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=20), zorder=5)

def Div(ax, y, W):
    ax.plot([1, W-1], [y, y], color="#CCCCCC", lw=1.8, zorder=1)

def StepBand(ax, y, h, label, num, W):
    """Coloured step band spanning full width."""
    p = FancyBboxPatch((0.5, y-h/2), W-1, h,
                       boxstyle="round,pad=0.1",
                       fc=LGRAY, ec="#CCCCCC", lw=1.0, alpha=0.7, zorder=1)
    ax.add_patch(p)
    ax.text(1.5, y, f"  {num}  {label}",
            ha="left", va="center", fontsize=13, color="#444",
            fontweight="bold", zorder=2)

def badge(ax, cx, cy):
    p = FancyBboxPatch((cx-0.6, cy-0.28), 1.2, 0.56,
                       boxstyle="round,pad=0.06", fc=RED, ec="none", zorder=8)
    ax.add_patch(p)
    ax.text(cx, cy, "NEW", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", zorder=9)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════
W1, H1 = 42, 58
fig1, ax1 = make_fig(W1, H1)

# Column centres for 5 streams (very wide apart)
XC = [5.5, 12.5, 19.5, 26.5, 34.5]
BW = 6.0   # box width per stream

# ── TITLE ────────────────────────────────────────────────────────────────────
ax1.text(W1/2, 57.2,
         "VLLA — Vision-Language-Language-Action Model  (Flat Pipeline View)",
         ha="center", va="center", fontsize=26, fontweight="bold", color=DARK)
ax1.text(W1/2, 56.5,
         "Five input streams  ▶  Frozen CLIP  ▶  LLaMA Fusion Transformer  ▶  Action logits"
         "       [See vlla_architecture_3d.png for isometric view]",
         ha="center", va="center", fontsize=14, color="#555")
Div(ax1, 56.0, W1)

# ── STEP 1 — INPUTS ──────────────────────────────────────────────────────────
StepBand(ax1, 55.3, 0.7, "INPUT STREAMS", "①", W1)

names  = ["VISION",  "INSTRUCTION",  "ACTION",  "CONSEQUENCE",  "HISTORY"]
descs  = [
    "frames\n(B, 3, 3, 224, 224)",
    "instruction tokens\n(B, 77)",
    "prev_action_idx  (B,)\nprev_reward  (B,)",
    "prev_reward  (B,)\nstate_delta  (B,)",
    "action_hist  (B, 4)\nreward_hist  (B, 4)",
]
Y_hdr = 54.15
Y_raw = 52.85
for x, col, nm, ds in zip(XC, COLS, names, descs):
    B(ax1, x, Y_hdr, BW, 0.85, col, nm, fs=15, bold=True)
    B(ax1, x, Y_raw, BW, 1.05, col, ds, fs=12.5, alpha=0.72)

badge(ax1, XC[3]+3.3, Y_hdr+0.50)
Div(ax1, 52.15, W1)

# ── STEP 2 — ENCODE ──────────────────────────────────────────────────────────
StepBand(ax1, 51.55, 0.65, "ENCODE  ─  Frozen CLIP ViT-B/32 (shared, never updated)  +  History Encoder", "②", W1)

# dashed CLIP panel behind streams 1-4
p = FancyBboxPatch((2.5, 49.35), 26.5, 1.90,
                   boxstyle="round,pad=0.1",
                   fc="#F0F3F4", ec=CLIP, lw=2.5, ls="--", alpha=0.9, zorder=2)
ax1.add_patch(p)

enc_txt = ["CLIP Image Encoder\n→ 512-dim",
           "CLIP Text Encoder\n→ 512-dim",
           "CLIP Text Encoder\n(same weights)\n→ 512-dim",
           "CLIP Text Encoder\n(same weights)\n→ 512-dim"]
Y_enc = 50.35
for x, txt in zip(XC[:4], enc_txt):
    B(ax1, x, Y_enc, BW-0.4, 1.55, CLIP, txt, fs=12.5)

B(ax1, XC[4], Y_enc, BW+0.4, 1.55, HIS,
  "History Encoder\nEmbedding + SinPE\n+ 2-layer Transformer\n→ 256-dim", fs=12)

for x, col in zip(XC, COLS):
    Arr(ax1, x, Y_raw-0.53, x, Y_enc+0.78, col)

Div(ax1, 49.15, W1)

# ── STEP 3 — PROJECT ─────────────────────────────────────────────────────────
StepBand(ax1, 48.55, 0.65, "PROJECT  →  256-dim  (each stream has independent projection weights)", "③", W1)

proj_txt = [
    "vis_proj\nLinear(512→256)\n+ RMSNorm\n\n→ 3 tokens\n(B, 3, 256)",
    "lang_proj\nLinear(512→256)\n+ RMSNorm\n\n→ 1 token\n(B, 1, 256)",
    "ActionLangEncoder\nLinear(512→256)\n+ RMSNorm\n+ RewardGate σ\n\n→ 1 token (B,1,256)",
    "ConsequenceEncoder\nLinear(512→256)\n+ RMSNorm\n(own weights)\n\n→ 1 token (B,1,256)",
    "Already 256-dim\n(no projection)\n\n→ 4 tokens\n(B, 4, 256)",
]
Y_proj = 46.75
for x, col, txt in zip(XC, COLS, proj_txt):
    B(ax1, x, Y_proj, BW, 2.80, col, txt, fs=12.5)
    Arr(ax1, x, Y_enc-0.78, x, Y_proj+1.40, col)

badge(ax1, XC[3]+3.3, Y_proj+1.50)
Div(ax1, 45.15, W1)

# ── STEP 4 — TOKEN SEQUENCE ──────────────────────────────────────────────────
StepBand(ax1, 44.55, 0.65, "TOKEN SEQUENCE  +  ViLT Modality-Type Embeddings", "④", W1)

ax1.text(W1/2, 43.85,
         "Concatenate all stream tokens into one sequence.\n"
         "Add a learnable type embedding to each token so the"
         " transformer knows which stream it came from.",
         ha="center", va="center", fontsize=13, color="#444", style="italic",
         linespacing=1.5)

for x, col in zip(XC, COLS):
    Arr(ax1, x, Y_proj-1.40, x, 42.80, col, lw=2.0)

# Token boxes — one per type, well spaced
toks = [
    ("L_instr\ntype = 0",         INS,  5.5),
    ("L_action\ntype = 1",        ACT,  5.5),
    ("L_consequence\ntype = 4",   CON,  6.5),
    ("V₁   V₂   V₃\ntype = 2",   VIS,  6.0),
    ("H₁  H₂  H₃  H₄\ntype = 3", HIS,  6.5),
    ("[ CLS ]\ntype = 0",         FUSE, 4.5),
]
tw = [w for _, _, w in toks]
GAP = 0.55
tot = sum(tw) + GAP*(len(tw)-1)
cur = (W1 - tot)/2
cxs = []
for w in tw:
    cxs.append(cur + w/2);  cur += w + GAP

Y_seq = 41.70
for (lbl, col, w), cx in zip(toks, cxs):
    B(ax1, cx, Y_seq, w, 1.75, col, lbl, fs=13)

badge(ax1, cxs[2]+3.45, Y_seq+1.00)

bx0 = cxs[0]-tw[0]/2;  bx1 = cxs[-1]+tw[-1]/2
ax1.annotate("", xy=(bx1, Y_seq-1.15), xytext=(bx0, Y_seq-1.15),
             arrowprops=dict(arrowstyle="|-|,widthA=0.7,widthB=0.7",
                             color="#999", lw=1.8))
ax1.text(W1/2, Y_seq-1.55,
         "S = 1 + 1 + 1 + 3 + 4 + 1 = 11 tokens   →   shape  (B, 11, 256)",
         ha="center", va="center", fontsize=14, color="#444", style="italic")

Div(ax1, 39.75, W1)

# ── STEP 5 — FUSION ──────────────────────────────────────────────────────────
StepBand(ax1, 39.15, 0.65, "LLAMA FUSION TRANSFORMER", "⑤", W1)

ax1.text(W1/2, 38.45,
         "Six identical LLaMA decoder blocks  ·  8 attention heads  ·  d_model = 256",
         ha="center", va="center", fontsize=13.5, color="#444", style="italic")

Arr(ax1, W1/2, Y_seq-1.80, W1/2, 37.50, FUSE, lw=3.5)

B(ax1, W1/2, 36.10, 33.0, 2.35, FUSE,
  "LLaMA Fusion Transformer  ×6\n\n"
  "[ RMSNorm → Multi-Head Attention (with RoPE) → Residual ]\n"
  "+  [ RMSNorm → SwiGLU FFN → Residual ]\n\n"
  "Causal attention mask  —  each token only attends to earlier positions",
  fs=14.5, bold=False)

Div(ax1, 34.70, W1)

# ── STEP 6 — ACTION HEAD ─────────────────────────────────────────────────────
StepBand(ax1, 34.10, 0.65, "ACTION HEAD", "⑥", W1)

ax1.text(W1/2, 33.40,
         "Extract the CLS token (last position) and classify with a 3-layer MLP",
         ha="center", va="center", fontsize=13.5, color="#444", style="italic")

Arr(ax1, W1/2, 34.95, W1/2, 32.60, FUSE, lw=3.5)

B(ax1, W1/2, 31.35, 33.0, 2.20, HEAD,
  "Action Head  (3-layer MLP)\n\n"
  "CLS  →  RMSNorm  →  Linear(256→256) + SiLU"
  "  →  Linear(256→128) + SiLU  →  Linear(128→A)\n\n"
  "Output:  Logits  (B, A)",
  fs=14.5)

Div(ax1, 30.10, W1)

# ── STEP 7 — AUXILIARY ALIGNMENT LOSS ────────────────────────────────────────
StepBand(ax1, 29.50, 0.65, "AUXILIARY ALIGNMENT LOSS  (training only)", "⑦", W1)

ax1.text(W1/2, 28.80,
         "Reward-weighted InfoNCE contrastive loss in CLIP's 512-dim space.\n"
         "Teaches: successful actions and their consequences"
         " should be semantically close to the task instruction.",
         ha="center", va="center", fontsize=13, color="#444",
         style="italic", linespacing=1.5)

# Three input boxes — left column
B(ax1,  6.5, 27.80, 7.5, 1.00, INS,  "instr_emb  (B, 512)",       fs=13.5)
B(ax1,  6.5, 26.65, 7.5, 1.00, ACT,  "action_lang_emb  (B, 512)", fs=13.5)
B(ax1,  6.5, 25.50, 7.5, 1.00, CON,  "consequence_emb  (B, 512)", fs=13.5)
badge(ax1, 10.35, 25.95)

# InfoNCE box — centre
B(ax1, 20.5, 26.65, 9.0, 3.60, "#6C3483",
  "Reward-weighted InfoNCE\n\n"
  "InfoNCE(instr, action, r)\n"
  "+\n"
  "InfoNCE(instr, consequence, r)\n"
  "────────────────\n"
  "divided by 2",
  fs=13.5)

# Total loss — right
B(ax1, 34.0, 26.65, 9.0, 3.40, DARK,
  "TOTAL LOSS\n\n"
  "CrossEntropy\n(logits, target)\n\n"
  "+ 0.1 × AlignLoss",
  fs=14.5, bold=True)

# Horizontal arrows
HAr(ax1, 10.25, 27.80, 16.0, INS)
HAr(ax1, 10.25, 26.65, 16.0, ACT)
HAr(ax1, 10.25, 25.50, 16.0, CON)
HAr(ax1, 25.0,  26.65, 29.5, "#6C3483")

Div(ax1, 24.3, W1)

# ── FOOTER ───────────────────────────────────────────────────────────────────
ax1.text(W1/2, 23.9,
         "See Page 2 for: Colour Legend · Token Sequence Key · Key Numbers",
         ha="center", va="center", fontsize=13, color="#888", style="italic")

# ════════════════════════════════════════════════════════════════════════════
#  FEEDBACK LOOP HIGHLIGHT — drawn LAST so it sits on top
# ════════════════════════════════════════════════════════════════════════════
#
#  Streams 3a (ACTION), 3b (CONSEQUENCE), and 4 (HISTORY) are the
#  feedback loop — they carry information from the PREVIOUS timestep.
#  Highlight them with a dashed orange border spanning Steps ①②③.

LOOP_COL = "#E67E22"   # orange

# Coordinates of the three feedback columns
FL_LEFT  = XC[2] - BW/2 - 0.5          # left edge of ACTION column
FL_RIGHT = XC[4] + BW/2 + 0.8 + 0.5   # right edge of HISTORY column (BW+0.4 wider)
FL_TOP   = Y_hdr + 0.85/2 + 0.55       # top of Step ① header boxes
FL_BOT   = Y_proj - 2.80/2 - 0.55      # bottom of Step ③ projection boxes

highlight = FancyBboxPatch(
    (FL_LEFT, FL_BOT), FL_RIGHT - FL_LEFT, FL_TOP - FL_BOT,
    boxstyle="round,pad=0.2,rounding_size=0.4",
    fc="none", ec=LOOP_COL, lw=4.5, ls="--", alpha=1.0, zorder=10,
)
ax1.add_patch(highlight)

# Label banner above the highlight
banner_cx = (FL_LEFT + FL_RIGHT) / 2
banner_y  = FL_TOP + 0.55
banner = FancyBboxPatch(
    (FL_LEFT + 1.0, banner_y - 0.42), FL_RIGHT - FL_LEFT - 2.0, 0.84,
    boxstyle="round,pad=0.1", fc=LOOP_COL, ec="none", zorder=11,
)
ax1.add_patch(banner)
ax1.text(banner_cx, banner_y,
         "◀  FEEDBACK LOOP  ▶   (outputs from timestep  t  become inputs at timestep  t+1)",
         ha="center", va="center", fontsize=13.5,
         color="white", fontweight="bold", zorder=12)

# Small callout labels on each feedback column
callouts = [
    (XC[2], FL_BOT - 0.50, "prev action\nverbalized"),
    (XC[3], FL_BOT - 0.50, "consequence\nverbalized  [NEW]"),
    (XC[4], FL_BOT - 0.50, "rolling\nhistory"),
]
for cx, cy, txt in callouts:
    ax1.text(cx, cy, txt, ha="center", va="top", fontsize=11,
             color=LOOP_COL, fontweight="bold", zorder=12,
             multialignment="center")

# Curved arrow on the RIGHT side:  action logits (bottom) ──▶ feedback inputs (top)
# Shows the temporal cycle visually
ax1.annotate(
    "",
    xy=(FL_RIGHT + 0.3, FL_TOP - 1.0),        # arrow tip — arrives at feedback input area
    xytext=(FL_RIGHT + 0.3, 30.25),              # arrow tail — leaves from bottom of action head box
    arrowprops=dict(
        arrowstyle="-|>",
        color=LOOP_COL,
        lw=3.0,
        mutation_scale=22,
        connectionstyle="arc3,rad=-0.0",
    ),
    zorder=12,
)
ax1.text(FL_RIGHT + 1.8, (FL_TOP - 1.0 + 30.25) / 2,
         "one\ntimestep\nlater",
         ha="center", va="center", fontsize=11,
         color=LOOP_COL, fontweight="bold", style="italic", zorder=12)

# ── save ─────────────────────────────────────────────────────────────────────
fig1.savefig("docs/vlla_architecture_p1.png", dpi=150,
             bbox_inches="tight", facecolor=BG)
print("Saved page 1.")
plt.close(fig1)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — LEGEND + TOKEN KEY + NUMBERS
# ════════════════════════════════════════════════════════════════════════════
W2, H2 = 32, 24
fig2, ax2 = make_fig(W2, H2)

ax2.text(W2/2, 23.4, "VLLA Architecture — Reference Page",
         ha="center", va="center", fontsize=22, fontweight="bold", color=DARK)
Div(ax2, 22.9, W2)

# Colour legend
LX, LY = 1.5, 22.4
ax2.text(LX, LY, "COLOUR LEGEND", fontsize=16, fontweight="bold", color=DARK)
leg = [
    (VIS,  "Stream 1 — VISION"),
    (INS,  "Stream 2 — INSTRUCTION"),
    (ACT,  "Stream 3a — ACTION  (what the agent DID)"),
    (CON,  "Stream 3b — CONSEQUENCE  [NEW]  (what HAPPENED as a result)"),
    (HIS,  "Stream 4 — HISTORY  (past actions + rewards)"),
    (CLIP, "Frozen CLIP ViT-B/32  (shared backbone, never updated)"),
    (FUSE, "LLaMA Fusion Transformer  (RMSNorm + RoPE + SwiGLU)"),
    (HEAD, "Action Head  (3-layer MLP)"),
]
for i, (col, lbl) in enumerate(leg):
    iy = LY - 1.15*(i+1)
    p = FancyBboxPatch((LX, iy-0.35), 1.1, 0.70,
                       boxstyle="round,pad=0.08", fc=col, ec="none", zorder=5)
    ax2.add_patch(p)
    ax2.text(LX+1.4, iy, lbl, ha="left", va="center",
             fontsize=13.5, color=DARK, zorder=6)

Div(ax2, 12.4, W2)

# Token sequence key
TX, TY = 1.5, 12.0
ax2.text(TX, TY, "TOKEN SEQUENCE KEY", fontsize=16, fontweight="bold", color=DARK)
tseq = [
    (INS,  "L_instr",       "type = 0   —   task instruction (what to do)"),
    (ACT,  "L_action",      "type = 1   —   what the agent DID last step"),
    (CON,  "L_consequence", "type = 4   —   what HAPPENED as a result  [NEW]"),
    (VIS,  "V₁  V₂  V₃",   "type = 2   —   3 vision frame features"),
    (HIS,  "H₁ H₂ H₃ H₄",  "type = 3   —   4 past action-reward pairs"),
    (FUSE, "[ CLS ]",       "type = 0   —   aggregation token (last position)"),
]
for i, (col, tok, desc) in enumerate(tseq):
    iy = TY - 1.2*(i+1)
    p = FancyBboxPatch((TX, iy-0.40), 3.0, 0.80,
                       boxstyle="round,pad=0.08", fc=col, ec="none", zorder=5)
    ax2.add_patch(p)
    ax2.text(TX+1.50, iy, tok, ha="center", va="center",
             fontsize=12, color="white", fontweight="bold", zorder=6)
    ax2.text(TX+3.30, iy, desc, ha="left", va="center",
             fontsize=13.5, color=DARK, zorder=6)

Div(ax2, 4.5, W2)

# Key numbers
KX, KY = 1.5, 4.1
ax2.text(KX, KY, "KEY NUMBERS", fontsize=16, fontweight="bold", color=DARK)
nums = [
    ("CLIP output dim",     "512"),
    ("d_model (fusion)",    "256"),
    ("Fusion layers",       "6"),
    ("Attention heads",     "8"),
    ("Vision frames T",     "3"),
    ("History length H",    "4"),
    ("Modality types",      "5  (0, 1, 2, 3, 4)"),
    ("Sequence length S",   "11  tokens  (default)"),
    ("Action vocab size",   "14 actions + 1 null"),
]
for i, (k, v) in enumerate(nums):
    iy = KY - 0.95*(i+1)
    ax2.text(KX+0.2, iy, k, ha="left", va="center", fontsize=13.5, color="#333")
    ax2.text(KX+15.0, iy, v, ha="left", va="center",
             fontsize=13.5, color=FUSE, fontweight="bold")
    if i < len(nums)-1:
        ax2.plot([KX, KX+22], [iy-0.46, iy-0.46], color="#DDD", lw=1.0)

fig2.savefig("docs/vlla_architecture_p2.png", dpi=150,
             bbox_inches="tight", facecolor=BG)
print("Saved page 2.")
plt.close(fig2)
