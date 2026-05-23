"""
Generates fig_ablation_bar.pdf — Language-Table ablation bar chart.
Legend: upper-left, compact.  Figure: sized for wrapfigure{r}{0.52\linewidth}.
Run: python3 docs/make_ablation_bar.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data (3 seeds per condition) ──────────────────────────────────────────────
# Short single-line x-labels — no line breaks so they don't collide when rotated
conditions = [
    "BC baseline",
    r"No $E_{\mathrm{emb}}$",
    "No lang. feed.",
    r"No $E_{\mathrm{act}}$",
    "No hist. TF",
    "Full TERA ★",          # ★ — highest bar, bolded
]

seeds = {
    "Seed 42":  [68.9, 69.0, 69.6, 69.6, 69.6, 70.0],
    "Seed 123": [67.4, 67.5, 67.6, 67.6, 67.6, 68.0],
    "Seed 456": [68.6, 69.0, 68.6, 68.6, 68.7, 69.0],
}
means   = [68.3, 68.5, 68.6, 68.6, 68.6, 69.0]
BC_MEAN = 68.3

n = len(conditions)
x = np.arange(n)

# Colour: grey ablations, blue TERA
COLORS = ["#CFD8DC"] * (n - 1) + ["#1565C0"]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(2.6, 1.9))   # fits 0.45 × 5.5" wrapfigure, [13]-line slot

# Bars
bars = ax.bar(x, means, color=COLORS, width=0.60, zorder=2,
              edgecolor="white", linewidth=0.5)

# Black edge on the TERA bar only
bars[-1].set_edgecolor("#0D47A1")
bars[-1].set_linewidth(1.0)

# Per-seed scatter points
seed_markers = ["o", "s", "^"]
seed_colors  = ["#E53935", "#43A047", "#FB8C00"]
for (sname, svals), mk, sc in zip(seeds.items(), seed_markers, seed_colors):
    ax.scatter(x, svals, marker=mk, color=sc, s=18, zorder=4,
               label=sname, linewidths=0.5, edgecolors="white")

# BC dashed reference line
ax.axhline(BC_MEAN, color="#B71C1C", linewidth=1.0, linestyle="--",
           zorder=3, label=f"BC ref ({BC_MEAN:.1f}%)")

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(
    conditions,
    fontsize=6.5,
    rotation=32,
    ha="right",
    rotation_mode="anchor",
)
ax.set_ylabel("Act.-Classif. Acc. (%)", fontsize=6.5, labelpad=2)
ax.set_ylim(66.5, 71.5)   # tight range — legend sits in the upper-left gap

ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.yaxis.grid(True, linestyle=":", linewidth=0.55, alpha=0.65, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=6.0)

# ── Legend — upper-left, compact, inside axes ──────────────────────────────────
leg = ax.legend(
    loc="upper left",
    fontsize=5.5,
    framealpha=0.92,
    edgecolor="#CCCCCC",
    handlelength=1.2,
    handleheight=0.8,
    borderpad=0.45,
    labelspacing=0.28,
    handletextpad=0.4,
    ncol=1,
    markerscale=0.85,
)
leg.get_frame().set_linewidth(0.5)

# ── Layout & save ─────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.35)

for dest in [
    os.path.join(OUT_DIR, "fig_ablation_bar.pdf"),
    "/Users/HP/Downloads/corl_2026_template_submission/fig_ablation_bar.pdf",
]:
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    print(f"Saved: {dest}")

plt.close(fig)
