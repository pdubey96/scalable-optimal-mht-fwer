#!/usr/bin/env python3
"""Main/supplement power-curve figures, compact two-row layout.

For each K in {3, 4, 5, 6}, produce a 2x4 panel: columns are the four
alternative families, and rows are average power Pi_K (the optimised objective)
and any-discovery power Pi_any. Corresponding panels across K share vertical
limits (per model and metric) so the K=3 and K=6 figures are directly
comparable; scales differ between models. K=3 and K=6 are main-text figures;
all four are used in the supplement.

Writes power_curves_K{3,4,5,6}.pdf into ``reproduced/plots`` by default
with an explicit white
background and fully opaque colours (no alpha).
Standalone; does not touch other figures or regenerate results.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from general_k import figstyle, paths
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# columns, left to right: (model key, display name, x-axis label)
MODELS = [
    ("trunc",   "Truncated normal", r"Signal $\theta$"),
    ("mixture", "Mixture normal",   r"Signal $\theta$"),
    ("t",       "Student-$t$",      r"Degrees of freedom"),
    ("beta",    "Beta",             r"Shape $\theta$"),
]
# The reference layout uses clean line work rather than a marker at every
# simulated setting. Six distinct dash patterns preserve that economy in
# greyscale; colour is a redundant on-screen cue and SPOT remains the darkest,
# slightly heaviest line.
STYLE = {
    "Algorithm":     {"color": (0.00, 0.00, 0.00), "lw": 1.45, "ls": "-"},
    "Bonferroni":    {"color": (0.00, 0.45, 0.70), "lw": 1.05, "ls": "--"},
    "Holm":          {"color": (0.00, 0.60, 0.50), "lw": 1.05,
                       "ls": (0, (5, 1.5, 1, 1.5))},
    "Hochberg":      {"color": (0.90, 0.60, 0.00), "lw": 1.05, "ls": ":"},
    "Hommel":        {"color": (0.80, 0.40, 0.00), "lw": 1.05,
                       "ls": (0, (6, 2))},
    "Fisher-closed": {"color": (0.60, 0.00, 0.45), "lw": 1.15,
                       "ls": (0, (3, 1, 1, 1))},
}
METHODS = ["Algorithm", "Bonferroni", "Holm", "Hochberg", "Hommel", "Fisher-closed"]
LEGEND = {"Algorithm": "SPOT", "Bonferroni": "Bonferroni", "Holm": "Holm",
          "Hochberg": "Hochberg", "Hommel": "Hommel",
          "Fisher-closed": "Closed Fisher"}
GRID_GREY = "0.89"
STRIP_FACE = "0.94"
STRIP_EDGE = "0.35"
KS_ALL = [3, 4, 5, 6]


def shared_ylims(data):
    """Per (model, metric): common y-limits over all K so corresponding panels match."""
    ylims = {}
    for mkey, _, _ in MODELS:
        for metric in ("pi_K", "pi_any"):
            vals = [suite[t]["results"][m][metric]
                    for K in KS_ALL
                    for t, suite in [(t, data[str(K)][mkey]) for t in data[str(K)][mkey]]
                    for m in METHODS]
            lo, hi = min(vals), max(vals)
            pad = 0.05 * (hi - lo) if hi > lo else 0.05
            ylims[(mkey, metric)] = (lo - pad, hi + pad)
    return ylims


# JCGS requires 10pt printed figure text. This setting, checked
# against the saved PDF's bounding box and its manuscript inclusion scale,
# prints at about 10.1pt: materially less obtrusive than the former 11.4pt but
# still above the stated floor.
PLOT_PT = 10.20
ASPECT = 0.60


def facet_strip(ax, label):
    """Add the slim, full-width grey facet strip used in the reference."""
    ax.add_patch(Rectangle(
        (0, 1.01), 1, 0.15, transform=ax.transAxes, clip_on=False,
        facecolor=STRIP_FACE, edgecolor=STRIP_EDGE, linewidth=0.65))
    ax.text(0.5, 1.085, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=PLOT_PT)


def make_figure(data, K, out, ylims):
    figstyle.apply()
    fig, axes = plt.subplots(
        2, 4, figsize=figstyle.size(aspect=ASPECT), facecolor="white",
        sharex="col", gridspec_kw={"hspace": 0.10, "wspace": 0.36})
    rows = [("pi_K", rf"$\Pi_{{{K}}}$"),
            ("pi_any", r"$\Pi_{\rm any}$")]
    for c, (mkey, mname, xlabel) in enumerate(MODELS):
        suite = data[str(K)][mkey]
        thetas = sorted(suite.keys(), key=float)
        xs = [float(t) for t in thetas]
        for r, (metric, row_label) in enumerate(rows):
            ax = axes[r, c]
            ax.set_facecolor("white")
            for m in METHODS:
                ys = [suite[t]["results"][m][metric] for t in thetas]
                st = STYLE[m]
                ax.plot(xs, ys, color=st["color"], ls=st["ls"],
                        lw=st["lw"], solid_capstyle="butt")
            ax.set_ylim(*ylims[(mkey, metric)])
            ax.grid(True, ls=(0, (2, 3)), color=GRID_GREY, lw=0.55)
            ax.tick_params(labelsize=PLOT_PT, length=2.5, width=0.65, pad=1.5)
            for spine in ax.spines.values():
                spine.set_color("0.25")
                spine.set_linewidth(0.7)
            ax.locator_params(axis="x", nbins=4)
            ax.locator_params(axis="y", nbins=4)
            if r == 0:
                ax.tick_params(axis="x", labelbottom=False)
                facet_strip(ax, mname)
            else:
                ax.set_xlabel(xlabel, fontsize=PLOT_PT, labelpad=2)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=PLOT_PT, labelpad=3)

    handles = [Line2D([0], [0], color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                      lw=STYLE[m]["lw"],
                      label=LEGEND[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=6,
               fontsize=PLOT_PT, frameon=False,
               columnspacing=0.60, handlelength=1.45, handletextpad=0.30,
               borderaxespad=0, bbox_to_anchor=(0.5, 0.985))

    # Manual margins avoid the large legend-to-strip and inter-panel gaps that
    # tight_layout introduced in this dense 2x4 design.
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.140, top=0.840,
                        hspace=0.10, wspace=0.36)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03,
                facecolor="white", transparent=False)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=paths.result("all_experiments.json"),
        help="Grid-result JSON (default: bundled reference results)",
    )
    parser.add_argument(
        "--out-dir", default=paths.REPRODUCED_PLOTS_DIR,
        help="Figure directory (default: reproduced/plots)",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)
    figdir = os.path.abspath(args.out_dir)
    os.makedirs(figdir, exist_ok=True)
    ylims = shared_ylims(data)
    for K in KS_ALL:
        make_figure(data, K, os.path.join(figdir, f"power_curves_K{K}.pdf"), ylims)


if __name__ == "__main__":
    main()
