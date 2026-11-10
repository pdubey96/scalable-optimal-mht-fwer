#!/usr/bin/env python3
"""Figures for SPOT under a prior on the number of non-null hypotheses.

Reads results/prior_objective.json and writes
reproduced/plots/prior_objective.{pdf,png} by default,
a 2x2 panel:

  (a) Pi_gamma against gamma at the headline setting, for the point prior at
      gamma = K, the prior-weighted objectives, and Hommel.
  (b) Margin over the eight-procedure envelope against the number of non-nulls,
      same setting:
      the gamma = K objective crosses zero, the prior-weighted ones do not.
  (c) The robustness-power tradeoff across every setting: power given up at
      gamma = K against worst-case gain over all gamma.
  (d) Realised FWER by configuration: the gamma = K objective leaves the
      constraint slack where it loses power; the prior-weighted one does not.

Standalone; reads frozen results and touches no other figure.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from general_k import figstyle, paths

from general_k.gamma_sweep import COMPARATORS

BASELINES = COMPARATORS
PRIOR_STYLE = {
    "point gamma=K":  {"color": (0.00, 0.00, 0.00), "ls": "-",         "marker": "o",
                       "label": r"$\gamma=K$ point prior"},
    "uniform":        {"color": (0.00, 0.45, 0.70), "ls": "--",        "marker": "s",
                       "label": "uniform prior"},
    "binomial p=0.5": {"color": (0.00, 0.60, 0.50), "ls": "-.",        "marker": "^",
                       "label": r"binomial $p=0.5$"},
    "binomial p=0.8": {"color": (0.90, 0.60, 0.00), "ls": ":",         "marker": "D",
                       "label": r"binomial $p=0.8$"},
}
HOMMEL = {"color": (0.80, 0.40, 0.00), "ls": (0, (5, 1)), "marker": "v",
          "label": "Hommel"}
FISHER = {"color": (0.60, 0.00, 0.45), "ls": (0, (3, 1)), "marker": "P",
          "label": "Closed Fisher"}
HEADLINE = ("trunc", -2.0, 6)
GRID_GREY = "0.82"
# Figure-local type can be slightly smaller than the shared default because the
# figure is printed at essentially its native width.  The saved-PDF audit gives
# an effective size just above 10.2 pt, retaining margin over the 10 pt floor.
PLOT_PT = 10.25


def rel_gain(curves, gamma):
    best = max(curves[n]["power"][gamma] for n in BASELINES)
    return (curves["SPOT"]["power"][gamma] - best) / best


def select(records, model, theta, K):
    return [r for r in records
            if r["model"] == model and abs(float(r["theta"]) - theta) < 1e-9
            and r["K"] == K]


def setting_label(rec):
    names = {"trunc": "Trunc. normal", "mixture": "Mixture normal",
             "t": "Student-$t$", "beta": "Beta"}
    theta = rec["theta"]
    unit = "df" if rec["model"] == "t" else r"$\theta$"
    return f"{names[rec['model']]}, {unit}={theta:g}, $K$={rec['K']}"


def panel_a(ax, records):
    group = select(records, *HEADLINE)
    K = HEADLINE[2]
    gammas = list(range(1, K + 1))
    for rec in group:
        st = PRIOR_STYLE[rec["prior_name"]]
        ax.plot(gammas, [rec["curves"]["SPOT"]["power"][g] for g in gammas],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                markersize=4, lw=1.5, label=st["label"])
    for st, key in ((HOMMEL, "Hommel"), (FISHER, "Fisher-closed")):
        ax.plot(gammas, [group[0]["curves"][key]["power"][g] for g in gammas],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                markersize=4, lw=1.2, label=st["label"])
    ax.set_xlabel(r"Number of non-null hypotheses $\gamma$")
    ax.set_ylabel(r"Average power $\Pi_\gamma$")
    ax.set_title("(a) Power by configuration", fontsize=PLOT_PT)
    ax.set_xticks(gammas)


def panel_b(ax, records):
    group = select(records, *HEADLINE)
    K = HEADLINE[2]
    gammas = list(range(1, K + 1))
    for rec in group:
        st = PRIOR_STYLE[rec["prior_name"]]
        ax.plot(gammas,
                [100 * rel_gain(rec["curves"], g) for g in gammas],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                markersize=4, lw=1.5, label=st["label"])
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_xlabel(r"Number of non-null hypotheses $\gamma$")
    ax.set_ylabel("Margin over envelope (%)")
    ax.set_title("(b) Margin over envelope", fontsize=PLOT_PT)
    ax.set_xticks(gammas)
    # The prior-weighted curves run along the top of this panel, so the legend
    # goes bottom right, where only the gamma = K curve's descent leaves space.


def panel_c(ax, records):
    """The trade, one point per solver run.

    Only solver runs that meet the residual criterion are plotted.  An uncertified
    run has not reached the constraint boundary, so both of its coordinates --
    the power surrendered at gamma = K and the worst-case margin -- are read off
    a policy that is not the one the objective defines, and a reader cannot tell
    the two kinds of point apart once they are on the same axes.  The dropped
    rows are named on stdout and in the caption.
    """
    dropped = []
    for rec in records:
        st = PRIOR_STYLE[rec["prior_name"]]
        if rec["price_at_K"] is None:
            continue
        if not rec["converged"]:
            dropped.append(rec)
            continue
        ax.scatter(100 * rec["price_at_K"], 100 * rec["min_gain"],
                   color=st["color"], marker=st["marker"], s=26,
                   label=st["label"])
    # One legend entry per prior, not per point.
    handles, labels = ax.get_legend_handles_labels()
    seen, keep = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            keep.append((h, l))
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_xlabel(r"Power given up at $\gamma=K$ (%)")
    ax.set_ylabel(r"Worst gain over all $\gamma$ (%)")
    ax.set_title("(c) Robustness tradeoff", fontsize=PLOT_PT)
    return dropped


def panel_d(ax, records):
    group = select(records, *HEADLINE)
    K = HEADLINE[2]
    gammas = list(range(K))                       # gamma = K has no null
    for rec in group:
        if rec["prior_name"] not in ("point gamma=K", "uniform"):
            continue
        st = PRIOR_STYLE[rec["prior_name"]]
        ax.plot(gammas, [rec["curves"]["SPOT"]["fwer"][g] for g in gammas],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                markersize=4, lw=1.5, label=st["label"])
    ax.axhline(0.05, color=(0.80, 0.40, 0.00), lw=0.9, ls="--")
    ax.text(0.05, 0.0505, r"$\alpha$", fontsize=PLOT_PT, color=(0.80, 0.40, 0.00))
    ax.set_xlabel(r"Number of non-null hypotheses $\gamma$")
    ax.set_ylabel("Realised FWER")
    # Kept short: the full statement is in the caption, and a longer title runs
    # past the panel at this width.
    ax.set_title("(d) FWER by configuration", fontsize=PLOT_PT)
    ax.set_xticks(gammas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=paths.result("prior_objective.json"),
        help="Prior-objective JSON (default: bundled reference results)",
    )
    parser.add_argument(
        "--out-dir", default=paths.REPRODUCED_PLOTS_DIR,
        help="Figure directory (default: reproduced/plots)",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    # Sized to the current article text measure so inclusion at \textwidth neither
    # shrinks nor enlarges the type; panel sizes are what the reader sees.
    figstyle.apply()
    plt.rcParams.update({
        "font.size": PLOT_PT,
        "axes.labelsize": PLOT_PT,
        "axes.titlesize": PLOT_PT,
        "xtick.labelsize": PLOT_PT,
        "ytick.labelsize": PLOT_PT,
        "legend.fontsize": PLOT_PT,
    })
    fig, axes = plt.subplots(
        2, 2, figsize=figstyle.size(aspect=0.70), facecolor="white")
    for ax in axes.ravel():
        ax.set_facecolor("white")
        ax.grid(True, color=GRID_GREY, lw=0.6)
        ax.set_axisbelow(True)

    panel_a(axes[0, 0], records)
    panel_b(axes[0, 1], records)
    dropped = panel_c(axes[1, 0], records)
    panel_d(axes[1, 1], records)

    print(f"panel (c): {len(records) - len(dropped)} of {len(records)} solver runs "
          f"plotted; {len(dropped)} dropped as uncertified")
    for rec in dropped:
        print(f"  dropped: {rec['model']} K={rec['K']} {rec['prior_name']} "
              f"(stopped on '{rec['reason']}' after {rec['sweeps']} sweeps, "
              f"residual {rec['final_residual']:.2e})")

    # One legend for the whole figure.  At 10pt or more a six-entry legend does
    # not fit inside a panel of this width without covering the curves.
    seen, handles, labels = set(), [], []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen.add(lab)
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=PLOT_PT, columnspacing=1.4, handlelength=2.0,
               bbox_to_anchor=(0.5, 1.0))
    # Reclaim internal legend and row-spacing reserve without making the outer
    # float taller: this keeps the table and complete figure together while
    # increasing each panel's plotting height by roughly one quarter.
    fig.tight_layout(rect=(0, 0, 1, 0.86), h_pad=0.6, w_pad=1.4)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"prior_objective.{ext}")
        fig.savefig(path, dpi=200, facecolor="white")
        print(f"wrote {os.path.normpath(path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
