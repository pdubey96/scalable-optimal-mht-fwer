#!/usr/bin/env python3
"""Figures for the power-versus-number-of-non-nulls study.

Reads results/gamma_sweep.json and writes
reproduced/plots/gamma_sweep.{pdf,png} by default, a 2x2 panel:

  (a) Pi_gamma against gamma for every procedure, at the headline setting
      (truncated normal, theta = -2, K = 6).
  (b) SPOT's relative gain over the best baseline against the non-null
      fraction gamma/K, one line per K, same alternative family.
  (c) The same relative gain at K = 6 across alternative families and signal
      strengths, showing that the crossover is family-dependent.
  (d) Relative gain against SPOT's realised FWER at that configuration: where
      the constraint goes slack off-design, the power is lost.

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
STYLE = {
    "SPOT":            {"color": (0.00, 0.00, 0.00), "lw": 1.8, "ls": "-",         "marker": "o"},
    "Bonferroni":      {"color": (0.00, 0.45, 0.70), "lw": 1.0, "ls": "--",        "marker": "s"},
    "Holm":            {"color": (0.00, 0.60, 0.50), "lw": 1.0, "ls": "-.",        "marker": "^"},
    "Hochberg":        {"color": (0.90, 0.60, 0.00), "lw": 1.0, "ls": ":",         "marker": "D"},
    "Hommel":          {"color": (0.80, 0.40, 0.00), "lw": 1.2, "ls": (0, (5, 1)), "marker": "v"},
    "Fisher-closed":   {"color": (0.60, 0.00, 0.45), "lw": 1.4, "ls": (0, (3, 1)), "marker": "P"},
    "Stouffer-closed": {"color": (0.35, 0.35, 0.35), "lw": 1.0, "ls": (0, (1, 1)), "marker": "X"},
    "Median-closed":   {"color": (0.20, 0.55, 0.20), "lw": 1.0, "ls": (0, (4, 2)), "marker": "*"},
}
# Panel (a) would be unreadable with all nine procedures; show SPOT, the
# strongest stepwise procedure, and the strongest closed test.
PANEL_A_METHODS = ["SPOT", "Hommel", "Fisher-closed"]
K_COLORS = {3: (0.00, 0.45, 0.70), 4: (0.00, 0.60, 0.50), 5: (0.35, 0.70, 0.90),
            6: (0.00, 0.00, 0.00), 8: (0.90, 0.60, 0.00), 10: (0.80, 0.40, 0.00),
            12: (0.60, 0.00, 0.45)}
GRID_GREY = "0.82"


def rel_gain(curves, gamma):
    """SPOT's power at gamma relative to the strongest baseline there."""
    best = max(curves[n]["power"][gamma] for n in BASELINES)
    return (curves["SPOT"]["power"][gamma] - best) / best


def find(records, model, theta, K):
    for r in records:
        if r["model"] == model and abs(float(r["theta"]) - theta) < 1e-9 and r["K"] == K:
            return r
    return None


def panel_a(ax, records):
    rec = find(records, "trunc", -2.0, 6)
    K, curves = rec["K"], rec["curves"]
    gammas = list(range(1, K + 1))
    for name in PANEL_A_METHODS:
        ax.plot(gammas, [curves[name]["power"][g] for g in gammas],
                label=name, **STYLE[name], markersize=4)
    ax.set_xlabel(r"Number of non-null hypotheses $\gamma$")
    ax.set_ylabel(r"Average power $\Pi_\gamma$")
    ax.set_title(r"(a) Truncated normal, $\theta=-2$, $K=6$", fontsize=figstyle.BASE_PT)
    ax.set_xticks(gammas)
    ax.legend(fontsize=figstyle.BASE_PT, frameon=False, loc="upper left")


def panel_b(ax, records):
    for rec in records:
        if rec["model"] != "trunc" or abs(float(rec["theta"]) + 2.0) > 1e-9:
            continue
        K, curves = rec["K"], rec["curves"]
        gammas = list(range(1, K + 1))
        ax.plot([g / K for g in gammas], [100 * rel_gain(curves, g) for g in gammas],
                color=K_COLORS[K], lw=1.4, marker="o", markersize=3,
                label=f"$K={K}$")
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_xlabel(r"Non-null fraction $\gamma/K$")
    ax.set_ylabel("Gain over best baseline (%)")
    ax.set_title(r"(b) Truncated normal, $\theta=-2$", fontsize=figstyle.BASE_PT)
    ax.legend(fontsize=figstyle.BASE_PT, frameon=False, ncol=2, loc="upper left")


def panel_c(ax, records):
    series = [
        ("trunc", -1.0, r"Trunc. normal $\theta=-1$", (0.00, 0.45, 0.70), "-"),
        ("trunc", -2.0, r"Trunc. normal $\theta=-2$", (0.00, 0.00, 0.00), "-"),
        ("trunc", -3.0, r"Trunc. normal $\theta=-3$", (0.35, 0.70, 0.90), "-"),
        ("mixture", -2.0, r"Mixture normal $\theta=-2$", (0.00, 0.60, 0.50), "--"),
        ("t", 6, r"Student-$t$, 6 df", (0.80, 0.40, 0.00), "-."),
        ("beta", 0.4, r"Beta $\theta=0.4$", (0.90, 0.60, 0.00), ":"),
    ]
    for model, theta, label, color, ls in series:
        rec = find(records, model, theta, 6)
        if rec is None:
            continue
        K, curves = rec["K"], rec["curves"]
        gammas = list(range(1, K + 1))
        ax.plot([g / K for g in gammas], [100 * rel_gain(curves, g) for g in gammas],
                color=color, ls=ls, lw=1.4, marker="o", markersize=3, label=label)
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_xlabel(r"Non-null fraction $\gamma/K$")
    ax.set_ylabel("Gain over best baseline (%)")
    ax.set_title(r"(c) Alternative families, $K=6$", fontsize=figstyle.BASE_PT)
    ax.legend(fontsize=figstyle.BASE_PT, frameon=False, loc="upper left")


def panel_d(ax, records):
    xs, ys = [], []
    for rec in records:
        K, curves = rec["K"], rec["curves"]
        for gamma in range(1, K):          # gamma = K has no null, hence no FWER
            xs.append(curves["SPOT"]["fwer"][gamma])
            ys.append(100 * rel_gain(curves, gamma))
    ax.scatter(xs, ys, s=14, facecolors="none", edgecolors=(0.0, 0.0, 0.0), lw=0.8)
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.axvline(0.05, color=(0.80, 0.40, 0.00), lw=0.9, ls="--")
    ax.text(0.0495, ax.get_ylim()[0] + 6, r"$\alpha$", fontsize=figstyle.BASE_PT, ha="right",
            color=(0.80, 0.40, 0.00))
    ax.set_xlabel(r"SPOT's realised FWER at $\gamma$")
    ax.set_ylabel("Gain over best baseline (%)")
    ax.set_title("(d) Slack off-design constraint costs power", fontsize=figstyle.BASE_PT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=paths.result("gamma_sweep.json"),
        help="Gamma-sweep JSON (default: bundled reference results)",
    )
    parser.add_argument(
        "--out-dir", default=paths.REPRODUCED_PLOTS_DIR,
        help="Figure directory (default: reproduced/plots)",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), facecolor="white")
    for ax in axes.ravel():
        ax.set_facecolor("white")
        ax.grid(True, color=GRID_GREY, lw=0.6)
        ax.set_axisbelow(True)

    panel_a(axes[0, 0], records)
    panel_b(axes[0, 1], records)
    panel_c(axes[1, 0], records)
    panel_d(axes[1, 1], records)

    fig.tight_layout()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"gamma_sweep.{ext}")
        fig.savefig(path, dpi=200, facecolor="white")
        print(f"wrote {os.path.normpath(path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
