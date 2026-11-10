#!/usr/bin/env python3
"""Emit the manuscript's scaling table from results/scaling_results.json.

Figure 2 (plots/scaling_with_K.pdf) is drawn from the same file, so the table and
the figure agree exactly rather than to within Monte Carlo error, which is what
happened while the table was maintained by hand from the configuration sweep.

Writes reproduced/results/table_scaling.tex by default, the tabular body only.
Use it as the body of manuscript Table 2, or compare it with the bundled
reference file at ../results/table_scaling.tex.

Usage:  python make_table_scaling.py
"""
import argparse
import json
import math
import os

from general_k import paths
from general_k.io import atomic_text_write

N_EVAL = 50_000          # legacy paper files predate embedded run configuration
Z = 1.959963985          # two-sided 95% normal quantile

# Printed left to right; "Algorithm" is SPOT and is set in bold.
COLUMNS = ["Bonferroni", "Holm", "Hommel", "Fisher-closed", "Algorithm"]
HEADINGS = ["Bonferroni", "Holm", "Hommel", "Fisher-closed", "SPOT"]


def max_fwer(entry):
    """Largest realised FWER over the null configurations, and where it occurs."""
    fwers = {int(k.split("_")[1]): v
             for k, v in entry.items() if k.startswith("fwer_")}
    gamma = max(fwers, key=lambda g: fwers[g])
    return fwers[gamma], gamma


def wald_ci(p, n=N_EVAL):
    """Pointwise binomial 95% interval, unadjusted for having selected the max."""
    half = Z * math.sqrt(p * (1.0 - p) / n)
    return p - half, p + half


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=paths.result("scaling_results.json"),
        help="Scaling-result JSON (default: bundled reference results)",
    )
    parser.add_argument(
        "--out", default=paths.reproduced_result("table_scaling.tex"),
        help="Generated LaTeX table body (default: reproduced/results/table_scaling.tex)",
    )
    parser.add_argument(
        "--n-eval", type=_positive_int, default=None,
        help="Override the evaluation sample size used for confidence intervals; "
             "otherwise read _run_config.n_eval (legacy paper files use 50000)",
    )
    args = parser.parse_args(argv)

    with open(args.input, encoding="utf-8") as f:
        scaling = json.load(f)

    rows = []
    for K_str in sorted(scaling, key=int):
        K = int(K_str)
        entry = scaling[K_str]
        res = entry["results"]

        powers = " & ".join(
            (r"$\mathbf{%.3f}$" if name == "Algorithm" else r"$%.3f$") % res[name]["pi_K"]
            for name in COLUMNS)
        spot_p, _ = max_fwer(res["Algorithm"])
        n_eval = args.n_eval
        if n_eval is None:
            n_eval = entry.get("_run_config", {}).get("n_eval", N_EVAL)
        lo, hi = wald_ci(spot_p, n_eval)
        rows.append(r"%d & %s & $%.3f\ (%.3f,%.3f)$ \\"
                    % (K, powers, spot_p, lo, hi))

    body = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            "$K$ & " + " & ".join(HEADINGS) +
            r" & \shortstack{SPOT max FWER\\(95\% CI)} \\",
            r"\midrule"]
    body += rows
    body += [r"\bottomrule", r"\end{tabular}"]

    out = os.path.abspath(args.out)
    atomic_text_write("\n".join(body) + "\n", out)
    print("\n".join(body))
    print("\nwrote", os.path.normpath(out))


if __name__ == "__main__":
    raise SystemExit(main())
