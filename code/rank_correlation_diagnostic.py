#!/usr/bin/env python3
"""Rank correlation between realised FWER and margin, quoted in Section 4.1.

The manuscript argues that a policy's realised family-wise error rate at a
configuration is a usable diagnostic for whether the objective is matched to
that configuration: slack error rate, small margin.  The number backing that
sentence had no generator, and went stale once the comparator envelope grew
from four procedures to eight; this script is the generator.

Population: the two objectives contrasted in Table 1 (the point prior at
gamma = K and the flat prior), in all six of its settings, at the configurations
gamma = 1, ..., K-1.  Configuration gamma = K is excluded because FWER_K is
identically zero there -- no hypothesis is null.

Usage:  python rank_correlation_diagnostic.py
"""
import argparse
import json

from general_k import paths

# The two objectives Table 1 contrasts, as named in results/prior_objective.json.
OBJECTIVES = ("point gamma=K", "uniform")


def ranks(xs):
    """Ranks with ties averaged, as Spearman's coefficient requires."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=paths.result("prior_objective.json"),
        help="Prior-objective JSON (default: bundled reference results)",
    )
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    fwers, margins = [], []
    for rec in records:
        if rec["prior_name"] not in OBJECTIVES:
            continue
        K = rec["K"]
        # curves[...]["fwer"] is indexed by gamma = 0..K; "gains" runs over
        # gamma = 1..K, so the margin at gamma sits at gains[gamma - 1].
        for gamma in range(1, K):
            fwers.append(rec["curves"]["SPOT"]["fwer"][gamma])
            margins.append(rec["gains"][gamma - 1])

    rho = pearson(ranks(fwers), ranks(margins))
    print("configurations pooled: %d" % len(fwers))
    print("Spearman rank correlation: %.4f  (quoted as %.2f)" % (rho, rho))
    print("Pearson, for reference:    %.4f" % pearson(fwers, margins))


if __name__ == "__main__":
    main()
