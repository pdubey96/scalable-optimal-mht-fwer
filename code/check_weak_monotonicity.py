#!/usr/bin/env python3
"""Are the computed policies "reasonable" in the sense of Rosset et al. (2022)?

This script answers that question empirically. The term is used informally in
that paper and is tied to monotonicity of the rejection count. Two versions
appear there:

  strict   if u <= u' elementwise and the policy rejects every hypothesis at u',
           it rejects every hypothesis at u;
  weak     smaller p-values cannot result in fewer rejections, i.e.
           u <= u' elementwise  =>  l*(u) >= l*(u').

Rosset et al. report that the optimal policies they derive satisfy neither in
general -- their Figure 3 and Figure 6 regions violate weak monotonicity -- and
they add it as an explicit constraint at a cost in power and computation.

The policies here have more structure: they reject a leading block of the
smallest p-values under the likelihood-ratio ordering, with the block length

    l*(u) = argmax_{0 <= l <= K} sum_{i<=l} R_i(mu, u).

Whether that length is monotone in u is not obvious, because every R_i depends
on the whole vector. This script settles it numerically on the policies the
paper actually reports, using the mu vectors stored with the results rather
than re-solving.

Sorting is not a complication: if u <= u' elementwise then the order statistics
satisfy u_(i) <= u'_(i) for every i, so the comparison is well posed on the
sorted vectors the decision rule consumes.

Usage:
    python check_weak_monotonicity.py                 # all stored policies
    python check_weak_monotonicity.py --n-pairs 200000
"""

import argparse
import json
import os
import sys

import numpy as np

from general_k import paths
from general_k.decision import l_star_batch
from run_all_experiments import MODEL_CONFIGS


DEFAULT_RESULTS = os.path.join(paths.RESULTS_DIR, "prior_objective.json")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Preserve examples while displaying argument defaults."""


def _positive_int(value):
    """An argparse type for positive Monte Carlo counts."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value):
    """An argparse type for NumPy-compatible random seeds."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _existing_file(value):
    """An argparse type for a readable regular file."""
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"file is not readable: {value}")
    return path


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--n-pairs", type=_positive_int, default=100000,
        help="Ordered p-value pairs sampled per policy and perturbation scheme",
    )
    parser.add_argument(
        "--seed", type=_nonnegative_int, default=20260812,
        help="Random seed for the empirical audit",
    )
    parser.add_argument(
        "--results", type=_existing_file, default=DEFAULT_RESULTS,
        help="Prior-objective JSON containing the policies to audit",
    )
    parser.add_argument(
        "--allow-violations", action="store_true",
        help="Exit zero after reporting observed violations; required for the "
             "paper's disclosed Student-t counterexamples",
    )
    return parser


def sample_pairs(rng, K, n, scheme):
    """Draw (u, u_prime) with u <= u_prime elementwise, K coordinates each."""
    up = rng.random((n, K))
    if scheme == "shrink":
        # u is a uniformly random point below u' in the box: the whole range
        # of "smaller", including much smaller.
        u = up * rng.random((n, K))
    elif scheme == "nudge":
        # u is just below u': the local case, where a non-monotonicity would
        # show up as a knife-edge flip rather than a gross one.
        u = up * (1.0 - 1e-3 * rng.random((n, K)))
    elif scheme == "one_coord":
        # only one coordinate decreases: the sharpest form of the requirement,
        # and the one Rosset et al.'s figures violate.
        u = up.copy()
        j = rng.integers(0, K, size=n)
        rows = np.arange(n)
        u[rows, j] = up[rows, j] * rng.random(n)
    else:
        raise ValueError(
            f"unknown sampling scheme {scheme!r}; expected shrink, nudge, or one_coord"
        )
    return np.sort(u, axis=1), np.sort(up, axis=1)


def check(mu, g_func, K, rng, n_pairs, scheme):
    """Count weak and strict monotonicity violations in sampled pairs."""
    u, up = sample_pairs(rng, K, n_pairs, scheme)
    l_u = l_star_batch(np.asarray(mu), u, g_func, K)
    l_up = l_star_batch(np.asarray(mu), up, g_func, K)
    viol = l_u < l_up                      # smaller p-values, fewer rejections
    strict_viol = viol & (l_up == K)       # ... and everything was rejected at u'
    return int(viol.sum()), int(strict_viol.sum()), l_u, l_up


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        with open(args.results, encoding="utf-8") as stream:
            records = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read {args.results}: {exc}")
    if not isinstance(records, list) or not records:
        parser.error("--results must contain a nonempty JSON list of policy records")

    required = {"model", "theta", "K", "prior_name", "mu"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            parser.error(f"record {index} in --results is not a JSON object")
        missing = sorted(required.difference(record))
        if missing:
            parser.error(
                f"record {index} in --results is missing: {', '.join(missing)}"
            )
        if record["model"] not in MODEL_CONFIGS:
            parser.error(
                f"record {index} uses unknown model {record['model']!r}; "
                f"expected one of {tuple(MODEL_CONFIGS)}"
            )

    rng = np.random.default_rng(args.seed)

    schemes = ("shrink", "nudge", "one_coord")
    print(f"{args.n_pairs:,} pairs per policy per scheme, "
          f"{len(records)} stored policies\n")
    header = f"{'policy':<44}" + "".join(f"{s:>12}" for s in schemes)
    print(header)
    print("-" * len(header))

    total_viol = 0
    total_strict = 0
    total_all_vs_none = 0
    for rec in records:
        g_func = MODEL_CONFIGS[rec["model"]]["factory"](rec["theta"])
        K = rec["K"]
        label = (f"{rec['model']} th={rec['theta']:g} K={K} "
                 f"{rec['prior_name']}")
        cells = []
        for scheme in schemes:
            v, sv, l_u, l_up = check(
                rec["mu"], g_func, K, rng, args.n_pairs, scheme
            )
            total_viol += v
            total_strict += sv
            total_all_vs_none += int(
                ((l_u == 0) & (l_up == K)).sum()
            )
            cells.append(f"{v:>12,}")
        print(f"{label:<44}" + "".join(cells))

    print()
    print(f"weak-monotonicity violations:   {total_viol:,}")
    print(f"  of which the policy rejected everything at the larger vector: "
          f"{total_strict:,}")
    print(f"  of which it rejected everything at the larger vector and "
          f"nothing at the smaller: {total_all_vs_none:,}")
    if total_viol == 0:
        print("\nNo violation found: on every policy the paper reports, a "
              "coordinatewise decrease in the p-values never decreased the "
              "number of rejections.")
    else:
        print("\nViolations found. These are counterexamples to weak "
              "monotonicity and must be reported as such, not explained away.")
        if not args.allow_violations:
            print(
                "Re-run with --allow-violations only when these "
                "counterexamples are intentionally acknowledged.",
                file=sys.stderr,
            )
            return 2
        print(
            "Continuing with exit status 0 because --allow-violations was set.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
