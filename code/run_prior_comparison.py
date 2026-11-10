"""
Is there a better prior on gamma than the uniform one?

The uniform prior on the number of non-nulls is the Bayes prior induced by a
flat Beta(1, 1) prior on the per-hypothesis non-null rate.  This script sweeps
the Beta(a, b) family around it -- including Jeffreys' Beta(1/2, 1/2), which is
U-shaped and favours both extremes -- and scores each by the two quantities
that matter:

  - worst gain over the best standard procedure, across all gamma;
  - price paid at gamma = K relative to the manuscript's own policy there.

Usage:
    python run_prior_comparison.py
    python run_prior_comparison.py --k 6
    python run_prior_comparison.py --model t --theta 6 --k 6 \\
        --allow-uncertified --out prior_comparison_t.json
    python run_prior_comparison.py --model beta --theta 0.4 \\
        --out prior_comparison_beta.json
"""

import argparse
import math
import time

from general_k.prior_objective import beta_binomial_prior, point_prior
from general_k import paths
from run_all_experiments import MODEL_CONFIGS, save_results
from run_prior_objective import _certification_status, run_one


CANDIDATES = [
    ("Beta(1,1) = uniform", 1.0, 1.0),
    ("Beta(0.5,0.5) Jeffreys", 0.5, 0.5),
    ("Beta(0.25,0.25)", 0.25, 0.25),
    ("Beta(1,2) sparse-tilt", 1.0, 2.0),
    ("Beta(1,3) sparse-tilt", 1.0, 3.0),
    ("Beta(2,1) dense-tilt", 2.0, 1.0),
]


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Preserve examples while displaying argument defaults."""


def _positive_int(value):
    """An argparse type for positive counts."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value):
    """An argparse type for positive numerical tolerances."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _probability(value):
    """An argparse type for probabilities strictly between zero and one."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return parsed


def _finite_float(value):
    """An argparse type for a finite model parameter."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--k", nargs="+", type=_positive_int, default=[6, 12],
        help="Hypothesis counts at which to compare the candidate priors",
    )
    parser.add_argument(
        "--model", choices=sorted(MODEL_CONFIGS), default="trunc",
        help="Alternative family; theta has the meaning defined by that family",
    )
    parser.add_argument(
        "--theta", type=_finite_float, default=-2.0,
        help="Alternative parameter (Student-t degrees of freedom or Beta shape)",
    )
    parser.add_argument(
        "--alpha", type=_probability, default=0.05,
        help="Family-wise error-rate level",
    )
    parser.add_argument(
        "--n-algo-samples", type=_positive_int, default=100000,
        help="Monte Carlo samples per configuration when solving each policy",
    )
    # The released prior comparison uses the same 400,000-replication budget as
    # run_prior_objective.py because the sparse-configuration margins are small.
    parser.add_argument(
        "--n-mc", type=_positive_int, default=400000,
        help="Paired Monte Carlo replications used to evaluate each policy",
    )
    parser.add_argument(
        "--tol", type=_positive_float, default=1e-6,
        help="Secondary iterate-change tolerance",
    )
    parser.add_argument(
        "--max-iter", type=_positive_int, default=400,
        help="Maximum coordinate-descent sweeps per policy",
    )
    parser.add_argument(
        "--bisect-tol", type=_positive_float, default=1e-4,
        help="Bisection bracket width; achievable |F-alpha| is about slope times width",
    )
    parser.add_argument(
        "--f-tol", type=_positive_float, default=None,
        help="Optionally refine bisection until |F-alpha| is at most this value",
    )
    parser.add_argument(
        "--allow-uncertified", action="store_true",
        help="Exit zero despite rows that miss the residual certificate; use only for diagnostics",
    )
    parser.add_argument(
        "--out", default=paths.reproduced_result("prior_comparison.json"),
        help="JSON output path; rows are saved incrementally",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.model in {"t", "beta"} and args.theta <= 0.0:
        parser.error(f"--theta must be positive for model {args.model!r}")

    records = []
    t0 = time.time()
    for K in args.k:
        print(f"\n=== K={K}, {MODEL_CONFIGS[args.model]['name']}, "
              f"theta={args.theta} ===")
        print(f"  {'prior':<24s} {'worst':>8s} {'gain@K':>8s} {'price@K':>8s} "
              f"{'maxFWER':>8s} {'conv':>5s} {'active':>7s}")

        ref = run_one(K, args.model, args.theta, point_prior(K, K), args.alpha,
                      args.n_algo_samples, args.n_mc, tol=args.tol,
                      max_iter=args.max_iter, bisect_tol=args.bisect_tol,
                      f_tol=args.f_tol, include_floor=True)
        ref["prior_name"] = "point gamma=K (manuscript)"
        ref["price_at_K"] = 0.0
        base_at_K = ref["curves"]["SPOT"]["power"][K]
        records.append(ref)
        print(f"  {ref['prior_name']:<24s} {100*ref['min_gain']:+7.1f}% "
              f"{100*ref['gains'][-1]:+7.1f}% {0.0:+7.1f}% "
              f"{ref['max_fwer']:8.4f} {str(ref['converged']):>5s} "
              f"{ref['n_active_multipliers']:5d}/{K}")

        for name, a, b in CANDIDATES:
            rec = run_one(K, args.model, args.theta, beta_binomial_prior(K, a, b),
                          args.alpha, args.n_algo_samples, args.n_mc,
                          tol=args.tol, max_iter=args.max_iter,
                          bisect_tol=args.bisect_tol, f_tol=args.f_tol,
                          include_floor=True)
            rec["prior_name"] = name
            rec["beta_a"], rec["beta_b"] = a, b
            rec["price_at_K"] = (
                base_at_K - rec["curves"]["SPOT"]["power"][K]) / base_at_K
            records.append(rec)
            print(f"  {name:<24s} {100*rec['min_gain']:+7.1f}% "
                  f"{100*rec['gains'][-1]:+7.1f}% {100*rec['price_at_K']:+7.1f}% "
                  f"{rec['max_fwer']:8.4f} {str(rec['converged']):>5s} "
                  f"{rec['n_active_multipliers']:5d}/{K}")
            save_results(records, args.out)

    print(f"\nTotal time: {(time.time() - t0)/60:.1f} minutes")
    return _certification_status(records, args.allow_uncertified)


if __name__ == "__main__":
    raise SystemExit(main())
