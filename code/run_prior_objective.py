"""
SPOT under a prior on the number of non-null hypotheses.

The point-prior special case uses the objective Pi_K, the average power when all
K hypotheses are false.  general_k.prior_objective supplies the power
coefficient for any prior pi on gamma, leaving the family-wise error constraints
untouched.  This script solves the problem under several priors and evaluates
each resulting policy at every configuration, so the trade can be read off
directly:

  - how much power a prior-weighted policy gives up at gamma = K, against
  - how much it gains at the configurations the manuscript's policy ignores.

Usage:
    python run_prior_objective.py
    python run_prior_objective.py --k 6                 # restrict K
    python run_prior_objective.py --quick               # certifying K=3 smoke test
"""

import argparse
import sys
import time

import numpy as np

from general_k.algorithm import coordinate_descent_fast
from general_k.gamma_sweep import COMPARATORS, gamma_sweep
from general_k.prior_objective import (
    binomial_prior, point_prior, prior_power_coefficients, uniform_prior,
)
from general_k import paths
from run_all_experiments import MODEL_CONFIGS, save_results


SETTINGS = [
    ("trunc", -2.0, 3),
    ("trunc", -2.0, 6),
    ("trunc", -2.0, 12),
    ("trunc", -3.0, 6),
    ("t", 6, 6),
    ("beta", 0.4, 6),
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
    if not np.isfinite(parsed) or parsed <= 0.0:
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


def _build_parser():
    available_models = sorted({model for model, _, _ in SETTINGS})
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--k", nargs="+", type=_positive_int, default=None,
        help="Restrict the predefined study to these hypothesis counts",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, choices=available_models,
        help="Restrict the predefined study to these alternative families",
    )
    parser.add_argument(
        "--alpha", type=_probability, default=0.05,
        help="Family-wise error-rate level",
    )
    parser.add_argument(
        "--n-algo-samples", type=_positive_int, default=100000,
        help="Monte Carlo samples per configuration when solving each policy",
    )
    # The margins here are sub-percent at gamma = 1, so the released analysis
    # uses 400,000 rather than the 50,000 replications used elsewhere.
    parser.add_argument(
        "--n-mc", type=_positive_int, default=400000,
        help="Paired Monte Carlo replications used to evaluate each policy",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use 20,000 samples; without filters, run the certifying K=3 "
             "truncated-normal slice only",
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
        "--out", default=argparse.SUPPRESS,
        help="JSON output path; defaults to reproduced/results/prior_objective.json, "
             "or prior_objective_quick.json with --quick",
    )
    return parser


def _parse_args(argv=None):
    """Parse arguments and apply the documented smoke-test preset."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.quick:
        args.n_algo_samples, args.n_mc = 20000, 20000
        if args.k is None and args.models is None:
            # This four-prior slice is fast and all four rows meet the residual
            # certificate at the quick Monte Carlo size. Explicit filters opt
            # into the requested broader diagnostic run.
            args.k = [3]
            args.models = ["trunc"]
    if not hasattr(args, "out"):
        filename = (
            "prior_objective_quick.json"
            if args.quick else "prior_objective.json"
        )
        args.out = paths.reproduced_result(filename)
    return parser, args


def priors_for(K):
    """The four objective priors compared in the manuscript."""
    return [
        ("point gamma=K", point_prior(K, K)),      # the manuscript's objective
        ("uniform", uniform_prior(K)),
        ("binomial p=0.5", binomial_prior(K, 0.5)),
        ("binomial p=0.8", binomial_prior(K, 0.8)),
    ]


def rel_gain(curves, gamma):
    """SPOT's relative power gain over the strongest comparator at gamma."""
    best = max(curves[n]["power"][gamma] for n in COMPARATORS)
    return (curves["SPOT"]["power"][gamma] - best) / best


def _certification_status(records, allow_uncertified=False):
    """Report non-certifying rows and return the appropriate process status."""
    uncertified = [record for record in records if not record["converged"]]
    if not uncertified:
        return 0

    print(
        f"\nWARNING: {len(uncertified)} of {len(records)} requested rows did "
        "not meet the residual certificate:",
        file=sys.stderr,
    )
    for record in uncertified:
        print(
            f"  model={record['model']} theta={record['theta']:g} "
            f"K={record['K']} prior={record['prior_name']!r}: "
            f"reason={record['reason']}, residual="
            f"{record['final_residual']:.3e}",
            file=sys.stderr,
        )
    if not allow_uncertified:
        print(
            "Results were saved for diagnosis. Re-run with tighter solver "
            "settings, or pass --allow-uncertified to accept a diagnostic "
            "non-certifying run.",
            file=sys.stderr,
        )
        return 2
    print(
        "Continuing with exit status 0 because --allow-uncertified was set.",
        file=sys.stderr,
    )
    return 0


def run_one(K, model_key, theta, prior, alpha, n_algo_samples, n_mc, seed=42,
            tol=1e-3, max_iter=400, bisect_tol=1e-4, f_tol=None,
            include_floor=False):
    """
    Solve under one prior and evaluate at every configuration.

    There are two distinct stopping tolerances and they fail in different ways.

    ``tol`` is the iterate-change threshold.  Too loose and the recursion stops
    while the residual is still falling; the truncated normal tolerates 1e-3 but
    near-null alternatives such as the Student-t need 1e-6 with a larger
    ``max_iter``.

    ``bisect_tol`` is the bisection bracket width.  Too coarse and no iterate can
    reach the residual threshold at all, because the achievable |F - alpha| is
    about slope * bisect_tol: the recursion then settles at an exact fixed point
    of the discretised map and reports ``delta`` with a large residual.  Sparse
    tilted priors trigger this by activating coordinate gamma=1, where the target
    can be far steeper than anywhere the other priors bind.  Tightening ``tol``
    does not help here -- it is the wrong knob.

    ``f_tol`` is the fix for that case: pass the residual tolerance and the
    bisection refines past ``bisect_tol`` until |F - alpha| <= f_tol.  It is
    strictly additive, so it cannot perturb a coordinate that was already
    resolved finely enough.
    """
    g_func = MODEL_CONFIGS[model_key]["factory"](theta)
    fn = prior_power_coefficients(prior)

    t0 = time.time()
    mu, _, diag = coordinate_descent_fast(
        g_func=g_func, K=K, alpha=alpha, n_samples=n_algo_samples,
        tol=tol, max_iter=max_iter, seed=seed, return_diagnostics=True,
        power_coeff_fn=fn, bisect_tol=bisect_tol, f_tol=f_tol,
    )
    solve_seconds = time.time() - t0

    curves = gamma_sweep(mu, g_func, K, alpha, n_mc=n_mc, seed=seed + 1,
                         power_coeff_fn=fn)

    gains = [rel_gain(curves, gamma) for gamma in range(1, K + 1)]
    record = {
        "K": K, "model": model_key, "theta": theta, "alpha": alpha,
        # Recorded because the margins this driver reports are sub-percent at
        # sparse configurations, so which evaluation budget produced a row is
        # part of the row.
        "n_algo_samples": n_algo_samples, "n_mc": n_mc,
        "prior": prior.tolist(),
        "mu": mu.tolist(),
        "n_active_multipliers": int(np.sum(np.asarray(mu) > 1e-6)),
        "converged": diag["converged"],
        # Carried through so an uncertified row says why it is uncertified.
        # Without this the diagnostic exists in the solver and never reaches
        # the file anyone reads, which is how the Student-t rows stayed
        # mysterious for as long as they did.
        "reason": diag["reason"],
        "sweeps": diag["sweeps"], "final_residual": diag["final_residual"],
        "solve_seconds": solve_seconds,
        "curves": curves,
        "gains": gains,
        "min_gain": float(min(gains)),
        "max_fwer": float(max(curves["SPOT"]["fwer"][:K])),
    }
    # The released prior-comparison files include the detailed bracket-floor
    # diagnostic; the prior-objective file predates that field.  Make the
    # distinction explicit so a full reproduction retains both exact schemas.
    if include_floor:
        record["floor"] = diag.get("floor")
    return record


def main(argv=None):
    parser, args = _parse_args(argv)

    settings = [
        (m, th, K) for (m, th, K) in SETTINGS
        if (args.k is None or K in args.k)
        and (args.models is None or m in args.models)
    ]
    if not settings:
        parser.error(
            "the requested --k/--models filters match no predefined setting"
        )

    records = []
    total_t0 = time.time()
    for model_key, theta, K in settings:
        label = MODEL_CONFIGS[model_key]["theta_label"]
        print(f"\n=== K={K}, {MODEL_CONFIGS[model_key]['name']}, "
              f"{label}={theta} ===")
        baseline_at_K = None
        for prior_name, prior in priors_for(K):
            rec = run_one(K, model_key, theta, prior, args.alpha,
                          args.n_algo_samples, args.n_mc,
                          tol=args.tol, max_iter=args.max_iter,
                          bisect_tol=args.bisect_tol, f_tol=args.f_tol)
            rec["prior_name"] = prior_name
            if prior_name == "point gamma=K":
                baseline_at_K = rec["curves"]["SPOT"]["power"][K]
            rec["price_at_K"] = (
                None if baseline_at_K is None else
                (baseline_at_K - rec["curves"]["SPOT"]["power"][K]) / baseline_at_K
            )
            records.append(rec)

            gains = " ".join(f"{100 * x:+6.1f}" for x in rec["gains"])
            price = ("    --" if rec["price_at_K"] is None
                     else f"{100 * rec['price_at_K']:+5.1f}%")
            print(f"  {prior_name:<15s} conv={str(rec['converged']):<5s} "
                  f"active={rec['n_active_multipliers']}/{K}  "
                  f"maxFWER={rec['max_fwer']:.4f}  "
                  f"price@K={price}  worst={100 * rec['min_gain']:+6.1f}%")
            print(f"  {'':15s} gain vs best baseline by gamma: {gains}")
            save_results(records, args.out)

    print(f"\nTotal time: {(time.time() - total_t0)/60:.1f} minutes")

    return _certification_status(records, args.allow_uncertified)


if __name__ == "__main__":
    raise SystemExit(main())
