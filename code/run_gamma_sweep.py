"""
Power as a function of the number of non-null hypotheses.

SPOT is tuned once at the all-alternatives configuration (gamma = K), then
evaluated at every configuration gamma = 1, ..., K, so the curves show how its
advantage changes as the family stops being signal-rich.

The multipliers are computed once per (K, model, theta) setting and reused
across gamma; every method sees the same p-value matrix at each gamma.

Usage:
    python run_gamma_sweep.py                       # default grid
    python run_gamma_sweep.py --k 3 6               # restrict K
    python run_gamma_sweep.py --models trunc t      # restrict models
    python run_gamma_sweep.py --quick               # small MC, for a smoke test
"""

import argparse
import sys
import time

from general_k.algorithm import coordinate_descent_fast
from general_k.gamma_sweep import COMPARATORS, gamma_sweep
from general_k import paths
from run_all_experiments import MODEL_CONFIGS, save_results


# The grid: the scaling study's headline setting across K, then robustness
# checks across alternative families and signal strengths at K = 6.
DEFAULT_SETTINGS = [
    ("trunc", -2.0, [3, 4, 5, 6, 8, 10, 12]),
    ("mixture", -2.0, [6]),
    ("t", 6, [6]),
    ("beta", 0.4, [6]),
    ("trunc", -1.0, [6]),
    ("trunc", -3.0, [6]),
    ("trunc", -4.0, [6]),
]


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Preserve examples while displaying argument defaults."""


def _positive_int(value):
    """An argparse type for positive sample sizes and hypothesis counts."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
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
    available_models = sorted({model for model, _, _ in DEFAULT_SETTINGS})
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--k", nargs="+", type=_positive_int, default=None,
        help="Restrict the predefined sweep to these hypothesis counts",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, choices=available_models,
        help="Restrict the predefined sweep to these alternative families",
    )
    parser.add_argument(
        "--alpha", type=_probability, default=0.05,
        help="Family-wise error-rate level",
    )
    parser.add_argument(
        "--n-algo-samples", type=_positive_int, default=100000,
        help="Monte Carlo samples per configuration when solving each policy",
    )
    # The crossover this driver locates sits where SPOT and Romano-Wolf are
    # within a few tenths of a percent, which 50,000 replications do not separate.
    parser.add_argument(
        "--n-mc", type=_positive_int, default=400000,
        help="Paired Monte Carlo replications used to evaluate each policy",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use 20,000 optimization and evaluation samples for a smoke test",
    )
    parser.add_argument(
        "--allow-uncertified", action="store_true",
        help="Exit zero despite solves that miss the residual certificate; diagnostic use only",
    )
    parser.add_argument(
        "--out", default=argparse.SUPPRESS,
        help="JSON output path; defaults to reproduced/results/gamma_sweep.json, "
             "or gamma_sweep_quick.json with --quick",
    )
    return parser


def _parse_args(argv=None):
    """Parse arguments and select a mode-specific, non-reference output path."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "out"):
        filename = "gamma_sweep_quick.json" if args.quick else "gamma_sweep.json"
        args.out = paths.reproduced_result(filename)
    return parser, args


def run_setting(K, model_key, theta, alpha, n_algo_samples, n_mc, seed=42):
    """Solve for mu at gamma = K, then sweep gamma = 0, ..., K."""
    g_func = MODEL_CONFIGS[model_key]["factory"](theta)

    t0 = time.time()
    mu_star, _, diagnostics = coordinate_descent_fast(
        g_func=g_func, K=K, alpha=alpha, n_samples=n_algo_samples,
        tol=1e-3, max_iter=400, seed=seed, return_diagnostics=True,
    )
    solve_seconds = time.time() - t0

    t1 = time.time()
    curves = gamma_sweep(mu_star, g_func, K, alpha, n_mc=n_mc, seed=seed + 1)
    sweep_seconds = time.time() - t1

    return {
        "K": K,
        "model": model_key,
        "theta": theta,
        "alpha": alpha,
        "n_algo_samples": n_algo_samples,
        "n_mc": n_mc,
        "mu": mu_star.tolist(),
        "converged": diagnostics["converged"],
        "sweeps": diagnostics["sweeps"],
        "final_residual": diagnostics["final_residual"],
        "solve_seconds": solve_seconds,
        "sweep_seconds": sweep_seconds,
        "curves": curves,
    }


def summarise(record):
    """Compact per-gamma comparison of SPOT against the strongest baseline."""
    K = record["K"]
    curves = record["curves"]
    spot = curves["SPOT"]["power"]

    lines = []
    header = f"  gamma  SPOT     Hommel   best-baseline  rel.gain  SPOT FWER"
    lines.append(header)
    for gamma in range(1, K + 1):
        best_name, best_val = None, -1.0
        for name in COMPARATORS:
            val = curves[name]["power"][gamma]
            if val > best_val:
                best_name, best_val = name, val
        hommel = curves["Hommel"]["power"][gamma]
        gain = (spot[gamma] - best_val) / best_val if best_val > 0 else float("nan")
        fwer = curves["SPOT"]["fwer"][gamma] if gamma < K else 0.0
        lines.append(
            f"  {gamma:>5d}  {spot[gamma]:.4f}   {hommel:.4f}   "
            f"{best_val:.4f} ({best_name[:4]})  {gain:+7.1%}   {fwer:.4f}"
        )
    return "\n".join(lines)


def crossover(record):
    """Smallest gamma at which SPOT still beats every baseline; None if never."""
    K = record["K"]
    curves = record["curves"]
    for gamma in range(1, K + 1):
        best = max(curves[n]["power"][gamma] for n in COMPARATORS)
        if curves["SPOT"]["power"][gamma] >= best:
            return gamma
    return None


def _certification_status(records, allow_uncertified=False):
    """Report non-certifying policies and return a process exit status."""
    failed = [record for record in records if not record["converged"]]
    if not failed:
        return 0
    print(
        f"\nWARNING: {len(failed)} of {len(records)} policies did not meet "
        "the residual certificate:",
        file=sys.stderr,
    )
    for record in failed:
        print(
            f"  model={record['model']} theta={record['theta']:g} "
            f"K={record['K']}: residual={record['final_residual']:.3e}",
            file=sys.stderr,
        )
    if allow_uncertified:
        print(
            "Continuing with exit status 0 because --allow-uncertified was set.",
            file=sys.stderr,
        )
        return 0
    print(
        "Results were saved for diagnosis; pass --allow-uncertified only if "
        "non-certifying output is intentionally acceptable.",
        file=sys.stderr,
    )
    return 2


def main(argv=None):
    parser, args = _parse_args(argv)

    if args.quick:
        args.n_algo_samples, args.n_mc = 20000, 20000

    settings = []
    for model_key, theta, k_values in DEFAULT_SETTINGS:
        if args.models is not None and model_key not in args.models:
            continue
        for K in k_values:
            if args.k is not None and K not in args.k:
                continue
            settings.append((K, model_key, theta))

    if not settings:
        parser.error(
            "the requested --k/--models filters match no predefined setting"
        )

    print(f"{len(settings)} setting(s); "
          f"n_algo_samples={args.n_algo_samples}, n_mc={args.n_mc}\n")

    records = []
    total_t0 = time.time()
    for i, (K, model_key, theta) in enumerate(settings, 1):
        label = MODEL_CONFIGS[model_key]["theta_label"]
        print(f"[{i}/{len(settings)}] K={K}, {MODEL_CONFIGS[model_key]['name']}, "
              f"{label}={theta}")
        record = run_setting(K, model_key, theta, args.alpha,
                             args.n_algo_samples, args.n_mc)
        record["crossover_gamma"] = crossover(record)
        records.append(record)
        print(summarise(record))
        print(f"  [solve {record['solve_seconds']:.1f}s, "
              f"sweep {record['sweep_seconds']:.1f}s, "
              f"converged={record['converged']}, "
              f"SPOT leads from gamma={record['crossover_gamma']}]\n")
        save_results(records, args.out)

    print(f"Total time: {(time.time() - total_t0)/60:.1f} minutes")
    return _certification_status(records, args.allow_uncertified)


if __name__ == "__main__":
    raise SystemExit(main())
