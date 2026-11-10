"""
Does f_tol plus a real sweep budget certify the sparse-tilt Student-t runs?

At bisect_tol=1e-6 with max_iter=150 these reached 1.87x and 2.03x res_tol and
were still descending when the cap hit.  The priors that do certify needed 225 to
449 sweeps.  So the open question is whether they certify given both the adaptive
bracket and enough sweeps, or whether the residual floors out.

Two beta-binomial priors, checked before committing to the full prior re-run.

Use ``--help`` to inspect or reduce the exploratory solve settings without
starting a computation.
"""
import argparse
import time

import numpy as np

from general_k.algorithm import coordinate_descent_fast, default_res_tol
from general_k.prior_objective import beta_binomial_prior, prior_power_coefficients
from run_all_experiments import MODEL_CONFIGS

def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _probability(value):
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must lie strictly between 0 and 1")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--k", type=_positive_int, default=6,
                        help="Number of hypotheses")
    parser.add_argument("--df", type=_positive_float, default=6.0,
                        help="Student-t degrees of freedom")
    parser.add_argument("--alpha", type=_probability, default=0.05,
                        help="Family-wise error-rate level")
    parser.add_argument("--n-samples", type=_positive_int, default=100_000,
                        help="Fixed Monte Carlo batch size")
    parser.add_argument("--seed", type=_nonnegative_int, default=42,
                        help="Monte Carlo seed")
    parser.add_argument("--tol", type=_positive_float, default=1e-6,
                        help="Iterate-change tolerance")
    parser.add_argument("--bisect-tol", type=_positive_float, default=1e-4,
                        help="Initial bisection bracket tolerance")
    parser.add_argument("--max-iter", type=_positive_int, default=800,
                        help="Maximum coordinate-descent sweeps")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    residual_tolerance = default_res_tol(args.alpha, args.n_samples)
    g = MODEL_CONFIGS["t"]["factory"](args.df)

    print(
        f"Student-t df={args.df:g}, K={args.k}, "
        f"res_tol={residual_tolerance:.4e}",
        flush=True,
    )
    print(
        f"{'prior':<12s} {'f_tol':>9s} {'max_iter':>9s} {'reason':>9s} "
        f"{'swp':>5s} {'residual':>11s} {'x rt':>7s} {'secs':>7s}  certified",
        flush=True,
    )

    all_certified = True
    for a, b in [(1.0, 2.0), (1.0, 3.0)]:
        fn = prior_power_coefficients(beta_binomial_prior(args.k, a, b))
        t0 = time.time()
        mu, _, diagnostics = coordinate_descent_fast(
            g_func=g, K=args.k, alpha=args.alpha,
            n_samples=args.n_samples, tol=args.tol,
            max_iter=args.max_iter, seed=args.seed,
            bisect_tol=args.bisect_tol, f_tol=residual_tolerance,
            return_diagnostics=True, power_coeff_fn=fn,
        )
        all_certified = all_certified and diagnostics["converged"]
        print(
            f"Beta({a},{b})".ljust(12)
            + f" {residual_tolerance:9.2e} {args.max_iter:9d} "
            + f"{diagnostics['reason']:>9s} {diagnostics['sweeps']:5d} "
            + f"{diagnostics['final_residual']:11.4e} "
            + f"{diagnostics['final_residual']/residual_tolerance:6.2f}x "
            + f"{time.time()-t0:7.0f}  {diagnostics['converged']}",
            flush=True,
        )
        print(f"             mu = {np.round(mu, 6).tolist()}", flush=True)

    print("SPARSE_DONE", flush=True)
    return 0 if all_certified else 2


if __name__ == "__main__":
    raise SystemExit(main())
