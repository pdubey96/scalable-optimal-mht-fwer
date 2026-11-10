"""
Numerical check that global monotonicity survives the prior-weighted objective.

The manuscript's global monotonicity theorem states that each constraint function
``F_gamma(mu_gamma; mu_{-gamma})`` is non-increasing in its own multiplier.  The
proof is written for the objective's power coefficient ``a(u)``, which at
gamma = K is constant in the position ``k``.  Under a prior on the number of
non-nulls the coefficient becomes position-dependent (general_k.prior_objective),
so the theorem has to be re-established -- this script checks whether the
property still holds numerically before anyone invests in a proof.

For each configuration gamma, mu_gamma is swept over a grid with the other
multipliers held at their solved values, and the realised family-wise error rate
is recomputed on a fixed sample batch (common random numbers, so the curve is
not contaminated by Monte Carlo noise between grid points).

Usage:
    python check_prior_monotonicity.py
    python check_prior_monotonicity.py --k 6 --theta -2.0 --grid 40
"""

import argparse

import numpy as np

from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import truncated_normal_g, student_t_g
from general_k.coefficients import precompute_coefficients
from general_k.decision import l_star_from_precomputed
from general_k.prior_objective import (
    binomial_prior, point_prior, prior_power_coefficients, uniform_prior,
)
from general_k.gamma_sweep import sample_configuration


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Preserve examples while displaying argument defaults."""


def _int_at_least(minimum):
    """Build an argparse integer type with a stated lower bound."""
    def parse(value):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return parsed
    return parse


def _positive_float(value):
    """An argparse type for positive numerical bounds."""
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


def _finite_float(value):
    """An argparse type for a finite model parameter."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--k", type=_int_at_least(1), default=6,
        help="Number of hypotheses",
    )
    parser.add_argument(
        "--theta", type=_finite_float, default=-2.0,
        help="Location parameter for the truncated-normal alternative",
    )
    parser.add_argument(
        "--alpha", type=_probability, default=0.05,
        help="Family-wise error-rate level",
    )
    parser.add_argument(
        "--grid", type=_int_at_least(2), default=40,
        help="Number of multiplier values in each monotonicity sweep",
    )
    parser.add_argument(
        "--mu-max", type=_positive_float, default=6.0,
        help="Upper endpoint of each multiplier grid",
    )
    parser.add_argument(
        "--n-mc", type=_int_at_least(1), default=50000,
        help="Fixed-batch Monte Carlo samples per FWER curve",
    )
    parser.add_argument(
        "--n-algo-samples", type=_int_at_least(1), default=100000,
        help="Monte Carlo samples per configuration when solving each policy",
    )
    return parser


def fwer_curve(mu_base, gamma, grid, g_func, K, power_coeff_fn, n_mc, seed):
    """Realised FWER at configuration gamma as mu_gamma sweeps the grid."""
    rng = np.random.default_rng(seed)
    sorted_p, sort_indices = sample_configuration(gamma, g_func, K, n_mc, rng)
    _, b_coeff = precompute_coefficients(sorted_p, g_func, K)
    a_batch = power_coeff_fn(sorted_p, g_func, K)
    is_null = sort_indices >= gamma

    out = []
    for value in grid:
        mu = mu_base.copy()
        mu[gamma] = value
        n_rej = l_star_from_precomputed(mu, a_batch, b_coeff)
        rejected = np.arange(K)[None, :] < n_rej[:, None]
        out.append(float(np.mean(np.any(rejected & is_null, axis=1))))
    return np.array(out)


def _mc_standard_error(alpha, n_mc):
    """Binomial standard error at the requested FWER level."""
    return float(np.sqrt(alpha * (1.0 - alpha) / n_mc))


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    K, alpha = args.k, args.alpha
    families = [
        (f"truncated normal, theta={args.theta}", truncated_normal_g(args.theta)),
        ("Student-t, 6 df", student_t_g(6)),
    ]
    priors = [
        ("point gamma=K", point_prior(K, K)),
        ("uniform", uniform_prior(K)),
        ("binomial p=0.5", binomial_prior(K, 0.5)),
        ("binomial p=0.8", binomial_prior(K, 0.8)),
    ]
    grid = np.linspace(0.0, args.mu_max, args.grid)

    print(f"K={K}, alpha={alpha}, grid of {args.grid} points on [0, {args.mu_max}]")
    print("Reporting the largest increase of F_gamma along the grid; exact "
          "monotonicity would give 0, and Monte Carlo ties give small positive "
          "values.\n")

    worst_overall = 0.0
    for fam_name, g_func in families:
        for prior_name, prior in priors:
            fn = prior_power_coefficients(prior)
            mu, _, diag = coordinate_descent_fast(
                g_func=g_func, K=K, alpha=alpha, n_samples=args.n_algo_samples,
                tol=1e-3, max_iter=400, seed=42, return_diagnostics=True,
                power_coeff_fn=fn,
            )
            worst, worst_gamma = 0.0, None
            for gamma in range(K):
                curve = fwer_curve(mu, gamma, grid, g_func, K, fn,
                                   args.n_mc, seed=5000 + gamma)
                rise = float(np.max(np.diff(curve)))
                if rise > worst:
                    worst, worst_gamma = rise, gamma
            worst_overall = max(worst_overall, worst)
            print(f"  {fam_name:<32s} {prior_name:<16s} "
                  f"converged={str(diag['converged']):<5s} "
                  f"max rise {worst:+.5f} (gamma={worst_gamma})")

    mc_se = _mc_standard_error(args.alpha, args.n_mc)
    print(f"\nLargest rise anywhere: {worst_overall:+.5f}")
    print(f"Monte Carlo standard error at n={args.n_mc}: {mc_se:.5f}")
    consistent = worst_overall < 2 * mc_se
    print("Monotonicity is consistent with the data."
          if consistent else
          "A rise exceeds two standard errors -- inspect before proceeding.")
    return 0 if consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
