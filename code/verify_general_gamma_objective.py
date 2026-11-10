"""
Numerical check that the general-gamma objective is an elementary symmetric
polynomial, exactly like the FWER constraint coefficients.

Why this matters
----------------
For the point prior at gamma = K, the objective is Pi_K and its power
coefficient is constant in k:

    a(u) = (K-1)! * prod_{j=1}^{K} g(u_j).

A prior over the number of non-nulls instead needs the power coefficient at a
general configuration h_gamma.  Fix a symmetric policy D and an alternative set
S with |S| = gamma.  The density is prod_{j in S} g(u_j), so by symmetry of D
the expected number of correct rejections is

    E_{h_gamma}[ sum_{k in S} D_k ]
        = C(K,gamma)^{-1} E_{u ~ U[0,1]^K} [ sum_k D_k(u) g(u_k)
                                             e_{gamma-1}( g(u_{-k}) ) ],       (*)

where e_m is the m-th elementary symmetric polynomial and g(u_{-k}) drops
coordinate k.  So the power coefficient at position k under h_gamma is

    a_{gamma,k}(u)  proportional to  g(u_k) * e_{gamma-1}( g(u_{-k}) ),

which is the same ESP machinery already used for b_{l,k}, and which collapses to
the manuscript's constant a(u) when gamma = K.

This script checks (*) by Monte Carlo: the right-hand side, evaluated by drawing
all K p-values from the *uniform* null and reweighting, is compared against the
direct simulation used in the gamma sweep.  Agreement within Monte Carlo error
confirms the representation.

Usage:
    python verify_general_gamma_objective.py
    python verify_general_gamma_objective.py --k 6 --theta -1.5 --n 400000
"""

import argparse
import math

import numpy as np

from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import truncated_normal_g
from general_k.coefficients import elem_sym_poly_batch
from general_k.decision import l_star_batch
from general_k.gamma_sweep import gamma_sweep


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


def _nonnegative_int(value):
    """An argparse type for NumPy-compatible random seeds."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--k", type=_int_at_least(1), default=5,
        help="Number of hypotheses",
    )
    parser.add_argument(
        "--theta", type=_finite_float, default=-1.5,
        help="Location parameter for the truncated-normal alternative",
    )
    parser.add_argument(
        "--alpha", type=_probability, default=0.05,
        help="Family-wise error-rate level",
    )
    parser.add_argument(
        "--n-algo-samples", type=_int_at_least(1), default=100000,
        help="Monte Carlo samples per configuration when solving the policy",
    )
    parser.add_argument(
        "--n", type=_int_at_least(2), default=400000,
        help="Monte Carlo draws for the ESP-reweighted estimator",
    )
    parser.add_argument(
        "--n-direct", type=_int_at_least(2), default=200000,
        help="Monte Carlo draws for the direct simulation",
    )
    parser.add_argument(
        "--seed", type=_nonnegative_int, default=2026,
        help="Base random seed for both estimators",
    )
    return parser


def esp_power_weight(rejected, g_vals, K, gamma):
    """
    Per-sample summand of the right-hand side of (*).

        C(K,gamma)^{-1} gamma^{-1} sum_k D_k g(u_k) e_{gamma-1}( g(u_{-k}) )

    Parameters
    ----------
    rejected : (n, K) boolean, in the original coordinate order
    g_vals   : (n, K) values of g at the same coordinates
    """
    n = g_vals.shape[0]
    total = np.zeros(n)
    for k in range(K):
        others = np.delete(g_vals, k, axis=1)               # (n, K-1)
        esp = elem_sym_poly_batch(others, gamma - 1)        # (n,)
        total += rejected[:, k] * g_vals[:, k] * esp

    return total / (math.comb(K, gamma) * gamma)


def esp_reweighted_power(mu, g_func, K, gamma, n, seed):
    """
    Right-hand side of (*): draw u iid uniform, reweight by the ESP coefficient.

    Returns
    -------
    (estimate of Pi_gamma, Monte Carlo standard error)
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=(n, K))

    sort_indices = np.argsort(u, axis=1)
    u_sorted = np.take_along_axis(u, sort_indices, axis=1)
    l_star = l_star_batch(mu, u_sorted, g_func, K)

    # Rejections back in the original coordinate order: column j is rejected
    # iff its rank among the row is below l*.
    ranks = np.empty((n, K), dtype=int)
    np.put_along_axis(ranks, sort_indices, np.tile(np.arange(K), (n, 1)), axis=1)
    rejected = ranks < l_star[:, None]                      # (n, K)

    values = esp_power_weight(rejected, g_func(u), K, gamma)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n))


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    K, alpha = args.k, args.alpha
    g_func = truncated_normal_g(args.theta)

    mu, _ = coordinate_descent_fast(
        g_func=g_func, K=K, alpha=alpha, n_samples=args.n_algo_samples,
        tol=1e-3, max_iter=400, seed=42,
    )

    direct = gamma_sweep(mu, g_func, K, alpha, n_mc=args.n_direct,
                         seed=args.seed, methods=[])

    print(f"Truncated normal, theta={args.theta}, K={K}, alpha={alpha}")
    print(f"direct n={args.n_direct}, ESP-reweighted n={args.n}\n")
    print("  gamma   direct Pi_gamma   ESP formula (se)      difference   z")
    ok = True
    for gamma in range(1, K + 1):
        d = direct["SPOT"]["power"][gamma]
        e, se_e = esp_reweighted_power(mu, g_func, K, gamma, args.n,
                                       args.seed + gamma)
        se_d = math.sqrt(max(d * (1 - d), 1e-12) / args.n_direct)
        se = math.sqrt(se_d**2 + se_e**2)
        z = (e - d) / se if se > 0 else float("nan")
        flag = "" if abs(z) < 4 else "   <-- MISMATCH"
        ok = ok and abs(z) < 4
        print(f"  {gamma:>5d}   {d:.5f}           {e:.5f} ({se_e:.5f})   "
              f"{e - d:+.5f}    {z:+.2f}{flag}")

    print("\n" + ("All gamma agree within Monte Carlo error."
                  if ok else "At least one gamma disagrees."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
