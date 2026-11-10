"""
Checks the polynomial facts behind the generalised tie-set lemma.

Supplementary Lemma S1.1 shows the tie set is Lebesgue-null by writing
``S_l - S_{l'}`` as a polynomial in the likelihood ratios ``v_j = g(u_j)`` and
exhibiting a monomial whose coefficient cannot vanish.  For the manuscript's
objective that monomial is the squarefree ``v_1 v_2 ... v_K``, whose coefficient
is ``(l - l')(K-1)!``, because every ``b_{gamma,i}`` has total degree
``gamma <= K-1`` and so cannot reach degree ``K``.

Under a prior on the number of non-nulls the objective coefficient becomes

    a_{pi,i}(v) = sum_gamma pi(gamma) (gamma-1)! (K-gamma)! v_i e_{gamma-1}(v_{-i}),

and the same argument goes through provided the coefficient of ``v_1 ... v_K``
survives.  This script checks the three facts the generalised proof needs:

1. ``a_{pi,i}`` and ``b_{gamma,i}`` are multilinear in ``v`` (degree <= 1 in
   each coordinate), so a monomial coefficient can be extracted exactly by the
   inclusion-exclusion identity
   ``c_{[K]} = sum_{S subset [K]} (-1)^{K-|S|} f(1_S)``;
2. the coefficient of ``v_1 ... v_K`` in ``a_{pi,i}`` equals ``pi(K) (K-1)!``
   for **every** position ``i`` -- so it is non-zero exactly when the prior puts
   mass on the all-alternatives configuration;
3. the coefficient of ``v_1 ... v_K`` in every ``b_{gamma,i}``, gamma <= K-1,
   is zero.

Together these give: the coefficient of ``v_1 ... v_K`` in ``S_l - S_{l'}`` is
``(l - l') pi(K) (K-1)!``, non-zero whenever ``pi(K) > 0``.

Usage:
    python verify_tie_set_polynomial.py
"""

import argparse
import itertools
import math

import numpy as np

from general_k.coefficients import elem_sym_poly
from general_k.prior_objective import (
    beta_binomial_prior, point_prior, uniform_prior,
)


def a_pi(v, i, prior, K):
    """a_{pi,i}(v), the prior-weighted power coefficient; i is 1-indexed."""
    others = np.delete(v, i - 1)
    total = 0.0
    for gamma in range(1, K + 1):
        if prior[gamma] == 0.0:
            continue
        coeff = math.factorial(gamma - 1) * math.factorial(K - gamma)
        total += prior[gamma] * coeff * v[i - 1] * elem_sym_poly(others, gamma - 1)
    return total


def b_coef(v, gamma, i, K):
    """b_{gamma,i}(v), the FWER coefficient; i is 1-indexed, 0 <= gamma <= K-1."""
    esp_index = gamma - i + 1
    n_tail = K - i
    if gamma < i - 1 or esp_index > n_tail:
        return 0.0
    prefix = float(np.prod(v[: i - 1])) if i > 1 else 1.0
    tail = v[i:]
    return (math.factorial(gamma) * math.factorial(K - gamma)
            * prefix * elem_sym_poly(tail, esp_index))


def top_monomial_coefficient(f, K):
    """
    Coefficient of v_1 v_2 ... v_K in a multilinear f, by inclusion-exclusion.

    For multilinear f, sum_{S subset [K]} (-1)^{K-|S|} f(1_S) isolates exactly
    the coefficient of the full squarefree monomial.
    """
    total = 0.0
    for r in range(K + 1):
        for S in itertools.combinations(range(K), r):
            v = np.zeros(K)
            v[list(S)] = 1.0
            total += (-1) ** (K - r) * f(v)
    return total


def is_multilinear(f, K, rng, tol=1e-9):
    """f is affine in each coordinate separately: check the second difference."""
    for j in range(K):
        v = rng.uniform(0.5, 2.0, size=K)
        xs = [0.3, 1.1, 2.7]
        ys = []
        for x in xs:
            w = v.copy()
            w[j] = x
            ys.append(f(w))
        # Affine in w[j] <=> the divided second difference vanishes.
        d1 = (ys[1] - ys[0]) / (xs[1] - xs[0])
        d2 = (ys[2] - ys[1]) / (xs[2] - xs[1])
        scale = max(1.0, abs(d1), abs(d2))
        if abs(d1 - d2) > tol * scale:
            return False
    return True


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
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
    parser.add_argument(
        "--k", nargs="+", type=_positive_int, default=[2, 3, 4, 5, 6],
        help="Hypothesis counts to verify",
    )
    parser.add_argument(
        "--seed", type=_nonnegative_int, default=0,
        help="Random seed for the multilinearity probes",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)
    print("Checking the tie-set polynomial facts.\n")

    ok = True
    for K in args.k:
        priors = [
            ("point gamma=K", point_prior(K, K)),
            ("uniform", uniform_prior(K)),
            ("Jeffreys Beta(.5,.5)", beta_binomial_prior(K, 0.5, 0.5)),
            ("Beta(2,1)", beta_binomial_prior(K, 2.0, 1.0)),
            ("point gamma=1", point_prior(K, 1)),
        ]
        for name, prior in priors:
            expected = prior[K] * math.factorial(K - 1)
            for i in range(1, K + 1):
                f = lambda v, i=i, prior=prior: a_pi(v, i, prior, K)
                if not is_multilinear(f, K, rng):
                    print(f"  K={K} {name} i={i}: a_pi NOT multilinear")
                    ok = False
                got = top_monomial_coefficient(f, K)
                if abs(got - expected) > 1e-8 * max(1.0, abs(expected)):
                    print(f"  K={K} {name} i={i}: coeff {got:.6g} != "
                          f"pi(K)(K-1)! = {expected:.6g}")
                    ok = False
            print(f"  K={K:<2d} {name:<22s} coeff of v_1...v_K in a_pi,i = "
                  f"{expected:.6g} for every i   "
                  f"({'non-zero' if expected > 0 else 'ZERO -- argument fails'})")

        # b coefficients can never reach the full squarefree monomial.
        worst = 0.0
        for gamma in range(K):
            for i in range(1, K + 1):
                f = lambda v, gamma=gamma, i=i: b_coef(v, gamma, i, K)
                if not is_multilinear(f, K, rng):
                    print(f"  K={K} b_{gamma},{i}: NOT multilinear")
                    ok = False
                worst = max(worst, abs(top_monomial_coefficient(f, K)))
        print(f"  K={K:<2d} {'all b_{gamma,i}, gamma<=K-1':<22s} "
              f"max |coeff of v_1...v_K| = {worst:.3g}\n")
        ok = ok and worst < 1e-9

    print("All facts hold." if ok else "A check FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
