"""
Cross-checks for the family-wise error constraint estimator.

Regression guards, not red-green cycles: they pin behaviour that is already
correct, because the analogous check was missing when the K=3 R implementation
of this same quantity shipped with all three of its constraint functions wrong
(F0 off by exactly 3! from a mishandled ordered-simplex normalisation, F1 able to
return a negative probability, F2 missing two of its three terms).  A single test
comparing the estimator against an independent route would have caught it.

Two independent routes to FWER_gamma are compared:

  A. labelled simulation -- draw gamma alternatives and K-gamma nulls, sort, apply
     the policy, count draws with at least one false rejection.  This is what the
     solver uses and it involves no simplex algebra.
  B. the ordered-simplex integral of Lemma 3.4 -- integrate sum_k b_{gamma,k} D_k
     over Q.  Sorted uniforms have density K! on Q, so the integral is
     mean(...) / K!.  This exercises the coefficients and the normalisation.
"""

import math

import numpy as np
import pytest

from general_k.alternatives import beta_g, truncated_normal_g
from general_k.coefficients import precompute_coefficients
from general_k.decision import l_star_from_precomputed
from general_k.target import prepare_samples, target_function_fast

ALPHA = 0.05
N = 200_000


def fwer_labelled(mu, gamma, g, K, seed):
    """Route A: the solver's own estimator, on labelled batches."""
    rng = np.random.default_rng(seed)
    batches = [prepare_samples(gm, g, K, N, rng) for gm in range(K)]
    ss, si, nl = batches[gamma]
    a_batch, b_coeff = precompute_coefficients(ss, g, K)
    return ALPHA + target_function_fast(
        gamma, mu[gamma], mu, K, ALPHA, si, nl, a_batch, b_coeff)


def fwer_simplex(mu, gamma, g, K, seed):
    """Route B: integrate the Lemma 3.4 coefficients over the ordered simplex."""
    rng = np.random.default_rng(seed + 1000)
    UQ = np.sort(rng.random((N, K)), axis=1)
    a_batch, b_coeff = precompute_coefficients(UQ, g, K)
    l_star = l_star_from_precomputed(mu, a_batch, b_coeff)
    rejected = np.arange(K)[None, :] < l_star[:, None]
    integrand = np.sum(b_coeff[gamma] * rejected, axis=1)
    return float(np.mean(integrand) / math.factorial(K))


@pytest.mark.parametrize("K", [3, 4])
@pytest.mark.parametrize("gamma", [0, 1])
def test_labelled_and_simplex_routes_agree(K, gamma):
    """The two routes must agree to within Monte Carlo error."""
    g = truncated_normal_g(-2.0)
    mu = np.full(K, 0.15)

    a = fwer_labelled(mu, gamma, g, K, seed=11)
    b = fwer_simplex(mu, gamma, g, K, seed=11)

    se = np.sqrt(max(a, 1e-6) * (1 - max(a, 1e-6)) / N)
    assert abs(a - b) < 6 * se + 2e-3, f"labelled={a:.5f} simplex={b:.5f}"


def test_estimator_is_a_probability():
    """
    Guards the failure mode directly: an FWER estimate must lie in [0, 1].

    The R implementation returned 6.0 and -1.2527 for this quantity.
    """
    g = beta_g(0.4)
    K = 3
    for mu_val in (0.0, 0.05, 0.2, 1.0):
        mu = np.full(K, mu_val)
        for gamma in range(K):
            f = fwer_labelled(mu, gamma, g, K, seed=5)
            assert 0.0 <= f <= 1.0, f"mu={mu_val} gamma={gamma} gave {f}"


def test_zero_multipliers_reject_everything_under_the_global_null():
    """
    At mu = 0 nothing penalises rejection, so the policy rejects all K and the
    global-null FWER is 1.  This pins the scale: a normalisation slip of K!
    shows up here immediately.
    """
    g = truncated_normal_g(-2.0)
    K = 3

    assert fwer_labelled(np.zeros(K), 0, g, K, seed=3) == pytest.approx(1.0)
