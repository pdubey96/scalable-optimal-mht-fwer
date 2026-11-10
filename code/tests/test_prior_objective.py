"""
Tests for general_k.prior_objective.

Covers:
- priors are proper distributions on gamma = 1, ..., K
- the gamma = K point prior reproduces the manuscript's power coefficient
- closed forms at gamma = 1 and gamma = K
- linearity of the prior-weighted coefficient in the prior
- the solver hook leaves the default path untouched
- the coefficient representation of Pi_pi matches direct simulation
- policies solved under a prior still control the family-wise error rate
"""

import math

import numpy as np
import pytest

from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import truncated_normal_g
from general_k.coefficients import precompute_coefficients
from general_k.gamma_sweep import gamma_sweep
from general_k.prior_objective import (
    beta_binomial_prior,
    binomial_prior,
    gamma_power_coefficients,
    objective_from_coefficients,
    point_prior,
    prior_power_coefficients,
    uniform_prior,
)


def _sorted_batch(K, n=300, seed=1):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, 1.0, size=(n, K)), axis=1)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

class TestPriors:
    @pytest.mark.parametrize("K", [2, 3, 6, 12])
    def test_priors_are_distributions_on_one_to_K(self, K):
        for prior in (uniform_prior(K), binomial_prior(K, 0.5),
                      binomial_prior(K, 0.8), point_prior(K, K)):
            assert prior.shape == (K + 1,)
            assert prior[0] == 0.0            # the global null carries no power
            assert prior.sum() == pytest.approx(1.0)
            assert np.all(prior >= 0.0)

    def test_binomial_prior_is_conditioned_not_truncated(self):
        """Mass is renormalised over gamma >= 1, not merely dropped."""
        K, p = 5, 0.3
        prior = binomial_prior(K, p)
        raw = np.array([math.comb(K, g) * p**g * (1 - p)**(K - g)
                        for g in range(1, K + 1)])
        assert np.allclose(prior[1:], raw / raw.sum())

    def test_point_prior_places_all_mass_at_gamma(self):
        prior = point_prior(6, 4)
        assert prior[4] == 1.0
        assert prior.sum() == 1.0

    @pytest.mark.parametrize("K", [2, 3, 6, 12, 21])
    def test_flat_beta_induces_the_uniform_prior(self, K):
        """
        Beta(1, 1) on the non-null rate induces a uniform prior on gamma.  This
        is why uniform_prior is the Bayes prior under a flat rate rather than an
        arbitrary default, so the identity is worth pinning.
        """
        assert np.allclose(beta_binomial_prior(K, 1.0, 1.0), uniform_prior(K),
                           rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("a", [0.25, 0.5, 2.0])
    def test_symmetric_beta_is_symmetric_before_the_global_null_is_dropped(self, a):
        """
        For a = b the beta-binomial pmf satisfies pmf(gamma) = pmf(K - gamma).
        Zeroing gamma = 0 and renormalising leaves that symmetry intact on
        1 <= gamma <= K-1, while gamma = K keeps the mass its partner lost.
        """
        K = 6
        prior = beta_binomial_prior(K, a, a)
        for gamma in range(1, K):
            assert prior[gamma] == pytest.approx(prior[K - gamma])

    def test_jeffreys_is_u_shaped(self):
        """Jeffreys' Beta(1/2, 1/2) puts more mass at the extremes than in the
        middle -- the property that makes it a candidate here."""
        K = 12
        prior = beta_binomial_prior(K, 0.5, 0.5)
        middle = prior[K // 2]
        assert prior[1] > middle
        assert prior[K] > middle

    @pytest.mark.parametrize("K", [6, 21])
    def test_large_K_is_numerically_safe(self, K):
        """The log-gamma implementation must not overflow or produce NaNs."""
        for a, b in [(0.25, 0.25), (0.5, 0.5), (1.0, 3.0), (3.0, 1.0)]:
            prior = beta_binomial_prior(K, a, b)
            assert np.all(np.isfinite(prior))
            assert prior.sum() == pytest.approx(1.0)
            assert np.all(prior >= 0.0)


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

class TestGammaPowerCoefficients:
    @pytest.mark.parametrize("K", [2, 3, 5, 6])
    def test_gamma_K_reproduces_manuscript_coefficient(self, K):
        """
        At gamma = K the coefficient must equal (K-1)! prod_j g(u_j), the value
        precompute_coefficients already returns, for every position.
        """
        u = _sorted_batch(K)
        g = truncated_normal_g(-2.0)
        a_ref, _ = precompute_coefficients(u, g, K)
        a_new = prior_power_coefficients(point_prior(K, K))(u, g, K)
        assert np.allclose(a_new, a_ref, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("K", [2, 4, 6])
    def test_gamma_one_is_proportional_to_g_at_that_position(self, K):
        """At gamma = 1, e_0 == 1, so a_{1,i} = (K-1)! g(u_i)."""
        u = _sorted_batch(K)
        g = truncated_normal_g(-2.0)
        A = gamma_power_coefficients(u, g, K)
        expected = math.factorial(K - 1) * g(u)
        assert np.allclose(A[1], expected)

    @pytest.mark.parametrize("K", [3, 5])
    def test_row_zero_is_empty(self, K):
        """The global null has no alternatives, hence no power coefficient."""
        A = gamma_power_coefficients(_sorted_batch(K), truncated_normal_g(-2.0), K)
        assert np.all(A[0] == 0.0)

    @pytest.mark.parametrize("K", [3, 5])
    def test_coefficients_are_non_negative(self, K):
        """Non-negativity is what the monotonicity argument leans on."""
        A = gamma_power_coefficients(_sorted_batch(K), truncated_normal_g(-2.0), K)
        assert np.all(A >= 0.0)

    def test_linear_in_the_prior(self):
        """a_pi is the prior-weighted mixture of the per-gamma coefficients."""
        K = 5
        u = _sorted_batch(K)
        g = truncated_normal_g(-2.0)
        mix = 0.25 * point_prior(K, 2) + 0.75 * point_prior(K, 5)

        got = prior_power_coefficients(mix)(u, g, K)
        expected = (0.25 * prior_power_coefficients(point_prior(K, 2))(u, g, K)
                    + 0.75 * prior_power_coefficients(point_prior(K, 5))(u, g, K))
        assert np.allclose(got, expected)

    def test_wrong_length_prior_is_rejected(self):
        K = 4
        fn = prior_power_coefficients(uniform_prior(6))
        with pytest.raises(ValueError, match="expected K \\+ 1"):
            fn(_sorted_batch(K), truncated_normal_g(-2.0), K)


# ---------------------------------------------------------------------------
# Solver hook
# ---------------------------------------------------------------------------

class TestSolverHook:
    def test_default_path_is_unchanged_by_the_hook(self):
        """
        Passing the gamma = K point prior must reproduce the multipliers the
        solver returns with no hook at all.
        """
        K, alpha = 4, 0.05
        g = truncated_normal_g(-2.0)
        kwargs = dict(g_func=g, K=K, alpha=alpha, n_samples=30000,
                      tol=1e-3, max_iter=200, seed=42)

        mu_default, _ = coordinate_descent_fast(**kwargs)
        mu_hooked, _ = coordinate_descent_fast(
            **kwargs, power_coeff_fn=prior_power_coefficients(point_prior(K, K))
        )
        assert np.allclose(mu_default, mu_hooked, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# The objective value, and validity of the resulting policy
# ---------------------------------------------------------------------------

class TestObjectiveAndValidity:
    def test_coefficient_objective_matches_direct_simulation(self):
        """
        The coefficient representation of Pi_pi integrates over the ordered
        simplex; direct simulation samples each configuration.  They must agree.
        """
        K, alpha = 4, 0.05
        g = truncated_normal_g(-1.0)
        prior = uniform_prior(K)
        fn = prior_power_coefficients(prior)

        mu, _ = coordinate_descent_fast(
            g_func=g, K=K, alpha=alpha, n_samples=40000, tol=1e-3,
            max_iter=200, seed=42, power_coeff_fn=fn,
        )

        n_direct = 200000
        curves = gamma_sweep(mu, g, K, alpha, n_mc=n_direct, seed=2026,
                             methods=[], power_coeff_fn=fn)
        direct = sum(prior[gamma] * curves["SPOT"]["power"][gamma]
                     for gamma in range(1, K + 1))

        est, se = objective_from_coefficients(mu, g, K, prior,
                                              n_samples=400000, seed=77)
        se_direct = math.sqrt(0.25 / n_direct)      # loose bound, power in [0,1]
        assert abs(est - direct) < 4 * math.sqrt(se**2 + se_direct**2), (est, direct)

    @pytest.mark.parametrize("prior_name", ["uniform", "binomial"])
    def test_prior_policy_controls_fwer(self, prior_name):
        """Changing the objective must not disturb the error guarantee."""
        K, alpha = 4, 0.05
        g = truncated_normal_g(-2.0)
        prior = uniform_prior(K) if prior_name == "uniform" else binomial_prior(K, 0.6)
        fn = prior_power_coefficients(prior)

        mu, _ = coordinate_descent_fast(
            g_func=g, K=K, alpha=alpha, n_samples=40000, tol=1e-3,
            max_iter=200, seed=42, power_coeff_fn=fn,
        )
        n_mc = 50000
        curves = gamma_sweep(mu, g, K, alpha, n_mc=n_mc, seed=2026,
                             methods=[], power_coeff_fn=fn)

        slack = 4 * math.sqrt(alpha * (1 - alpha) / n_mc)
        for gamma in range(K):
            assert curves["SPOT"]["fwer"][gamma] <= alpha + slack, gamma
