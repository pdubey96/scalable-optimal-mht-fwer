"""
Tests for the elementary-symmetric-polynomial form of the general-gamma
objective (verify_general_gamma_objective.py).

The claim under test is

    E_{h_gamma}[ sum_{k in S} D_k ]
        = C(K,gamma)^{-1} E_{u ~ U[0,1]^K} [ sum_k D_k(u) g(u_k)
                                             e_{gamma-1}( g(u_{-k}) ) ],

for any symmetric policy D.  Two checks:

- algebraic, with g == 1, where the ESP weight must collapse exactly to
  (number rejected) / K;
- algebraic, at gamma = K, where it must collapse to the manuscript's own
  constant power coefficient prod_j g(u_j) times (number rejected) / K;
- numerical, comparing the reweighted estimator against direct simulation.
"""

import math

import numpy as np
import pytest

from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import truncated_normal_g
from general_k.gamma_sweep import gamma_sweep
from verify_general_gamma_objective import esp_power_weight, esp_reweighted_power


# ---------------------------------------------------------------------------
# Algebraic collapses
# ---------------------------------------------------------------------------

class TestEspWeightCollapses:
    """Special cases where the ESP weight has a known closed form."""

    @pytest.mark.parametrize("K", [2, 3, 5, 7])
    @pytest.mark.parametrize("gamma_frac", [1, 2, None])
    def test_unit_density_gives_count_over_K(self, K, gamma_frac):
        """
        With g == 1 the alternative equals the null, so the weight must reduce
        to (number rejected) / K for every gamma, exactly.
        """
        gamma = K if gamma_frac is None else min(gamma_frac, K)
        rng = np.random.default_rng(19 + K)
        rejected = rng.random((64, K)) < 0.5
        g_vals = np.ones((64, K))

        got = esp_power_weight(rejected, g_vals, K, gamma)
        expected = rejected.sum(axis=1) / K
        assert np.allclose(got, expected)

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_gamma_equals_K_recovers_manuscript_coefficient(self, K):
        """
        At gamma = K, e_{K-1}(g(u_{-k})) = prod_{j != k} g(u_j), so the weight
        is (number rejected) * prod_j g(u_j) / K -- the constant-in-k power
        coefficient the manuscript already uses.
        """
        rng = np.random.default_rng(23 + K)
        rejected = rng.random((64, K)) < 0.5
        g_vals = rng.uniform(0.2, 3.0, size=(64, K))

        got = esp_power_weight(rejected, g_vals, K, K)
        expected = rejected.sum(axis=1) * np.prod(g_vals, axis=1) / K
        assert np.allclose(got, expected)

    def test_gamma_one_selects_the_rejected_coordinate_only(self):
        """At gamma = 1, e_0 == 1, so the weight is sum_k D_k g(u_k) / K."""
        K = 4
        rng = np.random.default_rng(29)
        rejected = rng.random((32, K)) < 0.5
        g_vals = rng.uniform(0.2, 3.0, size=(32, K))

        got = esp_power_weight(rejected, g_vals, K, 1)
        expected = np.sum(rejected * g_vals, axis=1) / K
        assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# Numerical agreement with direct simulation
# ---------------------------------------------------------------------------

class TestMatchesDirectSimulation:
    """The reweighted estimator reproduces the gamma sweep's power curve."""

    @pytest.mark.parametrize("gamma", [1, 2, 3, 4])
    def test_agrees_within_monte_carlo_error(self, gamma):
        K, alpha, theta = 4, 0.05, -1.0
        g = truncated_normal_g(theta)
        mu, _ = coordinate_descent_fast(
            g_func=g, K=K, alpha=alpha, n_samples=40000, tol=1e-3,
            max_iter=200, seed=42,
        )

        n_direct = 200000
        direct = gamma_sweep(mu, g, K, alpha, n_mc=n_direct, seed=2026,
                             methods=[])["SPOT"]["power"][gamma]

        est, se_est = esp_reweighted_power(mu, g, K, gamma, n=400000, seed=99)
        se_direct = math.sqrt(direct * (1 - direct) / n_direct)
        se = math.sqrt(se_direct**2 + se_est**2)

        assert abs(est - direct) < 4 * se, (gamma, direct, est, se)
