"""
Tests for general_k.decision module.

Covers:
- optimal_l_star: mu=0 (all R_i > 0 => l*=K), large mu (all R_i < 0 => l*=0)
- decisions: shape (K,), LR-ordering (monotone non-increasing), correct binary output
"""

import numpy as np
import pytest

from general_k.decision import optimal_l_star, decisions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def constant_g(factor):
    """Return a g_func that maps any array to factor * ones_like(array)."""
    return lambda u: np.full_like(u, factor, dtype=float)


# ---------------------------------------------------------------------------
# optimal_l_star
# ---------------------------------------------------------------------------

class TestOptimalLStar:
    """optimal_l_star(mu, u_sorted, g_func, K) -> int in {0, ..., K}."""

    def test_zero_mu_all_positive_R_returns_K(self):
        """
        mu=0, g=2, K=3: all R_i = 16 > 0, cumulative sum is always increasing,
        so l* = K = 3.
        """
        K = 3
        mu = np.zeros(K)
        u = np.array([0.1, 0.5, 0.9])
        g = constant_g(2.0)
        assert optimal_l_star(mu, u, g, K) == K

    def test_large_mu_all_negative_R_returns_0(self):
        """
        Very large mu (100), K=3: all R_i << 0, so l* = 0.
        """
        K = 3
        mu = np.full(K, 100.0)
        u = np.array([0.1, 0.5, 0.9])
        g = constant_g(2.0)
        assert optimal_l_star(mu, u, g, K) == 0

    def test_return_type_is_int(self):
        """optimal_l_star must return a Python int (or numpy integer)."""
        K = 3
        mu = np.zeros(K)
        u = np.array([0.1, 0.5, 0.9])
        g = constant_g(2.0)
        result = optimal_l_star(mu, u, g, K)
        assert isinstance(result, (int, np.integer))

    def test_result_in_valid_range(self):
        """l* must satisfy 0 <= l* <= K."""
        K = 4
        mu = np.array([1.0, 2.0, 0.5, 1.5])
        u = np.array([0.1, 0.3, 0.6, 0.9])
        g = constant_g(2.0)
        result = optimal_l_star(mu, u, g, K)
        assert 0 <= result <= K

    def test_zero_mu_K1_returns_1(self):
        """K=1, mu=0, g=2: R_1 = 1 > 0, so l* = 1."""
        K = 1
        mu = np.zeros(K)
        u = np.array([0.5])
        g = constant_g(2.0)
        assert optimal_l_star(mu, u, g, K) == 1


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

class TestDecisions:
    """decisions(mu, u_sorted, g_func, K) -> np.ndarray of shape (K,) with values in {0,1}."""

    def test_zero_mu_all_ones(self):
        """
        mu=0, g=2, K=3: l*=3, so D = [1, 1, 1].
        """
        K = 3
        mu = np.zeros(K)
        u = np.array([0.1, 0.5, 0.9])
        g = constant_g(2.0)
        D = decisions(mu, u, g, K)
        np.testing.assert_array_equal(D, np.ones(K))

    def test_large_mu_all_zeros(self):
        """
        Very large mu (100), K=3: l*=0, so D = [0, 0, 0].
        """
        K = 3
        mu = np.full(K, 100.0)
        u = np.array([0.1, 0.5, 0.9])
        g = constant_g(2.0)
        D = decisions(mu, u, g, K)
        np.testing.assert_array_equal(D, np.zeros(K))

    def test_shape_is_K(self):
        """decisions must return an array of shape (K,)."""
        K = 3
        mu = np.zeros(K)
        u = np.array([0.2, 0.5, 0.8])
        g = constant_g(2.0)
        D = decisions(mu, u, g, K)
        assert D.shape == (K,)

    def test_shape_K4(self):
        """decisions shape is (K,) for K=4."""
        K = 4
        mu = np.zeros(K)
        u = np.array([0.1, 0.3, 0.6, 0.9])
        g = constant_g(2.0)
        D = decisions(mu, u, g, K)
        assert D.shape == (K,)

    def test_lr_ordering_zero_mu(self):
        """
        LR-ordering: D[i] >= D[i+1] for all i (non-increasing).
        With mu=0, K=4: D=[1,1,1,1] trivially satisfies this.
        """
        K = 4
        mu = np.zeros(K)
        u = np.array([0.1, 0.3, 0.6, 0.9])
        g = constant_g(2.0)
        D = decisions(mu, u, g, K)
        for i in range(K - 1):
            assert D[i] >= D[i + 1], (
                f"LR-ordering violated at i={i}: D[{i}]={D[i]} < D[{i+1}]={D[i+1]}"
            )

    def test_lr_ordering_large_mu(self):
        """
        LR-ordering with large mu, K=4: D=[0,0,0,0] trivially satisfies this.
        """
        K = 4
        mu = np.full(K, 100.0)
        u = np.array([0.1, 0.3, 0.6, 0.9])
        g = constant_g(2.0)
        D = decisions(mu, u, g, K)
        for i in range(K - 1):
            assert D[i] >= D[i + 1], (
                f"LR-ordering violated at i={i}: D[{i}]={D[i]} < D[{i+1}]={D[i+1]}"
            )

    def test_lr_ordering_partial_rejection(self):
        """
        LR-ordering with intermediate mu producing partial rejection.
        The construction D_i=1 iff i<=l* guarantees non-increasing by design,
        so this test verifies the output is consistent with that construction
        for any mu producing 0 < l* < K.
        """
        K = 4
        # Tune mu to get partial rejection: moderate positive mu
        mu = np.array([0.5, 0.5, 0.5, 0.5])
        u = np.array([0.1, 0.3, 0.6, 0.9])
        g = constant_g(2.0)
        D = decisions(mu, u, g, K)
        # Regardless of l*, non-increasing must hold
        for i in range(K - 1):
            assert D[i] >= D[i + 1], (
                f"LR-ordering violated at i={i}: D[{i}]={D[i]} < D[{i+1}]={D[i+1]}"
            )

    def test_values_binary(self):
        """All entries in D must be 0 or 1."""
        K = 3
        mu = np.zeros(K)
        u = np.array([0.1, 0.5, 0.9])
        g = constant_g(2.0)
        D = decisions(mu, u, g, K)
        assert set(D.tolist()).issubset({0, 1})
