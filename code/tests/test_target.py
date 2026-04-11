"""
Tests for general_k.alternatives and general_k.target modules.

Covers:
- test_sample_shape         : prepare_samples returns (n_samples, K) sorted arrays
- test_sample_null_uniform  : under gamma=0, mean of u_1 (smallest of K=3 uniforms) ~ 1/(K+1)
- test_fwer_null_zero_mu    : mu=0, g=constant(2), K=3 -> FWER_0 > 0.99
- test_fwer_large_mu        : mu=[1000,...], l*=0 always -> FWER_0 = 0.0
- test_target_function_sign : mu=0 -> F_0(0) - alpha > 0
"""

import numpy as np
import pytest

from general_k.alternatives import constant_g, truncated_normal_g, sample_from_g
from general_k.target import (
    prepare_samples,
    sample_and_estimate_fwer,
    target_function,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_rng(seed=42):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# prepare_samples
# ---------------------------------------------------------------------------

class TestPrepareSamples:

    def test_sample_shape(self):
        """prepare_samples returns (n_samples, K) sorted array and matching index array."""
        K, n_samples = 3, 200
        gamma = 1
        g = constant_g(2.0)
        rng = make_rng(0)

        samples_sorted, sort_indices, null_label = prepare_samples(
            gamma, g, K, n_samples, rng
        )

        assert samples_sorted.shape == (n_samples, K), (
            f"Expected shape ({n_samples}, {K}), got {samples_sorted.shape}"
        )
        assert sort_indices.shape == (n_samples, K)
        assert null_label.shape == (K,)

    def test_samples_sorted_ascending(self):
        """Each row of samples_sorted must be non-decreasing."""
        K, n_samples = 4, 100
        gamma = 2
        g = constant_g(1.5)
        rng = make_rng(1)

        samples_sorted, _, _ = prepare_samples(gamma, g, K, n_samples, rng)

        for i in range(n_samples):
            row = samples_sorted[i]
            assert np.all(row[:-1] <= row[1:]), (
                f"Row {i} not sorted: {row}"
            )

    def test_null_label_positions(self):
        """null_label should be True for columns gamma..K-1 and False for 0..gamma-1."""
        K, gamma = 5, 2
        g = constant_g(1.0)
        rng = make_rng(2)
        _, _, null_label = prepare_samples(gamma, g, K, 10, rng)

        assert not np.any(null_label[:gamma]), "Alternative columns should not be null"
        assert np.all(null_label[gamma:]), "Null columns should be null"

    def test_sample_null_uniform(self):
        """
        Under gamma=0 (all null), E[u_(1)] = 1/(K+1) for K uniform order statistics.
        With K=3, this is 0.25.  Check within 0.01.
        """
        K = 3
        n_samples = 20_000
        gamma = 0
        g = constant_g(1.0)          # doesn't matter; no alternative samples
        rng = make_rng(3)

        samples_sorted, _, _ = prepare_samples(gamma, g, K, n_samples, rng)
        mean_u1 = float(np.mean(samples_sorted[:, 0]))
        expected = 1.0 / (K + 1)     # = 0.25

        assert abs(mean_u1 - expected) < 0.01, (
            f"E[u_(1)] = {mean_u1:.4f}, expected ~{expected:.4f}"
        )


# ---------------------------------------------------------------------------
# sample_and_estimate_fwer
# ---------------------------------------------------------------------------

class TestSampleAndEstimateFWER:

    def test_fwer_null_zero_mu(self):
        """
        mu=0, g=constant(2), K=3, gamma=0:
        With mu=0 all R_i > 0 so l*=K=3, every null is rejected -> FWER > 0.99.
        """
        K = 3
        mu = np.zeros(K)
        g = constant_g(2.0)
        gamma = 0
        n_samples = 2_000
        rng = make_rng(10)

        fwer = sample_and_estimate_fwer(gamma, mu, g, K, n_samples, rng)
        assert fwer > 0.99, f"Expected FWER > 0.99, got {fwer:.4f}"

    def test_fwer_large_mu(self):
        """
        mu=[1000,...], g=constant(2), K=3, gamma=0:
        All R_i << 0 so l*=0 always -> FWER = 0.
        """
        K = 3
        mu = np.full(K, 1000.0)
        g = constant_g(2.0)
        gamma = 0
        n_samples = 1_000
        rng = make_rng(11)

        fwer = sample_and_estimate_fwer(gamma, mu, g, K, n_samples, rng)
        assert fwer == 0.0, f"Expected FWER = 0.0, got {fwer:.4f}"

    def test_fwer_all_alternative_no_nulls(self):
        """
        When gamma=K (all alternative), there are no null hypotheses.
        FWER should be 0 regardless of mu.
        """
        K = 3
        mu = np.zeros(K)
        g = constant_g(2.0)
        gamma = K           # no nulls
        n_samples = 1_000
        rng = make_rng(12)

        fwer = sample_and_estimate_fwer(gamma, mu, g, K, n_samples, rng)
        assert fwer == 0.0, f"With no nulls, FWER should be 0, got {fwer:.4f}"

    def test_fwer_in_unit_interval(self):
        """FWER estimate must be in [0, 1]."""
        K = 4
        mu = np.array([1.0, 2.0, 1.5, 0.5])
        g = constant_g(1.5)
        gamma = 2
        n_samples = 500
        rng = make_rng(13)

        fwer = sample_and_estimate_fwer(gamma, mu, g, K, n_samples, rng)
        assert 0.0 <= fwer <= 1.0, f"FWER out of [0,1]: {fwer}"


# ---------------------------------------------------------------------------
# target_function
# ---------------------------------------------------------------------------

class TestTargetFunction:

    def _make_predrawn(self, gamma, g, K, n_samples, rng):
        return prepare_samples(gamma, g, K, n_samples, rng)

    def test_target_function_sign(self):
        """
        mu=0, g=constant(2), K=3, alpha=0.05:
        With mu_gamma=0, FWER >> alpha so target_function > 0.
        """
        K = 3
        gamma = 0
        alpha = 0.05
        g = constant_g(2.0)
        n_samples = 2_000
        rng = make_rng(20)

        samples_sorted, sort_indices, null_label = self._make_predrawn(
            gamma, g, K, n_samples, rng
        )
        mu_rest = np.zeros(K)
        mu_gamma = 0.0

        result = target_function(
            gamma, mu_gamma, mu_rest, g, K, alpha,
            samples_sorted, sort_indices, null_label
        )
        assert result > 0, (
            f"Expected target > 0 (FWER >> alpha), got {result:.4f}"
        )

    def test_target_function_large_mu_negative(self):
        """
        Very large mu_gamma -> l*=0 -> FWER=0, so target = 0 - alpha < 0.
        """
        K = 3
        gamma = 0
        alpha = 0.05
        g = constant_g(2.0)
        n_samples = 1_000
        rng = make_rng(21)

        samples_sorted, sort_indices, null_label = self._make_predrawn(
            gamma, g, K, n_samples, rng
        )
        mu_rest = np.full(K, 1000.0)
        mu_gamma = 1000.0

        result = target_function(
            gamma, mu_gamma, mu_rest, g, K, alpha,
            samples_sorted, sort_indices, null_label
        )
        assert result < 0, (
            f"Expected target < 0 (FWER=0 < alpha), got {result:.4f}"
        )

    def test_target_function_uses_mu_gamma(self):
        """
        target_function must use mu_gamma (not the value already in mu_rest[gamma]).
        Changing mu_gamma from 0 to large should flip the sign.
        """
        K = 3
        gamma = 1
        alpha = 0.05
        g = constant_g(2.0)
        n_samples = 2_000
        rng = make_rng(22)

        samples_sorted, sort_indices, null_label = self._make_predrawn(
            gamma, g, K, n_samples, rng
        )
        mu_rest = np.zeros(K)       # mu_rest[gamma] = 0; will be overwritten

        result_low = target_function(
            gamma, 0.0, mu_rest, g, K, alpha,
            samples_sorted, sort_indices, null_label
        )
        result_high = target_function(
            gamma, 1000.0, mu_rest, g, K, alpha,
            samples_sorted, sort_indices, null_label
        )

        assert result_low > result_high, (
            "Higher mu_gamma should reduce FWER, lowering target value"
        )

    def test_target_function_does_not_mutate_mu_rest(self):
        """target_function must not modify mu_rest in place."""
        K = 3
        gamma = 0
        alpha = 0.05
        g = constant_g(2.0)
        n_samples = 500
        rng = make_rng(23)

        samples_sorted, sort_indices, null_label = self._make_predrawn(
            gamma, g, K, n_samples, rng
        )
        mu_rest = np.zeros(K)
        mu_rest_copy = mu_rest.copy()

        target_function(
            gamma, 5.0, mu_rest, g, K, alpha,
            samples_sorted, sort_indices, null_label
        )

        np.testing.assert_array_equal(
            mu_rest, mu_rest_copy,
            err_msg="target_function mutated mu_rest"
        )


# ---------------------------------------------------------------------------
# alternatives module: constant_g and truncated_normal_g
# ---------------------------------------------------------------------------

class TestConstantG:

    def test_constant_g_scalar(self):
        g = constant_g(3.0)
        result = g(np.array([0.5]))
        np.testing.assert_allclose(result, [3.0])

    def test_constant_g_shape(self):
        g = constant_g(2.5)
        u = np.linspace(0, 1, 50)
        result = g(u)
        assert result.shape == u.shape
        np.testing.assert_allclose(result, np.full_like(u, 2.5))


class TestTruncatedNormalG:

    def test_returns_callable(self):
        g = truncated_normal_g(theta=1.0)
        assert callable(g)

    def test_output_shape(self):
        g = truncated_normal_g(theta=1.0)
        u = np.linspace(0.01, 0.99, 100)
        result = g(u)
        assert result.shape == u.shape

    def test_nonnegative(self):
        g = truncated_normal_g(theta=2.0)
        u = np.linspace(0.01, 0.99, 200)
        assert np.all(g(u) >= 0), "Density ratio must be non-negative"

    def test_zero_theta_near_one(self):
        """With theta=0, the null and alt are the same -> g(u) ~ 1 everywhere."""
        g = truncated_normal_g(theta=0.0)
        u = np.linspace(0.05, 0.95, 50)
        vals = g(u)
        np.testing.assert_allclose(vals, np.ones_like(vals), atol=1e-6)


class TestSampleFromG:

    def test_sample_shape(self):
        g = constant_g(1.0)
        rng = make_rng(30)
        samples = sample_from_g(g, 200, rng)
        assert samples.shape == (200,)

    def test_sample_in_unit_interval(self):
        g = truncated_normal_g(theta=1.0)
        rng = make_rng(31)
        samples = sample_from_g(g, 500, rng)
        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)

    def test_constant_g_uniform_distribution(self):
        """
        With g=constant(c), the rejection sampler should produce U(0,1) samples
        (acceptance probability is constant -> uniform).  Check mean ~ 0.5.
        """
        g = constant_g(2.0)
        rng = make_rng(32)
        samples = sample_from_g(g, 5_000, rng)
        mean = float(np.mean(samples))
        assert abs(mean - 0.5) < 0.05, f"Mean {mean:.4f} far from 0.5 for uniform draw"
