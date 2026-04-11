"""
Tests for general_k.algorithm module.

Covers:
- test_bisection_finds_root       : bisection_update returns mu0 in (0, 50) for K=3
- test_coordinate_descent_converges: coordinate_descent returns valid mu and history for K=3
- test_coordinate_descent_k4      : coordinate_descent works without error for K=4
"""

import numpy as np
import pytest

from general_k.alternatives import constant_g
from general_k.algorithm import bisection_update, coordinate_descent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_rng(seed=42):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# bisection_update
# ---------------------------------------------------------------------------

class TestBisectionUpdate:

    def test_bisection_finds_root(self):
        """
        g=constant(2), K=3, alpha=0.05, mu_current=[0,0,0].
        Bisection for gamma=0 should return mu0 > 0 and < 50.
        """
        K = 3
        alpha = 0.05
        g = constant_g(2.0)
        mu_current = np.zeros(K)
        rng = make_rng(42)
        n_samples = 30000

        mu0 = bisection_update(
            gamma=0,
            mu_current=mu_current,
            g_func=g,
            K=K,
            alpha=alpha,
            n_samples=n_samples,
            rng=rng,
        )

        assert mu0 > 0.0, f"Expected mu0 > 0, got {mu0}"
        assert mu0 < 50.0, f"Expected mu0 < 50, got {mu0}"

    def test_bisection_returns_float(self):
        """bisection_update must return a scalar float."""
        K = 3
        alpha = 0.05
        g = constant_g(2.0)
        mu_current = np.zeros(K)
        rng = make_rng(7)

        result = bisection_update(
            gamma=0,
            mu_current=mu_current,
            g_func=g,
            K=K,
            alpha=alpha,
            n_samples=10000,
            rng=rng,
        )

        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_bisection_does_not_mutate_mu_current(self):
        """bisection_update must not modify mu_current in place."""
        K = 3
        alpha = 0.05
        g = constant_g(2.0)
        mu_current = np.zeros(K)
        mu_copy = mu_current.copy()
        rng = make_rng(8)

        bisection_update(
            gamma=0,
            mu_current=mu_current,
            g_func=g,
            K=K,
            alpha=alpha,
            n_samples=10000,
            rng=rng,
        )

        np.testing.assert_array_equal(
            mu_current, mu_copy, err_msg="bisection_update mutated mu_current"
        )


# ---------------------------------------------------------------------------
# coordinate_descent
# ---------------------------------------------------------------------------

class TestCoordinateDescent:

    def test_coordinate_descent_converges(self):
        """
        g=constant(2), K=3, alpha=0.05, n_samples=50000.
        Should return mu of shape (K,), all >= 0, and non-empty history.
        """
        K = 3
        alpha = 0.05
        g = constant_g(2.0)

        mu, history = coordinate_descent(
            g_func=g,
            K=K,
            alpha=alpha,
            n_samples=50000,
            seed=42,
        )

        assert mu.shape == (K,), f"Expected shape ({K},), got {mu.shape}"
        assert np.all(mu >= 0.0), f"All mu values must be >= 0, got {mu}"
        assert len(history) > 0, "history must be non-empty"

    def test_coordinate_descent_k4(self):
        """
        g=constant(2), K=4, alpha=0.05, n_samples=30000.
        Should work without error.
        """
        K = 4
        alpha = 0.05
        g = constant_g(2.0)

        mu, history = coordinate_descent(
            g_func=g,
            K=K,
            alpha=alpha,
            n_samples=30000,
            seed=0,
        )

        assert mu.shape == (K,), f"Expected shape ({K},), got {mu.shape}"
        assert np.all(mu >= 0.0), f"All mu values must be >= 0, got {mu}"
        assert len(history) > 0, "history must be non-empty"

    def test_coordinate_descent_history_shape(self):
        """Each entry in history should be an array of shape (K,)."""
        K = 3
        alpha = 0.05
        g = constant_g(2.0)

        mu, history = coordinate_descent(
            g_func=g,
            K=K,
            alpha=alpha,
            n_samples=20000,
            seed=1,
        )

        for i, h in enumerate(history):
            assert np.array(h).shape == (K,), (
                f"history[{i}] has wrong shape: {np.array(h).shape}"
            )

    def test_coordinate_descent_returns_tuple(self):
        """coordinate_descent must return (mu, history) tuple."""
        K = 3
        alpha = 0.05
        g = constant_g(2.0)

        result = coordinate_descent(
            g_func=g,
            K=K,
            alpha=alpha,
            n_samples=10000,
            seed=99,
        )

        assert isinstance(result, tuple) and len(result) == 2, (
            "Expected a 2-tuple (mu, history)"
        )
        mu, history = result
        assert isinstance(history, list), "history must be a list"
