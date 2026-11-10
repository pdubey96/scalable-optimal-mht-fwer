"""
Tests for general_k.closed_testing.

The O(K^2) shortcut is the only subtle part: rather than enumerating all
2^K - 1 intersections, it tests each hypothesis against the least-favourable
subset of each size. These tests check it against brute-force enumeration and
check that the resulting procedures are valid. A separate numerical benchmark
reproduces the two quantities that Rosset et al.'s published 2022 Table 2
actually reports for this setting.
"""

import functools
import math

import numpy as np
import pytest

from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import truncated_normal_g
from general_k.closed_testing import (
    CLOSED_TESTS,
    closed_mask_bruteforce,
    fisher_closed_mask,
    fisher_statistic,
    median_closed_mask,
    median_statistic,
    stouffer_closed_mask,
    stouffer_statistic,
)
from general_k.gamma_sweep import gamma_sweep

METHODS = [
    ("Fisher", fisher_closed_mask, fisher_statistic),
    ("Stouffer", stouffer_closed_mask, stouffer_statistic),
    ("Median", median_closed_mask, median_statistic),
]


class TestShortcutMatchesBruteForce:
    """The least-favourable-subset shortcut must give the closed test exactly."""

    @pytest.mark.parametrize("name,fast,stat", METHODS)
    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    @pytest.mark.parametrize("hi", [0.05, 0.3, 1.0])
    def test_agrees_on_random_draws(self, name, fast, stat, K, hi):
        """
        ``hi`` controls how much signal there is: small ``hi`` makes almost
        everything reject, ``hi = 1`` is the global null where almost nothing
        does, so both branches of the shortcut are exercised.
        """
        # Python salts ``hash(str)`` per process, so deriving this seed from a
        # tuple containing ``name`` made the supposedly reproducible test data
        # change from one interpreter launch to the next.
        method_index = next(i for i, (label, _, _) in enumerate(METHODS)
                            if label == name)
        seed = 10_000 * K + 1_000 * method_index + round(100 * hi)
        rng = np.random.default_rng(seed)
        p = np.sort(rng.uniform(0.0, hi, size=(60, K)), axis=1)

        got = fast(p, 0.05)
        expected = closed_mask_bruteforce(
            p, 0.05, functools.partial(stat, alpha=0.05)
        )
        assert np.array_equal(got, expected)

    @pytest.mark.parametrize("name,fast,stat", METHODS)
    def test_agrees_on_ties_and_extremes(self, name, fast, stat):
        """Duplicated and boundary p-values must not break the index arithmetic."""
        p = np.array([
            [0.0, 0.0, 0.5, 1.0],
            [0.01, 0.01, 0.01, 0.01],
            [1.0, 1.0, 1.0, 1.0],
            [1e-12, 0.2, 0.2, 0.9],
        ])
        got = fast(p, 0.05)
        expected = closed_mask_bruteforce(
            p, 0.05, functools.partial(stat, alpha=0.05)
        )
        assert np.array_equal(got, expected)
        assert np.all(np.isfinite(got.astype(float)))


class TestProcedureProperties:
    """Properties every closed test must have."""

    @pytest.mark.parametrize("name", sorted(CLOSED_TESTS))
    def test_rejects_a_leading_block(self, name):
        """
        A symmetric monotone combining function cannot reject a larger p-value
        while retaining a smaller one, so the rejection set is a leading block.
        This is what places these procedures inside the class over which SPOT
        is optimal.
        """
        rng = np.random.default_rng(5)
        p = np.sort(rng.uniform(0.0, 0.3, size=(500, 6)), axis=1)
        mask = CLOSED_TESTS[name](p, 0.05)
        assert np.all(mask[:, :-1] >= mask[:, 1:])

    @pytest.mark.parametrize("name", sorted(CLOSED_TESTS))
    def test_controls_fwer_under_the_global_null(self, name):
        """Under the global null every rejection is false, so the rejection
        probability is the family-wise error rate."""
        rng = np.random.default_rng(9)
        n, K, alpha = 40000, 5, 0.05
        p = np.sort(rng.uniform(0.0, 1.0, size=(n, K)), axis=1)
        any_rejection = CLOSED_TESTS[name](p, alpha).any(axis=1).mean()
        se = math.sqrt(alpha * (1 - alpha) / n)
        assert any_rejection <= alpha + 4 * se, (name, any_rejection)

    @pytest.mark.parametrize("name", sorted(CLOSED_TESTS))
    def test_more_signal_never_rejects_less(self, name):
        """Monotonicity: shrinking every p-value cannot lose a rejection."""
        rng = np.random.default_rng(13)
        p = np.sort(rng.uniform(0.0, 0.4, size=(300, 5)), axis=1)
        fewer = CLOSED_TESTS[name](p, 0.05).sum(axis=1)
        more = CLOSED_TESTS[name](p * 0.5, 0.05).sum(axis=1)
        assert np.all(more >= fewer)


class TestAgainstPublishedNumbers:
    """
    Rosset et al.'s published 2022 Table 2 reports average power at K = 3,
    theta = -2, alpha = 0.05 of 0.634 for their OMT-Pi3 policy and 0.554 for
    Hommel. Their table does not report closed-Stouffer or Holm. Our M=6
    truncated-normal model approximates that untruncated normal-means setting.
    """

    def test_reproduces_values_reported_in_rosset_table_2(self):
        K, alpha = 3, 0.05
        g = truncated_normal_g(-2.0)
        mu, _ = coordinate_descent_fast(
            g_func=g, K=K, alpha=alpha, n_samples=100000, tol=1e-3,
            max_iter=400, seed=42,
        )
        n_mc = 100000
        curves = gamma_sweep(mu, g, K, alpha, n_mc=n_mc, seed=2026)
        tol = 6 * math.sqrt(0.25 / n_mc)          # ~0.009

        assert curves["SPOT"]["power"][K] == pytest.approx(0.634, abs=tol)
        assert curves["Hommel"]["power"][K] == pytest.approx(0.554, abs=tol)

    def test_closed_tests_beat_hommel_when_all_are_false(self):
        """For dense signals the rank-based procedures are the wrong
        benchmark."""
        K, alpha = 3, 0.05
        g = truncated_normal_g(-2.0)
        mu, _ = coordinate_descent_fast(
            g_func=g, K=K, alpha=alpha, n_samples=100000, tol=1e-3,
            max_iter=400, seed=42,
        )
        curves = gamma_sweep(mu, g, K, alpha, n_mc=100000, seed=2026)
        hommel = curves["Hommel"]["power"][K]
        assert curves["Fisher-closed"]["power"][K] > hommel
        assert curves["Stouffer-closed"]["power"][K] > hommel
