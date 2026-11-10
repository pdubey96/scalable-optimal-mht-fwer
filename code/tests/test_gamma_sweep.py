"""
Tests for general_k.gamma_sweep module.

Covers:
- batch baselines agree with the per-sample reference implementations
- sample_configuration puts alternatives in columns 0..gamma-1
- _power_and_fwer counts true and false rejections in the original column order,
  taking a rejection mask so closed testing is scored by the same code
- gamma_sweep returns well-formed curves and respects the FWER constraint
"""

import numpy as np
import pytest

from general_k.alternatives import truncated_normal_g
from general_k.algorithm import coordinate_descent_fast
from general_k.baselines import bonferroni, holm, hochberg, hommel, romano_wolf
from general_k.gamma_sweep import (
    BATCH_BASELINES,
    _mask_from_count,
    _power_and_fwer,
    gamma_sweep,
    sample_configuration,
)

REFERENCE = {
    "Bonferroni": bonferroni,
    "Holm": holm,
    "Hochberg": hochberg,
    "Hommel": hommel,
    "Romano-Wolf": romano_wolf,
}


def test_every_batch_baseline_has_a_reference():
    """A new entry in BATCH_BASELINES must bring a reference to check against."""
    assert set(BATCH_BASELINES) == set(REFERENCE)


# ---------------------------------------------------------------------------
# Batch baselines vs. reference implementations
# ---------------------------------------------------------------------------

class TestBatchBaselinesMatchReference:
    """The vectorised counts must reproduce baselines.py exactly."""

    @pytest.mark.parametrize("name", sorted(REFERENCE))
    @pytest.mark.parametrize("K", [2, 3, 4, 6, 9])
    def test_matches_on_random_p_values(self, name, K):
        rng = np.random.default_rng(7 + K)
        p = rng.uniform(0.0, 1.0, size=(400, K))
        sorted_p = np.sort(p, axis=1)

        batch = BATCH_BASELINES[name](sorted_p, alpha=0.05)
        expected = np.array(
            [int(REFERENCE[name](row, 0.05).sum()) for row in p]
        )
        assert np.array_equal(batch, expected)

    @pytest.mark.parametrize("name", sorted(REFERENCE))
    def test_matches_on_small_p_values(self, name):
        """Signal-rich draws exercise the all-rejected branch."""
        rng = np.random.default_rng(11)
        p = rng.uniform(0.0, 0.004, size=(200, 5))
        sorted_p = np.sort(p, axis=1)

        batch = BATCH_BASELINES[name](sorted_p, alpha=0.05)
        expected = np.array([int(REFERENCE[name](row, 0.05).sum()) for row in p])
        assert np.array_equal(batch, expected)
        assert np.all(batch == 5)

    @pytest.mark.parametrize("name", sorted(REFERENCE))
    def test_matches_on_large_p_values(self, name):
        """Null-only draws exercise the none-rejected branch."""
        rng = np.random.default_rng(13)
        p = rng.uniform(0.5, 1.0, size=(200, 5))
        sorted_p = np.sort(p, axis=1)

        batch = BATCH_BASELINES[name](sorted_p, alpha=0.05)
        assert np.all(batch == 0)


# ---------------------------------------------------------------------------
# Sampling layout
# ---------------------------------------------------------------------------

class TestSampleConfiguration:
    """Columns 0..gamma-1 are alternatives, the rest are uniform nulls."""

    @pytest.mark.parametrize("gamma", [0, 1, 3, 5])
    def test_shapes_and_sorting(self, gamma):
        K = 5
        rng = np.random.default_rng(3)
        sorted_p, sort_indices = sample_configuration(
            gamma, truncated_normal_g(-2.0), K, n_mc=50, rng=rng
        )
        assert sorted_p.shape == (50, K)
        assert sort_indices.shape == (50, K)
        assert np.all(np.diff(sorted_p, axis=1) >= 0)
        # sort_indices is a permutation of 0..K-1 in every row
        assert np.array_equal(np.sort(sort_indices, axis=1),
                              np.tile(np.arange(K), (50, 1)))

    def test_alternatives_are_stochastically_smaller(self):
        """With a strong alternative, columns 0..gamma-1 have smaller p-values."""
        K, gamma = 6, 3
        rng = np.random.default_rng(5)
        sorted_p, sort_indices = sample_configuration(
            gamma, truncated_normal_g(-3.0), K, n_mc=4000, rng=rng
        )
        # Recover the unsorted matrix and compare column means.
        unsorted = np.empty_like(sorted_p)
        np.put_along_axis(unsorted, sort_indices, sorted_p, axis=1)
        assert unsorted[:, :gamma].mean() < unsorted[:, gamma:].mean()


# ---------------------------------------------------------------------------
# Power / FWER accounting
# ---------------------------------------------------------------------------

class TestPowerAndFwer:
    """Rejections are attributed to original columns, not sorted positions."""

    def test_hand_worked_example(self):
        """
        K=4, gamma=2 (columns 0,1 are alternatives).
        Row 0: sorted order came from columns [1, 3, 0, 2]; 2 rejected
               -> true rejections {1}, false rejections {3} -> power 1/2, error.
        Row 1: sorted order came from columns [0, 1, 2, 3]; 2 rejected
               -> true rejections {0, 1}, no false rejection -> power 1.
        """
        sort_indices = np.array([[1, 3, 0, 2],
                                 [0, 1, 2, 3]])
        n_rej = np.array([2, 2])
        power, fwer = _power_and_fwer(
            _mask_from_count(n_rej, 4), sort_indices, gamma=2, K=4)
        assert power == pytest.approx((0.5 + 1.0) / 2)
        assert fwer == pytest.approx(0.5)

    def test_gamma_zero_has_no_power(self):
        sort_indices = np.array([[0, 1, 2]])
        power, fwer = _power_and_fwer(
            _mask_from_count(np.array([1]), 3), sort_indices, gamma=0, K=3)
        assert power is None
        assert fwer == pytest.approx(1.0)

    def test_gamma_equals_K_has_no_error(self):
        sort_indices = np.array([[0, 1, 2]])
        power, fwer = _power_and_fwer(
            _mask_from_count(np.array([3]), 3), sort_indices, gamma=3, K=3)
        assert power == pytest.approx(1.0)
        assert fwer == 0.0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestGammaSweep:
    """The sweep is well-formed and every method controls FWER."""

    def test_curves_are_well_formed_and_fwer_controlled(self):
        K, alpha = 3, 0.05
        g = truncated_normal_g(-2.0)
        mu, _ = coordinate_descent_fast(
            g_func=g, K=K, alpha=alpha, n_samples=20000, tol=1e-3,
            max_iter=100, seed=42,
        )
        out = gamma_sweep(mu, g, K, alpha, n_mc=20000, seed=2026)

        assert set(out) == {"SPOT", "Bonferroni", "Holm", "Hochberg", "Hommel",
                            "Romano-Wolf",
                            "Fisher-closed", "Stouffer-closed", "Median-closed"}
        for name, curves in out.items():
            assert len(curves["power"]) == K + 1
            assert len(curves["fwer"]) == K + 1
            assert curves["power"][0] is None
            assert curves["fwer"][K] == 0.0
            for gamma in range(K):
                # Monte Carlo slack at n_mc = 20000 is well under 0.01.
                assert curves["fwer"][gamma] <= alpha + 0.01, (name, gamma)

    def test_spot_leads_at_the_design_configuration(self):
        """SPOT is optimal for Pi_K, so it must lead there."""
        K, alpha = 3, 0.05
        g = truncated_normal_g(-2.0)
        mu, _ = coordinate_descent_fast(
            g_func=g, K=K, alpha=alpha, n_samples=20000, tol=1e-3,
            max_iter=100, seed=42,
        )
        out = gamma_sweep(mu, g, K, alpha, n_mc=20000, seed=2026)
        best_baseline = max(out[n]["power"][K] for n in BATCH_BASELINES)
        assert out["SPOT"]["power"][K] >= best_baseline
