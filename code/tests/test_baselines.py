"""
Tests for general_k.baselines module.

Covers:
- bonferroni: threshold=alpha/K, rejects p_k <= threshold
- holm: step-down; stops at first non-rejection
- hochberg: step-up; reject all at once when p_{(i)} <= alpha/(K-i+1)
- hommel: at least as powerful as Holm (dominates Holm)
"""

import numpy as np
import pytest

from general_k.baselines import bonferroni, holm, hochberg, hommel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decisions(arr):
    """Return a plain Python list of ints for readable assertions."""
    return list(arr.astype(int))


# ---------------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------------

class TestBonferroni:
    """Bonferroni: reject p_k <= alpha/K."""

    def test_mixed_decisions(self):
        """
        K=3, alpha=0.05 -> threshold=0.0167.
        p=[0.001, 0.01, 0.03] -> D=[1, 1, 0].
        """
        p = np.array([0.001, 0.01, 0.03])
        D = bonferroni(p, alpha=0.05)
        assert _decisions(D) == [1, 1, 0]

    def test_all_reject(self):
        """All p-values well below threshold -> all rejected."""
        p = np.array([0.001, 0.002, 0.003])
        D = bonferroni(p, alpha=0.05)
        assert _decisions(D) == [1, 1, 1]

    def test_none_reject(self):
        """All p-values above threshold -> none rejected."""
        p = np.array([0.1, 0.2, 0.3])
        D = bonferroni(p, alpha=0.05)
        assert _decisions(D) == [0, 0, 0]

    def test_exactly_at_threshold_rejects(self):
        """p_k == alpha/K is rejected (<=)."""
        alpha = 0.05
        K = 4
        p = np.array([alpha / K, alpha / K + 1e-10, 0.5, 0.9])
        D = bonferroni(p, alpha=alpha)
        assert D[0] == 1
        assert D[1] == 0

    def test_single_hypothesis(self):
        """K=1: threshold is alpha itself."""
        D_reject = bonferroni(np.array([0.04]), alpha=0.05)
        D_keep   = bonferroni(np.array([0.06]), alpha=0.05)
        assert _decisions(D_reject) == [1]
        assert _decisions(D_keep)   == [0]

    def test_output_is_ndarray_of_01(self):
        """Return type is np.ndarray with values in {0, 1}."""
        p = np.array([0.01, 0.04, 0.1])
        D = bonferroni(p, alpha=0.05)
        assert isinstance(D, np.ndarray)
        assert set(D.tolist()).issubset({0, 1})

    def test_unsorted_input_preserved(self):
        """Output order matches input order (no sorting)."""
        p = np.array([0.03, 0.001, 0.01])   # unsorted
        D = bonferroni(p, alpha=0.05)
        # threshold = 0.05/3 ≈ 0.0167; only index 1 (0.001) and index 2 (0.01) reject
        assert _decisions(D) == [0, 1, 1]


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------

class TestHolm:
    """Holm step-down: sort ascending, reject p_{(i)} while <= alpha/(K-i+1)."""

    def test_all_reject(self):
        """
        K=3, alpha=0.05.
        Sorted p=[0.001, 0.02, 0.04].
        Thresholds: 0.05/3≈0.0167, 0.05/2=0.025, 0.05/1=0.05.
        0.001<0.0167 -> reject; 0.02<0.025 -> reject; 0.04<0.05 -> reject.
        D=[1,1,1].
        """
        p = np.array([0.001, 0.02, 0.04])
        D = holm(p, alpha=0.05)
        assert _decisions(D) == [1, 1, 1]

    def test_stops_early(self):
        """
        K=3, alpha=0.05.
        Sorted p=[0.001, 0.03, 0.04].
        0.001<0.0167 -> reject; 0.03>0.025 -> STOP.
        D=[1,0,0].
        """
        p = np.array([0.001, 0.03, 0.04])
        D = holm(p, alpha=0.05)
        assert _decisions(D) == [1, 0, 0]

    def test_none_reject(self):
        """First sorted p-value already above threshold -> none rejected."""
        p = np.array([0.1, 0.2, 0.3])
        D = holm(p, alpha=0.05)
        assert _decisions(D) == [0, 0, 0]

    def test_unsorted_input(self):
        """
        Input [0.04, 0.001, 0.02] should give same rejections as sorted version.
        After sorting: 0.001, 0.02, 0.04 -> all reject.
        Decision array aligned with original order.
        """
        p = np.array([0.04, 0.001, 0.02])
        D = holm(p, alpha=0.05)
        # original index 0 (0.04): reject; 1 (0.001): reject; 2 (0.02): reject
        assert _decisions(D) == [1, 1, 1]

    def test_single_hypothesis(self):
        """K=1: threshold is alpha; behaves like Bonferroni."""
        assert _decisions(holm(np.array([0.04]), 0.05)) == [1]
        assert _decisions(holm(np.array([0.06]), 0.05)) == [0]

    def test_output_shape_and_type(self):
        """Output is np.ndarray of same length as input, values in {0,1}."""
        p = np.array([0.01, 0.04, 0.1])
        D = holm(p, alpha=0.05)
        assert isinstance(D, np.ndarray)
        assert D.shape == p.shape
        assert set(D.tolist()).issubset({0, 1})

    def test_stops_in_middle(self):
        """
        K=4, alpha=0.05.
        Thresholds: 0.0125, 0.0167, 0.025, 0.05.
        p=[0.005, 0.015, 0.03, 0.045] (sorted).
        0.005<0.0125 -> R; 0.015<0.0167 -> R; 0.03>0.025 -> STOP.
        D=[1,1,0,0].
        """
        p = np.array([0.005, 0.015, 0.03, 0.045])
        D = holm(p, alpha=0.05)
        assert _decisions(D) == [1, 1, 0, 0]


# ---------------------------------------------------------------------------
# Hochberg
# ---------------------------------------------------------------------------

class TestHochberg:
    """Hochberg step-up: scan from largest p, find first p_{(i)} <= alpha/(K-i+1)."""

    def test_largest_rejects_all(self):
        """
        K=3, alpha=0.05.
        Sorted p=[0.001, 0.03, 0.04].
        Check from top: p_{(3)}=0.04 <= 0.05/1=0.05 -> reject all.
        D=[1,1,1].
        """
        p = np.array([0.001, 0.03, 0.04])
        D = hochberg(p, alpha=0.05)
        assert _decisions(D) == [1, 1, 1]

    def test_only_smallest_rejects(self):
        """
        K=3, alpha=0.05.
        Sorted p=[0.001, 0.03, 0.06].
        p_{(3)}=0.06>0.05; p_{(2)}=0.03>0.025; p_{(1)}=0.001<=0.0167 -> reject p_{(1)} only.
        D=[1,0,0].
        """
        p = np.array([0.001, 0.03, 0.06])
        D = hochberg(p, alpha=0.05)
        assert _decisions(D) == [1, 0, 0]

    def test_none_reject(self):
        """All p-values too large -> none rejected."""
        p = np.array([0.1, 0.2, 0.3])
        D = hochberg(p, alpha=0.05)
        assert _decisions(D) == [0, 0, 0]

    def test_unsorted_input(self):
        """Output aligned with original input order."""
        # Same as largest_rejects_all but input is scrambled
        p = np.array([0.04, 0.001, 0.03])
        D = hochberg(p, alpha=0.05)
        # all three rejected
        assert _decisions(D) == [1, 1, 1]

    def test_single_hypothesis(self):
        """K=1: threshold is alpha; same as Bonferroni/Holm."""
        assert _decisions(hochberg(np.array([0.04]), 0.05)) == [1]
        assert _decisions(hochberg(np.array([0.06]), 0.05)) == [0]

    def test_output_shape_and_type(self):
        """Output is np.ndarray of same length as input, values in {0,1}."""
        p = np.array([0.01, 0.04, 0.1])
        D = hochberg(p, alpha=0.05)
        assert isinstance(D, np.ndarray)
        assert D.shape == p.shape
        assert set(D.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Hommel
# ---------------------------------------------------------------------------

class TestHommel:
    """Hommel: at least as powerful as Holm (dominates Holm)."""

    def test_at_least_as_powerful_as_holm_case1(self):
        """
        Hommel must reject wherever Holm rejects (dominance property).
        p=[0.001, 0.02, 0.04], alpha=0.05.
        Holm rejects all three; Hommel must also reject all three.
        """
        p = np.array([0.001, 0.02, 0.04])
        D_holm   = holm(p, alpha=0.05)
        D_hommel = hommel(p, alpha=0.05)
        # Hommel >= Holm everywhere
        assert np.all(D_hommel >= D_holm)

    def test_at_least_as_powerful_as_holm_case2(self):
        """
        p=[0.001, 0.03, 0.04], alpha=0.05.
        Holm: D=[1,0,0]. Hommel must have D[0]=1 at minimum.
        """
        p = np.array([0.001, 0.03, 0.04])
        D_holm   = holm(p, alpha=0.05)
        D_hommel = hommel(p, alpha=0.05)
        assert np.all(D_hommel >= D_holm)

    def test_at_least_as_powerful_as_holm_larger(self):
        """Dominance check with K=5."""
        p = np.array([0.005, 0.011, 0.02, 0.04, 0.10])
        D_holm   = holm(p, alpha=0.05)
        D_hommel = hommel(p, alpha=0.05)
        assert np.all(D_hommel >= D_holm)

    def test_none_reject_when_all_large(self):
        """No p-value should be rejected when all are large."""
        p = np.array([0.5, 0.6, 0.7])
        D = hommel(p, alpha=0.05)
        assert _decisions(D) == [0, 0, 0]

    def test_all_reject_when_all_tiny(self):
        """All very small p-values should all be rejected."""
        p = np.array([1e-6, 2e-6, 3e-6])
        D = hommel(p, alpha=0.05)
        assert _decisions(D) == [1, 1, 1]

    def test_single_hypothesis(self):
        """K=1: j_star=1, threshold=alpha."""
        assert _decisions(hommel(np.array([0.04]), 0.05)) == [1]
        assert _decisions(hommel(np.array([0.06]), 0.05)) == [0]

    def test_output_shape_and_type(self):
        """Output is np.ndarray of same length as input, values in {0,1}."""
        p = np.array([0.01, 0.04, 0.1])
        D = hommel(p, alpha=0.05)
        assert isinstance(D, np.ndarray)
        assert D.shape == p.shape
        assert set(D.tolist()).issubset({0, 1})

    def test_unsorted_input_preserved(self):
        """Output is aligned with the original (unsorted) input order."""
        p_sorted   = np.array([0.001, 0.02, 0.04])
        p_unsorted = np.array([0.04, 0.001, 0.02])
        D_sorted   = hommel(p_sorted,   alpha=0.05)
        D_unsorted = hommel(p_unsorted, alpha=0.05)
        # Same set of rejections regardless of order
        assert set(np.where(D_sorted == 1)[0].tolist())    # noqa – just check both run
        # Specific: all three should be rejected in both cases
        assert np.all(D_sorted   == 1)
        assert np.all(D_unsorted == 1)
