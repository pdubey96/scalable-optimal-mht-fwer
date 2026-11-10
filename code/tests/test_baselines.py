"""
Tests for general_k.baselines module.

Covers:
- bonferroni: threshold=alpha/K, rejects p_k <= threshold
- holm: step-down; stops at first non-rejection
- hochberg: step-up; reject all at once when p_{(i)} <= alpha/(K-i+1)
- hommel: at least as powerful as Holm (dominates Holm)
- romano_wolf: stepdown min-P under independence; dominates Holm; exact FWER
"""

import numpy as np
import pytest

from general_k.baselines import bonferroni, holm, hochberg, hommel, romano_wolf


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


# ---------------------------------------------------------------------------
# Romano-Wolf
# ---------------------------------------------------------------------------

class TestRomanoWolf:
    """Stepdown min-P under independence, i.e. the stepdown Sidak thresholds."""

    def test_thresholds_are_the_sidak_sequence(self):
        """
        K=3, alpha=0.05.  Step i leaves m = K-i+1 hypotheses, so the thresholds
        are 1-(1-alpha)^(1/m) = 0.016952, 0.025321, 0.05.
        """
        alpha = 0.05
        expected = [1 - (1 - alpha) ** (1 / m) for m in (3, 2, 1)]
        # Just below each threshold: everything is rejected.
        p = np.array([expected[0] - 1e-9, expected[1] - 1e-9, expected[2] - 1e-9])
        assert _decisions(romano_wolf(p, alpha)) == [1, 1, 1]
        # Smallest p-value just above the first threshold: the stepdown stops
        # at step one and rejects nothing.
        p = np.array([expected[0] + 1e-9, 0.5, 0.9])
        assert _decisions(romano_wolf(p, alpha)) == [0, 0, 0]

    def test_strictly_beats_holm_in_the_gap(self):
        """
        p_(1) = 0.0168 sits between Holm's alpha/3 = 0.016667 and Romano-Wolf's
        0.016952, so Holm rejects nothing and Romano-Wolf rejects one.  The
        second step then stops it: 0.026 > 0.025321.
        """
        p = np.array([0.0168, 0.026, 0.9])
        assert _decisions(holm(p, alpha=0.05)) == [0, 0, 0]
        assert _decisions(romano_wolf(p, alpha=0.05)) == [1, 0, 0]

    @pytest.mark.parametrize("K", [2, 3, 4, 6, 9])
    def test_dominates_holm(self, K):
        """
        1-(1-alpha)^(1/m) >= alpha/m for every m, so Romano-Wolf rejects a
        superset of Holm's rejections on every draw.
        """
        rng = np.random.default_rng(101 + K)
        for row in rng.uniform(0.0, 1.0, size=(500, K)):
            d_holm = holm(row, alpha=0.05)
            d_rw = romano_wolf(row, alpha=0.05)
            assert np.all(d_rw >= d_holm)

    def test_enforces_monotone_adjusted_p_values(self):
        """
        The raw adjusted sequence can decrease: at K=3, p=(0.02, 0.021, 0.9)
        gives 1-0.98^3 = 0.0588 then 1-0.979^2 = 0.0416.  The running maximum
        keeps the second from being rejected while the first is not.
        """
        p = np.array([0.02, 0.021, 0.9])
        assert 1 - (1 - p[1]) ** 2 < 1 - (1 - p[0]) ** 3    # the raw sequence dips
        assert _decisions(romano_wolf(p, alpha=0.05)) == [0, 0, 0]

    def test_rejections_are_a_leading_block(self):
        """A stepdown never rejects a larger p-value while retaining a smaller."""
        rng = np.random.default_rng(23)
        for row in rng.uniform(0.0, 0.06, size=(300, 5)):
            d = romano_wolf(row, alpha=0.05)
            rejected_p = row[d == 1]
            retained_p = row[d == 0]
            if rejected_p.size and retained_p.size:
                assert rejected_p.max() < retained_p.min()

    def test_fwer_is_exactly_alpha_under_the_complete_null(self):
        """
        Under the complete null the procedure rejects something iff the smallest
        of K independent uniforms falls below 1-(1-alpha)^(1/K), an event of
        probability exactly alpha.  200,000 draws give a standard error of
        0.0005, so 0.003 is a wide tolerance.
        """
        alpha, K, n = 0.05, 4, 200_000
        rng = np.random.default_rng(2026)
        p = rng.uniform(0.0, 1.0, size=(n, K))
        any_rejection = np.array([romano_wolf(row, alpha).sum() > 0 for row in p])
        assert abs(any_rejection.mean() - alpha) < 0.003

    def test_single_hypothesis(self):
        """K=1: m=1, so the threshold is alpha itself."""
        assert _decisions(romano_wolf(np.array([0.04]), 0.05)) == [1]
        assert _decisions(romano_wolf(np.array([0.06]), 0.05)) == [0]

    def test_unsorted_input_preserved(self):
        """Decisions follow the hypotheses, not the sort order."""
        p = np.array([0.9, 0.0001, 0.02])
        assert _decisions(romano_wolf(p, alpha=0.05)) == [0, 1, 1]

    def test_output_shape_and_type(self):
        p = np.array([0.01, 0.04, 0.1])
        D = romano_wolf(p, alpha=0.05)
        assert isinstance(D, np.ndarray)
        assert D.shape == p.shape
        assert set(D.tolist()).issubset({0, 1})

    def test_reproduces_the_submitted_R_implementation(self):
        """
        The earlier K=3 version of this work computed this in R, as
        ``romano_wolf_stepdown_indep`` in its simulation scripts.  The Python
        code must not silently redefine a comparator it carries forward, so
        the R function is transcribed
        literally here and required to agree exactly.  Checked against
        statsmodels' ``holm-sidak`` as well when writing this, also exactly.
        """
        def submitted_R(p, alpha):
            order = np.argsort(p)
            ps = p[order]
            m = len(ps)
            padj = np.empty(m)
            for i in range(1, m + 1):
                padj[i - 1] = 1 - (1 - ps[i - 1]) ** (m - i + 1)
            padj = np.maximum.accumulate(padj)
            rej = np.zeros(m, dtype=int)
            rej[order] = (padj <= alpha).astype(int)
            return rej

        rng = np.random.default_rng(0)
        for K in (1, 2, 3, 4, 6, 12, 21):
            draws = np.concatenate([
                rng.uniform(0.0, 1.0, size=(200, K)),        # mostly null
                rng.uniform(0.0, 0.06, size=(200, K)),       # signal-rich
            ])
            for row in draws:
                assert np.array_equal(
                    romano_wolf(row, 0.05), submitted_R(row, 0.05)
                ), (K, row)
