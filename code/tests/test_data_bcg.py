"""The frozen BCG data reproduces the R analysis it replaces.

The submitted script read ``dat.bcg`` out of the R ``metadat`` package at run
time, so the result depended on an R install and its version.  ``data_bcg``
carries the counts instead.  That is only an improvement if the frozen copy
gives the same answers, so every subgroup value here is checked against numbers
computed independently in R from ``metadat`` 1.6.0 on 12 August 2026, using
``escalc(measure = "RR")``'s formulae and ``rma.uni(method = "FE")``'s pooling.

The reference values below are that R output, to twelve significant figures.
"""

import numpy as np
import pytest

from general_k import data_bcg
from general_k.baselines import bonferroni, holm, hommel, romano_wolf


# split -> subgroup -> (n, beta, se, p), from R.
R_REFERENCE = {
    "alloc": {
        "alternate":  (2, -0.718555367634601, 0.0780067766160431, 3.21754733030296e-20),
        "random":     (7, -0.294536663614775, 0.0546177938139501, 6.94118232647077e-08),
        "systematic": (4, -0.413328379603451, 0.0952846101388717, 1.43896802001412e-05),
    },
    "lat": {
        "low_lat":  (5, -0.117841618037171, 0.0516589667359684, 0.0225398401600342),
        "mid_lat":  (4, -0.670012854768936, 0.176820671169484, 0.000151119554604177),
        "high_lat": (4, -0.969272458064511, 0.0701870710306006, 2.22314413319359e-43),
    },
    "risk": {
        "low_risk":  (5, -0.0874177336814501, 0.0522285623991375, 0.0941790749750009),
        "mid_risk":  (4, -1.16715872752715, 0.106794560764227, 8.3758014204958e-28),
        "high_risk": (4, -0.823287386666686, 0.0802113258894967, 1.02398114155068e-24),
    },
    "era": {
        "era1": (5, -0.867664215621852, 0.0768953843760551, 1.57964984166376e-29),
        "era2": (4, -0.435615280443648, 0.151430021441786, 0.00401886560620913),
        "era3": (4, -0.243362351209033, 0.0501904572384768, 1.24224768253706e-06),
    },
}


class TestTheFrozenTable:

    def test_thirteen_trials(self):
        assert len(data_bcg.TRIALS) == 13

    def test_checksum_pins_the_counts(self):
        assert data_bcg.checksum() == data_bcg.CHECKSUM

    def test_counts_are_positive_integers(self):
        for t in data_bcg.TRIALS:
            for value in t[3:7]:
                assert isinstance(value, int) and value > 0, t

    def test_allocation_levels_are_the_three_expected(self):
        levels = {t[8] for t in data_bcg.TRIALS}
        assert levels == {"random", "alternate", "systematic"}


class TestAgreementWithR:

    @pytest.mark.parametrize("split", sorted(R_REFERENCE))
    def test_every_subgroup_matches(self, split):
        got = data_bcg.pooled(split)
        want = R_REFERENCE[split]
        assert set(got) == set(want), split
        for name, (n, beta, se, p) in want.items():
            assert got[name]["n"] == n, (split, name)
            assert got[name]["beta"] == pytest.approx(beta, rel=1e-10), (split, name)
            assert got[name]["se"] == pytest.approx(se, rel=1e-10), (split, name)
            assert got[name]["p"] == pytest.approx(p, rel=1e-8), (split, name)

    def test_every_split_is_a_partition_into_three(self):
        for split in data_bcg.SPLITS:
            labels, order = data_bcg.subgroups(split)
            assert len(order) == 3, split
            assert len(labels) == 13, split
            assert set(labels) == set(order), split


class TestTheResultTheManuscriptReports:

    def test_baseline_risk_is_the_borderline_split(self):
        """
        Supplementary Section S7.1 turns on this split: p_low ~ 0.094 against
        two values below 1e-24, and every method retains the low-risk subgroup.
        If these move, the section's argument moves with them.
        """
        p = data_bcg.pooled("risk")
        assert p["low_risk"]["p"] == pytest.approx(0.0942, abs=5e-5)
        assert p["mid_risk"]["p"] < 1e-27
        assert p["high_risk"]["p"] < 1e-23

    @pytest.mark.parametrize("split", ["alloc", "lat", "era"])
    def test_the_other_three_splits_are_rejected_throughout(self, split):
        """
        Supplementary Section S7.1 says the evidence is uniformly strong in the
        other three splits and every method rejects all three subgroups.  The
        stepwise methods checked here are Holm, Hommel and Romano-Wolf, not
        Bonferroni, and the distinction matters: at latitude the low tertile
        has p = 0.0225, above Bonferroni's 0.05/3 but below every stepwise
        threshold.
        """
        p = data_bcg.p_values(split)
        for name, fn in (("Holm", holm), ("Hommel", hommel),
                         ("Romano-Wolf", romano_wolf)):
            assert np.all(fn(p, 0.05) == 1), (split, name, p)

    def test_bonferroni_alone_retains_the_low_latitude_tertile(self):
        """
        Recorded because it is the one place the splits are not unanimous, and
        Supplementary Section S7.1 should not claim more agreement than there
        is.
        """
        p = data_bcg.p_values("lat")
        d = bonferroni(p, 0.05)
        assert d.sum() == 2
        assert p[np.where(d == 0)[0][0]] == pytest.approx(0.0225398, abs=1e-6)

    def test_p_values_helper_matches_pooled(self):
        for split in data_bcg.SPLITS:
            pooled = data_bcg.pooled(split)
            assert np.allclose(data_bcg.p_values(split),
                               [pooled[n]["p"] for n in pooled])
