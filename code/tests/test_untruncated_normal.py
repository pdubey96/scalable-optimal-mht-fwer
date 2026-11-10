"""
Tests for the classical, non-truncated normal alternative.

The untruncated statistic shows what happens where the standard procedures
should already do well, separating the power gain due to truncation from the
gain due to the policy.  NormalG supplies that arm.

Covers:
- it is the M -> infinity limit of TruncatedNormalG, to machine precision
- how far M = 6 actually is from the limit, which is the size of the
  truncation effect
- the closed form, the monotone likelihood ratio, and sampler/density
  agreement
- that it sits outside Assumptions 3.4 and 3.5, which is why it is an
  out-of-model arm rather than a fifth in-model one
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import norm

from general_k.alternatives import normal_g, truncated_normal_g, sample_from_g


THETAS = [-1.0, -2.0, -3.0, -4.0]


def _mass(a, b, theta):
    """Exact mass g puts on (a, b).

    Substituting u = Phi(x) turns int g(u) du into int phi(x - theta) dx, so
    the mass is a difference of normal CDFs and needs no quadrature.
    """
    return float(norm.cdf(norm.ppf(b) - theta) - norm.cdf(norm.ppf(a) - theta))


class TestItIsTheLimitOfTheTruncatedModel:

    @pytest.mark.parametrize("theta", THETAS)
    def test_matches_truncation_at_M_12_to_machine_precision(self, theta):
        """At M = 12 the discarded mass is ~1e-30, so the two must coincide."""
        u = np.linspace(1e-8, 1 - 1e-8, 20001)
        g_inf = normal_g(theta)(u)
        g_12 = truncated_normal_g(theta, M=12.0)(u)
        assert np.max(np.abs(g_12 - g_inf) / g_inf) < 1e-12

    @pytest.mark.parametrize(
        "theta,expected_gap",
        [(-1.0, 0.0164), (-2.0, 0.0332), (-3.0, 0.0515), (-4.0, 0.0922)],
    )
    def test_gap_at_the_paper_s_M_is_what_the_AE_is_asking_about(
        self, theta, expected_gap
    ):
        """
        At M = 6 the two densities are not interchangeable, and the gap grows
        with the signal because more of the alternative's mass is cut off.
        These are the numbers quoted in the NormalG docstring; pinning them
        keeps the manuscript's account of the truncation honest.
        """
        u = np.linspace(1e-8, 1 - 1e-8, 20001)
        g_inf = normal_g(theta)(u)
        g_6 = truncated_normal_g(theta, M=6.0)(u)
        gap = np.max(np.abs(g_6 - g_inf) / g_inf)
        assert gap == pytest.approx(expected_gap, abs=5e-4)


class TestTheDensityItself:

    @pytest.mark.parametrize("theta", THETAS)
    def test_closed_form(self, theta):
        u = np.array([1e-6, 0.001, 0.01, 0.1, 0.5, 0.9, 0.999])
        want = np.exp(theta * norm.ppf(u) - theta ** 2 / 2.0)
        assert normal_g(theta)(u) == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("theta", THETAS)
    def test_monotone_likelihood_ratio(self, theta):
        """Non-increasing in u, so p-value order is likelihood-ratio order."""
        u = np.linspace(1e-6, 1 - 1e-6, 5000)
        assert np.all(np.diff(normal_g(theta)(u)) < 0)

    @pytest.mark.parametrize("theta", THETAS)
    def test_integrates_to_one_away_from_the_singularity(self, theta):
        """
        g is a density on [0,1], but with an integrable singularity at u = 0:
        it is unbounded there, so quadrature on a grid reaching 0 diverges.
        Substituting u = Phi(x) gives the mass in closed form,

            int_a^b g = Phi(Phi^{-1}(b) - theta) - Phi(Phi^{-1}(a) - theta),

        which is 1 over the whole interval.  Checked against quadrature on
        [0.001, 1], where g is bounded and the trapezoid rule is reliable, so
        this tests the implementation of g and not just the algebra.
        """
        assert _mass(0.0, 1.0, theta) == pytest.approx(1.0, abs=1e-12)

        a = 0.001
        u = np.linspace(a, 1.0, 2_000_001)
        quad = trapezoid(normal_g(theta)(u), u)
        assert quad == pytest.approx(_mass(a, 1.0, theta), rel=1e-4)

    @pytest.mark.parametrize("theta", THETAS)
    def test_sampler_matches_the_density(self, theta):
        """
        Draws must follow g, or the optimiser and the evaluation would be
        solving different problems.  Bin frequencies against the exact bin
        masses, so the tolerance is Monte Carlo error alone.
        """
        rng = np.random.default_rng(4)
        n = 400_000
        draws = sample_from_g(normal_g(theta), n, rng)
        edges = np.linspace(0.0, 1.0, 21)
        observed = np.histogram(draws, bins=edges)[0] / n
        expected = np.array([
            _mass(a, b, theta) for a, b in zip(edges[:-1], edges[1:])
        ])
        # 4 binomial standard errors at the largest bin mass.
        tol = 4.0 * np.sqrt(expected.max() * (1 - expected.max()) / n)
        assert np.max(np.abs(observed - expected)) < tol

    def test_it_is_out_of_model_for_assumptions_3_4_and_3_5(self):
        """
        Unbounded as u -> 0 and vanishing as u -> 1: no c_4 > 0 bounds it
        below and no c_5 bounds it above.  That is why this arm is reported
        as an out-of-model comparison rather than a fifth in-model one.
        """
        g = normal_g(-2.0)
        assert g(np.array([1e-12]))[0] > 1e3          # no finite upper bound
        assert g(np.array([1.0 - 1e-12]))[0] < 1e-6   # no positive lower bound
