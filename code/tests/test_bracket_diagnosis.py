"""
Tests for the uncertified-exit report, general_k.algorithm.diagnose_bracket.

A run that stops on delta/cycle/max_iter has not certified, and the operator's
first question is whether a finer bisection bracket would have helped.  The
report answers it from the local slope of the binding target.  Covers:

- a synthetic linear target, where the slope, the attainable residual and the
  required bracket are all known in closed form
- the sign convention: F_gamma is non-increasing in its own coordinate
- a jumping target, where no bracket certifies
- a flat target, where sufficient_bisect_tol is None rather than infinite
- which of the two residual terms binds
- that a certifying run carries floor=None and an uncertified one does not
"""

import numpy as np
import pytest

from general_k.algorithm import (
    diagnose_bracket,
    coordinate_descent_fast,
    default_res_tol,
)
from general_k.alternatives import truncated_normal_g


ALPHA = 0.05


class TestOnASyntheticTarget:
    """Closed-form cases, so the arithmetic is checkable by hand."""

    def test_linear_target_recovers_its_slope(self):
        """
        F_gamma(mu) = alpha - s * (mu_gamma - 1) has slope -s everywhere.
        At mu = 1 the gap is zero, so the multiplier binds, but the slope
        report is about the bracket and must still read -s.
        """
        s = 4.0
        mu = np.array([1.0, 1.0])

        def fwer_at(gamma, m):
            return ALPHA - s * (m[gamma] - 1.0)

        out = diagnose_bracket(mu, ALPHA, fwer_at, bisect_tol=1e-4, res_tol=1e-3)
        assert out["slopes"] == pytest.approx([-s, -s])
        assert out["max_abs_slope"] == pytest.approx(s)

    def test_gap_bound_and_required_bracket(self):
        """
        With slope s the midpoint of a width-w bracket is within w/2 of the
        crossing, so the attainable gap is s*w/2, and certifying at res_tol
        needs w <= 2*res_tol/s.  Here s=4, w=1e-4, res_tol=1e-3: the floor is
        2e-4 and the bracket needed is 5e-4.
        """
        s, w, res_tol = 4.0, 1e-4, 1e-3
        mu = np.array([2.0])

        def fwer_at(gamma, m):
            return ALPHA - s * (m[gamma] - 2.0)

        out = diagnose_bracket(mu, ALPHA, fwer_at, bisect_tol=w, res_tol=res_tol)
        assert out["gap_bound"] == pytest.approx(s * w / 2.0)
        assert out["sufficient_bisect_tol"] == pytest.approx(2.0 * res_tol / s)
        # And that bracket does what it claims.
        assert (s * out["sufficient_bisect_tol"] / 2.0) == pytest.approx(res_tol)

    def test_flat_target_reports_none_not_infinity(self):
        """A zero slope means the bracket is not what binds; say so."""
        mu = np.array([0.5])

        def fwer_at(gamma, m):
            return ALPHA - 0.01          # constant in mu

        out = diagnose_bracket(mu, ALPHA, fwer_at, bisect_tol=1e-4, res_tol=1e-3)
        assert out["binding_slope"] == 0.0
        assert out["sufficient_bisect_tol"] is None
        assert out["gap_bound"] == 0.0

    def test_jumping_target_demands_an_absurd_bracket(self):
        """
        A target that steps across alpha has an enormous finite-difference
        slope, so the bracket it asks for is far below any usable width.  That
        is the honest signal that refining will not rescue the run.
        """
        mu = np.array([1.0])

        def fwer_at(gamma, m):
            return ALPHA - 0.02 if m[gamma] >= 1.0 else ALPHA + 0.02

        out = diagnose_bracket(mu, ALPHA, fwer_at, bisect_tol=1e-6, res_tol=1e-3)
        assert out["binding_slope"] > 1e3
        assert out["sufficient_bisect_tol"] < 1e-5

    def test_binding_term_distinguishes_multiplier_from_gap(self):
        """
        The residual term is |min(mu_gamma, alpha - F_gamma)|.  Coordinate 0
        has a tiny multiplier and a large gap, coordinate 1 the reverse; the
        larger of the two mins is what binds.
        """
        mu = np.array([0.30, 0.02])

        def fwer_at(gamma, m):
            return ALPHA - (0.5 if gamma == 0 else 0.001)

        out = diagnose_bracket(mu, ALPHA, fwer_at, bisect_tol=1e-4, res_tol=1e-3)
        # gaps are 0.5 and 0.001; mins are 0.30 and 0.001; coordinate 0 binds
        # and it binds through its multiplier, not its constraint gap.
        assert out["binding_gamma"] == 0
        assert out["binding_term"] == "multiplier"


class TestWiredIntoTheSolver:
    """The solver attaches the report exactly when it fails to certify."""

    def test_certifying_run_has_no_floor(self):
        K = 3
        mu, _, diag = coordinate_descent_fast(
            g_func=truncated_normal_g(-2.0), K=K, alpha=ALPHA,
            n_samples=20000, tol=1e-3, max_iter=100, seed=42,
            return_diagnostics=True,
        )
        assert diag["converged"]
        assert diag["reason"] == "residual"
        assert diag["floor"] is None

    def test_uncertified_run_reports_a_floor(self):
        """One sweep cannot certify, so the report must be there and usable."""
        K = 3
        n = 20000
        mu, _, diag = coordinate_descent_fast(
            g_func=truncated_normal_g(-2.0), K=K, alpha=ALPHA,
            n_samples=n, tol=1e-3, max_iter=1, seed=42,
            return_diagnostics=True,
        )
        assert not diag["converged"]
        assert diag["reason"] == "max_iter"

        floor = diag["floor"]
        assert floor is not None
        assert len(floor["slopes"]) == K
        assert 0 <= floor["binding_gamma"] < K
        assert floor["binding_term"] in {"gap", "multiplier"}
        assert floor["gap_bound"] >= 0.0
        if floor["sufficient_bisect_tol"] is not None:
            assert floor["sufficient_bisect_tol"] > 0.0

    def test_targets_are_non_increasing_in_their_own_coordinate(self):
        """
        F_gamma decreases as mu_gamma rises -- more penalty, fewer rejections.
        Every estimated slope should therefore be non-positive, up to Monte
        Carlo noise on a finite batch.
        """
        _, _, diag = coordinate_descent_fast(
            g_func=truncated_normal_g(-2.0), K=3, alpha=ALPHA,
            n_samples=20000, tol=1e-3, max_iter=1, seed=42,
            return_diagnostics=True,
        )
        assert all(s <= 1e-9 for s in diag["floor"]["slopes"]), diag["floor"]["slopes"]


class TestTheBoundIsAnUpperBound:
    """
    gap_bound is the largest gap the bracket permits, not a floor the run
    cannot beat.  This was got wrong once: a live uncertified run reached a
    residual of 1.59e-3 while its gap_bound was 3.65e-3, which reads as a
    contradiction only if the quantity is misread as a floor.
    """

    def test_midpoint_can_land_nearer_than_the_worst_case(self):
        """
        With slope s the bound is s*w/2, attained only when the crossing sits
        at a bracket endpoint.  Here it sits at the midpoint, so the true gap
        is zero while the bound is not.
        """
        s, w = 50.0, 1e-4
        mu = np.array([1.0])

        def fwer_at(gamma, m):
            return ALPHA - s * (m[gamma] - 1.0)      # crossing exactly at mu = 1

        out = diagnose_bracket(mu, ALPHA, fwer_at, bisect_tol=w, res_tol=1e-3)
        assert out["gap_bound"] == pytest.approx(s * w / 2)
        assert abs(fwer_at(0, mu) - ALPHA) == 0.0     # the achieved gap
        assert out["gap_bound"] > 0.0                 # the permitted one

    def test_the_quoted_width_is_sufficient(self):
        """
        The claim attached to sufficient_bisect_tol is that it makes the bound
        fall to res_tol -- not that no coarser width could ever certify.
        """
        s, res_tol = 73.0, 6.892e-4
        mu = np.array([2.0])

        def fwer_at(gamma, m):
            return ALPHA - s * (m[gamma] - 2.0)

        out = diagnose_bracket(mu, ALPHA, fwer_at, bisect_tol=1e-4, res_tol=res_tol)
        w = out["sufficient_bisect_tol"]
        assert s * w / 2 == pytest.approx(res_tol)
        tighter = diagnose_bracket(mu, ALPHA, fwer_at, bisect_tol=w, res_tol=res_tol)
        assert tighter["gap_bound"] <= res_tol * (1 + 1e-12)
