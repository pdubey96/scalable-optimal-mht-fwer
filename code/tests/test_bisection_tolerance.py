"""
Tests for the bisection stopping rule.

A coordinate can only be certified if the bisection resolves the crossing finely
enough: the achievable |F(mu) - alpha| is about slope * bracket_width, and that
must fall below res_tol.  With a fixed bracket width this fails silently on steep
targets -- the recursion reaches an exact fixed point of the discretised map and
reports 'delta' while the residual is stuck well above res_tol.

These cover the fix: the bisection takes a target tolerance on |F| and keeps
refining until it is met, so the certification criterion is attainable by
construction rather than by luck.
"""

import numpy as np

from general_k.algorithm import _bracket_and_bisect, coordinate_descent_fast
from general_k.alternatives import constant_g


class TestAdaptiveBisection:

    def test_steep_target_is_refined_until_f_tol_is_met(self):
        """
        Slope 200 with a root at 0.37, which bisection midpoints never hit
        exactly.  A 1e-4 bracket guarantees only |F| <= 200 * 1e-4 = 0.02, so
        f_tol = 1e-3 is unreachable without further refinement.
        """
        F = lambda x: 200.0 * (0.37 - x)

        root = _bracket_and_bisect(F, u_max=1.0, tol=1e-4, f_tol=1e-3)

        assert abs(F(root)) <= 1e-3

    def test_flat_target_does_not_pay_for_the_extra_refinement(self):
        """A gentle slope meets f_tol almost immediately; no deep bisection."""
        calls = []

        def F(x):
            calls.append(x)
            return 0.001 * (x - 0.5)

        root = _bracket_and_bisect(F, u_max=1.0, tol=1e-4, f_tol=1e-3)

        assert abs(F(root)) <= 1e-3
        assert len(calls) < 20

    def test_zero_at_origin_still_short_circuits(self):
        """F(0) <= 0 means the constraint is slack; mu = 0 regardless of f_tol."""
        F = lambda x: -1.0

        assert _bracket_and_bisect(F, u_max=1.0, tol=1e-4, f_tol=1e-3) == 0.0

    def test_discontinuous_target_terminates(self):
        """A jump straddling the root must not spin forever."""
        F = lambda x: 1.0 if x < 0.5 else -1.0

        root = _bracket_and_bisect(F, u_max=1.0, tol=1e-4, f_tol=1e-9)

        assert 0.0 <= root <= 1.0
        assert abs(root - 0.5) < 1e-6


class TestCertificationDiagnostics:

    def test_uncertified_run_reports_what_it_would_take(self):
        """
        A run that stops short of the residual certificate should say so
        actionably: the slope it saw and the bracket width that would certify.
        """
        g = constant_g(1.0)
        _, _, d = coordinate_descent_fast(
            g_func=g, K=3, alpha=0.05, n_samples=2000,
            tol=1e-3, max_iter=2, seed=0, return_diagnostics=True,
        )

        assert "bisect_tol" in d
        assert "f_tol" in d

    def test_diagnostics_record_the_sweep_budget(self):
        """
        Whether a run was cut off by the cap or settled on its own can only be
        read from the diagnostics if the cap is recorded alongside the sweeps
        actually taken.  Settings re-solved at a raised budget are otherwise
        indistinguishable from the rest of the grid.
        """
        g = constant_g(1.0)
        _, _, d = coordinate_descent_fast(
            g_func=g, K=3, alpha=0.05, n_samples=2000,
            tol=1e-3, max_iter=2, seed=0, return_diagnostics=True,
        )

        assert d["max_iter"] == 2
        assert d["sweeps"] <= d["max_iter"]


class TestGridStoppingToleranceIsInert:

    def test_tightening_algo_tol_cannot_change_a_certified_run(self):
        """
        The residual criterion is checked before the delta criterion, so a run
        that certifies is bit-identical under a tighter delta threshold.  This is
        what makes it safe to re-run only the settings that failed to certify.
        """
        g = constant_g(1.0)
        kw = dict(g_func=g, K=3, alpha=0.05, n_samples=4000,
                  max_iter=50, seed=1, return_diagnostics=True)

        mu_loose, _, d_loose = coordinate_descent_fast(tol=1e-3, **kw)
        mu_tight, _, d_tight = coordinate_descent_fast(tol=1e-9, **kw)

        if d_loose["reason"] == "residual":
            assert d_loose["sweeps"] == d_tight["sweeps"]
            np.testing.assert_array_equal(mu_loose, mu_tight)
