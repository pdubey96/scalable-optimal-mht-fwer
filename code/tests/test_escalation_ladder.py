"""
Tests for the grid driver's escalation ladder.

A handful of settings do not certify at the default sweep budget.  Re-solving
them by hand leaves the released results irreproducible: the recorded solve is
then whatever was typed at a shell, not what the driver produces.  The ladder
makes the escalation part of the driver -- a setting that exits without the
residual certificate is re-solved at the next stage, from scratch and with the
same seed -- so a full grid re-run reproduces the recorded solve exactly.

These pin the properties that claim rests on: the ladder stops at the first
stage that certifies, it records which stage produced the result and what the
abandoned stages did, and it leaves a setting that certifies immediately exactly
as it was.
"""

import pytest

from general_k.algorithm import default_res_tol
from run_all_experiments import run_single


# Small enough to run in seconds; the ladder logic is independent of size.
SMALL = dict(K=3, model_key="trunc", theta=-2.0, n_algo_samples=2000, n_mc=500)


def _solve(stages, **overrides):
    kw = dict(SMALL)
    kw.update(overrides)
    K = kw.pop("K")
    model_key = kw.pop("model_key")
    theta = kw.pop("theta")
    return run_single(K, model_key, theta, stages=stages, **kw)["diagnostics"]


class TestEscalationLadder:

    def test_a_setting_that_certifies_at_the_first_stage_is_solved_once(self):
        d = _solve(({"max_iter": 200, "f_tol": None},))

        assert d["converged"]
        assert d["stage"] == 1
        assert d["escalation"] == []

    def test_a_setting_that_does_not_certify_is_re_solved_at_the_next_stage(self):
        """Stage 1 is capped at a single sweep, which cannot certify here."""
        d = _solve((
            {"max_iter": 1, "f_tol": None},
            {"max_iter": 200, "f_tol": None},
        ))

        assert d["stage"] == 2
        assert d["max_iter"] == 200
        assert d["converged"]

    def test_the_abandoned_stages_are_recorded(self):
        d = _solve((
            {"max_iter": 1, "f_tol": None},
            {"max_iter": 200, "f_tol": None},
        ))

        assert len(d["escalation"]) == 1
        first = d["escalation"][0]
        assert first["stage"] == 1
        assert first["max_iter"] == 1
        assert first["reason"] == "max_iter"
        assert not first["converged"]

    def test_later_stages_do_not_run_once_a_stage_certifies(self):
        """The second stage would raise if it were reached."""
        d = _solve((
            {"max_iter": 200, "f_tol": None},
            {"max_iter": "not a number", "f_tol": None},
        ))

        assert d["stage"] == 1

    def test_the_last_stage_stands_when_no_stage_certifies(self):
        d = _solve((
            {"max_iter": 1, "f_tol": None},
            {"max_iter": 2, "f_tol": None},
        ))

        assert not d["converged"]
        assert d["stage"] == 2
        assert d["sweeps"] == 2
        assert len(d["escalation"]) == 1

    def test_res_tol_names_the_residual_tolerance_as_an_f_tol(self):
        """
        The stage that matters asks bisection to resolve the crossing to within
        the tolerance the run is judged against.  Spelling that dependence out
        in the ladder keeps the two in step if the sample size changes.
        """
        d = _solve(({"max_iter": 1, "f_tol": "res_tol"},))

        assert d["f_tol"] == pytest.approx(default_res_tol(0.05, 2000))
        assert d["f_tol"] == pytest.approx(d["res_tol"])
