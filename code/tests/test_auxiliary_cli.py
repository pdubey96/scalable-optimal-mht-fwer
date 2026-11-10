"""Focused checks for the documented auxiliary command-line interfaces."""

import argparse
from pathlib import Path

import pytest

import check_prior_monotonicity
import check_weak_monotonicity
import diag_residual_terms
import run_all_experiments
import run_gamma_sweep
import run_prior_comparison
import run_prior_objective
import test_sparse_tilt_ftol
import verify_general_gamma_objective
import verify_tie_set_polynomial
from general_k import alternatives, paths


CLI_MODULES = (
    diag_residual_terms,
    run_all_experiments,
    run_gamma_sweep,
    run_prior_objective,
    run_prior_comparison,
    check_prior_monotonicity,
    check_weak_monotonicity,
    verify_general_gamma_objective,
    verify_tie_set_polynomial,
    test_sparse_tilt_ftol,
)


@pytest.mark.parametrize("module", CLI_MODULES, ids=lambda module: module.__name__)
def test_every_cli_option_has_help(module):
    parser = module._build_parser()
    assert parser.description
    for action in parser._actions:
        if action.option_strings:
            assert action.help not in (None, argparse.SUPPRESS), action.option_strings
    assert "usage:" in parser.format_help()


@pytest.mark.parametrize(
    ("module", "filename"),
    (
        (run_gamma_sweep, "gamma_sweep.json"),
        (run_prior_objective, "prior_objective.json"),
        (run_prior_comparison, "prior_comparison.json"),
    ),
)
def test_result_driver_defaults_do_not_target_released_results(module, filename):
    if hasattr(module, "_parse_args"):
        _, args = module._parse_args([])
    else:
        args = module._build_parser().parse_args([])
    assert Path(args.out) == Path(paths.REPRODUCED_RESULTS_DIR, filename)
    assert Path(args.out).parent != Path(paths.RESULTS_DIR)


def test_quick_modes_use_distinct_output_names_and_known_small_preset(tmp_path):
    _, gamma = run_gamma_sweep._parse_args(["--quick"])
    assert Path(gamma.out).name == "gamma_sweep_quick.json"

    _, prior = run_prior_objective._parse_args(["--quick"])
    assert Path(prior.out).name == "prior_objective_quick.json"
    assert prior.k == [3]
    assert prior.models == ["trunc"]
    assert (prior.n_algo_samples, prior.n_mc) == (20000, 20000)

    explicit_out = str(tmp_path / "chosen.json")
    _, filtered = run_prior_objective._parse_args([
        "--quick", "--models", "t", "--out", explicit_out,
    ])
    assert filtered.out == explicit_out
    assert filtered.k is None
    assert filtered.models == ["t"]


def test_diagnostic_default_input_is_package_anchored():
    args = diag_residual_terms._build_parser().parse_args([])
    assert Path(args.results) == Path(paths.RESULTS_DIR, "prior_comparison_t.json")
    assert callable(diag_residual_terms.diagnose)


@pytest.mark.parametrize(
    ("parser", "argv"),
    (
        (run_gamma_sweep._build_parser(), ["--alpha", "1"]),
        (run_prior_objective._build_parser(), ["--n-mc", "0"]),
        (run_prior_comparison._build_parser(), ["--model", "unknown"]),
        (run_prior_comparison._build_parser(), ["--theta", "nan"]),
        (check_prior_monotonicity._build_parser(), ["--grid", "1"]),
        (check_weak_monotonicity._build_parser(), ["--seed", "-1"]),
        (verify_general_gamma_objective._build_parser(), ["--n", "1"]),
    ),
)
def test_invalid_cli_values_fail_during_parsing(parser, argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(argv)
    assert exc_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_empty_predefined_filters_fail_before_computation(tmp_path, capsys):
    out = str(tmp_path / "unused.json")
    with pytest.raises(SystemExit) as gamma_exit:
        run_gamma_sweep.main(["--k", "7", "--out", out])
    assert gamma_exit.value.code == 2

    with pytest.raises(SystemExit) as prior_exit:
        run_prior_objective.main(["--k", "7", "--out", out])
    assert prior_exit.value.code == 2
    assert "match no predefined setting" in capsys.readouterr().err


def test_model_specific_theta_validation_precedes_computation(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        run_prior_comparison.main([
            "--model", "t", "--theta", "-2", "--out",
            str(tmp_path / "unused.json"),
        ])
    assert exc_info.value.code == 2
    assert "must be positive" in capsys.readouterr().err


def test_prior_objective_certification_controls_exit_status(capsys):
    certified = {"converged": True}
    uncertified = {
        "converged": False,
        "model": "t",
        "theta": 6.0,
        "K": 6,
        "prior_name": "binomial p=0.5",
        "reason": "delta",
        "final_residual": 0.00159,
    }

    assert run_prior_objective._certification_status([certified]) == 0
    assert run_prior_objective._certification_status([uncertified]) == 2
    assert "Results were saved for diagnosis" in capsys.readouterr().err

    assert run_prior_objective._certification_status(
        [uncertified], allow_uncertified=True,
    ) == 0
    assert "--allow-uncertified was set" in capsys.readouterr().err


def test_prior_comparison_passes_certification_policy_to_exit_status(
        monkeypatch, tmp_path):
    def fake_run_one(K, model, theta, *_args, **_kwargs):
        return {
            "K": K,
            "model": model,
            "theta": theta,
            "curves": {"SPOT": {"power": [0.0, 0.5]}},
            "min_gain": 0.1,
            "gains": [0.1],
            "max_fwer": 0.05,
            "converged": False,
            "n_active_multipliers": 1,
            "reason": "delta",
            "final_residual": 0.01,
        }

    calls = []

    def fake_status(records, allow_uncertified):
        calls.append((len(records), allow_uncertified))
        return 0 if allow_uncertified else 2

    monkeypatch.setattr(run_prior_comparison, "run_one", fake_run_one)
    monkeypatch.setattr(run_prior_comparison, "CANDIDATES", [])
    monkeypatch.setattr(
        run_prior_comparison, "_certification_status", fake_status,
    )
    out = str(tmp_path / "unused.json")

    assert run_prior_comparison.main(["--k", "1", "--out", out]) == 2
    assert run_prior_comparison.main([
        "--k", "1", "--out", out, "--allow-uncertified",
    ]) == 0
    assert calls == [(1, False), (1, True)]


def test_prior_objective_and_comparison_keep_their_released_schemas(monkeypatch):
    diagnostics = {
        "converged": False,
        "reason": "delta",
        "sweeps": 2,
        "final_residual": 0.01,
        "floor": {"binding_gamma": 0},
    }
    monkeypatch.setattr(
        run_prior_objective,
        "coordinate_descent_fast",
        lambda **_kwargs: (run_prior_objective.np.array([0.1]), None, diagnostics),
    )
    curves = {
        name: {"power": [0.0, 0.5], "fwer": [0.05, 0.0]}
        for name in (*run_prior_objective.COMPARATORS, "SPOT")
    }
    monkeypatch.setattr(
        run_prior_objective, "gamma_sweep", lambda *_args, **_kwargs: curves,
    )
    prior = run_prior_objective.np.array([0.0, 1.0])

    objective = run_prior_objective.run_one(
        1, "trunc", -2.0, prior, 0.05, 10, 10,
    )
    comparison = run_prior_objective.run_one(
        1, "trunc", -2.0, prior, 0.05, 10, 10, include_floor=True,
    )

    assert "floor" not in objective
    assert comparison["floor"] == {"binding_gamma": 0}


def test_gamma_sweep_uncertified_policy_controls_exit_status(
        monkeypatch, tmp_path, capsys):
    record = {
        "K": 3,
        "model": "trunc",
        "theta": -2.0,
        "converged": False,
        "final_residual": 0.01,
        "solve_seconds": 0.0,
        "sweep_seconds": 0.0,
    }
    monkeypatch.setattr(run_gamma_sweep, "run_setting", lambda *_a, **_k: dict(record))
    monkeypatch.setattr(run_gamma_sweep, "crossover", lambda _record: None)
    monkeypatch.setattr(run_gamma_sweep, "summarise", lambda _record: "summary")
    monkeypatch.setattr(run_gamma_sweep, "save_results", lambda *_a, **_k: None)
    base = ["--quick", "--k", "3", "--models", "trunc",
            "--out", str(tmp_path / "gamma.json")]

    assert run_gamma_sweep.main(base) == 2
    assert "did not meet the residual certificate" in capsys.readouterr().err
    assert run_gamma_sweep.main([*base, "--allow-uncertified"]) == 0
    assert "--allow-uncertified was set" in capsys.readouterr().err


def test_grid_uncertified_policy_controls_exit_status(monkeypatch, tmp_path, capsys):
    failed = {
        3: {"trunc": {-2.0: {"diagnostics": {
            "converged": False,
            "reason": "max_iter",
            "final_residual": 0.01,
        }}}},
    }
    monkeypatch.setattr(run_all_experiments, "estimate_time", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(run_all_experiments, "run_all", lambda *_a, **_k: failed)
    base = ["--quick", "--out", str(tmp_path / "grid.json")]

    assert run_all_experiments.main(base) == 2
    assert "did not meet the residual certificate" in capsys.readouterr().err
    assert run_all_experiments.main([*base, "--allow-uncertified"]) == 0
    assert "--allow-uncertified was set" in capsys.readouterr().err


def test_grid_estimate_rejects_an_empty_filtered_design(capsys):
    with pytest.raises(SystemExit) as exc_info:
        run_all_experiments.main(["--quick", "--models", "t", "--estimate"])
    assert exc_info.value.code == 2
    assert "contain no settings" in capsys.readouterr().err


def test_prior_monotonicity_standard_error_uses_requested_alpha():
    assert check_prior_monotonicity._mc_standard_error(0.2, 100) == pytest.approx(0.04)


def test_weak_monotonicity_requires_explicit_acknowledgement(
        monkeypatch, tmp_path, capsys):
    source = tmp_path / "policies.json"
    source.write_text(
        '[{"model":"trunc","theta":-2,"K":1,'
        '"prior_name":"test","mu":[0.0]}]\n',
        encoding="utf-8",
    )
    values = run_prior_objective.np.array([0])
    monkeypatch.setattr(
        check_weak_monotonicity,
        "check",
        lambda *_args, **_kwargs: (1, 0, values, values),
    )
    base = ["--results", str(source), "--n-pairs", "1"]

    assert check_weak_monotonicity.main(base) == 2
    assert "intentionally acknowledged" in capsys.readouterr().err
    assert check_weak_monotonicity.main([*base, "--allow-violations"]) == 0
    assert "--allow-violations was set" in capsys.readouterr().err


def test_diagnostic_rejects_mixed_settings():
    base = {
        "K": 6,
        "model": "t",
        "theta": 6.0,
        "alpha": 0.05,
        "n_algo_samples": 100000,
        "prior_name": "Beta(1,1) = uniform",
        "mu": [0.0] * 6,
        "converged": True,
        "reason": "residual",
        "sweeps": 1,
        "final_residual": 0.0,
    }
    other = dict(base, theta=8.0)
    with pytest.raises(ValueError, match="multiple settings"):
        diag_residual_terms._setting([base, other])


@pytest.mark.parametrize(
    "factory",
    (
        alternatives.constant_g,
        alternatives.truncated_normal_g,
        alternatives.normal_g,
        alternatives.mixture_normal_g,
        alternatives.student_t_g,
        alternatives.beta_g,
    ),
)
def test_convenience_factories_are_documented(factory):
    assert factory.__doc__
