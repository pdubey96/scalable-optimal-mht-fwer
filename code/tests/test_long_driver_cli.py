"""Fast CLI contract tests for the two expensive paper drivers.

The real numerical workloads are covered elsewhere.  These tests replace the
solver at the driver boundary so they can verify mode defaults, resource
choices, argument propagation, output isolation, and atomic writes without
turning a command-line test into an hours-long reproduction run.
"""

import json

import numpy as np
import pytest

import run_applications as applications
import run_scaling_convergence as scaling


@pytest.mark.parametrize("module", [applications, scaling])
def test_long_drivers_require_an_explicit_mode(module):
    with pytest.raises(SystemExit) as exc:
        module._parse_args([])
    assert exc.value.code == 2


@pytest.mark.parametrize("module", [applications, scaling])
@pytest.mark.parametrize(
    ("text", "expected"),
    [("auto", "auto"), ("0", 0), ("2GiB", 2 * 1024 ** 3),
     ("3.5GB", 3_500_000_000)],
)
def test_cache_size_parser(module, text, expected):
    assert module.parse_cache_bytes(text) == expected


@pytest.mark.parametrize("module", [applications, scaling])
def test_cache_size_parser_rejects_ambiguous_values(module):
    with pytest.raises(Exception):
        module.parse_cache_bytes("lots")


def test_application_modes_resolve_to_documented_budgets(tmp_path):
    quick = applications._resolve_args(applications._parse_args([
        "--quick", "--out", str(tmp_path / "quick.json")
    ]))
    assert (quick.n_opt, quick.tol, quick.max_iter, quick.seed) == (
        2_000, 1e-3, 25, 42
    )

    paper = applications._resolve_args(applications._parse_args([
        "--paper", "--out", str(tmp_path / "paper.json")
    ]))
    assert (paper.n_opt, paper.tol, paper.max_iter, paper.seed) == (
        100_000, 1e-3, 200, 42
    )


def test_scaling_paper_mode_is_the_exact_submitted_design(tmp_path):
    config = scaling._resolve_args(scaling._parse_args([
        "--paper",
        "--scaling-out", str(tmp_path / "scaling.json"),
        "--sensitivity-out", str(tmp_path / "sensitivity.json"),
        "--convergence-plot", str(tmp_path / "convergence.pdf"),
        "--scaling-plot", str(tmp_path / "scaling.pdf"),
    ]))
    assert config.k_values == [3, 4, 5, 6, 8, 10, 12]
    assert (config.alpha, config.theta) == (0.05, -2.0)
    assert (config.n_opt, config.n_eval) == (100_000, 50_000)
    assert (config.tol, config.max_iter) == (1e-3, 100)
    assert (config.seed_opt, config.seed_eval) == (42, 43)
    assert config.sensitivity_k == 6
    assert config.alpha_sensitivity == [0.01, 0.10]
    assert config.n_opt_sensitivity == [50_000, 200_000]


def test_default_outputs_are_reproduced_artifacts(monkeypatch, tmp_path):
    result_dir = tmp_path / "reproduced" / "results"
    plot_dir = tmp_path / "reproduced" / "plots"
    monkeypatch.setattr(
        applications.paths, "reproduced_result",
        lambda name: str(result_dir / name),
    )
    app_config = applications._resolve_args(
        applications._parse_args(["--quick"])
    )
    assert app_config.out == str(result_dir / "applications_quick.json")

    monkeypatch.setattr(
        scaling.paths, "reproduced_result", lambda name: str(result_dir / name)
    )
    monkeypatch.setattr(
        scaling.paths, "reproduced_plot", lambda name: str(plot_dir / name)
    )
    scale_config = scaling._resolve_args(scaling._parse_args(["--paper"]))
    assert scale_config.scaling_out == str(result_dir / "scaling_results.json")
    assert scale_config.sensitivity_out == str(
        result_dir / "sensitivity_results.json"
    )
    assert scale_config.convergence_plot == str(
        plot_dir / "convergence_trajectory.pdf"
    )
    assert scale_config.scaling_plot == str(plot_dir / "scaling_with_K.pdf")


def test_k21_auto_cache_streams_for_paper_but_caches_quick_run():
    paper = applications.optimizer_memory_plan(21, 100_000, "auto")
    quick = applications.optimizer_memory_plan(21, 2_000, "auto")
    explicit = applications.optimizer_memory_plan(21, 100_000, 9_000_000_000)

    assert paper["fixed_batches"] == 16 * 100_000 * 21 ** 2
    assert paper["streaming_lower_bound"] == (
        24 * 100_000 * 21 ** 2 + 8 * 100_000 * 21
    )
    assert paper["full_cache_lower_bound"] == (
        24 * 100_000 * 21 ** 2 + 8 * 100_000 * 21 ** 3
    )
    assert not paper["cache_enabled"]
    assert paper["resolved_cache_bytes"] == 0
    assert quick["cache_enabled"]
    assert explicit["cache_enabled"]


def test_scaling_auto_cache_preserves_paper_k12_cache_behavior():
    plan = scaling.optimizer_memory_plan(12, 100_000, "auto")
    assert plan["cache_enabled"]
    assert plan["full_cache_lower_bound"] < 2 * 1024 ** 3
    assert plan["coefficient_cache"] < plan["resolved_cache_bytes"]


def _fake_application_result(K):
    return {
        "l_star": 0,
        "rejected": [],
        "sweeps": 1,
        "seconds": 0.0,
        "converged": True,
        "reason": "residual",
        "final_residual": 0.0,
        "res_tol": 1.0,
        "mu": np.zeros(K).tolist(),
    }


def test_application_cli_controls_reach_every_solver_call(
        monkeypatch, tmp_path):
    calls = []

    def fake_decision(_g, K, _p_values, **kwargs):
        calls.append((K, kwargs))
        return _fake_application_result(K)

    monkeypatch.setattr(applications, "algorithm_decision", fake_decision)
    monkeypatch.setattr(applications, "optimal_l_star", lambda *_args: 0)
    monkeypatch.setattr(applications, "report", lambda *_args: None)
    output = tmp_path / "applications.json"
    config = applications._resolve_args(applications._parse_args([
        "--quick", "--n-opt", "7", "--tol", "0.02", "--max-iter", "3",
        "--seed", "99", "--cache-bytes", "0", "--out", str(output),
    ]))

    applications.main(config)

    assert len(calls) == 15
    assert sum(K == 21 for K, _ in calls) == 4
    for _K, kwargs in calls:
        assert kwargs["n_opt"] == 7
        assert kwargs["tol"] == 0.02
        assert kwargs["max_iter"] == 3
        assert kwargs["seed"] == 99
        assert kwargs["cache_bytes"] == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == {
        "camerer", "camerer_sensitivity_1e-10", "sprint", "osc", "bcg",
        "_run_config",
    }
    assert payload["_run_config"] == {
        "mode": "quick",
        "n_opt": 7,
        "tol": 0.02,
        "max_iter": 3,
        "seed": 99,
        "cache_bytes_requested": 0,
        "cache_bytes_resolved_by_k": {"2": 0, "3": 0, "5": 0,
                                         "6": 0, "21": 0},
        "allow_uncertified": False,
    }
    assert not list(tmp_path.glob(".applications.json.*.tmp"))


def _fake_scaling_result(K):
    method = {"pi_K": 0.5, "pi_any": 0.6, "fwer_0": 0.01}
    return {
        "mu": [0.0] * K,
        "sweeps": 1,
        "opt_seconds": 0.01,
        "total_seconds": 0.02,
        "deltas": [0.1],
        "residuals": [0.0],
        "converged": True,
        "reason": "residual",
        "res_tol": 1.0,
        "results": {"Algorithm": method},
    }


def test_scaling_cli_controls_and_sensitivity_arms_reach_run_setting(
        monkeypatch, tmp_path):
    calls = []
    plotted_tolerances = []

    def fake_run_setting(K, **kwargs):
        calls.append((K, kwargs))
        return _fake_scaling_result(K)

    monkeypatch.setattr(scaling, "run_setting", fake_run_setting)
    monkeypatch.setattr(
        scaling, "plot_convergence_figure",
        lambda _results, _output, *, tol: plotted_tolerances.append(tol),
    )
    monkeypatch.setattr(
        scaling, "plot_scaling_figure",
        lambda _results, _output, _k_values: None,
    )
    scaling_out = tmp_path / "scaling.json"
    sensitivity_out = tmp_path / "sensitivity.json"
    config = scaling._resolve_args(scaling._parse_args([
        "--quick", "--k-values", "3", "4", "--alpha", "0.04",
        "--theta", "-1.5", "--n-opt", "7", "--n-eval", "11",
        "--tol", "0.02", "--max-iter", "3", "--seed-opt", "97",
        "--seed-eval", "98", "--cache-bytes", "0",
        "--sensitivity-k", "5", "--alpha-sensitivity", "0.02", "0.08",
        "--n-opt-sensitivity", "2", "9",
        "--scaling-out", str(scaling_out),
        "--sensitivity-out", str(sensitivity_out),
        "--convergence-plot", str(tmp_path / "convergence.pdf"),
        "--scaling-plot", str(tmp_path / "scaling.pdf"),
    ]))

    scaling.main(config)

    assert [(K, call["alpha"], call["n_opt"]) for K, call in calls] == [
        (3, 0.04, 7), (4, 0.04, 7),
        (5, 0.02, 7), (5, 0.08, 7),
        (5, 0.04, 2), (5, 0.04, 9),
    ]
    for _K, call in calls:
        assert call["n_eval"] == 11
        assert call["theta"] == -1.5
        assert call["tol"] == 0.02
        assert call["max_iter"] == 3
        assert (call["seed_opt"], call["seed_eval"]) == (97, 98)
        assert call["cache_bytes"] == 0
    scaling_payload = json.loads(scaling_out.read_text(encoding="utf-8"))
    assert set(scaling_payload) == {"3", "4"}
    assert scaling_payload["3"]["_run_config"]["arm"] == "scaling"
    assert scaling_payload["3"]["_run_config"]["tol"] == 0.02
    assert scaling_payload["3"]["_run_config"]["cache_bytes_resolved"] == 0
    sensitivity = json.loads(sensitivity_out.read_text(encoding="utf-8"))
    assert set(sensitivity["alpha"]) == {"0.02", "0.08"}
    assert set(sensitivity["n_opt"]) == {"2", "9"}
    assert sensitivity["alpha"]["0.02"]["_run_config"]["arm"] == (
        "alpha_sensitivity"
    )
    assert sensitivity["n_opt"]["9"]["_run_config"]["arm"] == (
        "n_opt_sensitivity"
    )
    assert plotted_tolerances == [0.02]
    assert not list(tmp_path.glob(".*.tmp"))


def test_application_report_exposes_certified_and_uncertified_status(capsys):
    certified = _fake_application_result(1)
    uncertified = _fake_application_result(1)
    uncertified.update({
        "converged": False,
        "reason": "max_iter",
        "final_residual": 0.125,
        "res_tol": 0.01,
    })

    applications.report(
        "example", ["study"], [0.5],
        {"certified": certified, "failed": uncertified}, {},
    )

    output = capsys.readouterr().out
    assert "status=certified" in output
    assert "status=UNCERTIFIED" in output
    assert "reason=max_iter" in output
    assert "residual=1.25e-01" in output
    assert "certificate_tol=1.00e-02" in output


@pytest.mark.parametrize(("allow", "expected_code"), [(False, 2), (True, 0)])
def test_application_cli_requires_explicit_acceptance_of_uncertified_run(
        monkeypatch, tmp_path, capsys, allow, expected_code):
    call_count = 0

    def fake_decision(_g, K, _p_values, **_kwargs):
        nonlocal call_count
        call_count += 1
        result = _fake_application_result(K)
        # The eighth solve is hierarchical and bypasses report().
        if call_count == 8:
            result.update({
                "converged": False,
                "reason": "max_iter",
                "final_residual": 0.2,
                "res_tol": 0.01,
            })
        return result

    monkeypatch.setattr(applications, "algorithm_decision", fake_decision)
    monkeypatch.setattr(applications, "optimal_l_star", lambda *_args: 0)
    output = tmp_path / f"applications-{allow}.json"
    args = ["--quick", "--out", str(output)]
    if allow:
        args.append("--allow-uncertified")

    assert applications.cli(args) == expected_code
    captured = capsys.readouterr()
    assert call_count == 15
    assert output.is_file()
    assert "group A (strong-effect)" in captured.out
    assert "status=UNCERTIFIED" in captured.out
    assert "uncertified optimizer runs: 1 of 15" in captured.err
    assert "--allow-uncertified" in captured.err
    assert ("WARNING:" in captured.err) is allow


@pytest.mark.parametrize(("allow", "expected_code"), [(False, 2), (True, 0)])
def test_scaling_cli_checks_sensitivity_certificates_too(
        monkeypatch, tmp_path, capsys, allow, expected_code):
    call_count = 0

    def fake_run_setting(K, **_kwargs):
        nonlocal call_count
        call_count += 1
        result = _fake_scaling_result(K)
        if call_count == 3:
            result.update({"converged": False, "reason": "max_iter",
                           "residuals": [0.2], "res_tol": 0.01})
        return result

    monkeypatch.setattr(scaling, "run_setting", fake_run_setting)
    args = [
        "--quick", "--k-values", "3", "--no-plots",
        "--scaling-out", str(tmp_path / f"scaling-{allow}.json"),
        "--sensitivity-out", str(tmp_path / f"sensitivity-{allow}.json"),
    ]
    if allow:
        args.append("--allow-uncertified")

    assert scaling.cli(args) == expected_code
    captured = capsys.readouterr()
    assert call_count == 5
    assert "status=UNCERTIFIED" in captured.out
    assert "uncertified optimizer runs: 1 of 5" in captured.err
    assert "--allow-uncertified" in captured.err
    assert ("WARNING:" in captured.err) is allow


def test_scaling_explicit_active_output_never_resolves_inactive_defaults(
        monkeypatch, tmp_path):
    def unexpected_default(_name):
        raise AssertionError("an inactive/default output helper was called")

    monkeypatch.setattr(scaling.paths, "reproduced_result", unexpected_default)
    monkeypatch.setattr(scaling.paths, "reproduced_plot", unexpected_default)

    config = scaling._resolve_args(scaling._parse_args([
        "--quick", "--skip-sensitivity", "--no-plots",
        "--scaling-out", str(tmp_path / "scaling.json"),
    ]))

    assert config.scaling_out == str(tmp_path / "scaling.json")
    assert config.sensitivity_out is None
    assert config.convergence_plot is None
    assert config.scaling_plot is None


def test_scaling_rejects_normalized_active_output_collision(tmp_path):
    first = tmp_path / "same.json"
    alias = tmp_path / "unused" / ".." / "same.json"
    with pytest.raises(ValueError, match="same output path"):
        scaling._resolve_args(scaling._parse_args([
            "--quick", "--no-plots",
            "--scaling-out", str(first),
            "--sensitivity-out", str(alias),
        ]))


@pytest.mark.parametrize(
    "args",
    [
        ["--quick", "--skip-sensitivity", "--no-plots",
         "--scaling-out", "scaling.pdf"],
        ["--quick", "--skip-sensitivity", "--scaling-out", "scaling.json",
         "--convergence-plot", "convergence.json",
         "--scaling-plot", "scaling.pdf"],
    ],
)
def test_scaling_rejects_wrong_active_output_suffix(args):
    with pytest.raises(ValueError, match="suffix"):
        scaling._resolve_args(scaling._parse_args(args))


def test_inactive_scaling_outputs_need_not_be_distinct(tmp_path):
    shared = tmp_path / "shared.json"
    config = scaling._resolve_args(scaling._parse_args([
        "--quick", "--skip-sensitivity", "--no-plots",
        "--scaling-out", str(shared), "--sensitivity-out", str(shared),
    ]))
    assert config.scaling_out == config.sensitivity_out


def test_application_rejects_non_json_output_before_computation(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        applications, "main",
        lambda _config: pytest.fail("invalid output started computation"),
    )
    assert applications.cli([
        "--quick", "--out", str(tmp_path / "applications.txt")
    ]) == 2
    assert ".json suffix" in capsys.readouterr().err


def test_custom_tolerance_controls_convergence_line_and_label(
        monkeypatch, tmp_path):
    observed = {}

    def inspect_figure(figure, _output_path, **_kwargs):
        axis = figure.axes[0]
        observed["line_y"] = [np.asarray(line.get_ydata(), dtype=float)
                              for line in axis.lines]
        observed["text"] = [item.get_text() for item in axis.texts]

    monkeypatch.setattr(scaling, "_atomic_savefig", inspect_figure)
    scaling.plot_convergence_figure(
        {3: {"deltas": [0.1, 0.01]}}, tmp_path / "convergence.pdf",
        tol=0.02,
    )

    assert any(np.allclose(values, [0.02, 0.02])
               for values in observed["line_y"] if values.size == 2)
    assert any(r"2\times10^{-2}" in text for text in observed["text"])
    assert all(r"10^{-3}" not in text for text in observed["text"])
    assert scaling._tolerance_mathtext(scaling.TOL) == "10^{-3}"


def test_exact_paper_application_keeps_strict_reference_schema(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        applications, "algorithm_decision",
        lambda _g, K, _p_values, **_kwargs: _fake_application_result(K),
    )
    monkeypatch.setattr(applications, "optimal_l_star", lambda *_args: 0)
    monkeypatch.setattr(applications, "report", lambda *_args: None)
    output = tmp_path / "applications.json"
    config = applications._resolve_args(applications._parse_args([
        "--paper", "--allow-uncertified", "--out", str(output),
    ]))

    applications.main(config)

    assert set(json.loads(output.read_text(encoding="utf-8"))) == {
        "camerer", "camerer_sensitivity_1e-10", "sprint", "osc", "bcg"
    }


def test_exact_paper_scaling_keeps_strict_reference_record_schema(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        scaling, "run_setting",
        lambda K, **_kwargs: _fake_scaling_result(K),
    )
    scaling_out = tmp_path / "scaling.json"
    sensitivity_out = tmp_path / "sensitivity.json"
    config = scaling._resolve_args(scaling._parse_args([
        "--paper", "--no-plots", "--allow-uncertified",
        "--scaling-out", str(scaling_out),
        "--sensitivity-out", str(sensitivity_out),
    ]))

    scaling.main(config)

    payload = json.loads(scaling_out.read_text(encoding="utf-8"))
    assert set(payload) == {"3", "4", "5", "6", "8", "10", "12"}
    assert all("_run_config" not in record for record in payload.values())
    sensitivity = json.loads(sensitivity_out.read_text(encoding="utf-8"))
    assert all("_run_config" not in record
               for arm in sensitivity.values() for record in arm.values())


@pytest.mark.parametrize("module", [applications, scaling])
def test_show_plan_never_starts_the_expensive_main(module, monkeypatch,
                                                   tmp_path, capsys):
    monkeypatch.setattr(
        module, "main",
        lambda _config: pytest.fail("--show-plan started the computation"),
    )
    args = ["--quick", "--show-plan"]
    if module is applications:
        args += ["--out", str(tmp_path / "applications.json")]
    else:
        args += [
            "--skip-sensitivity", "--no-plots",
            "--scaling-out", str(tmp_path / "scaling.json"),
        ]
    assert module.cli(args) == 0
    assert "Resource plan" in capsys.readouterr().out
