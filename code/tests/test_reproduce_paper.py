"""Tests for the one-command quick/full reproduction orchestrator."""

import io
import json
import sys
from pathlib import Path

import pytest

import reproduce_paper as reproduce


def _argument(command, flag):
    return command[command.index(flag) + 1]


class TestPlans:

    def test_quick_is_the_default_and_does_not_verify_reduced_results(self, tmp_path):
        args = reproduce._build_parser().parse_args(["--output-dir", str(tmp_path)])
        assert args.mode == "quick"

        steps = reproduce.build_steps("quick", tmp_path.resolve(), jobs=2)
        names = [step.name for step in steps]
        assert names[:10] == [
            "environment",
            "grid_quick",
            "untruncated_normal_quick",
            "scaling_and_sensitivity_quick",
            "gamma_sweep_quick",
            "prior_objective_quick",
            "prior_comparison_trunc_quick",
            "prior_comparison_t_quick",
            "prior_comparison_beta_quick",
            "applications_quick",
        ]
        assert "verify_results" not in names
        assert "--quick" in steps[1].command
        assert "--quick" in steps[3].command
        assert "--quick" in steps[4].command
        assert "--quick" in steps[5].command
        assert "--quick" in steps[9].command
        assert "--allow-uncertified" not in steps[5].command
        for step in steps[6:9]:
            assert "--allow-uncertified" in step.command

        by_name = {step.name: step for step in steps}
        for name, reference_name in [
            ("table_scaling", "scaling_results.json"),
            ("plot_power_curves", "all_experiments.json"),
            ("plot_prior_objective", "prior_objective.json"),
            ("plot_gamma_sweep", "gamma_sweep.json"),
        ]:
            assert Path(_argument(by_name[name].command, "--input")) == (
                reproduce.REFERENCE_RESULTS_DIR / reference_name)
        assert Path(_argument(
            by_name["weak_monotonicity"].command, "--results")) == (
                reproduce.REFERENCE_RESULTS_DIR / "prior_objective.json")
        assert "--allow-violations" in by_name["weak_monotonicity"].command

    def test_full_plan_has_paper_order_and_exact_prior_comparisons(self, tmp_path):
        steps = reproduce.build_steps("full", tmp_path.resolve(), jobs=6)
        names = [step.name for step in steps]
        assert names[:11] == [
            "check_packages",
            "environment",
            "grid_112_settings",
            "untruncated_normal",
            "scaling_and_sensitivity",
            "gamma_sweep",
            "prior_objective",
            "prior_comparison_trunc",
            "prior_comparison_t",
            "prior_comparison_beta",
            "applications",
        ]
        assert names[-1] == "verify_results"

        by_name = {step.name: step for step in steps}
        assert by_name["check_packages"].command == (
            sys.executable, "-m", "general_k.environment", "--check-packages")
        assert by_name["environment"].command[:4] == (
            sys.executable, "-m", "general_k.environment", "--out")
        grid = by_name["grid_112_settings"].command
        assert grid[grid.index("--k") + 1:grid.index("--models")] == (
            "3", "4", "5", "6")
        assert grid[grid.index("--models") + 1:grid.index("--jobs")] == (
            "trunc", "mixture", "t", "beta")
        assert _argument(grid, "--jobs") == "6"

        expected = {
            "prior_comparison_trunc": (("6", "12"), "trunc", "-2"),
            "prior_comparison_t": (("6",), "t", "6"),
            "prior_comparison_beta": (("6", "12"), "beta", "0.4"),
        }
        for name, (ks, model, theta) in expected.items():
            command = by_name[name].command
            assert command[command.index("--k") + 1:command.index("--model")] == ks
            assert _argument(command, "--model") == model
            assert _argument(command, "--theta") == theta
        assert "--allow-uncertified" in by_name["prior_objective"].command
        assert "--allow-uncertified" in by_name["prior_comparison_t"].command
        assert "--allow-uncertified" not in by_name["prior_comparison_trunc"].command
        assert "--allow-uncertified" not in by_name["prior_comparison_beta"].command

        results = tmp_path.resolve() / "results"
        assert by_name["verify_results"].command[-2:] == (
            "--reference", str(reproduce.REFERENCE_RESULTS_DIR))
        assert str(results) in by_name["verify_results"].command

    def test_all_commands_use_current_python_and_outputs_stay_below_root(self, tmp_path):
        root = tmp_path.resolve()
        for mode in ("quick", "full"):
            for step in reproduce.build_steps(mode, root, jobs=1):
                assert step.command[0] == sys.executable
                for output in step.outputs:
                    output.relative_to(root)


class TestSafetyAndExecution:

    @pytest.mark.parametrize("protected", [
        reproduce.REFERENCE_RESULTS_DIR,
        reproduce.REFERENCE_PLOTS_DIR,
        reproduce.MANUSCRIPT_PLOTS_DIR,
    ])
    def test_rejects_bundled_artifact_locations(self, protected):
        with pytest.raises(ValueError, match="reference artifacts"):
            reproduce.validate_output_dir(protected)

    def test_rejects_ancestor_of_bundled_artifacts(self):
        with pytest.raises(ValueError, match="reference artifacts"):
            reproduce.validate_output_dir(reproduce.REFERENCE_RESULTS_DIR.parent)

    def test_dry_run_writes_manifest_and_logs_without_spawning(
            self, tmp_path, monkeypatch):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("dry run spawned a process")

        monkeypatch.setattr(reproduce.subprocess, "Popen", forbidden)
        assert reproduce.main(["--dry-run", "--output-dir", str(tmp_path)]) == 0

        manifest = json.loads(
            (tmp_path / reproduce.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["mode"] == "quick"
        assert manifest["status"] == "dry-run"
        assert all(item["status"] == "dry-run" for item in manifest["steps"])
        logs = sorted((tmp_path / "logs").glob("*.log"))
        assert len(logs) == len(manifest["steps"])
        assert all("DRY RUN" in path.read_text(encoding="utf-8") for path in logs)

    @pytest.mark.parametrize("name", ["results", "plots", "logs"])
    def test_rejects_symlinked_workflow_directories(self, tmp_path, name):
        output_dir = tmp_path / "run"
        outside = tmp_path / "outside"
        output_dir.mkdir()
        outside.mkdir()
        try:
            (output_dir / name).symlink_to(outside, target_is_directory=True)
        except OSError as exc:  # Windows may deny unprivileged symlink creation.
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match="symlinked workflow directory"):
            reproduce.execute_plan(
                [], mode="quick", output_dir=output_dir, jobs=1,
                dry_run=True, resume=False,
            )

        assert list(outside.iterdir()) == []

    def test_child_process_uses_argv_without_a_shell(self, tmp_path, monkeypatch):
        captured = {}

        class FakeProcess:
            stdout = io.StringIO("child output\n")

            @staticmethod
            def wait(timeout=None):
                assert timeout is None
                return 0

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            return FakeProcess()

        monkeypatch.setattr(reproduce.subprocess, "Popen", fake_popen)
        step = reproduce.Step("probe", (sys.executable, "-c", "print('ok')"))
        assert reproduce._run_step(step, tmp_path / "probe.log", {}) == 0
        assert captured["command"] == list(step.command)
        assert captured["shell"] is False
        assert captured["cwd"] == str(reproduce.CODE_DIR)

    def test_failure_stops_the_plan_and_is_recorded(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(step, log_path, _environment):
            calls.append(step.name)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("fake\n", encoding="utf-8")
            if len(calls) == 2:
                return 7
            for output in step.outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"artifact")
            return 0

        monkeypatch.setattr(reproduce, "_run_step", fake_run)
        rc = reproduce.main(["--output-dir", str(tmp_path)])
        assert rc == 7
        assert calls == ["environment", "grid_quick"]
        manifest = json.loads(
            (tmp_path / reproduce.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"
        assert manifest["steps"][0]["status"] == "completed"
        assert manifest["steps"][1]["status"] == "failed"
        assert manifest["steps"][2]["status"] == "pending"
        assert len(manifest["steps"]) > 2

    def test_resume_skips_verified_prefix_then_runs_downstream(
            self, tmp_path, monkeypatch):
        steps = [
            reproduce.Step("first", (sys.executable, "-c", "first"),
                           (tmp_path / "results" / "first.json",)),
            reproduce.Step("second", (sys.executable, "-c", "second"),
                           (tmp_path / "results" / "second.json",)),
        ]
        calls = []

        def successful(step, log_path, _environment):
            calls.append(step.name)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("done\n", encoding="utf-8")
            for output in step.outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(step.name, encoding="utf-8")
            return 0

        monkeypatch.setattr(reproduce, "_run_step", successful)
        assert reproduce.execute_plan(
            steps, mode="quick", output_dir=tmp_path, jobs=1,
            dry_run=False, resume=False,
        ) == 0
        previous = json.loads(
            (tmp_path / reproduce.MANIFEST_NAME).read_text(encoding="utf-8"))

        # Damage the second artifact. The first should be hash-verified and
        # skipped; the second and anything after it must run again.
        steps[1].outputs[0].write_text("changed", encoding="utf-8")
        calls.clear()
        assert reproduce.execute_plan(
            steps, mode="quick", output_dir=tmp_path, jobs=1,
            dry_run=False, resume=True, previous_manifest=previous,
        ) == 0
        assert calls == ["second"]

    def test_rerun_cannot_accept_a_stale_declared_output(
            self, tmp_path, monkeypatch):
        output = tmp_path / "results" / "artifact.json"
        step = reproduce.Step(
            "producer", (sys.executable, "-c", "pass"), (output,),
        )

        def writes_once(_step, log_path, _environment):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("first run\n", encoding="utf-8")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("old", encoding="utf-8")
            return 0

        monkeypatch.setattr(reproduce, "_run_step", writes_once)
        assert reproduce.execute_plan(
            [step], mode="quick", output_dir=tmp_path, jobs=1,
            dry_run=False, resume=False,
        ) == 0
        previous = json.loads(
            (tmp_path / reproduce.MANIFEST_NAME).read_text(encoding="utf-8"))

        # Force a rerun, then simulate a child that exits zero without writing.
        output.write_text("stale", encoding="utf-8")

        def writes_nothing(_step, log_path, _environment):
            log_path.write_text("second run\n", encoding="utf-8")
            return 0

        monkeypatch.setattr(reproduce, "_run_step", writes_nothing)
        assert reproduce.execute_plan(
            [step], mode="quick", output_dir=tmp_path, jobs=1,
            dry_run=False, resume=True, previous_manifest=previous,
        ) == 1
        assert not output.exists()
        manifest = json.loads(
            (tmp_path / reproduce.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"
        assert manifest["steps"][0]["status"] == "failed"

    def test_resume_requires_an_existing_manifest(self, tmp_path, capsys):
        occupied = tmp_path / "results" / "all_experiments_quick.json"
        occupied.parent.mkdir(parents=True)
        occupied.write_text("stale", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            reproduce.main(["--resume", "--output-dir", str(tmp_path)])

        assert exc_info.value.code == 2
        assert "requires an existing manifest" in capsys.readouterr().err

    def test_resume_rejects_an_incomplete_manifest(self, tmp_path, capsys):
        (tmp_path / reproduce.MANIFEST_NAME).write_text(
            '{"mode": "quick"}\n', encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc_info:
            reproduce.main(["--resume", "--output-dir", str(tmp_path)])

        assert exc_info.value.code == 2
        assert "incomplete manifest schema" in capsys.readouterr().err

    def test_resume_rejects_source_or_environment_drift(
            self, tmp_path, monkeypatch, capsys):
        identity = {
            "source_sha256": "source-a",
            "reference_sha256": "reference-a",
            "runtime": {"python": "test-a"},
        }
        monkeypatch.setattr(reproduce, "_workflow_identity", lambda: identity)
        assert reproduce.main(["--dry-run", "--output-dir", str(tmp_path)]) == 0

        changed = dict(identity, source_sha256="source-b")
        monkeypatch.setattr(reproduce, "_workflow_identity", lambda: changed)
        with pytest.raises(SystemExit) as exc_info:
            reproduce.main(["--resume", "--output-dir", str(tmp_path)])

        assert exc_info.value.code == 2
        error = capsys.readouterr().err
        assert "workflow identity drift" in error
        assert "source_sha256" in error
