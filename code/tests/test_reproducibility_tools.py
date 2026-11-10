import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from general_k import paths
from general_k.io import atomic_json_dump, atomic_text_write
import make_table_scaling
from verify_reproduction import compare_file, compare_values, main


CODE_DIR = Path(__file__).resolve().parent.parent


def test_reproduced_outputs_are_separate_from_references():
    assert Path(paths.REPRODUCED_RESULTS_DIR) != Path(paths.RESULTS_DIR)
    assert Path(paths.REPRODUCED_PLOTS_DIR) != Path(paths.PLOTS_DIR)
    assert paths.reproduced_result("x.json").startswith(paths.OUTPUT_ROOT)


def test_atomic_json_is_strict_and_converts_numpy(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    atomic_json_dump({"x": np.array([np.int64(2), np.float64(3.5)])}, destination)
    assert json.loads(destination.read_text()) == {"x": [2.0, 3.5]}
    assert not list(destination.parent.glob("*.tmp"))


def test_atomic_text_write(tmp_path):
    destination = tmp_path / "table.tex"
    atomic_text_write("first\n", destination)
    atomic_text_write("second\n", destination)
    assert destination.read_text() == "second\n"


def test_comparator_ignores_time_but_checks_science():
    reference = {"power": [0.5, 0.75], "elapsed_seconds": 10.0, "ok": True}
    candidate = {"power": [0.500001, 0.75], "elapsed_seconds": 999.0, "ok": True}
    assert compare_values(reference, candidate, rtol=1e-5, atol=5e-5) == []
    candidate["power"][1] = 0.7
    assert "$.power[1]" in compare_values(
        reference, candidate, rtol=1e-5, atol=5e-5
    )[0]


def test_comparator_requires_exact_configuration_and_source_values():
    reference = {
        "alpha": 0.05,
        "prior": [0.0, 1.0],
        "pooled": {"beta": -0.7},
        "power": 0.5,
    }
    candidate = {
        "alpha": 0.050049,
        "prior": [0.0, 1.00001],
        "pooled": {"beta": -0.70001},
        "power": 0.50001,
    }
    mismatches = compare_values(reference, candidate, rtol=1e-5, atol=5e-5)
    assert any("$.alpha" in mismatch for mismatch in mismatches)
    assert any("$.prior[1]" in mismatch for mismatch in mismatches)
    assert any("$.pooled.beta" in mismatch for mismatch in mismatches)
    assert not any("$.power" in mismatch for mismatch in mismatches)


def test_exact_scalar_name_does_not_make_a_whole_result_branch_exact():
    reference = {"alpha": {"0.05": {"power": 0.5}}}
    candidate = {"alpha": {"0.05": {"power": 0.50001}}}
    assert compare_values(reference, candidate, rtol=1e-5, atol=5e-5) == []


def test_text_comparison_is_exact(tmp_path):
    reference = tmp_path / "reference.tex"
    candidate = tmp_path / "candidate.tex"
    reference.write_text("same\n")
    candidate.write_text("same\n")
    assert compare_file(reference, candidate, rtol=0, atol=0,
                        ignored_keys=frozenset()) == []
    candidate.write_text("different\n")
    assert compare_file(reference, candidate, rtol=0, atol=0,
                        ignored_keys=frozenset())


def test_cli_reports_pass_and_missing(tmp_path, capsys):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    (reference / "one.json").write_text('{"value": 1, "seconds": 2}\n')
    (candidate / "one.json").write_text('{"value": 1, "seconds": 20}\n')

    assert main([str(candidate), "--reference", str(reference),
                 "--files", "one.json"]) == 0
    assert "PASS one.json" in capsys.readouterr().out

    assert main([str(candidate), "--reference", str(reference),
                 "--files", "missing.json"]) == 2
    assert "missing" in capsys.readouterr().out


def test_verifier_rejects_self_comparison(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path), "--reference", str(tmp_path), "--files", "x.json"])
    assert exc_info.value.code == 2
    assert "must be different" in capsys.readouterr().err


def test_verifier_rejects_hardlinked_candidate(tmp_path, capsys):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    source = reference / "one.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    try:
        (candidate / "one.json").hardlink_to(source)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    assert main([str(candidate), "--reference", str(reference),
                 "--files", "one.json"]) == 1
    assert "same file" in capsys.readouterr().out


def test_scaling_table_uses_recorded_evaluation_budget(tmp_path):
    methods = {
        name: {"pi_K": 0.5}
        for name in make_table_scaling.COLUMNS
    }
    methods["Algorithm"].update({"fwer_0": 0.05})
    source = tmp_path / "scaling.json"
    destination = tmp_path / "table.tex"
    source.write_text(json.dumps({
        "3": {
            "results": methods,
            "_run_config": {"n_eval": 100},
        },
    }), encoding="utf-8")

    assert make_table_scaling.main([
        "--input", str(source), "--out", str(destination),
    ]) is None
    assert r"$0.050\ (0.007,0.093)$" in destination.read_text(encoding="utf-8")


def test_grid_cli_defaults_to_the_complete_published_grid(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(CODE_DIR / "run_all_experiments.py"), "--estimate"],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "K=6, Beta" in completed.stdout
    assert "112" not in completed.stderr


def test_grid_quick_preset_is_one_small_setting(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(CODE_DIR / "run_all_experiments.py"),
         "--quick", "--estimate"],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "K=3, Truncated Normal" in completed.stdout
    assert "K=4" not in completed.stdout
    assert "1 points" in completed.stdout
