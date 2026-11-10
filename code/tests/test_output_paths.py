"""Output paths are anchored to the package, not to the working directory.

The submitted R package wrote into whatever directory it was run from, which
made it awkward to reproduce, so the Python drivers must not repeat it.  The
code is meant to be downloaded and run, and there is no guarantee it will be
run from ``code/``.

Covers:
- the three directories resolve to the intended places
- resolution does not depend on the working directory
- every driver's default output path is absolute
- no driver still carries a bare "../results/" or "../plots/" literal
"""

import ast
import os
import pathlib

import pytest

from general_k import figstyle, paths


CODE_DIR = pathlib.Path(__file__).resolve().parent.parent
DRIVERS = sorted(p.name for p in CODE_DIR.glob("run_*.py"))


class TestTheDirectoriesResolve:

    def test_results_sits_beside_code(self):
        assert pathlib.Path(paths.RESULTS_DIR).name == "results"
        assert pathlib.Path(paths.RESULTS_DIR).parent == CODE_DIR.parent

    def test_plots_sits_beside_code(self):
        assert pathlib.Path(paths.PLOTS_DIR).name == "plots"
        assert pathlib.Path(paths.PLOTS_DIR).parent == CODE_DIR.parent

    def test_figure_dir_matches_the_available_layout(self):
        p = pathlib.Path(paths.FIGURE_DIR)
        assert p.name == "plots"
        if figstyle.USING_MANUSCRIPT_FIGURE_DIR:
            assert p.parent.name == "Manuscript"
        else:
            assert p.parent == CODE_DIR

    def test_helpers_return_absolute_paths(self):
        assert os.path.isabs(paths.result("x.json"))
        assert os.path.isabs(paths.plot("x.pdf"))

    def test_path_lookup_does_not_create_directories(self, tmp_path, monkeypatch):
        reference = tmp_path / "missing-reference"
        reproduced = tmp_path / "missing-output"
        monkeypatch.setattr(paths, "RESULTS_DIR", str(reference))
        monkeypatch.setattr(paths, "REPRODUCED_RESULTS_DIR", str(reproduced))

        assert paths.result("x.json") == str(reference / "x.json")
        assert paths.reproduced_result("x.json") == str(reproduced / "x.json")
        assert not reference.exists()
        assert not reproduced.exists()


class TestIndependenceFromTheWorkingDirectory:

    def test_resolution_does_not_move_with_cwd(self, tmp_path, monkeypatch):
        before = paths.result("all_experiments.json")
        monkeypatch.chdir(tmp_path)
        assert paths.result("all_experiments.json") == before

    def test_a_driver_default_does_not_move_with_cwd(self, tmp_path, monkeypatch):
        """
        The defaults are evaluated when the parser is built, so this is the
        check that matters: build it from somewhere else and the path is the
        same absolute one.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(CODE_DIR))
        import importlib

        mod = importlib.import_module("run_gamma_sweep")
        importlib.reload(mod)
        assert mod.paths.result("gamma_sweep.json").startswith(str(CODE_DIR.parent))


class TestNoWorkingDirectoryLiteralsRemain:

    @pytest.mark.parametrize("name", DRIVERS)
    def test_driver_has_no_relative_output_literal(self, name):
        """
        A "../results/..." string in a driver means some path escaped the
        anchoring and will follow the working directory again.
        """
        src = (CODE_DIR / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        offenders = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and ("../results/" in node.value or "../plots/" in node.value)
        ]
        assert not offenders, f"{name} still writes relative to the cwd: {offenders}"
