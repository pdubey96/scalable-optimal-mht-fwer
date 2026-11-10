"""Tests for general_k.environment.

The supplement must state the environment the results were produced in, and the
failure this guards against has already happened once: the paragraph described
a MacBook Pro with Python 3.12.7 and NumPy 1.26.4 while settings re-solved in
August came off Windows with Python 3.14.3 and NumPy 2.4.3, and nothing noticed,
because nothing was recording it.

Covers capture, round-trip, strict comparison, and the package-only comparison
used to verify dependencies on a different clean machine.
"""

import json

import pytest

from general_k import environment


class TestCapture:

    def test_has_the_fields_the_supplement_quotes(self):
        env = environment.capture()
        assert env["python"].count(".") == 2
        for name in ("numpy", "scipy"):
            assert env["packages"][name], f"{name} version not captured"
        assert env["platform"]["system"]
        assert env["platform"]["machine"]
        assert isinstance(env["cpu_count"], int) and env["cpu_count"] > 0

    def test_is_json_serialisable(self):
        json.dumps(environment.capture())


class TestRecordAndLoad:

    def test_round_trip(self, tmp_path):
        path = tmp_path / "environment.json"
        environment.record(str(path))
        assert environment.load(str(path)) == environment.capture()

    def test_load_is_none_when_nothing_recorded(self, tmp_path):
        assert environment.load(str(tmp_path / "absent.json")) is None


class TestCompare:

    def test_identical_environments_differ_in_nothing(self):
        env = environment.capture()
        assert environment.compare(env, env) == []

    def test_detects_the_drift_that_actually_happened(self):
        """The real case: a Mac/3.12.7/NumPy 1.26.4 record against Windows."""
        recorded = {
            "python": "3.12.7",
            "packages": {"numpy": "1.26.4", "scipy": "1.14.1",
                         "matplotlib": "3.9.2"},
            "platform": {"system": "Darwin", "release": "24.0.0",
                         "machine": "x86_64", "processor": "i386"},
            "cpu_count": 8,
        }
        current = {
            "python": "3.14.3",
            "packages": {"numpy": "2.4.3", "scipy": "1.17.1",
                         "matplotlib": "3.10.8"},
            "platform": {"system": "Windows", "release": "11",
                         "machine": "AMD64", "processor": "Intel64"},
            "cpu_count": 12,
        }
        fields = {d[0] for d in environment.compare(recorded, current)}
        assert fields == {"python", "numpy", "scipy", "matplotlib",
                          "platform.system", "platform.machine"}

    def test_a_single_package_bump_is_caught(self):
        env = environment.capture()
        bumped = json.loads(json.dumps(env))
        bumped["packages"]["numpy"] = "0.0.1"
        diffs = environment.compare(bumped, env)
        assert [d[0] for d in diffs] == ["numpy"]

    def test_release_alone_is_not_treated_as_drift(self):
        """
        A patched OS is not a reason to distrust a number; the system and the
        architecture are. Keeping release out of the comparison stops the check
        from crying wolf on every update.
        """
        env = environment.capture()
        other = json.loads(json.dumps(env))
        other["platform"]["release"] = "something-else"
        assert environment.compare(other, env) == []


class TestComparePackages:

    @staticmethod
    def _different_platforms_with_matching_packages():
        recorded = {
            "python": "3.14.3",
            "packages": {"numpy": "2.4.3", "scipy": "1.17.1",
                         "matplotlib": "3.10.8"},
            "platform": {"system": "Windows", "machine": "AMD64"},
        }
        current = {
            "python": "3.12.7",
            "packages": {"numpy": "2.4.3", "scipy": "1.17.1",
                         "matplotlib": "3.10.8"},
            "platform": {"system": "Darwin", "machine": "x86_64"},
        }
        return recorded, current

    def test_ignores_python_and_platform_drift(self):
        recorded, current = self._different_platforms_with_matching_packages()
        assert environment.compare_packages(recorded, current) == []
        strict_fields = {d[0] for d in environment.compare(recorded, current)}
        assert strict_fields == {"python", "platform.system", "platform.machine"}

    def test_detects_package_drift(self):
        recorded, current = self._different_platforms_with_matching_packages()
        current["packages"]["scipy"] = "0.0.1"
        assert environment.compare_packages(recorded, current) == [
            ("scipy", "1.17.1", "0.0.1")
        ]


class TestCommandLine:

    def test_package_check_succeeds_across_platforms(
            self, monkeypatch, capsys):
        recorded, current = (
            TestComparePackages._different_platforms_with_matching_packages())
        monkeypatch.setattr(environment, "load", lambda: recorded)
        monkeypatch.setattr(environment, "capture", lambda: current)

        assert environment._main(["--check-packages"]) == 0
        assert "Python and platform not checked" in capsys.readouterr().out

    def test_strict_check_remains_strict(self, monkeypatch, capsys):
        recorded, current = (
            TestComparePackages._different_platforms_with_matching_packages())
        monkeypatch.setattr(environment, "load", lambda: recorded)
        monkeypatch.setattr(environment, "capture", lambda: current)

        assert environment._main(["--check"]) == 1
        output = capsys.readouterr().out
        assert "environment differs from the recorded one" in output
        assert "recorded 3.14.3, now 3.12.7" in output

    def test_check_modes_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            environment._main(["--check", "--check-packages"])
        assert exc.value.code == 2

    def test_help_distinguishes_the_two_checks(self, capsys):
        with pytest.raises(SystemExit) as exc:
            environment._main(["--help"])
        assert exc.value.code == 0
        output = capsys.readouterr().out
        assert "--check-dependencies" in output
        assert "original strict audit" in output
        assert "different platform" in output
