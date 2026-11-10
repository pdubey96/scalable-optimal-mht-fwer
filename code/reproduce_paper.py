#!/usr/bin/env python3
"""Run the paper's computational workflow without touching bundled results.

The default is a bounded smoke workflow.  Paper-scale Monte Carlo settings are
available only through the explicit ``--full`` flag.  Every child process uses
the current interpreter, every command is printed and logged, and all generated
artifacts live below one output directory.

Examples
--------
Run the bounded installation/reproducibility smoke test::

    python reproduce_paper.py --output-dir /tmp/spot-quick

Print the same plan without executing it::

    python reproduce_paper.py --dry-run --output-dir /tmp/spot-plan

Run the complete paper workflow and later resume an interrupted run::

    python reproduce_paper.py --full --jobs 6 --output-dir /scratch/spot-full
    python reproduce_paper.py --full --jobs 6 --output-dir /scratch/spot-full --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


CODE_DIR = Path(__file__).resolve().parent
REFERENCE_RESULTS_DIR = (CODE_DIR.parent / "results").resolve()
REFERENCE_PLOTS_DIR = (CODE_DIR.parent / "plots").resolve()
MANUSCRIPT_PLOTS_DIR = (CODE_DIR.parent.parent / "Manuscript" / "plots").resolve()
DEFAULT_OUTPUT_DIR = CODE_DIR / "reproduced"
MANIFEST_NAME = "reproduction_manifest.json"
MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Step:
    """One subprocess and the artifacts that prove it completed."""

    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...] = ()


def _now() -> str:
    """Current UTC time in a stable, timezone-explicit representation."""
    return datetime.now(timezone.utc).isoformat()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _script(name: str, *arguments: object) -> tuple[str, ...]:
    """Command for a repository driver under the running interpreter."""
    return (sys.executable, str(CODE_DIR / name), *(str(item) for item in arguments))


def _environment_command(destination: Path) -> tuple[str, ...]:
    """Record through the public CLI to an explicit, output-safe path."""
    return (
        sys.executable, "-m", "general_k.environment", "--out", str(destination)
    )


def _display_command(command: Sequence[str]) -> str:
    """Render an argv vector for display without using it for execution."""
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _inside(path: Path, directory: Path) -> bool:
    """Whether *path* is *directory* or is below it."""
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def validate_output_dir(value: os.PathLike[str] | str) -> Path:
    """Resolve and reject output roots that could overwrite shipped evidence."""
    output = Path(value).expanduser().resolve()
    if output == Path(output.anchor) or output == Path.home().resolve():
        raise ValueError("--output-dir must not be a filesystem root or home directory")

    protected = (REFERENCE_RESULTS_DIR, REFERENCE_PLOTS_DIR, MANUSCRIPT_PLOTS_DIR)
    generated_locations = (output, output / "results", output / "plots")
    for generated in generated_locations:
        for reference in protected:
            if _inside(generated, reference) or _inside(reference, generated):
                raise ValueError(
                    f"refusing output location {generated}: it overlaps bundled "
                    f"reference artifacts at {reference}"
                )
    return output


def build_steps(mode: str, output_dir: Path, jobs: int) -> list[Step]:
    """Construct the ordered quick or paper-scale command plan."""
    if mode not in {"quick", "full"}:
        raise ValueError(f"unknown reproduction mode: {mode}")

    output_dir = output_dir.resolve()
    results = output_dir / "results"
    plots = output_dir / "plots"
    steps: list[Step] = []
    if mode == "full":
        # Fail before any multi-hour computation when the numerical dependency
        # versions differ from those that produced the bundled results.
        steps.append(Step(
            "check_packages",
            (sys.executable, "-m", "general_k.environment", "--check-packages"),
        ))
    steps.append(Step(
        "environment",
        _environment_command(results / "environment.json"),
        (results / "environment.json",),
    ))

    if mode == "full":
        steps.extend([
            Step(
                "grid_112_settings",
                _script(
                    "run_all_experiments.py",
                    "--k", 3, 4, 5, 6,
                    "--models", "trunc", "mixture", "t", "beta",
                    "--jobs", jobs,
                    "--out", results / "all_experiments.json",
                ),
                (results / "all_experiments.json",),
            ),
            Step(
                "untruncated_normal",
                _script(
                    "run_all_experiments.py",
                    "--k", 3, 4, 5, 6,
                    "--models", "normal",
                    "--jobs", jobs,
                    "--out", results / "untruncated_normal.json",
                ),
                (results / "untruncated_normal.json",),
            ),
            Step(
                "scaling_and_sensitivity",
                _script(
                    "run_scaling_convergence.py", "--paper",
                    "--scaling-out", results / "scaling_results.json",
                    "--sensitivity-out", results / "sensitivity_results.json",
                    "--convergence-plot", plots / "convergence_trajectory.pdf",
                    "--scaling-plot", plots / "scaling_with_K.pdf",
                ),
                (
                    results / "scaling_results.json",
                    results / "sensitivity_results.json",
                    plots / "convergence_trajectory.pdf",
                    plots / "scaling_with_K.pdf",
                ),
            ),
            Step(
                "gamma_sweep",
                _script(
                    "run_gamma_sweep.py",
                    "--out", results / "gamma_sweep.json",
                ),
                (results / "gamma_sweep.json",),
            ),
            Step(
                "prior_objective",
                _script(
                    "run_prior_objective.py",
                    "--allow-uncertified",
                    "--out", results / "prior_objective.json",
                ),
                (results / "prior_objective.json",),
            ),
            Step(
                "prior_comparison_trunc",
                _script(
                    "run_prior_comparison.py",
                    "--k", 6, 12, "--model", "trunc", "--theta", -2,
                    "--out", results / "prior_comparison.json",
                ),
                (results / "prior_comparison.json",),
            ),
            Step(
                "prior_comparison_t",
                _script(
                    "run_prior_comparison.py",
                    "--k", 6, "--model", "t", "--theta", 6,
                    "--allow-uncertified",
                    "--out", results / "prior_comparison_t.json",
                ),
                (results / "prior_comparison_t.json",),
            ),
            Step(
                "prior_comparison_beta",
                _script(
                    "run_prior_comparison.py",
                    "--k", 6, 12, "--model", "beta", "--theta", 0.4,
                    "--out", results / "prior_comparison_beta.json",
                ),
                (results / "prior_comparison_beta.json",),
            ),
            Step(
                "applications",
                _script(
                    "run_applications.py", "--paper",
                    "--out", results / "applications.json",
                ),
                (results / "applications.json",),
            ),
        ])
        post_inputs = {
            "grid": results / "all_experiments.json",
            "scaling": results / "scaling_results.json",
            "gamma": results / "gamma_sweep.json",
            "prior": results / "prior_objective.json",
        }
    else:
        small = ("--n-algo-samples", 2_000, "--n-mc", 2_000, "--max-iter", 25)
        steps.extend([
            Step(
                "grid_quick",
                _script(
                    "run_all_experiments.py", "--quick", "--jobs", jobs,
                    "--out", results / "all_experiments_quick.json",
                ),
                (results / "all_experiments_quick.json",),
            ),
            Step(
                "untruncated_normal_quick",
                _script(
                    "run_all_experiments.py", "--quick", "--models", "normal",
                    "--thetas", -2, "--jobs", jobs,
                    "--out", results / "untruncated_normal_quick.json",
                ),
                (results / "untruncated_normal_quick.json",),
            ),
            Step(
                "scaling_and_sensitivity_quick",
                _script(
                    "run_scaling_convergence.py", "--quick", "--k-values", 3,
                    "--sensitivity-k", 3,
                    "--scaling-out", results / "scaling_results_quick.json",
                    "--sensitivity-out", results / "sensitivity_results_quick.json",
                    "--convergence-plot", plots / "convergence_quick.pdf",
                    "--scaling-plot", plots / "scaling_quick.pdf",
                ),
                (
                    results / "scaling_results_quick.json",
                    results / "sensitivity_results_quick.json",
                    plots / "convergence_quick.pdf",
                    plots / "scaling_quick.pdf",
                ),
            ),
            Step(
                "gamma_sweep_quick",
                _script(
                    "run_gamma_sweep.py", "--quick", "--k", 3,
                    "--models", "trunc",
                    "--out", results / "gamma_sweep_quick.json",
                ),
                (results / "gamma_sweep_quick.json",),
            ),
            Step(
                "prior_objective_quick",
                _script(
                    "run_prior_objective.py", "--quick", "--k", 3,
                    "--models", "trunc",
                    "--out", results / "prior_objective_quick.json",
                ),
                (results / "prior_objective_quick.json",),
            ),
            Step(
                "prior_comparison_trunc_quick",
                _script(
                    "run_prior_comparison.py", "--k", 3,
                    "--model", "trunc", "--theta", -2, *small,
                    "--allow-uncertified",
                    "--out", results / "prior_comparison_quick.json",
                ),
                (results / "prior_comparison_quick.json",),
            ),
            Step(
                "prior_comparison_t_quick",
                _script(
                    "run_prior_comparison.py", "--k", 3,
                    "--model", "t", "--theta", 6, *small,
                    "--allow-uncertified",
                    "--out", results / "prior_comparison_t_quick.json",
                ),
                (results / "prior_comparison_t_quick.json",),
            ),
            Step(
                "prior_comparison_beta_quick",
                _script(
                    "run_prior_comparison.py", "--k", 3,
                    "--model", "beta", "--theta", 0.4, *small,
                    "--allow-uncertified",
                    "--out", results / "prior_comparison_beta_quick.json",
                ),
                (results / "prior_comparison_beta_quick.json",),
            ),
            Step(
                "applications_quick",
                _script(
                    "run_applications.py", "--quick",
                    "--out", results / "applications_quick.json",
                ),
                (results / "applications_quick.json",),
            ),
        ])
        # Reduced Monte Carlo files are smoke artifacts, not estimators to compare
        # with the paper.  Postprocessors instead exercise their supported paths
        # against the complete bundled inputs.
        post_inputs = {
            "grid": REFERENCE_RESULTS_DIR / "all_experiments.json",
            "scaling": REFERENCE_RESULTS_DIR / "scaling_results.json",
            "gamma": REFERENCE_RESULTS_DIR / "gamma_sweep.json",
            "prior": REFERENCE_RESULTS_DIR / "prior_objective.json",
        }

    steps.extend([
        Step(
            "table_scaling",
            _script(
                "make_table_scaling.py", "--input", post_inputs["scaling"],
                "--out", results / "table_scaling.tex",
            ),
            (results / "table_scaling.tex",),
        ),
        Step(
            "plot_power_curves",
            _script(
                "plot_power_curves.py", "--input", post_inputs["grid"],
                "--out-dir", plots,
            ),
            tuple(plots / f"power_curves_K{K}.pdf" for K in (3, 4, 5, 6)),
        ),
        Step(
            "plot_prior_objective",
            _script(
                "plot_prior_objective.py", "--input", post_inputs["prior"],
                "--out-dir", plots,
            ),
            (plots / "prior_objective.pdf", plots / "prior_objective.png"),
        ),
        Step(
            "plot_gamma_sweep",
            _script(
                "plot_gamma_sweep.py", "--input", post_inputs["gamma"],
                "--out-dir", plots,
            ),
            (plots / "gamma_sweep.pdf", plots / "gamma_sweep.png"),
        ),
        Step(
            "rank_correlation",
            _script(
                "rank_correlation_diagnostic.py", "--input", post_inputs["prior"],
            ),
        ),
        Step(
            "weak_monotonicity",
            _script(
                "check_weak_monotonicity.py",
                *(("--n-pairs", 1_000) if mode == "quick" else ()),
                "--allow-violations",
                "--results", post_inputs["prior"],
            ),
        ),
    ])

    if mode == "full":
        steps.append(Step(
            "verify_results",
            _script(
                "verify_reproduction.py", results,
                "--reference", REFERENCE_RESULTS_DIR,
            ),
        ))

    return steps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_fingerprint(root: Path, files: Sequence[Path]) -> str:
    """Hash file names and contents in a stable order."""
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix()):
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _workflow_identity() -> dict[str, object]:
    """Identity of code, dependencies, platform, and immutable input evidence."""
    source_files = [
        *CODE_DIR.glob("*.py"),
        *(CODE_DIR / "general_k").rglob("*.py"),
        CODE_DIR / "requirements.txt",
        CODE_DIR / "pyproject.toml",
    ]
    reference_files = [
        path for path in REFERENCE_RESULTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".tex"}
    ]
    packages = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "source_sha256": _collection_fingerprint(CODE_DIR, source_files),
        "reference_sha256": _collection_fingerprint(
            REFERENCE_RESULTS_DIR, reference_files,
        ),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "packages": packages,
        },
    }


def _identity_drift(previous: object, current: dict[str, object]) -> list[str]:
    """Top-level workflow-identity fields that changed since the prior run."""
    if not isinstance(previous, dict):
        return ["workflow identity is absent"]
    return [key for key, value in current.items() if previous.get(key) != value]


def _artifacts(step: Step) -> list[dict[str, object]]:
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in step.outputs
    ]


def _atomic_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _run_step(step: Step, log_path: Path, environment: dict[str, str]) -> int:
    """Run one argv vector directly, while teeing combined output to its log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write(f"command: {_display_command(step.command)}\n")
        log.write(f"cwd: {CODE_DIR}\n\n")
        log.flush()
        process = subprocess.Popen(
            list(step.command), cwd=str(CODE_DIR), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            shell=False,
        )
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def _write_dry_run_log(step: Step, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"command: {_display_command(step.command)}\n"
        f"cwd: {CODE_DIR}\n\nDRY RUN: command not executed.\n",
        encoding="utf-8",
    )


def _previous_steps(manifest: dict[str, object] | None) -> dict[str, dict[str, object]]:
    if not manifest or not isinstance(manifest.get("steps"), list):
        return {}
    return {
        item["name"]: item
        for item in manifest["steps"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _can_resume(previous: dict[str, object] | None, step: Step,
                log_path: Path) -> bool:
    if previous is None or previous.get("status") != "completed":
        return False
    if previous.get("command") != list(step.command) or not log_path.is_file():
        return False
    if any(not path.is_file() for path in step.outputs):
        return False
    recorded = previous.get("artifacts")
    if step.outputs and not isinstance(recorded, list):
        return False
    expected_hashes = {
        item.get("path"): item.get("sha256")
        for item in (recorded or []) if isinstance(item, dict)
    }
    return all(expected_hashes.get(str(path)) == _sha256(path) for path in step.outputs)


def _clear_declared_outputs(step: Step, output_dir: Path) -> None:
    """Remove a rerun step's old files so they cannot masquerade as new output."""
    root = output_dir.resolve()
    for path in step.outputs:
        resolved = path.resolve(strict=False)
        if not _inside(resolved, root):
            raise ValueError(
                f"refusing to clear declared output outside --output-dir: {path}"
            )
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            raise ValueError(f"declared output is not a regular file: {path}")


def _prepare_output_layout(output_dir: Path) -> None:
    """Create real workflow directories without following internal symlinks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError(f"--output-dir is not a directory: {output_dir}")

    for name in ("results", "plots", "logs"):
        path = output_dir / name
        if path.is_symlink():
            raise ValueError(
                f"refusing symlinked workflow directory: {path}; "
                "choose a fresh --output-dir"
            )
        if path.exists() and not path.is_dir():
            raise ValueError(f"workflow path is not a directory: {path}")
        path.mkdir(exist_ok=True)


def execute_plan(steps: Sequence[Step], *, mode: str, output_dir: Path,
                 jobs: int, dry_run: bool, resume: bool,
                 previous_manifest: dict[str, object] | None = None,
                 workflow_identity: dict[str, object] | None = None) -> int:
    """Execute a plan, updating its durable manifest after every step."""
    logs = output_dir / "logs"
    manifest_path = output_dir / MANIFEST_NAME
    _prepare_output_layout(output_dir)

    identity = workflow_identity or _workflow_identity()
    if resume:
        drift = _identity_drift(
            None if previous_manifest is None
            else previous_manifest.get("workflow_identity"),
            identity,
        )
        if drift:
            raise ValueError(
                "cannot resume after workflow identity drift in: "
                + ", ".join(drift)
                + "; use a new --output-dir"
            )

    old_steps = _previous_steps(previous_manifest)
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "dry-run" if dry_run else "running",
        "mode": mode,
        "output_dir": str(output_dir),
        "reference_results": str(REFERENCE_RESULTS_DIR),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "jobs": jobs,
        "started_at": _now(),
        "updated_at": _now(),
        "environment_overrides": {
            "OPTIMAL_FWER_OUTPUT_DIR": str(output_dir),
            "PYTHONUNBUFFERED": "1",
        },
        "resume": resume,
        "workflow_identity": identity,
        "steps": [
            {
                "index": index,
                "name": step.name,
                "command": list(step.command),
                "display_command": _display_command(step.command),
                "cwd": str(CODE_DIR),
                "log": str(logs / f"{index:02d}_{step.name}.log"),
                "expected_outputs": [str(path) for path in step.outputs],
                "status": "pending",
            }
            for index, step in enumerate(steps, 1)
        ],
    }
    records: list[dict[str, object]] = manifest["steps"]  # type: ignore[assignment]

    child_environment = os.environ.copy()
    child_environment.update(manifest["environment_overrides"])  # type: ignore[arg-type]
    resumable_prefix = resume
    _atomic_manifest(manifest_path, manifest)

    for index, (step, record) in enumerate(zip(steps, records), 1):
        log_path = logs / f"{index:02d}_{step.name}.log"

        if dry_run:
            _write_dry_run_log(step, log_path)
            record["status"] = "dry-run"
            continue

        previous = old_steps.get(step.name)
        if resumable_prefix and _can_resume(previous, step, log_path):
            print(f"[{index}/{len(steps)}] RESUME {step.name}: already complete",
                  flush=True)
            record.update({
                "status": "completed",
                "resumed": True,
                "artifacts": _artifacts(step),
            })
            manifest["updated_at"] = _now()
            _atomic_manifest(manifest_path, manifest)
            continue

        # Once one step needs to run, rerun everything downstream so derived
        # artifacts and the final verifier cannot be stale relative to it.
        resumable_prefix = False
        print(f"[{index}/{len(steps)}] RUN {step.name}", flush=True)
        print(f"  {_display_command(step.command)}", flush=True)
        print(f"  log: {log_path}", flush=True)
        record["status"] = "running"
        record["started_at"] = _now()
        manifest["updated_at"] = _now()
        _atomic_manifest(manifest_path, manifest)

        try:
            # A successful child must recreate every declared artifact.  This
            # prevents an interrupted or changed step from being certified by
            # a file left behind by an earlier attempt.
            _clear_declared_outputs(step, output_dir)
            returncode = _run_step(step, log_path, child_environment)
        except KeyboardInterrupt:
            record.update({"status": "interrupted", "returncode": 130,
                           "finished_at": _now()})
            manifest.update({"status": "interrupted", "finished_at": _now(),
                             "updated_at": _now()})
            _atomic_manifest(manifest_path, manifest)
            print("Interrupted; rerun the same command with --resume.", file=sys.stderr)
            return 130
        except (OSError, ValueError) as exc:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nORCHESTRATOR ERROR: {exc}\n")
            returncode = 1

        missing = [path for path in step.outputs if not path.is_file()]
        if returncode == 0 and missing:
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\nORCHESTRATOR ERROR: command returned zero but did not "
                          "produce:\n")
                for path in missing:
                    log.write(f"  {path}\n")
            returncode = 1

        record.update({"returncode": returncode, "finished_at": _now()})
        if returncode != 0:
            record["status"] = "failed"
            manifest.update({"status": "failed", "finished_at": _now(),
                             "updated_at": _now()})
            _atomic_manifest(manifest_path, manifest)
            print(f"FAILED: {step.name} exited {returncode}; see {log_path}",
                  file=sys.stderr)
            return returncode if returncode > 0 else 1

        record.update({"status": "completed", "artifacts": _artifacts(step)})
        manifest["updated_at"] = _now()
        _atomic_manifest(manifest_path, manifest)

    manifest.update({
        "status": "dry-run" if dry_run else "completed",
        "finished_at": _now(),
        "updated_at": _now(),
    })
    if mode == "quick":
        manifest["verification"] = (
            "Reduced Monte Carlo outputs are smoke artifacts and were not "
            "compared with paper-scale references."
        )
    _atomic_manifest(manifest_path, manifest)
    return 0


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"cannot resume from {path}: manifest is not a JSON object")
    if (value.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or not isinstance(value.get("steps"), list)
            or not isinstance(value.get("workflow_identity"), dict)):
        raise ValueError(
            f"cannot resume from {path}: unsupported or incomplete manifest schema"
        )
    if value.get("output_dir") != str(path.parent.resolve()):
        raise ValueError(
            f"cannot resume from {path}: manifest output_dir does not match its location"
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce or smoke-test every computation behind the paper.",
        epilog=(
            "Quick mode is the default and never compares reduced Monte Carlo "
            "outputs with paper-scale references. Full mode is expensive and "
            "finishes by running verify_reproduction.py."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", dest="mode", action="store_const", const="quick",
        help="bounded smoke workflow (default)",
    )
    mode.add_argument(
        "--full", dest="mode", action="store_const", const="full",
        help="paper-scale workflow; requires many hours",
    )
    parser.set_defaults(mode="quick")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="root for results, plots, logs, and manifest",
    )
    parser.add_argument(
        "--jobs", type=_positive_int, default=1,
        help="worker count for the simulation-grid drivers (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write and print the command plan without running child processes",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume a matching manifest and rerun from the first incomplete step",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume cannot be combined")

    try:
        output_dir = validate_output_dir(args.output_dir)
        steps = build_steps(args.mode, output_dir, args.jobs)
    except ValueError as exc:
        parser.error(str(exc))

    missing_scripts = sorted({
        command
        for step in steps
        for command in step.command[1:2]
        if command.endswith(".py") and not Path(command).is_file()
    })
    if missing_scripts:
        parser.error("required script(s) missing: " + ", ".join(missing_scripts))

    manifest_path = output_dir / MANIFEST_NAME
    previous_manifest = None
    workflow_identity = _workflow_identity()
    if args.resume:
        if not manifest_path.is_file():
            parser.error(
                f"--resume requires an existing manifest at {manifest_path}; "
                "omit --resume for a new empty output directory"
            )
        try:
            previous_manifest = _load_manifest(manifest_path)
        except ValueError as exc:
            parser.error(str(exc))
        if previous_manifest.get("mode") != args.mode:
            parser.error(
                f"existing manifest is mode {previous_manifest.get('mode')!r}; "
                f"requested {args.mode!r}. Use a different --output-dir."
            )
        drift = _identity_drift(
            previous_manifest.get("workflow_identity"), workflow_identity,
        )
        if drift:
            parser.error(
                "cannot resume after workflow identity drift in: "
                + ", ".join(drift)
                + "; use a new --output-dir"
            )
    else:
        collisions = [path for step in steps for path in step.outputs if path.exists()]
        if manifest_path.exists() or collisions:
            parser.error(
                "output directory already contains this workflow's artifacts; "
                "use --resume or choose a new --output-dir"
            )

    print(f"Mode: {args.mode}")
    print(f"Python: {sys.executable}")
    print(f"Output: {output_dir}")
    print(f"Commands ({len(steps)}):")
    for index, step in enumerate(steps, 1):
        print(f"  {index:02d}. {step.name}")
        print(f"      {_display_command(step.command)}")
    if args.mode == "full":
        print("FULL mode selected explicitly: this paper-scale workflow may take many hours.")
    elif not args.dry_run:
        print("Quick outputs are smoke artifacts; they will not be compared to paper results.")

    return execute_plan(
        steps, mode=args.mode, output_dir=output_dir, jobs=args.jobs,
        dry_run=args.dry_run, resume=args.resume,
        previous_manifest=previous_manifest,
        workflow_identity=workflow_identity,
    )


if __name__ == "__main__":
    raise SystemExit(main())
