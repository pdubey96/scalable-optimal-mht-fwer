#!/usr/bin/env python3
"""Compare a fresh paper-scale run with the bundled reference results.

Timing fields are intentionally ignored: elapsed time depends on hardware and
system load.  Scientific values, convergence diagnostics, decisions, and data
provenance must retain the same structure and agree numerically within the
reported Monte Carlo precision.  The generated scaling-table body is compared
byte for byte.

Examples
--------
Verify every paper result after a full reproduction::

    python3 verify_reproduction.py reproduced/results

Verify only selected result files::

    python3 verify_reproduction.py /tmp/my-run/results \
        --files applications.json scaling_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from general_k import paths


PAPER_RESULTS = (
    "all_experiments.json",
    "untruncated_normal.json",
    "scaling_results.json",
    "sensitivity_results.json",
    "gamma_sweep.json",
    "prior_objective.json",
    "prior_comparison.json",
    "prior_comparison_t.json",
    "prior_comparison_beta.json",
    "applications.json",
    "table_scaling.tex",
)

DEFAULT_IGNORED_KEYS = frozenset({
    "elapsed_seconds",
    "solve_seconds",
    "sweep_seconds",
    "opt_seconds",
    "total_seconds",
    "seconds",
})

# These numbers define the experiment or are embedded source data rather than
# Monte Carlo estimates.  They must match exactly even though result floats are
# compared at a tolerance appropriate to cross-platform numerical variation.
DEFAULT_EXACT_NUMERIC_KEYS = frozenset({
    "K", "alpha", "theta", "n_algo_samples", "n_mc", "n_opt", "n_eval",
    "seed", "seed_opt", "seed_eval", "tol", "max_iter", "bisect_tol",
    "f_tol", "beta_a", "beta_b", "prior", "p", "z", "se", "a_hat",
    "beta", "n", "N_rep",
})
DEFAULT_EXACT_NUMERIC_CONTAINERS = frozenset({
    "prior", "p", "z", "se", "a_hat", "beta", "n", "N_rep",
})


def compare_values(
    reference: Any,
    candidate: Any,
    *,
    rtol: float,
    atol: float,
    ignored_keys: frozenset[str] = DEFAULT_IGNORED_KEYS,
    exact_numeric_keys: frozenset[str] = DEFAULT_EXACT_NUMERIC_KEYS,
    exact_numeric_containers: frozenset[str] = DEFAULT_EXACT_NUMERIC_CONTAINERS,
    exact_numeric: bool = False,
    location: str = "$",
    mismatches: list[str] | None = None,
    limit: int = 25,
) -> list[str]:
    """Return human-readable structural and numerical mismatches."""
    if mismatches is None:
        mismatches = []
    if len(mismatches) >= limit:
        return mismatches

    if isinstance(reference, dict) and isinstance(candidate, dict):
        reference_keys = set(reference) - ignored_keys
        candidate_keys = set(candidate) - ignored_keys
        for key in sorted(reference_keys - candidate_keys):
            if len(mismatches) >= limit:
                return mismatches
            mismatches.append(f"{location}: candidate is missing key {key!r}")
        for key in sorted(candidate_keys - reference_keys):
            if len(mismatches) >= limit:
                return mismatches
            mismatches.append(f"{location}: candidate has unexpected key {key!r}")
        for key in sorted(reference_keys & candidate_keys):
            value = reference[key]
            child_exact = (
                exact_numeric
                or key in exact_numeric_containers
                or (key in exact_numeric_keys and isinstance(value, (int, float)))
            )
            compare_values(
                value, candidate[key], rtol=rtol, atol=atol,
                ignored_keys=ignored_keys,
                exact_numeric_keys=exact_numeric_keys,
                exact_numeric_containers=exact_numeric_containers,
                exact_numeric=child_exact,
                location=f"{location}.{key}",
                mismatches=mismatches, limit=limit,
            )
            if len(mismatches) >= limit:
                break
        return mismatches

    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            mismatches.append(
                f"{location}: list lengths differ "
                f"({len(reference)} reference, {len(candidate)} candidate)"
            )
            return mismatches
        for index, (ref_item, candidate_item) in enumerate(zip(reference, candidate)):
            compare_values(
                ref_item, candidate_item, rtol=rtol, atol=atol,
                ignored_keys=ignored_keys,
                exact_numeric_keys=exact_numeric_keys,
                exact_numeric_containers=exact_numeric_containers,
                exact_numeric=exact_numeric,
                location=f"{location}[{index}]",
                mismatches=mismatches, limit=limit,
            )
            if len(mismatches) >= limit:
                break
        return mismatches

    # bool is an int subclass, but changing a certificate to 0/1 is a schema
    # change rather than a harmless numerical difference.
    if isinstance(reference, bool) or isinstance(candidate, bool):
        if type(reference) is not type(candidate) or reference != candidate:
            mismatches.append(f"{location}: {reference!r} != {candidate!r}")
        return mismatches

    if (isinstance(reference, (int, float))
            and isinstance(candidate, (int, float))):
        if isinstance(reference, int) and isinstance(candidate, int):
            if reference != candidate:
                mismatches.append(f"{location}: {reference} != {candidate}")
        elif exact_numeric and reference != candidate:
            mismatches.append(
                f"{location}: {reference!r} != {candidate!r} "
                "(configuration/source values require an exact match)"
            )
        elif not exact_numeric and not math.isclose(
                float(reference), float(candidate), rel_tol=rtol, abs_tol=atol):
            mismatches.append(
                f"{location}: {reference!r} != {candidate!r} "
                f"(rtol={rtol:g}, atol={atol:g})"
            )
        return mismatches

    if type(reference) is not type(candidate) or reference != candidate:
        mismatches.append(f"{location}: {reference!r} != {candidate!r}")
    return mismatches


def compare_file(reference: Path, candidate: Path, *, rtol: float, atol: float,
                 ignored_keys: frozenset[str]) -> list[str]:
    """Compare one JSON or text result file."""
    if reference.suffix.lower() != ".json":
        if reference.read_bytes() == candidate.read_bytes():
            return []
        return ["file contents differ (text artifacts require an exact match)"]

    with reference.open(encoding="utf-8") as handle:
        reference_data = json.load(handle)
    with candidate.open(encoding="utf-8") as handle:
        candidate_data = json.load(handle)
    return compare_values(
        reference_data, candidate_data, rtol=rtol, atol=atol,
        ignored_keys=ignored_keys,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", nargs="?", default=paths.REPRODUCED_RESULTS_DIR,
        help="Directory containing fresh results (default: reproduced/results)",
    )
    parser.add_argument(
        "--reference", default=paths.RESULTS_DIR,
        help="Bundled reference-result directory",
    )
    parser.add_argument(
        "--files", nargs="+", default=list(PAPER_RESULTS),
        help="Result filenames to compare (default: every paper result)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-5,
        help="Relative tolerance for floating-point values (default: 1e-5)",
    )
    parser.add_argument(
        "--atol", type=float, default=5e-5,
        help="Absolute tolerance for floating-point values (default: 5e-5)",
    )
    parser.add_argument(
        "--ignore-key", action="append", default=[],
        help="Additional JSON key to ignore everywhere; repeat as needed",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Verify files that exist and do not fail for selected missing files",
    )
    args = parser.parse_args(argv)

    if args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be non-negative")

    candidate_dir = Path(args.candidate).expanduser().resolve()
    reference_dir = Path(args.reference).expanduser().resolve()
    try:
        same_directories = os.path.samefile(candidate_dir, reference_dir)
    except OSError:
        same_directories = candidate_dir == reference_dir
    if same_directories:
        parser.error("candidate and reference directories must be different")
    ignored = DEFAULT_IGNORED_KEYS | frozenset(args.ignore_key)
    failures = 0
    compared = 0

    for name in args.files:
        reference = reference_dir / name
        candidate = candidate_dir / name
        missing = []
        if not reference.is_file():
            missing.append(f"reference {reference}")
        if not candidate.is_file():
            missing.append(f"candidate {candidate}")
        if missing:
            if args.allow_missing and reference.is_file() and not candidate.exists():
                print(f"SKIP {name}: candidate file is absent")
                continue
            print(f"FAIL {name}: missing " + " and ".join(missing))
            failures += 1
            continue

        try:
            if os.path.samefile(reference, candidate):
                print(f"FAIL {name}: candidate and reference are the same file")
                failures += 1
                compared += 1
                continue
        except OSError:
            pass

        try:
            mismatches = compare_file(
                reference, candidate, rtol=args.rtol, atol=args.atol,
                ignored_keys=ignored,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(f"FAIL {name}: cannot read a valid result ({exc})")
            failures += 1
            continue

        compared += 1
        if mismatches:
            failures += 1
            print(f"FAIL {name}: {len(mismatches)} mismatch(es), showing up to 25")
            for mismatch in mismatches:
                print(f"  {mismatch}")
        else:
            qualifier = "excluding timing metadata" if reference.suffix == ".json" else "exact"
            print(f"PASS {name} ({qualifier})")

    if compared == 0:
        print("FAIL: no result files were compared")
        return 2
    if failures:
        print(f"\nVerification failed: {failures} file(s) did not match.")
        return 1
    print(f"\nVerification passed: {compared} file(s) match the bundled results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
