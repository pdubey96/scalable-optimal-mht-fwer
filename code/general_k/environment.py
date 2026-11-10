"""Record the environment a result was produced in.

The supplement has to state the software environment, and stating it by hand is
how it drifts: the paragraph described a MacBook Pro running Python 3.12.7 with
NumPy 1.26.4 while settings re-solved in August came off Windows with Python
3.14.3 and NumPy 2.4.3.  Nothing in the pipeline noticed, because nothing in
the pipeline was recording it.

So the environment is captured from the interpreter that actually produced the
run, written next to the results, and quoted from there.  ``compare()`` reports
what has changed since a recorded capture, which is what makes a mixed-machine
release visible instead of silent.  A separate package-only comparison verifies
the numerically relevant dependencies on another machine without treating its
Python version or operating system as a failure.

    python -m general_k.environment                 # record under reproduced/
    python -m general_k.environment --check         # Python/packages/OS/architecture
    python -m general_k.environment --check-packages  # dependencies only
"""

import argparse
import json
import os
import platform
import sys

from general_k import paths
from general_k.io import atomic_json_dump

FILENAME = "environment.json"

# Packages whose version can move a numerical result.
_PACKAGES = ("numpy", "scipy", "matplotlib")


def capture():
    """The environment of the running interpreter, as a plain dict."""
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None

    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu_count": os.cpu_count(),
    }


def record(path=None):
    """Write the current environment with new-run outputs. Returns the path."""
    path = path or paths.reproduced_result(FILENAME)
    atomic_json_dump(capture(), path)
    return path


def load(path=None):
    """The recorded environment, or None if nothing has been recorded."""
    path = path or paths.result(FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compare(recorded, current=None):
    """Differences between a recorded environment and another one.

    Returns a list of ``(field, recorded_value, current_value)``.  Empty means
    the two agree on everything that could move a number.
    """
    current = current or capture()
    out = []
    if recorded["python"] != current["python"]:
        out.append(("python", recorded["python"], current["python"]))
    out.extend(compare_packages(recorded, current))
    for key in ("system", "machine"):
        a = recorded.get("platform", {}).get(key)
        b = current["platform"].get(key)
        if a != b:
            out.append((f"platform.{key}", a, b))
    return out


def compare_packages(recorded, current=None):
    """Differences in numerically relevant package versions only.

    Unlike :func:`compare`, this deliberately ignores the Python version,
    operating system, and machine architecture.  It is therefore suitable for
    checking that the pinned dependencies were installed correctly on a clean
    machine that does not match the reference platform.
    """
    current = current or capture()
    out = []
    for name in _PACKAGES:
        a = recorded.get("packages", {}).get(name)
        b = current.get("packages", {}).get(name)
        if a != b:
            out.append((name, a, b))
    return out


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="Record or verify the software environment used for the "
                    "released numerical results.",
        epilog=(
            "With no option, write the current environment to the safe "
            "reproduced/results/environment.json path. --check is the original strict "
            "audit and fails on any recorded Python, package, operating-system, "
            "or architecture drift. --check-packages checks only NumPy, SciPy, "
            "and Matplotlib, so it can pass on a different platform."
        ),
    )
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--check",
        action="store_true",
        help="strictly compare Python, package, OS, and architecture fields",
    )
    checks.add_argument(
        "--check-packages",
        "--check-dependencies",
        dest="check_packages",
        action="store_true",
        help="compare NumPy, SciPy, and Matplotlib versions only",
    )
    parser.add_argument(
        "--out",
        help="environment JSON to write (recording mode only)",
    )
    args = parser.parse_args(argv)

    if not (args.check or args.check_packages):
        print(f"wrote {record(args.out)}")
        return 0

    if args.out:
        parser.error("--out is only valid when recording, without a check option")

    recorded = load()
    if recorded is None:
        print("no environment.json recorded yet")
        return 1
    if args.check_packages:
        diffs = compare_packages(recorded)
        if not diffs:
            print("package versions match the recorded environment "
                  "(Python and platform not checked)")
            return 0
        print("package versions differ from the recorded environment:")
        for field, was, now in diffs:
            print(f"  {field}: recorded {was}, now {now}")
        return 1

    diffs = compare(recorded)
    if not diffs:
        print("environment matches the recorded one")
        return 0
    print("environment differs from the recorded one:")
    for field, was, now in diffs:
        print(f"  {field}: recorded {was}, now {now}")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
