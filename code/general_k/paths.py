"""Output locations, anchored to this file rather than to the working directory.

The submitted R package wrote into whatever directory it happened to be run
from, which is one of the defects that made it awkward to reproduce, so the
Python drivers should not repeat it.  Every default output path is resolved
from the location of the installed package, so

    python run_all_experiments.py
    cd /somewhere/else && python /path/to/code/run_all_experiments.py

write to the same place.  An explicit ``--out`` still wins, and is interpreted
relative to the working directory in the ordinary way, so a caller who wants a
scratch file still gets one where they expect.

The files shipped in ``results/`` and the manuscript figure directory are
reference artifacts.  Command-line drivers should normally write a new run to
``code/reproduced/`` instead, so that an interrupted or exploratory run cannot
replace the published evidence.  Set ``OPTIMAL_FWER_OUTPUT_DIR`` to put that
run elsewhere (for example, on a larger scratch volume).

Five directories matter:

``RESULTS_DIR``   JSON results backing the tables
``PLOTS_DIR``     driver figure output
``FIGURE_DIR``    the manuscript's figure directory when available, otherwise
                  ``code/plots`` (defined in ``figstyle`` and re-exported here)
``REPRODUCED_RESULTS_DIR``  JSON output from a new run
``REPRODUCED_PLOTS_DIR``    figure output from a new run
"""

import os

from general_k.figstyle import FIGURE_DIR  # noqa: F401  (re-exported)

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(_HERE, ".."))

RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "results"))
PLOTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "plots"))

_DEFAULT_OUTPUT_ROOT = os.path.join(CODE_DIR, "reproduced")
OUTPUT_ROOT = os.path.abspath(os.environ.get(
    "OPTIMAL_FWER_OUTPUT_DIR", _DEFAULT_OUTPUT_ROOT))
REPRODUCED_RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
REPRODUCED_PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")


def result(name):
    """Absolute path to a bundled reference result (a read-only lookup)."""
    return os.path.join(RESULTS_DIR, name)


def plot(name):
    """Absolute path to a bundled driver plot (a read-only lookup)."""
    return os.path.join(PLOTS_DIR, name)


def reproduced_result(name):
    """Absolute path for a new result; the writer creates its parent lazily."""
    return os.path.join(REPRODUCED_RESULTS_DIR, name)


def reproduced_plot(name):
    """Absolute path for a new plot; the writer creates its parent lazily."""
    return os.path.join(REPRODUCED_PLOTS_DIR, name)
