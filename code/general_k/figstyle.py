"""
One figure style for the whole paper.

JCGS requires every label and any text inside a figure to print at 10pt or
larger, and requires figures to be interpretable in black and white.  Both are
properties of the printed page rather than of the plotting call, which is why
they are easy to lose on a redraw:

  - Text size.  A figure drawn wider than the text block is scaled down by
    \\includegraphics and its labels shrink with it.  Drawing at exactly the
    text width makes the scale 1, so the nominal size is the printed size.
    Narrower figures are drawn at the same fraction at which LaTeX prints them;
    the artifact tests measure the remaining bounding-box adjustment.

  - Colour.  Every series must be separable in greyscale, so colour is only ever
    a redundant encoding: each series also carries its own dash pattern and,
    where useful, its own marker.  The palette is the Wong colour-blind-safe set,
    which additionally keeps the series distinguishable on screen.

`tests/test_figure_font_sizes.py` measures the produced PDFs and fails if any
text would print below 10pt.
"""

import os

# \textwidth of the compiled manuscript, which is `article` at 12pt with the
# enlarged text block of mht.tex.  This number was 404.9 while the sources were
# still typeset in the imsart class; leaving it stale after the retarget meant
# every figure was drawn narrow and then enlarged ~14% by \includegraphics,
# which printed the labels larger than intended and, on the four-row power-curve
# figures, made the float taller than \textheight.
TEXT_WIDTH_PT = 462.27
TEXT_HEIGHT_PT = 671.36
TEXT_WIDTH_IN = TEXT_WIDTH_PT / 72.0  # 6.42 in

# 10pt is the floor.  The margin absorbs the modest enlargement
# bbox_inches='tight' can cause when labels extend past the figure rectangle.
# Most figures are included at \linewidth. Scripts drawing narrower figures
# use the matching `fraction` argument, and the build test checks the actual
# saved-PDF and inclusion scales.
BASE_PT = 11.0

_HERE = os.path.dirname(os.path.abspath(__file__))

_CODE_DIR = os.path.normpath(os.path.join(_HERE, ".."))
_REPOSITORY_MANUSCRIPT_DIR = os.path.normpath(
    os.path.join(_CODE_DIR, "..", "..", "Manuscript"))

# The figures the manuscript actually includes.  Single source of truth: a
# redraw must land where LaTeX reads from, or the manuscript keeps the old copy.
# A standalone code bundle has no sibling Manuscript directory; in that layout
# keep generated output inside the bundle instead of escaping into its parent.
USING_MANUSCRIPT_FIGURE_DIR = os.path.isdir(_REPOSITORY_MANUSCRIPT_DIR)
FIGURE_DIR = os.path.join(
    _REPOSITORY_MANUSCRIPT_DIR if USING_MANUSCRIPT_FIGURE_DIR else _CODE_DIR,
    "plots",
)


def figure_path(name):
    """Absolute figure-output path, creating the directory if needed."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return os.path.join(FIGURE_DIR, name)


def apply():
    """Install the shared rcParams.  Call once, before creating any figure."""
    import matplotlib

    matplotlib.rcParams.update({
        "font.size": BASE_PT,
        "axes.labelsize": BASE_PT,
        "axes.titlesize": BASE_PT,
        "xtick.labelsize": BASE_PT,
        "ytick.labelsize": BASE_PT,
        "legend.fontsize": BASE_PT,
        "figure.titlesize": BASE_PT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        # Keep tick labels short so 10.5pt text fits in a narrow panel.
        "axes.formatter.limits": (-3, 4),
    })


def width(fraction=1.0):
    """Figure width in inches: the text width, or a fraction of it."""
    return TEXT_WIDTH_IN * fraction


def size(aspect=0.62, fraction=1.0):
    """(width, height) in inches for a figure of the given height/width ratio."""
    w = width(fraction)
    return (w, w * aspect)
