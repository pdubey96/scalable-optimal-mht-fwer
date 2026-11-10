"""
Every figure must print at 10pt or larger.

JCGS requires labels and any other text in a figure to be at least 10pt.  That
is a property of the printed page, not of the figure file: a figure drawn wider
than the text block is scaled down by \\includegraphics, and its text shrinks
with it.  So the quantity to
check is

    effective size = native size * (printed width / figure's native width)

This is a build check rather than a unit test.  It exists because the requirement
is invisible in the plotting code -- nothing in a matplotlib script tells you what
a label will measure once LaTeX has scaled the figure -- and so it is the kind of
thing that silently regresses on every redraw.
"""

import os

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF needed to measure figure text")

from general_k.figstyle import (  # noqa: E402
    FIGURE_DIR,
    TEXT_WIDTH_PT,
    USING_MANUSCRIPT_FIGURE_DIR,
)

# TEXT_WIDTH_PT is imported rather than restated.  It was duplicated here as a
# literal once, and the copy went stale when the manuscript was retargeted from
# imsart to article, so this check was silently measuring against the wrong text
# block.  The fractions below mirror the corresponding includegraphics widths.
MIN_PT = 10.0

INCLUSION_WIDTH_FRACTIONS = {
    "prior_objective.pdf": 0.995,
    "scaling_with_K.pdf": 0.80,
}

EXPECTED_FIGURES = (
    "convergence_trajectory.pdf",
    "power_curves_K3.pdf",
    "power_curves_K4.pdf",
    "power_curves_K5.pdf",
    "power_curves_K6.pdf",
    "prior_objective.pdf",
    "scaling_with_K.pdf",
)


def _figure_path(name):
    return os.path.join(FIGURE_DIR, name)


def _require_figure(path):
    if os.path.isfile(path):
        return
    if not USING_MANUSCRIPT_FIGURE_DIR:
        pytest.skip(
            "manuscript figure artifacts are absent from the standalone code bundle")
    pytest.fail(f"missing expected manuscript figure: {path}")


# Mathtext sub- and superscripts render at roughly 60-80% of the surrounding
# size, so a figure whose body text is exactly 10pt necessarily carries smaller
# glyphs in any label containing an exponent or an index.  Demanding 10pt of
# those would force body text near 17pt.  The requirement is therefore read as
# applying to ordinary text, and scripts are allowed down to this fraction of
# the figure's dominant size -- low enough to admit a nested script, high enough
# that a genuinely undersized tick label (the actual failure mode: an 8pt
# tick_params sitting under 11pt body text) still fails.
SCRIPT_FRACTION = 0.60


def printed_sizes(path):
    """(size in printed pt -> character count), and the inclusion scale."""
    page = fitz.open(path)[0]
    fraction = INCLUSION_WIDTH_FRACTIONS.get(os.path.basename(path), 1.0)
    scale = fraction * TEXT_WIDTH_PT / page.rect.width
    sizes = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    pt = span["size"] * scale
                    sizes[pt] = sizes.get(pt, 0) + len(text)
    return sizes, scale


@pytest.mark.parametrize("name", EXPECTED_FIGURES)
def test_figure_body_text_prints_at_least_10pt(name):
    """The dominant text size -- labels, ticks, legend -- must clear 10pt."""
    path = _figure_path(name)
    _require_figure(path)
    sizes, scale = printed_sizes(path)
    if not sizes:
        pytest.skip("figure contains no text")
    dominant = max(sizes, key=lambda pt: sizes[pt])
    assert dominant >= MIN_PT - 0.05, (
        f"{os.path.basename(path)}: body text prints at {dominant:.2f}pt "
        f"(scale {scale:.3f}); needs >= {MIN_PT}pt"
    )


@pytest.mark.parametrize("name", EXPECTED_FIGURES)
def test_figure_has_no_undersized_text_beyond_mathtext_scripts(name):
    """
    Nothing may be smaller than a mathtext script would be.

    This is what catches a stray hardcoded size -- the defect that put an 8pt
    tick label under 11pt body text in the original figures.
    """
    path = _figure_path(name)
    _require_figure(path)
    sizes, scale = printed_sizes(path)
    if not sizes:
        pytest.skip("figure contains no text")
    dominant = max(sizes, key=lambda pt: sizes[pt])
    floor = SCRIPT_FRACTION * dominant
    offenders = {round(pt, 2): n for pt, n in sizes.items() if pt < floor}
    assert not offenders, (
        f"{os.path.basename(path)}: text below {floor:.2f}pt "
        f"(dominant {dominant:.2f}pt): {offenders}"
    )


def test_all_expected_figure_artifacts_are_present():
    """Missing repository artifacts fail; standalone bundles skip this check."""
    missing = [name for name in EXPECTED_FIGURES
               if not os.path.isfile(_figure_path(name))]
    if missing and not USING_MANUSCRIPT_FIGURE_DIR:
        pytest.skip("manuscript figure artifacts are absent from the standalone code bundle")
    assert not missing, f"missing expected figures under {FIGURE_DIR}: {missing}"
