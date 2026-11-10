"""Figures stay readable in black and white.

The journal's figure requirement has two halves: in-figure text at 10pt or
larger, and figures that remain interpretable printed without colour.
``tests/test_figure_font_sizes.py`` measures the first half.  This file covers
the second.

The plotting modules already intend a redundant encoding: every series carries
a distinct dash pattern, with markers added where useful, on top of a
colour-blind-safe palette. Nothing asserted it, however, so a later edit could
quietly introduce two series that differ only by hue. These tests fail if that
happens.

Greyscale separation is checked on relative luminance, using the Rec. 709
coefficients that a printer's colour-to-grey conversion approximates.
"""

import importlib
import itertools
import sys
import pathlib

import pytest

CODE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _luminance(rgb):
    """Rec. 709 relative luminance of an (r, g, b) triple in [0, 1]."""
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _styles():
    """Every series-style mapping the figure scripts define."""
    out = {}
    for module_name, attr in [("plot_power_curves", "STYLE"),
                              ("plot_gamma_sweep", "STYLE"),
                              ("plot_prior_objective", "PRIOR_STYLE")]:
        mod = importlib.import_module(module_name)
        out[f"{module_name}.{attr}"] = getattr(mod, attr)
    return out


STYLE_SETS = list(_styles().items())


@pytest.mark.parametrize("name,style", STYLE_SETS, ids=[n for n, _ in STYLE_SETS])
class TestRedundantEncoding:
    """Colour must never be the only thing separating two series."""

    def test_line_styles_are_distinct(self, name, style):
        seen = {}
        for series, spec in style.items():
            ls = spec.get("ls", spec.get("linestyle"))
            if ls is None:
                continue
            key = str(ls)
            assert key not in seen, (
                f"{name}: {series} and {seen[key]} share the dash pattern {ls}")
            seen[key] = series

    def test_markers_are_distinct(self, name, style):
        seen = {}
        for series, spec in style.items():
            marker = spec.get("marker")
            if marker is None:
                continue
            assert marker not in seen, (
                f"{name}: {series} and {seen[marker]} share the marker {marker!r}")
            seen[marker] = series

    def test_no_two_series_are_separated_by_colour_alone(self, name, style):
        """
        The real requirement: for every pair, at least one of dash pattern or
        marker differs.  This is what survives a greyscale print.
        """
        for a, b in itertools.combinations(style, 2):
            sa, sb = style[a], style[b]
            same_ls = str(sa.get("ls", sa.get("linestyle"))) == str(
                sb.get("ls", sb.get("linestyle")))
            same_marker = sa.get("marker") == sb.get("marker")
            assert not (same_ls and same_marker), (
                f"{name}: {a} and {b} differ only in colour")


@pytest.mark.parametrize("name,style", STYLE_SETS, ids=[n for n, _ in STYLE_SETS])
def test_greyscale_luminances_are_not_all_bunched(name, style):
    """
    Dash patterns carry the identification, so luminance need not separate every
    pair.  What would be a problem is every series converting to nearly the same
    grey, leaving no tonal variation at all in a printed figure.
    """
    lums = [_luminance(spec["color"]) for spec in style.values()
            if isinstance(spec.get("color"), (tuple, list))]
    if len(lums) < 2:
        pytest.skip("fewer than two colour-bearing series")
    assert max(lums) - min(lums) > 0.15, (
        f"{name}: luminances span only {max(lums) - min(lums):.3f}")


def test_the_emphasised_series_is_the_darkest():
    """
    SPOT is the line the reader should find first, and it is drawn heaviest.
    In greyscale it should also be the darkest, not merely the thickest.
    """
    import plot_power_curves as ppc

    lums = {k: _luminance(v["color"]) for k, v in ppc.STYLE.items()}
    assert min(lums, key=lums.get) == "Algorithm", lums
    assert ppc.STYLE["Algorithm"]["lw"] == max(
        s["lw"] for s in ppc.STYLE.values())
