"""The BCG vaccine trials, frozen, with the four three-way subgroup splits.

Provenance
----------
Thirteen randomised trials of BCG vaccination against tuberculosis, as
collected by Colditz et al. (1994, JAMA 271:698-702) and distributed as
``dat.bcg`` in the R package ``metadat`` (version 1.6.0, extracted 12 August
2026).  The counts below are that dataset verbatim.

The submitted analysis read the data out of ``metadat`` at run time, which
makes the result depend on an R package being installed and on its version.
The counts are small and public, so the honest fix is to carry them here and
say where they came from.  ``CHECKSUM`` pins the table; ``tests/test_data_bcg.py``
checks it, and checks every derived p-value against values computed
independently in R.

Columns
-------
``tpos``/``tneg``  tuberculosis cases and non-cases in the vaccinated arm
``cpos``/``cneg``  the same in the control arm
``ablat``          absolute latitude of the trial site, in degrees
``year``           publication year
``alloc``          allocation method as recorded by the original trial

Analysis
--------
Per trial the log risk ratio and its variance are the standard ones,

    y = log{ (tpos/(tpos+tneg)) / (cpos/(cpos+cneg)) },
    v = 1/tpos - 1/(tpos+tneg) + 1/cpos - 1/(cpos+cneg),

which is what ``metafor::escalc(measure = "RR")`` computes, and subgroups are
pooled by inverse variance, which is ``metafor::rma.uni(method = "FE")``.  The
two-sided p-value is then ``2 * Phi(-|beta / se|)``.

Four three-way splits are defined, following the submitted analysis: allocation
as recorded, and tertiles of absolute latitude, of control-arm event rate, and
of year.  Tertile cuts follow R's ``cut(x, quantile(x, c(0, 1/3, 2/3, 1)),
include.lowest = TRUE)``: intervals are closed on the right, and the lowest is
closed on both sides.
"""

import hashlib

import numpy as np
from scipy.stats import norm

# trial, author, year, tpos, tneg, cpos, cneg, ablat, alloc
TRIALS = [
    (1,  "Aronson",              1948,   4,   119,  11,   128, 44, "random"),
    (2,  "Ferguson & Simes",     1949,   6,   300,  29,   274, 55, "random"),
    (3,  "Rosenthal et al",      1960,   3,   228,  11,   209, 42, "random"),
    (4,  "Hart & Sutherland",    1977,  62, 13536, 248, 12619, 52, "random"),
    (5,  "Frimodt-Moller et al", 1973,  33,  5036,  47,  5761, 13, "alternate"),
    (6,  "Stein & Aronson",      1953, 180,  1361, 372,  1079, 44, "alternate"),
    (7,  "Vandiviere et al",     1973,   8,  2537,  10,   619, 19, "random"),
    (8,  "TPT Madras",           1980, 505, 87886, 499, 87892, 13, "random"),
    (9,  "Coetzee & Berjak",     1968,  29,  7470,  45,  7232, 27, "random"),
    (10, "Rosenthal et al",      1961,  17,  1699,  65,  1600, 42, "systematic"),
    (11, "Comstock et al",       1974, 186, 50448, 141, 27197, 18, "systematic"),
    (12, "Comstock & Webster",   1969,   5,  2493,   3,  2338, 33, "systematic"),
    (13, "Comstock et al",       1976,  27, 16886,  29, 17825, 33, "systematic"),
]

SOURCE = "metadat 1.6.0, dat.bcg; Colditz et al. (1994) JAMA 271:698-702"

# sha256 of the numeric columns, so an accidental edit is caught rather than shipped
CHECKSUM = "f063d60d1f150aa5429ecce874d9c6de2ec0676beb2537ddb52adccb8838d133"

SPLITS = ("alloc", "lat", "risk", "era")

_LABELS = {
    "lat": ("low_lat", "mid_lat", "high_lat"),
    "risk": ("low_risk", "mid_risk", "high_risk"),
    "era": ("era1", "era2", "era3"),
}


def checksum():
    """sha256 over the numeric columns, in table order."""
    h = hashlib.sha256()
    for t in TRIALS:
        h.update(",".join(str(x) for x in (t[0], t[2], t[3], t[4], t[5], t[6], t[7])).encode())
        h.update(b";")
    return h.hexdigest()


def _columns():
    tpos = np.array([t[3] for t in TRIALS], dtype=float)
    tneg = np.array([t[4] for t in TRIALS], dtype=float)
    cpos = np.array([t[5] for t in TRIALS], dtype=float)
    cneg = np.array([t[6] for t in TRIALS], dtype=float)
    return tpos, tneg, cpos, cneg


def effect_sizes():
    """Per-trial log risk ratio and its variance."""
    tpos, tneg, cpos, cneg = _columns()
    y = np.log((tpos / (tpos + tneg)) / (cpos / (cpos + cneg)))
    v = 1 / tpos - 1 / (tpos + tneg) + 1 / cpos - 1 / (cpos + cneg)
    return y, v


def _tertile_labels(x, labels):
    """R's cut(x, quantile(x, c(0,1/3,2/3,1)), include.lowest=TRUE)."""
    q1, q2 = np.quantile(x, 1 / 3), np.quantile(x, 2 / 3)
    out = np.empty(len(x), dtype=object)
    for i, xi in enumerate(x):
        out[i] = labels[0] if xi <= q1 else (labels[1] if xi <= q2 else labels[2])
    return out


def subgroups(split):
    """Subgroup label per trial, and the label order, for one split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")

    if split == "alloc":
        labels = np.array([t[8] for t in TRIALS], dtype=object)
        return labels, tuple(sorted(set(labels)))

    if split == "lat":
        x = np.array([t[7] for t in TRIALS], dtype=float)
    elif split == "era":
        x = np.array([t[2] for t in TRIALS], dtype=float)
    else:                                   # control-arm event rate
        _, _, cpos, cneg = _columns()
        x = cpos / (cpos + cneg)

    names = _LABELS[split]
    return _tertile_labels(x, names), names


def pooled(split):
    """Fixed-effect pooled log risk ratio, se, z and two-sided p per subgroup.

    Returns a dict keyed by subgroup label, in the split's label order.
    """
    y, v = effect_sizes()
    labels, order = subgroups(split)

    out = {}
    for name in order:
        k = labels == name
        w = 1.0 / v[k]
        beta = float(np.sum(w * y[k]) / np.sum(w))
        se = float(np.sqrt(1.0 / np.sum(w)))
        z = beta / se
        out[name] = {"n": int(k.sum()), "beta": beta, "se": se, "z": z,
                     "p": float(2 * norm.cdf(-abs(z)))}
    return out


def p_values(split):
    """The K = 3 vector of subgroup p-values, in the split's label order."""
    pooled_ = pooled(split)
    return np.array([pooled_[name]["p"] for name in pooled_])
