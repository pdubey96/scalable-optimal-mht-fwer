"""
Closed testing on symmetric combination statistics.

These are the right benchmarks when many hypotheses are false. The rank-based
procedures (Bonferroni, Holm, Hochberg, Hommel) use only the order of the
p-values; a combination statistic uses their magnitudes, and closed testing
\\citep{Marcus1976} turns any family of level-alpha intersection tests into a
procedure with strong family-wise error control.

Three combining functions are provided, all symmetric in the subset's p-values
and monotone in each:

- **Fisher**: ``-2 sum log p`` against ``chi^2_{2m}``. The classical choice.
- **Stouffer**: mean z-score times ``sqrt(m)`` against the normal. This is the
  benchmark Rosset et al. (2022) use, so it also serves as an external check on
  this implementation.
- **Median**: the median order statistic of the subset against its
  ``Beta(r, m-r+1)`` null distribution, with ``r = ceil(m/2)``. Designed for
  dense signals: it ignores how extreme the smallest p-value is and asks whether
  the bulk of the subset is small.

Why this is cheap
-----------------
Closed testing over all ``2^K - 1`` intersections is not needed. For a
combining function that is symmetric and non-decreasing in each p-value, the
subset of a given size ``m`` containing hypothesis ``k`` that is *hardest* to
reject is the one pairing ``p_k`` with the ``m-1`` largest of the other
p-values. Rejecting that one implies rejecting every subset of size ``m``
containing ``k``. So only ``K`` intersections per hypothesis need testing,
``O(K^2)`` in total rather than ``2^K``, which is what makes the ``K = 21``
application feasible.

Writing the sorted p-values ascending, the least-favourable subset of size ``m``
for position ``k`` is
``{k} u {K-m+1, ..., K-1}`` when ``k < K-m+1``, and ``{K-m, ..., K-1}``
otherwise, so every statistic below is a suffix sum plus at most one extra term.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, norm

# p-values are clipped away from the endpoints before any log or probit, so a
# sampled 0 or 1 cannot produce an infinite statistic.
_EPS = 1e-300
_EPS_UPPER = 1.0 - 1e-16


def _suffix_sums(x):
    """``out[:, j] = x[:, j:].sum(axis=1)``, with a zero column at the end."""
    n, K = x.shape
    out = np.zeros((n, K + 1))
    out[:, :K] = np.cumsum(x[:, ::-1], axis=1)[:, ::-1]
    return out


def _additive_closed_mask(sorted_p, alpha, transform, threshold):
    """
    Closed test for a statistic that is a *sum* over the subset.

    Parameters
    ----------
    sorted_p  : (n, K) p-values, ascending along each row
    alpha     : level
    transform : elementwise map from p-value to summand; larger means more
                evidence against the null
    threshold : callable m -> critical value; reject the size-m intersection
                when the summed statistic is at least this
    """
    n, K = sorted_p.shape
    terms = transform(np.clip(sorted_p, _EPS, _EPS_UPPER))
    suffix = _suffix_sums(terms)
    positions = np.arange(K)

    rejected = np.ones((n, K), dtype=bool)
    for m in range(1, K + 1):
        # k with the m-1 largest others, versus the top m positions outright.
        head = terms + suffix[:, K - m + 1][:, None]
        tail = np.broadcast_to(suffix[:, K - m][:, None], (n, K))
        stat = np.where(positions[None, :] < K - m + 1, head, tail)
        rejected &= stat >= threshold(m)
    return rejected


def fisher_closed_mask(sorted_p, alpha):
    """Closed testing on Fisher's combination statistic."""
    return _additive_closed_mask(
        sorted_p, alpha,
        transform=lambda p: -2.0 * np.log(p),
        threshold=lambda m: chi2.isf(alpha, 2 * m),
    )


def stouffer_closed_mask(sorted_p, alpha):
    """Closed testing on Stouffer's combination statistic."""
    z_alpha = norm.isf(alpha)
    return _additive_closed_mask(
        sorted_p, alpha,
        transform=lambda p: norm.isf(p),
        threshold=lambda m: z_alpha * np.sqrt(m),
    )


def median_closed_mask(sorted_p, alpha):
    """
    Closed testing on the median order statistic of the subset.

    For a size-``m`` subset the ``r``th smallest of ``m`` independent uniforms
    is ``Beta(r, m-r+1)`` under the intersection null, so rejecting when it
    falls below that law's ``alpha`` quantile is an exact level-``alpha`` test.
    We take ``r = ceil(m/2)``, the median rank.
    """
    n, K = sorted_p.shape
    positions = np.arange(K)
    rejected = np.ones((n, K), dtype=bool)

    for m in range(1, K + 1):
        r = (m + 1) // 2                       # ceil(m / 2)
        crit = beta_dist.ppf(alpha, r, m - r + 1)

        # Subset {k} u {K-m+1, ..., K-1}, already ascending: its rth smallest is
        # p_k when r == 1, else the (r-2)th of the tail block.
        if r == 1:
            head = sorted_p
        else:
            head = np.broadcast_to(sorted_p[:, K - m + r - 1][:, None], (n, K))
        # Subset {K-m, ..., K-1}: its rth smallest sits at K-m+r-1.
        tail = np.broadcast_to(sorted_p[:, K - m + r - 1][:, None], (n, K))

        stat = np.where(positions[None, :] < K - m + 1, head, tail)
        rejected &= stat <= crit
    return rejected


CLOSED_TESTS = {
    "Fisher-closed": fisher_closed_mask,
    "Stouffer-closed": stouffer_closed_mask,
    "Median-closed": median_closed_mask,
}


# ---------------------------------------------------------------------------
# Reference implementation, for tests
# ---------------------------------------------------------------------------

def closed_mask_bruteforce(sorted_p, alpha, statistic):
    """
    Closed testing by enumerating every one of the ``2^K - 1`` intersections.

    Exponential in ``K`` and intended only to verify the shortcut used above on
    small ``K``.  ``statistic`` maps a 1-D array of p-values to True when that
    intersection hypothesis is rejected.
    """
    from itertools import combinations

    n, K = sorted_p.shape
    out = np.ones((n, K), dtype=bool)
    for i in range(n):
        row = sorted_p[i]
        for size in range(1, K + 1):
            for subset in combinations(range(K), size):
                if not statistic(row[list(subset)]):
                    out[i, list(subset)] = False
    return out


def fisher_statistic(p, alpha):
    m = len(p)
    return -2.0 * np.log(np.clip(p, _EPS, _EPS_UPPER)).sum() >= chi2.isf(alpha, 2 * m)


def stouffer_statistic(p, alpha):
    m = len(p)
    z = norm.isf(np.clip(p, _EPS, _EPS_UPPER)).sum()
    return z >= norm.isf(alpha) * np.sqrt(m)


def median_statistic(p, alpha):
    m = len(p)
    r = (m + 1) // 2
    return np.sort(p)[r - 1] <= beta_dist.ppf(alpha, r, m - r + 1)
