"""
Standard FWER-controlling multiple testing procedures.

Each function takes:
    p_values : np.ndarray, shape (K,)  -- raw p-values (any order)
    alpha    : float                   -- familywise error rate level

and returns:
    decisions : np.ndarray, shape (K,), dtype int  -- 1 = reject, 0 = retain

The output is aligned with the *original* input order.

References
----------
Bonferroni (1936); Holm (1979); Hochberg (1988); Hommel (1988).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------------

def bonferroni(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Bonferroni correction.

    Reject hypothesis i if p_i <= alpha / K.

    Parameters
    ----------
    p_values : 1-D array of p-values, length K
    alpha    : FWER level

    Returns
    -------
    np.ndarray of int (0/1), same shape as p_values
    """
    K = len(p_values)
    threshold = alpha / K
    return (p_values <= threshold).astype(int)


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------

def holm(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Holm (1979) step-down procedure.

    Algorithm
    ---------
    1. Sort p-values ascending: p_{(1)} <= ... <= p_{(K)}.
    2. For i = 1, ..., K (1-indexed):
       - If p_{(i)} <= alpha / (K - i + 1): reject and continue.
       - Otherwise: stop; retain this and all remaining.

    Parameters
    ----------
    p_values : 1-D array of p-values, length K
    alpha    : FWER level

    Returns
    -------
    np.ndarray of int (0/1), aligned with original input order
    """
    K = len(p_values)
    sort_idx = np.argsort(p_values)          # indices that sort p ascending
    sorted_p = p_values[sort_idx]

    decisions = np.zeros(K, dtype=int)

    for i in range(K):                       # i is 0-indexed; step is i+1 (1-indexed)
        threshold = alpha / (K - i)          # alpha / (K - (i+1) + 1) = alpha / (K - i)
        if sorted_p[i] <= threshold:
            decisions[sort_idx[i]] = 1
        else:
            break                            # stop at first non-rejection

    return decisions


# ---------------------------------------------------------------------------
# Hochberg
# ---------------------------------------------------------------------------

def hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Hochberg (1988) step-up procedure.

    Algorithm
    ---------
    1. Sort p-values ascending: p_{(1)} <= ... <= p_{(K)}.
    2. Scan from the largest (i = K) downward:
       - Find the first i where p_{(i)} <= alpha / (K - i + 1).
       - Reject p_{(1)}, ..., p_{(i)}.

    Parameters
    ----------
    p_values : 1-D array of p-values, length K
    alpha    : FWER level

    Returns
    -------
    np.ndarray of int (0/1), aligned with original input order
    """
    K = len(p_values)
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]

    decisions = np.zeros(K, dtype=int)

    # Scan from largest p-value down (0-indexed i goes from K-1 to 0)
    for i in range(K - 1, -1, -1):
        threshold = alpha / (K - i)          # alpha / (K - i); when i=K-1 -> alpha/1
        if sorted_p[i] <= threshold:
            # Reject this and all smaller p-values (indices 0 .. i)
            decisions[sort_idx[: i + 1]] = 1
            break

    return decisions


# ---------------------------------------------------------------------------
# Hommel
# ---------------------------------------------------------------------------

def hommel(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Hommel (1988) procedure.

    Algorithm
    ---------
    Let p_{(1)} <= ... <= p_{(K)} be the sorted p-values.

    1. Find the largest j in {1, ..., K} such that for ALL k = 1, ..., j:
           p_{(K - j + k)} > k * alpha / j
       (i.e. *no* rejection criterion is met for any k).
       Call this largest j satisfying the condition j_no_reject.
       Then j_star = j_no_reject (the largest j for which the null holds).

       Equivalently: j_star is the largest j such that
           min_{k=1..j}  p_{(K-j+k)} / (k/j)  > alpha,
       which means reject threshold is alpha / j_star.

    2. If no such j exists (j=1 always fails), set j_star = K.

    3. Reject hypothesis i if p_i <= alpha / j_star.

    Note: j_star = 1 is always valid (reduces to Bonferroni); we search
    for the largest valid j to maximise power.

    Parameters
    ----------
    p_values : 1-D array of p-values, length K
    alpha    : FWER level

    Returns
    -------
    np.ndarray of int (0/1), aligned with original input order
    """
    K = len(p_values)
    sorted_p = np.sort(p_values)             # ascending, 0-indexed

    # Find largest j in {1..K} such that for ALL k=1..j:
    #   p_{(K-j+k)} > k*alpha/j
    # In 0-indexed terms: sorted_p[K - j + k - 1] > k*alpha/j
    #                     for k = 1, ..., j
    j_star = 1                               # fallback (Bonferroni)

    for j in range(1, K + 1):
        # Check whether ALL k=1..j satisfy the strict inequality
        all_ok = True
        for k in range(1, j + 1):
            idx = K - j + k - 1             # 0-indexed position of p_{(K-j+k)}
            if sorted_p[idx] <= k * alpha / j:
                all_ok = False
                break
        if all_ok:
            j_star = j

    threshold = alpha / j_star
    return (p_values <= threshold).astype(int)
