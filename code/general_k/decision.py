"""
Optimal decision rule for the general-K testing problem.

Given Lagrange multipliers mu, sorted p-values u_sorted, density function
g_func, and number of hypotheses K, this module computes:

  l* = argmax_{0 <= l <= K} sum_{i=1}^{l} R_i(mu, u)

and produces the corresponding binary decision vector D^mu(u) of shape (K,),
where D_i = 1 if i <= l*, else 0.

This implements a likelihood-ratio ordered threshold rule: because R_i values
are used in a cumulative sum, rejecting the first l* hypotheses (by LR order)
is always optimal.
"""

import numpy as np

from general_k.coefficients import compute_R, compute_R_batch, compute_R_from_precomputed


def optimal_l_star(mu: np.ndarray, u_sorted: np.ndarray, g_func, K: int) -> int:
    """
    Compute l* = argmax_{0 <= l <= K} sum_{i=1}^{l} R_i(mu, u).

    The objective at l=0 is defined as 0 (empty sum). We evaluate the
    cumulative sum of R_i for l = 1, ..., K and pick the l that maximises it,
    keeping l=0 as the default when no positive gain exists.

    Parameters
    ----------
    mu       : 1-D array of shape (K,); Lagrange multipliers mu_0, ..., mu_{K-1}
    u_sorted : 1-D array of shape (K,); sorted p-values u_1 <= ... <= u_K
    g_func   : callable, maps np.ndarray -> np.ndarray of same shape (alternative density)
    K        : number of hypotheses

    Returns
    -------
    int
        l* in {0, 1, ..., K}.
    """
    R = compute_R(mu, u_sorted, g_func, K)          # shape (K,)
    cumsum = np.cumsum(R)                            # cumsum[l-1] = sum_{i=1}^l R_i

    # Objective values: obj[0] = 0 (l=0), obj[l] = cumsum[l-1] for l=1..K
    obj = np.empty(K + 1)
    obj[0] = 0.0
    obj[1:] = cumsum

    return int(np.argmax(obj))


def l_star_batch(
    mu: np.ndarray,
    u_sorted_batch: np.ndarray,
    g_func,
    K: int,
) -> np.ndarray:
    """
    Compute l* for a batch of samples.

    Vectorised version of optimal_l_star that processes all samples at once.

    Parameters
    ----------
    mu              : shape (K,); Lagrange multipliers
    u_sorted_batch  : shape (n_samples, K); sorted p-values, one row per sample
    g_func          : callable, vectorised (takes array, returns same-shape array)
    K               : number of hypotheses

    Returns
    -------
    np.ndarray of int, shape (n_samples,)
        l*[i] in {0, 1, ..., K} for each sample i.
    """
    R_batch = compute_R_batch(mu, u_sorted_batch, g_func, K)  # (n_samples, K)

    # cumulative sums along positions axis
    cumsum = np.cumsum(R_batch, axis=1)   # (n_samples, K)

    # obj[:, 0] = 0  (l=0, empty sum), obj[:, l] = cumsum[:, l-1] for l=1..K
    obj = np.empty((u_sorted_batch.shape[0], K + 1))
    obj[:, 0] = 0.0
    obj[:, 1:] = cumsum

    return np.argmax(obj, axis=1).astype(int)


def l_star_from_precomputed(mu, a_batch, b_coeff):
    """
    Compute l* for a batch using precomputed a_batch and b_coeff.

    Avoids recomputing g_func and ESPs — just linear algebra.
    """
    R_batch = compute_R_from_precomputed(mu, a_batch, b_coeff)
    cumsum = np.cumsum(R_batch, axis=1)
    n_samples, K = R_batch.shape
    obj = np.empty((n_samples, K + 1))
    obj[:, 0] = 0.0
    obj[:, 1:] = cumsum
    return np.argmax(obj, axis=1).astype(int)


def decisions(mu: np.ndarray, u_sorted: np.ndarray, g_func, K: int) -> np.ndarray:
    """
    Return the binary decision vector D^mu(u) of shape (K,).

    D_i = 1 if i <= l*, else 0  (1-indexed i).
    Equivalently, in 0-indexed terms: D[j] = 1 if j < l*, else 0.

    Parameters
    ----------
    mu       : 1-D array of shape (K,); Lagrange multipliers
    u_sorted : 1-D array of shape (K,); sorted p-values
    g_func   : callable, maps np.ndarray -> np.ndarray (alternative density)
    K        : number of hypotheses

    Returns
    -------
    np.ndarray of shape (K,) with dtype int, entries in {0, 1}.
    """
    l_star = optimal_l_star(mu, u_sorted, g_func, K)
    D = np.zeros(K, dtype=int)
    D[:l_star] = 1
    return D
