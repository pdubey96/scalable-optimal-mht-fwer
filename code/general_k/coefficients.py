"""
Elementary symmetric polynomials and FWER coefficient functions
for the general-K optimal testing algorithm.

Notation
--------
u_sorted : 1-D numpy array, length K, sorted p-values u_1 <= u_2 <= ... <= u_K
g_func   : callable u_array -> array of same shape; alternative density values
K        : number of hypotheses
k        : 1-indexed position (k = 1, ..., K)
l        : index for FWER constraint (l = 0, ..., K-1)
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Elementary symmetric polynomial
# ---------------------------------------------------------------------------

def elem_sym_poly(values: np.ndarray, m: int) -> float:
    """
    Compute the m-th elementary symmetric polynomial of ``values``.

    Uses an O(n*m) DP recurrence:
        e[0]   = 1
        e[j]   = e[j] + values[i] * e[j-1]   (updated in reverse order)

    Parameters
    ----------
    values : 1-D numpy array of real numbers
    m      : non-negative integer order

    Returns
    -------
    float
        e_m(values).
        Returns 1.0 if m == 0.
        Returns 0.0 if m > len(values).
    """
    if m == 0:
        return 1.0

    n = len(values)
    if m > n:
        return 0.0

    # dp[j] = e_j(values[:i]) after processing the first i elements
    dp = np.zeros(m + 1)
    dp[0] = 1.0

    for i, v in enumerate(values):
        # Update in reverse to avoid using the updated value in the same pass
        for j in range(min(i + 1, m), 0, -1):
            dp[j] += v * dp[j - 1]

    return float(dp[m])


# ---------------------------------------------------------------------------
# Power coefficient  a_k(u)
# ---------------------------------------------------------------------------

def compute_a(u_sorted: np.ndarray, g_func, K: int) -> float:
    """
    Compute the power coefficient a_k(u).

    Formula (same for all k):
        a = (K-1)! * prod_{j=1}^{K} g(u_j)

    Parameters
    ----------
    u_sorted : sorted p-values, length K
    g_func   : callable, maps array -> array of alternative densities
    K        : number of hypotheses

    Returns
    -------
    float
    """
    g_vals = g_func(u_sorted)          # shape (K,)
    return math.factorial(K - 1) * float(np.prod(g_vals))


# ---------------------------------------------------------------------------
# FWER constraint coefficient  b_{l,k}(u)
# ---------------------------------------------------------------------------

def compute_b(l: int, k: int, u_sorted: np.ndarray, g_func, K: int) -> float:
    """
    Compute the FWER constraint coefficient b_{l,k}(u).

    Formula (k is 1-indexed):
        b_{l,k}(u) = l! * (K-l)!
                     * prod_{j=1}^{k-1} g(u_j)
                     * e_{l-k+1}( g(u_{k+1}), ..., g(u_K) )

    Returns 0 when:
        - l < k-1              (prefix product would require more terms than allowed)
        - l - k + 1 > K - k   (ESP index exceeds the number of available tail elements)

    Parameters
    ----------
    l        : FWER constraint index, 0-indexed (l = 0, ..., K-1)
    k        : position, 1-indexed (k = 1, ..., K)
    u_sorted : sorted p-values, length K
    g_func   : callable, maps array -> array of alternative densities
    K        : number of hypotheses

    Returns
    -------
    float
    """
    esp_index = l - k + 1   # order of the ESP over the tail elements
    n_tail    = K - k       # number of elements after position k

    # Boundary conditions
    if l < k - 1:
        return 0.0
    if esp_index > n_tail:
        return 0.0

    g_vals = g_func(u_sorted)  # shape (K,); 0-indexed internally

    # Prefix product: g(u_1) * ... * g(u_{k-1})  (0-indexed: indices 0..k-2)
    prefix_prod = float(np.prod(g_vals[:k - 1])) if k > 1 else 1.0

    # Tail values: g(u_{k+1}), ..., g(u_K)  (0-indexed: indices k..K-1)
    tail_vals = g_vals[k:]  # length n_tail

    esp_val = elem_sym_poly(tail_vals, esp_index)

    factorial_coeff = math.factorial(l) * math.factorial(K - l)

    return factorial_coeff * prefix_prod * esp_val


# ---------------------------------------------------------------------------
# R_i vector
# ---------------------------------------------------------------------------

def compute_R(mu: np.ndarray, u_sorted: np.ndarray, g_func, K: int) -> np.ndarray:
    """
    Compute R_i = a_i - sum_{l=0}^{K-1} mu_l * b_{l,i}  for i = 1, ..., K.

    Parameters
    ----------
    mu       : 1-D array of shape (K,); Lagrange multipliers mu_0, ..., mu_{K-1}
    u_sorted : sorted p-values, length K
    g_func   : callable, maps array -> array of alternative densities
    K        : number of hypotheses

    Returns
    -------
    np.ndarray of shape (K,)
        R[i-1] = R_i  (0-indexed storage, 1-indexed formula).
    """
    a = compute_a(u_sorted, g_func, K)
    R = np.full(K, a)

    for k in range(1, K + 1):       # k is 1-indexed
        for l in range(K):          # l = 0, ..., K-1
            b_val = compute_b(l, k, u_sorted, g_func, K)
            R[k - 1] -= mu[l] * b_val

    return R


# ---------------------------------------------------------------------------
# Batch elementary symmetric polynomial
# ---------------------------------------------------------------------------

def elem_sym_poly_batch(values_2d: np.ndarray, m: int) -> np.ndarray:
    """
    Compute the m-th elementary symmetric polynomial for each row of values_2d.

    Uses the same O(n*m) DP recurrence as elem_sym_poly but operates on 2-D
    arrays so the entire batch is processed with vectorised NumPy operations.

    Parameters
    ----------
    values_2d : 2-D numpy array, shape (n_samples, n)
    m         : non-negative integer order

    Returns
    -------
    np.ndarray of shape (n_samples,)
        e_m applied to each row.
    """
    n_samples, n = values_2d.shape

    if m == 0:
        return np.ones(n_samples)

    if m > n:
        return np.zeros(n_samples)

    # dp[:, j] = e_j(values_2d[:, :i]) after processing the first i columns
    dp = np.zeros((n_samples, m + 1))
    dp[:, 0] = 1.0

    for i in range(n):
        v = values_2d[:, i]          # shape (n_samples,)
        # Update in reverse to avoid using the updated value in the same pass
        for j in range(min(i + 1, m), 0, -1):
            dp[:, j] += v * dp[:, j - 1]

    return dp[:, m]


# ---------------------------------------------------------------------------
# Batch R vector
# ---------------------------------------------------------------------------

def compute_R_batch(
    mu: np.ndarray,
    u_sorted_batch: np.ndarray,
    g_func,
    K: int,
) -> np.ndarray:
    """
    Compute R_i for a batch of samples.

    Vectorised version of compute_R that processes all samples simultaneously
    using 2-D NumPy operations.

    Parameters
    ----------
    mu              : shape (K,); Lagrange multipliers
    u_sorted_batch  : shape (n_samples, K); sorted p-values, one row per sample
    g_func          : callable, vectorised (takes array, returns same-shape array)
    K               : number of hypotheses

    Returns
    -------
    np.ndarray of shape (n_samples, K)
        R_batch[i, k-1] = R_k for sample i.
    """
    n_samples = u_sorted_batch.shape[0]

    # g_vals[i, j] = g(u_sorted_batch[i, j]),  shape (n_samples, K)
    g_vals = g_func(u_sorted_batch)

    # Power coefficient a = (K-1)! * prod_j g(u_j),  shape (n_samples,)
    fact_km1 = float(math.factorial(K - 1))
    a_batch = fact_km1 * np.prod(g_vals, axis=1)   # (n_samples,)

    # Initialise R with a for every position k
    R = np.tile(a_batch[:, np.newaxis], (1, K))    # (n_samples, K)

    # Precompute prefix products for all k:
    #   prefix_prod[i, k] = prod_{j=0}^{k-1} g_vals[i, j]   (0-indexed)
    # prefix_prod[:, 0] = 1  (empty product, k=1 means no prefix)
    # prefix_prod[:, k] = prefix_prod[:, k-1] * g_vals[:, k-1]
    prefix_prod = np.ones((n_samples, K + 1))
    for j in range(K):
        prefix_prod[:, j + 1] = prefix_prod[:, j] * g_vals[:, j]

    # Subtract mu[l] * b_{l,k} for every (l, k) pair
    for k in range(1, K + 1):          # 1-indexed
        for l in range(K):             # l = 0, ..., K-1
            esp_index = l - k + 1
            n_tail    = K - k

            # Boundary conditions (same as scalar compute_b)
            if l < k - 1:
                continue
            if esp_index > n_tail:
                continue

            # Prefix product: prod_{j=1}^{k-1} g(u_j)  => prefix_prod[:, k-1]
            pp = prefix_prod[:, k - 1]   # shape (n_samples,)

            # Tail values: columns k..K-1 of g_vals (0-indexed), length n_tail
            tail_vals = g_vals[:, k:]    # shape (n_samples, n_tail)

            esp_val = elem_sym_poly_batch(tail_vals, esp_index)  # (n_samples,)

            factorial_coeff = float(
                math.factorial(l) * math.factorial(K - l)
            )

            b_batch = factorial_coeff * pp * esp_val  # (n_samples,)

            R[:, k - 1] -= mu[l] * b_batch

    return R


# ---------------------------------------------------------------------------
# Precomputed coefficients for fast bisection
# ---------------------------------------------------------------------------

def precompute_coefficients(u_sorted_batch, g_func, K):
    """
    Precompute a_batch and b_coeff for all samples, so that R can be computed
    as a_batch - mu @ b_coeff without recomputing g_vals or ESPs.

    Returns
    -------
    a_batch : np.ndarray, shape (n_samples, K)
        a_batch[:, k] = a_k for each sample (same value for all k).
    b_coeff : np.ndarray, shape (K, n_samples, K)
        b_coeff[l, :, k] = b_{l,k+1} for each sample (k is 0-indexed).
    """
    n_samples = u_sorted_batch.shape[0]

    g_vals = g_func(u_sorted_batch)  # (n_samples, K)

    # Power coefficient
    fact_km1 = float(math.factorial(K - 1))
    a_val = fact_km1 * np.prod(g_vals, axis=1)  # (n_samples,)
    a_batch = np.tile(a_val[:, np.newaxis], (1, K))  # (n_samples, K)

    # Prefix products
    prefix_prod = np.ones((n_samples, K + 1))
    for j in range(K):
        prefix_prod[:, j + 1] = prefix_prod[:, j] * g_vals[:, j]

    # Precompute all b_{l,k} values
    b_coeff = np.zeros((K, n_samples, K))  # b_coeff[l, :, k-1]

    for k in range(1, K + 1):
        n_tail = K - k
        tail_vals = g_vals[:, k:] if n_tail > 0 else np.empty((n_samples, 0))
        pp = prefix_prod[:, k - 1]

        l_min = k - 1
        l_max = min(K - 1, k - 1 + n_tail)

        if l_min > l_max:
            continue

        max_order = l_max - k + 1

        if n_tail > 0 and max_order >= 0:
            # Full DP table for ESPs of this tail
            dp = np.zeros((n_samples, max_order + 2))
            dp[:, 0] = 1.0
            for i in range(n_tail):
                v = tail_vals[:, i]
                for j in range(min(i + 1, max_order + 1), 0, -1):
                    dp[:, j] += v * dp[:, j - 1]

            for l in range(l_min, l_max + 1):
                esp_index = l - k + 1
                factorial_coeff = float(math.factorial(l) * math.factorial(K - l))
                b_coeff[l, :, k - 1] = factorial_coeff * pp * dp[:, esp_index]
        elif max_order == 0:
            for l in range(l_min, l_max + 1):
                esp_index = l - k + 1
                if esp_index == 0:
                    factorial_coeff = float(math.factorial(l) * math.factorial(K - l))
                    b_coeff[l, :, k - 1] = factorial_coeff * pp

    return a_batch, b_coeff


def compute_R_from_precomputed(mu, a_batch, b_coeff):
    """
    Compute R_batch from precomputed coefficients.

    R[:, k] = a_batch[:, k] - sum_l mu[l] * b_coeff[l, :, k]

    O(K^2 * n_samples) with no g_func calls or ESP computation.
    """
    # b_coeff shape: (K, n_samples, K_pos), mu shape: (K,)
    mu_b = np.einsum('l,lnk->nk', mu, b_coeff)
    return a_batch - mu_b
