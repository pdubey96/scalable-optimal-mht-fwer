"""
Monte Carlo FWER estimation and bisection target functions.

Key design point
----------------
FWER estimation requires tracking which p-values are null vs alternative
AFTER sorting.  The decision rule operates on sorted p-values, but to count
false rejections we need to know which of the rejected (smallest) p-values
originally came from null hypotheses.

Provides
--------
- prepare_samples         : pre-draw labeled samples (common random numbers)
- sample_and_estimate_fwer: convenience wrapper (prepares + estimates)
- target_function         : F_gamma(mu_gamma; mu_{-gamma}) - alpha for bisection
"""

import numpy as np

from general_k.alternatives import sample_from_g
from general_k.decision import optimal_l_star, l_star_batch, l_star_from_precomputed
from general_k.coefficients import precompute_coefficients


# ---------------------------------------------------------------------------
# prepare_samples
# ---------------------------------------------------------------------------

def prepare_samples(gamma, g_func, K, n_samples, rng):
    """
    Pre-draw labeled p-value samples for use in target_function during bisection.

    Layout of each row before sorting:
        positions 0 .. gamma-1   : alternative p-values (drawn from g via rejection sampling)
        positions gamma .. K-1   : null p-values (drawn from U(0,1))

    After sorting each row, ``sort_indices`` records the original column each
    sorted position came from.  ``null_label[j]`` is True iff original column j
    is a null hypothesis (j >= gamma).

    Parameters
    ----------
    gamma    : int
        Number of alternative hypotheses (0 <= gamma <= K).
    g_func   : callable
        Likelihood-ratio density on [0, 1].
    K        : int
        Total number of hypotheses.
    n_samples : int
        Number of Monte Carlo samples.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    samples_sorted : np.ndarray, shape (n_samples, K)
        Sorted p-values (ascending in each row).
    sort_indices : np.ndarray of int, shape (n_samples, K)
        sort_indices[i, j] = original column (before sorting) of position j
        in sample i.
    null_label : np.ndarray of bool, shape (K,)
        null_label[j] is True iff original column j is a null hypothesis.
    """
    samples = np.empty((n_samples, K), dtype=float)

    # Alternative p-values (columns 0..gamma-1)
    if gamma > 0:
        alt_flat = sample_from_g(g_func, n_samples * gamma, rng)
        samples[:, :gamma] = alt_flat.reshape(n_samples, gamma)

    # Null p-values (columns gamma..K-1)
    n_null = K - gamma
    if n_null > 0:
        samples[:, gamma:] = rng.uniform(0.0, 1.0, size=(n_samples, n_null))

    # Sort each row and remember original column indices
    sort_indices = np.argsort(samples, axis=1)          # (n_samples, K)
    samples_sorted = np.take_along_axis(samples, sort_indices, axis=1)

    # null_label: column j is null iff j >= gamma
    null_label = np.arange(K) >= gamma                  # shape (K,)

    return samples_sorted, sort_indices, null_label


# ---------------------------------------------------------------------------
# sample_and_estimate_fwer
# ---------------------------------------------------------------------------

def sample_and_estimate_fwer(gamma, mu, g_func, K, n_samples, rng):
    """
    Prepare fresh samples and estimate FWER_gamma(D^mu).

    A false rejection occurs when at least one null hypothesis is rejected.
    After sorting, position j in the sorted row came from original column
    sort_indices[i, j].  The decision rejects positions 0..l*-1 (sorted).
    A false rejection happens when any of those original columns is null.

    Parameters
    ----------
    gamma    : int
        Number of alternative hypotheses.
    mu : np.ndarray, shape (K,)
        Lagrange multipliers.
    g_func   : callable
        Likelihood-ratio density on [0, 1].
    K        : int
        Total number of hypotheses.
    n_samples : int
        Number of Monte Carlo samples.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    float
        Estimated FWER_gamma(D^mu) in [0, 1].
    """
    samples_sorted, sort_indices, null_label = prepare_samples(
        gamma, g_func, K, n_samples, rng
    )
    return _estimate_fwer_from_samples(
        mu, g_func, K, samples_sorted, sort_indices, null_label
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _estimate_fwer_from_samples(mu, g_func, K, samples_sorted, sort_indices, null_label):
    """
    Estimate FWER using pre-drawn labeled samples.

    For each Monte Carlo replicate:
      1. Compute l* = optimal_l_star(mu, u_sorted, g_func, K).
      2. Check whether any of the l* rejected positions maps back to a null
         hypothesis via sort_indices and null_label.

    Returns
    -------
    float
    """
    n_samples = samples_sorted.shape[0]

    # Vectorised: compute l* for all samples at once
    l_stars = l_star_batch(mu, samples_sorted, g_func, K)  # shape (n_samples,)

    # Vectorised false-rejection check:
    # For each sample i and sorted position j, check if j < l_stars[i]
    # AND null_label[sort_indices[i, j]] is True.
    # A false rejection occurs in sample i if any such j exists.
    position_mask = np.arange(K)[None, :] < l_stars[:, None]  # (n_samples, K) bool: rejected positions
    orig_is_null = null_label[sort_indices]  # (n_samples, K) bool: whether each sorted position is null
    false_rejection = np.any(position_mask & orig_is_null, axis=1)  # (n_samples,)

    return np.mean(false_rejection)


# ---------------------------------------------------------------------------
# target_function
# ---------------------------------------------------------------------------

def target_function(
    gamma,
    mu_gamma,
    mu_rest,
    g_func,
    K,
    alpha,
    samples_sorted,
    sort_indices,
    null_label,
):
    """
    Compute F_gamma(mu_gamma; mu_{-gamma}) - alpha using pre-drawn samples.

    Used as the function whose root is found by bisection.  Common random
    numbers are achieved by reusing the same pre-drawn samples for every
    evaluation of mu_gamma during bisection.

    Parameters
    ----------
    gamma     : int
        Index of the multiplier being optimised (0-indexed, 0 <= gamma < K).
    mu_gamma  : float
        Current candidate value for mu[gamma].
    mu_rest   : np.ndarray, shape (K,)
        Full mu vector with all K entries set; entry gamma will be overwritten
        by mu_gamma before evaluation.
    g_func    : callable
        Likelihood-ratio density on [0, 1].
    K         : int
        Total number of hypotheses.
    alpha     : float
        Target FWER level.
    samples_sorted : np.ndarray, shape (n_samples, K)
        Pre-drawn sorted p-value matrix.
    sort_indices : np.ndarray of int, shape (n_samples, K)
        Original-column indices after sorting.
    null_label : np.ndarray of bool, shape (K,)
        True where the original column is a null hypothesis.

    Returns
    -------
    float
        Estimated FWER_gamma(D^mu) - alpha.
        Positive => FWER too high (mu_gamma too small).
        Negative => FWER too low (mu_gamma too large).
    """
    mu = mu_rest.copy()
    mu[gamma] = mu_gamma

    fwer = _estimate_fwer_from_samples(
        mu, g_func, K, samples_sorted, sort_indices, null_label
    )
    return fwer - alpha


# ---------------------------------------------------------------------------
# Fast versions using precomputed coefficients
# ---------------------------------------------------------------------------

def prepare_samples_fast(gamma, g_func, K, n_samples, rng):
    """
    Like prepare_samples, but also precomputes a_batch and b_coeff
    so that bisection iterations avoid recomputing g_func and ESPs.
    """
    samples_sorted, sort_indices, null_label = prepare_samples(
        gamma, g_func, K, n_samples, rng
    )
    a_batch, b_coeff = precompute_coefficients(samples_sorted, g_func, K)
    return samples_sorted, sort_indices, null_label, a_batch, b_coeff


def target_function_fast(
    gamma, mu_gamma, mu_rest, K, alpha,
    sort_indices, null_label, a_batch, b_coeff,
):
    """
    Fast version of target_function using precomputed coefficients.
    No g_func calls or ESP computation during bisection.
    """
    mu = mu_rest.copy()
    mu[gamma] = mu_gamma

    l_stars = l_star_from_precomputed(mu, a_batch, b_coeff)

    position_mask = np.arange(K)[None, :] < l_stars[:, None]
    orig_is_null = null_label[sort_indices]
    false_rejection = np.any(position_mask & orig_is_null, axis=1)
    fwer = np.mean(false_rejection)

    return fwer - alpha
