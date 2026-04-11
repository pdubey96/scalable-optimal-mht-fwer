"""
Coordinate-descent algorithm for finding Lagrange multipliers mu.

Provides
--------
- bisection_update    : find mu_gamma* by bisection on F_gamma(mu_gamma; mu_{-gamma}) = alpha
- coordinate_descent  : main coordinate-descent loop cycling through all gamma
"""

import numpy as np

from general_k.target import prepare_samples, target_function, prepare_samples_fast, target_function_fast


# ---------------------------------------------------------------------------
# bisection_update
# ---------------------------------------------------------------------------

def bisection_update(
    gamma,
    mu_current,
    g_func,
    K,
    alpha,
    n_samples,
    rng,
    tol=1e-4,
    u_max=50.0,
):
    """
    Find mu_gamma* by bisection on F_gamma(mu_gamma; mu_{-gamma}) = alpha.

    Uses common random numbers: samples are drawn once and reused for every
    function evaluation during the bisection.

    Parameters
    ----------
    gamma       : int
        Index of the coordinate being updated (0-indexed, 0 <= gamma < K).
    mu_current  : np.ndarray, shape (K,)
        Current mu vector.  Entry ``gamma`` will be overwritten; the array is
        not modified in place.
    g_func      : callable
        Likelihood-ratio density on [0, 1].
    K           : int
        Total number of hypotheses.
    alpha       : float
        Target FWER level.
    n_samples   : int
        Number of Monte Carlo samples for FWER estimation.
    rng         : np.random.Generator
        Random number generator.
    tol         : float, optional
        Bisection terminates when hi - lo < tol (default 1e-4).
    u_max       : float, optional
        Upper bound for bisection search (default 50.0).

    Returns
    -------
    float
        Optimal mu_gamma value (>= 0).
    """
    # Draw samples once (common random numbers for the whole bisection)
    samples_sorted, sort_indices, null_label = prepare_samples(
        gamma, g_func, K, n_samples, rng
    )

    # Use mu_current as mu_rest (target_function will overwrite entry gamma)
    mu_rest = mu_current.copy()

    def F(mu_gamma):
        """Returns FWER_gamma - alpha (positive => FWER too high)."""
        return target_function(
            gamma, mu_gamma, mu_rest, g_func, K, alpha,
            samples_sorted, sort_indices, null_label,
        )

    # Check if the constraint is already slack at mu_gamma = 0
    if F(0.0) <= 0.0:
        return 0.0

    # Check if the root lies beyond u_max
    if F(u_max) > 0.0:
        return u_max

    # Bisect in [0, u_max]
    lo, hi = 0.0, u_max
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if F(mid) > 0.0:
            # FWER still above alpha -> mu_gamma too small, raise lower bound
            lo = mid
        else:
            # FWER at or below alpha -> mu_gamma too large, lower upper bound
            hi = mid

    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# coordinate_descent
# ---------------------------------------------------------------------------

def coordinate_descent(
    g_func,
    K,
    alpha,
    n_samples=500000,
    tol=1e-3,
    max_iter=50,
    u_max=50.0,
    seed=42,
):
    """
    Coordinate-descent algorithm for Lagrange multipliers.

    Cycles through gamma = 0, 1, ..., K-1, updating each mu[gamma] via
    ``bisection_update`` while holding the others fixed.  Stops when the
    L2-norm of the change between successive full sweeps is below ``tol``
    or ``max_iter`` sweeps have been completed.

    Parameters
    ----------
    g_func    : callable
        Likelihood-ratio density on [0, 1].
    K         : int
        Total number of hypotheses.
    alpha     : float
        Target FWER level.
    n_samples : int, optional
        Monte Carlo samples per bisection call (default 500000).
    tol       : float, optional
        Convergence tolerance on ||mu^(t) - mu^(t-1)||_2 (default 1e-3).
    max_iter  : int, optional
        Maximum number of full coordinate sweeps (default 50).
    u_max     : float, optional
        Upper bound for bisection search (default 50.0).
    seed      : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    mu      : np.ndarray, shape (K,)
        Converged Lagrange multiplier vector.
    history : list of np.ndarray
        List of mu vectors at the end of each full sweep.
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros(K)
    history = []

    for iteration in range(max_iter):
        mu_prev = mu.copy()

        for gamma in range(K):
            mu[gamma] = bisection_update(
                gamma=gamma,
                mu_current=mu,
                g_func=g_func,
                K=K,
                alpha=alpha,
                n_samples=n_samples,
                rng=rng,
                u_max=u_max,
            )

        history.append(mu.copy())

        # Check convergence
        delta = np.linalg.norm(mu - mu_prev)
        if delta < tol:
            break

    return mu, history


# ---------------------------------------------------------------------------
# Fast coordinate descent using precomputed coefficients
# ---------------------------------------------------------------------------

def bisection_update_fast(
    gamma, mu_current, g_func, K, alpha, n_samples, rng,
    tol=1e-4, u_max=50.0,
):
    """
    Like bisection_update, but precomputes a_batch/b_coeff once and reuses
    them for all bisection iterations. ~10-30x faster for K>=5.
    """
    samples_sorted, sort_indices, null_label, a_batch, b_coeff = \
        prepare_samples_fast(gamma, g_func, K, n_samples, rng)

    mu_rest = mu_current.copy()

    def F(mu_gamma):
        return target_function_fast(
            gamma, mu_gamma, mu_rest, K, alpha,
            sort_indices, null_label, a_batch, b_coeff,
        )

    if F(0.0) <= 0.0:
        return 0.0
    if F(u_max) > 0.0:
        return u_max

    lo, hi = 0.0, u_max
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if F(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def coordinate_descent_fast(
    g_func, K, alpha, n_samples=500000, tol=1e-3,
    max_iter=50, u_max=50.0, seed=42,
):
    """
    Fast coordinate descent using precomputed coefficients.
    Same interface as coordinate_descent but much faster for K>=5.
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros(K)
    history = []

    for iteration in range(max_iter):
        mu_prev = mu.copy()

        for gamma in range(K):
            mu[gamma] = bisection_update_fast(
                gamma=gamma, mu_current=mu, g_func=g_func,
                K=K, alpha=alpha, n_samples=n_samples, rng=rng,
                u_max=u_max,
            )

        history.append(mu.copy())
        delta = np.linalg.norm(mu - mu_prev)
        if delta < tol:
            break

    return mu, history
