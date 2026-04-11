"""
Simulation framework for evaluating the optimal policy and baseline methods.

Provides
--------
- evaluate_optimal_policy : Power and FWER of D^mu (the optimal algorithm).
- evaluate_baseline       : Power and FWER of a standard baseline procedure.
- compare_methods         : Full comparison across Algorithm + four baselines.
"""

import numpy as np

from general_k.algorithm import coordinate_descent, coordinate_descent_fast
from general_k.decision import l_star_batch
from general_k.alternatives import sample_from_g
from general_k.baselines import bonferroni, holm, hochberg, hommel


# ---------------------------------------------------------------------------
# evaluate_optimal_policy
# ---------------------------------------------------------------------------

def evaluate_optimal_policy(mu, g_func, K, n_mc, seed=123):
    """
    Evaluate the power and FWER of D^mu (the optimal policy).

    Power metrics
    -------------
    - pi_K   : average fraction of alternatives rejected per sample.
    - pi_any : fraction of samples with at least one rejection.

    FWER metrics
    ------------
    For each gamma in {0, 1, ..., K-1}: generate gamma alternative and
    (K - gamma) null p-values, compute l* via l_star_batch, and check
    whether any null hypothesis falls among the rejected positions.

    Parameters
    ----------
    mu     : np.ndarray, shape (K,)
        Lagrange multiplier vector.
    g_func : callable
        Likelihood-ratio density on [0, 1].
    K      : int
        Number of hypotheses.
    n_mc   : int
        Number of Monte Carlo replications.
    seed   : int, optional
        RNG seed (default 123).

    Returns
    -------
    dict with keys:
        "pi_K"              : float  -- average rejection fraction
        "pi_any"            : float  -- probability of any rejection
        "fwer_0", ..., "fwer_{K-1}" : float  -- FWER for each gamma
    """
    rng = np.random.default_rng(seed)
    results = {}

    # ------------------------------------------------------------------
    # Power: all K p-values drawn from g (all alternatives)
    # ------------------------------------------------------------------
    # Draw n_mc * K samples from the alternative distribution
    alt_flat = sample_from_g(g_func, n_mc * K, rng)
    samples_alt = alt_flat.reshape(n_mc, K)          # (n_mc, K)

    # Sort each row
    u_sorted_alt = np.sort(samples_alt, axis=1)       # (n_mc, K)

    # Compute l* for each sample
    l_stars = l_star_batch(mu, u_sorted_alt, g_func, K)  # (n_mc,)

    # pi_K: average fraction of K alternatives rejected
    results["pi_K"] = float(np.mean(l_stars / K))

    # pi_any: fraction of samples with at least one rejection
    results["pi_any"] = float(np.mean(l_stars > 0))

    # ------------------------------------------------------------------
    # FWER: for gamma = 0, 1, ..., K-1
    # ------------------------------------------------------------------
    for gamma in range(K):
        # Draw samples: gamma from g, K-gamma from U(0,1)
        samples = np.empty((n_mc, K), dtype=float)

        if gamma > 0:
            alt_flat_g = sample_from_g(g_func, n_mc * gamma, rng)
            samples[:, :gamma] = alt_flat_g.reshape(n_mc, gamma)

        n_null = K - gamma
        if n_null > 0:
            samples[:, gamma:] = rng.uniform(0.0, 1.0, size=(n_mc, n_null))

        # Sort and keep track of original column indices
        sort_indices = np.argsort(samples, axis=1)               # (n_mc, K)
        u_sorted = np.take_along_axis(samples, sort_indices, axis=1)

        # null_label[j] is True iff original column j is null (j >= gamma)
        null_label = np.arange(K) >= gamma                        # shape (K,)

        # Compute l* for all samples at once
        l_stars_g = l_star_batch(mu, u_sorted, g_func, K)        # (n_mc,)

        # Vectorised false-rejection check
        position_mask = np.arange(K)[None, :] < l_stars_g[:, None]  # (n_mc, K)
        orig_is_null = null_label[sort_indices]  # (n_mc, K)
        false_rejection = np.any(position_mask & orig_is_null, axis=1)
        results[f"fwer_{gamma}"] = float(np.mean(false_rejection))

    return results


# ---------------------------------------------------------------------------
# evaluate_baseline
# ---------------------------------------------------------------------------

def evaluate_baseline(method_func, g_func, K, alpha, n_mc, seed=123):
    """
    Evaluate the power and FWER of a standard multiple testing procedure.

    Parameters
    ----------
    method_func : callable
        Signature: (p_values: np.ndarray, alpha: float) -> np.ndarray of int.
        Receives *unsorted* p-values and returns a 0/1 decision array aligned
        with the input order.
    g_func      : callable
        Likelihood-ratio density on [0, 1].
    K           : int
        Number of hypotheses.
    alpha       : float
        FWER level passed to method_func.
    n_mc        : int
        Number of Monte Carlo replications.
    seed        : int, optional
        RNG seed (default 123).

    Returns
    -------
    dict with keys:
        "pi_K", "pi_any", "fwer_0", ..., "fwer_{K-1}"
    """
    rng = np.random.default_rng(seed)
    results = {}

    # ------------------------------------------------------------------
    # Power: all K hypotheses are alternatives
    # ------------------------------------------------------------------
    alt_flat = sample_from_g(g_func, n_mc * K, rng)
    samples_alt = alt_flat.reshape(n_mc, K)   # (n_mc, K) unsorted

    total_rejections = 0
    any_rejection_count = 0

    for i in range(n_mc):
        d = method_func(samples_alt[i], alpha)  # shape (K,)
        n_rej = int(np.sum(d))
        total_rejections += n_rej
        if n_rej > 0:
            any_rejection_count += 1

    results["pi_K"] = float(total_rejections / (n_mc * K))
    results["pi_any"] = float(any_rejection_count / n_mc)

    # ------------------------------------------------------------------
    # FWER: gamma alternatives + (K - gamma) nulls
    # ------------------------------------------------------------------
    for gamma in range(K):
        samples = np.empty((n_mc, K), dtype=float)

        if gamma > 0:
            alt_flat_g = sample_from_g(g_func, n_mc * gamma, rng)
            samples[:, :gamma] = alt_flat_g.reshape(n_mc, gamma)

        n_null = K - gamma
        if n_null > 0:
            samples[:, gamma:] = rng.uniform(0.0, 1.0, size=(n_mc, n_null))

        # null positions: columns gamma..K-1 in the *original* (unsorted) order
        null_cols = np.arange(gamma, K)

        false_count = 0
        for i in range(n_mc):
            d = method_func(samples[i], alpha)   # shape (K,), aligned with original order
            if np.any(d[null_cols] == 1):
                false_count += 1

        results[f"fwer_{gamma}"] = float(false_count / n_mc)

    return results


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------

def compare_methods(
    K,
    g_func,
    alpha=0.05,
    n_algo_samples=100000,
    n_mc=50000,
    algo_tol=1e-2,
    max_iter=100,
    seed=42,
):
    """
    Full comparison of the optimal algorithm vs. standard baselines.

    Steps
    -----
    1. Run coordinate_descent to obtain mu_star.
    2. Evaluate the optimal policy (Algorithm).
    3. Evaluate Bonferroni, Holm, Hochberg, Hommel.

    Parameters
    ----------
    K               : int
        Number of hypotheses.
    g_func          : callable
        Likelihood-ratio density on [0, 1].
    alpha           : float, optional
        FWER level (default 0.05).
    n_algo_samples  : int, optional
        Monte Carlo samples for coordinate_descent (default 100000).
    n_mc            : int, optional
        Monte Carlo samples for evaluation (default 50000).
    algo_tol        : float, optional
        Convergence tolerance for coordinate_descent (default 1e-2).
    seed            : int, optional
        RNG seed (default 42).

    Returns
    -------
    dict mapping method name (str) -> results dict
        Each results dict has keys "pi_K", "pi_any", "fwer_0", ..., "fwer_{K-1}".
    """
    # ------------------------------------------------------------------
    # Step 1: find mu_star via coordinate descent
    # ------------------------------------------------------------------
    mu_star, history = coordinate_descent_fast(
        g_func=g_func,
        K=K,
        alpha=alpha,
        n_samples=n_algo_samples,
        tol=algo_tol,
        max_iter=max_iter,
        seed=seed,
    )

    n_iters = len(history)
    converged = (n_iters < 50)  # default max_iter in coordinate_descent is 50

    print(f"mu_star = {mu_star}")
    print(f"Coordinate descent: {n_iters} iteration(s), converged={converged}")

    # ------------------------------------------------------------------
    # Step 2: evaluate Algorithm
    # ------------------------------------------------------------------
    all_results = {}

    all_results["Algorithm"] = evaluate_optimal_policy(
        mu=mu_star,
        g_func=g_func,
        K=K,
        n_mc=n_mc,
        seed=seed + 1,
    )

    # ------------------------------------------------------------------
    # Step 3: evaluate baselines
    # ------------------------------------------------------------------
    baseline_methods = {
        "Bonferroni": bonferroni,
        "Holm":       holm,
        "Hochberg":   hochberg,
        "Hommel":     hommel,
    }

    for name, method_func in baseline_methods.items():
        all_results[name] = evaluate_baseline(
            method_func=method_func,
            g_func=g_func,
            K=K,
            alpha=alpha,
            n_mc=n_mc,
            seed=seed + 1,
        )

    return all_results
