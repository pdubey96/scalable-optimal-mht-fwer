"""
Coordinate-descent algorithm for finding Lagrange multipliers mu.

Provides
--------
- bisection_update      : find mu_gamma* by bisection on F_gamma(mu_gamma; mu_{-gamma}) = alpha
- coordinate_descent    : main coordinate-descent loop cycling through all gamma
- *_fast variants       : same, with precomputed coefficient batches

Monte Carlo design
------------------
One batch of N labeled samples is drawn for each configuration gamma BEFORE
the outer loop starts, and that same batch is reused for every update of
coordinate gamma across all sweeps and all bisection evaluations (common
random numbers).  Each estimator F_hat_gamma is therefore a single, fixed
empirical target function throughout the run, matching the Monte Carlo
analysis in the paper's supplement.  Batches are drawn independently across
configurations, so the asymptotic covariance of (F_hat_gamma(mu*))_gamma is
diagonal.

Stopping rule
-------------
After every sweep the algorithm computes the empirical complementarity
residual

    r(mu) = max_gamma | min{ mu_gamma, alpha - F_hat_gamma(mu) } |,

the plug-in Karush-Kuhn-Tucker residual of the empirical system evaluated at
the current full iterate.  The run stops when r(mu) <= res_tol, whose default
sqrt(alpha * (1 - alpha) / N) is the Monte Carlo standard error of an FWER
estimate at the constraint level (residuals below one standard error are
statistically indistinguishable from zero).  Two iterate-based criteria are
retained: ||mu^(t) - mu^(t-1)|| < tol (the recursion has settled), and
||mu^(t) - mu^(t-2)|| < tol (the recursion has settled into a period-two
orbit, which occurs when the empirical target jumps across the constraint
level so that no iterate can drive the residual below res_tol; this arises
for near-null alternatives such as Student-t with large df).  The stopping
reason and full per-sweep diagnostics (residuals, iterate changes) are
available via return_diagnostics=True.

An exit on delta/cycle/max_iter is a termination short of the certificate, and
those three look alike in a log.  Whenever one occurs the diagnostics carry a
"floor" entry from diagnose_bracket(): the local slope of the binding target,
an upper bound on the constraint gap the bracket width permits, and a bisect_tol
that would push that bound under res_tol -- or None for that last, when the
bracket is not what binds.
That distinguishes a bracket too coarse for a steep target, which is fixable,
from a target that jumps across the constraint level, which is not.
"""

import math

import numpy as np

from general_k.target import (
    prepare_samples,
    target_function,
    prepare_samples_fast,
    target_function_fast,
)
from general_k.coefficients import precompute_coefficients


# ---------------------------------------------------------------------------
# Bracketing + bisection helper
# ---------------------------------------------------------------------------

_MAX_DOUBLINGS = 60  # 50 * 2^60 ~ 5.8e19: far beyond any multiplier scale


_MIN_BRACKET_WIDTH = 1e-12  # hard floor; guarantees termination on a jump


def _bracket_and_bisect(F, u_max, tol, f_tol=None):
    """
    Solve F(mu_gamma) = 0 for the leftmost crossing of a non-increasing F.

    F returns FWER_gamma - alpha (positive => constraint violated).  Follows
    Algorithm 1 in the paper: if the initial upper bound does not bracket the
    crossing, it is doubled until it does; then bisection refines the bracket
    and the midpoint is returned.

    ``tol`` is the bracket width at which refinement may stop.  It is a width in
    mu, so the accuracy it buys on F depends on the local slope: the achievable
    |F| is about slope * tol.  On a steep target that can sit far above the
    residual tolerance the run is judged against, which stalls the outer
    recursion at a fixed point of the discretised map while its residual is
    still large.

    ``f_tol`` closes that gap.  It is strictly additive: bisection first refines
    to ``tol`` exactly as before, and only if |F| at that point still exceeds
    f_tol does it keep going, down to a hard floor of 1e-12 on the bracket width
    (reached only when F jumps across the level, in which case no mu attains
    f_tol and the floor is the honest answer).  Whenever the old width was
    already good enough the returned value is unchanged, so enabling f_tol
    cannot perturb a run that did not need it.  Passing the residual tolerance
    here makes the certification criterion attainable by construction.
    """
    if F(0.0) <= 0.0:
        return 0.0

    lo, hi = 0.0, u_max
    if F(hi) > 0.0:
        # Double the upper bound until the crossing is bracketed
        # (finite by monotonicity and F -> -alpha as mu_gamma -> infinity).
        for _ in range(_MAX_DOUBLINGS):
            lo, hi = hi, 2.0 * hi
            if F(hi) <= 0.0:
                break
        else:
            raise RuntimeError(
                f"bisection bracket not found: F(u) > 0 for u up to {hi:g}"
            )

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if F(mid) > 0.0:
            # FWER still above alpha -> mu_gamma too small, raise lower bound
            lo = mid
        else:
            # FWER at or below alpha -> mu_gamma large enough, lower upper bound
            hi = mid

    mid = (lo + hi) / 2.0
    if f_tol is None or abs(F(mid)) <= f_tol:
        return mid

    # The bracket width was too coarse for this slope: keep refining.
    while hi - lo > _MIN_BRACKET_WIDTH:
        mid = (lo + hi) / 2.0
        f_mid = F(mid)
        if abs(f_mid) <= f_tol:
            return mid
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def default_res_tol(alpha, n_samples):
    """Default residual tolerance: one Monte Carlo standard error of an FWER
    estimate at the constraint level."""
    return math.sqrt(alpha * (1.0 - alpha) / n_samples)


def diagnose_bracket(mu, alpha, fwer_at, bisect_tol, res_tol):
    """
    On an uncertified exit: how much of the residual the bracket width explains,
    and the width that would rule it out.

    The ``delta``, ``cycle`` and ``max_iter`` exits look alike in a log and
    have two quite different causes.  Either the bracket is too coarse for a
    steep target, which a smaller ``bisect_tol`` fixes, or the empirical target
    jumps across the constraint level, which no bracket fixes.  Separating them
    takes the local slope of ``F_gamma`` in its own coordinate: bisection
    returns the midpoint of a bracket of width ``bisect_tol``, so the
    coordinate is located to within ``bisect_tol / 2`` and the constraint gap
    is uncertain by at most ``|slope| * bisect_tol / 2``.

    That quantity is an upper bound on the bracket's contribution, not a floor
    the run cannot beat: the midpoint often lands nearer the crossing than the
    worst case, so an achieved residual below the bound is ordinary and not a
    contradiction.  Read it the one way it is sound -- a width satisfying
    ``|slope| * w / 2 <= res_tol`` is *sufficient* for the bracket to stop
    being the obstacle, and is not claimed to be necessary.

    A coordinate can also bind through ``mu_gamma`` itself rather than through
    its gap, since the residual term is ``|min(mu_gamma, alpha - F_gamma)|``.
    Refining the bracket does not address that case either, so which of the two
    binds is reported rather than assumed.

    Costs ``2K`` extra target evaluations, and is computed only when a run
    fails to certify, so a certifying run pays nothing for it.

    Parameters
    ----------
    mu         : final iterate
    alpha      : FWER level
    fwer_at    : callable ``(gamma, mu_vector) -> F_hat_gamma`` at that iterate
    bisect_tol : the bracket width the run used
    res_tol    : the residual tolerance the run was judged against

    Returns
    -------
    dict
        ``slopes`` per coordinate; ``binding_gamma`` and ``binding_term``
        (``"gap"`` or ``"multiplier"``) for the coordinate that sets the
        current residual; ``gap_bound``, the largest constraint gap the bracket
        permits there; and ``sufficient_bisect_tol``, the width that would push
        that bound under ``res_tol``, which is ``None`` when the binding slope
        is zero, since the bracket is then not what binds.
    """
    K = len(mu)
    mu = np.asarray(mu, dtype=float)

    slopes = np.zeros(K)
    for gamma in range(K):
        lo = max(mu[gamma] - bisect_tol, 0.0)
        hi = mu[gamma] + bisect_tol
        if hi <= lo:
            continue
        mu_lo, mu_hi = mu.copy(), mu.copy()
        mu_lo[gamma], mu_hi[gamma] = lo, hi
        slopes[gamma] = (fwer_at(gamma, mu_hi) - fwer_at(gamma, mu_lo)) / (hi - lo)

    # Which coordinate sets the residual, and through which of its two terms.
    gaps = np.array([alpha - fwer_at(g, mu) for g in range(K)])
    terms = np.abs(np.minimum(mu, gaps))
    binding = int(np.argmax(terms)) if K else -1
    binding_term = "multiplier" if mu[binding] < gaps[binding] else "gap"

    slope = abs(float(slopes[binding]))
    return {
        "slopes": [float(s) for s in slopes],
        "binding_gamma": binding,
        "binding_term": binding_term,
        "binding_slope": slope,
        "max_abs_slope": float(np.abs(slopes).max()) if K else 0.0,
        "gap_bound": slope * bisect_tol / 2.0,
        "sufficient_bisect_tol": (2.0 * res_tol / slope) if slope > 0.0 else None,
    }


def _residual(mu, alpha, fwer_of_gamma):
    """Empirical complementarity residual max_gamma |min(mu_g, alpha - Fhat_g)|.

    fwer_of_gamma(gamma) must return F_hat_gamma evaluated at the current
    full iterate mu, using configuration gamma's fixed batch.
    """
    r = 0.0
    for gamma in range(len(mu)):
        r = max(r, abs(min(mu[gamma], alpha - fwer_of_gamma(gamma))))
    return r


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
    samples=None,
    f_tol=None,
):
    """
    Find mu_gamma* by bisection on F_gamma(mu_gamma; mu_{-gamma}) = alpha.

    Uses common random numbers: the same batch is reused for every function
    evaluation during the bisection.  When ``samples`` is provided (the tuple
    returned by ``prepare_samples``), that fixed batch is used and ``rng`` is
    not consumed; this is how ``coordinate_descent`` keeps one batch per
    configuration across all sweeps.  When ``samples`` is None a fresh batch
    is drawn (one-shot use).

    Parameters
    ----------
    gamma       : int
        Index of the coordinate being updated (0-indexed, 0 <= gamma < K).
    mu_current  : np.ndarray, shape (K,)
        Current mu vector.  Entry ``gamma`` will be overwritten; the array is
        not modified in place.
    g_func      : callable
        Likelihood-ratio density on [0, 1] with a callable
        ``sample(n_samples, rng)`` method.
    K           : int
        Total number of hypotheses.
    alpha       : float
        Target FWER level.
    n_samples   : int
        Number of Monte Carlo samples for FWER estimation.
    rng         : np.random.Generator
        Random number generator (used only when ``samples`` is None).
    tol         : float, optional
        Bisection terminates when hi - lo < tol (default 1e-4).
    u_max       : float, optional
        Initial upper bound for the bisection bracket (default 50.0); doubled
        as needed until the crossing is bracketed.
    samples     : tuple, optional
        (samples_sorted, sort_indices, null_label) from ``prepare_samples``.

    Returns
    -------
    float
        Optimal mu_gamma value (>= 0).
    """
    if samples is None:
        samples = prepare_samples(gamma, g_func, K, n_samples, rng)
    samples_sorted, sort_indices, null_label = samples

    # Use mu_current as mu_rest (target_function will overwrite entry gamma)
    mu_rest = mu_current.copy()

    def F(mu_gamma):
        """Returns FWER_gamma - alpha (positive => FWER too high)."""
        return target_function(
            gamma, mu_gamma, mu_rest, g_func, K, alpha,
            samples_sorted, sort_indices, null_label,
        )

    return _bracket_and_bisect(F, u_max, tol, f_tol)


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
    res_tol=None,
    bisect_tol=1e-4,
    f_tol=None,
    return_diagnostics=False,
):
    """
    Coordinate-descent algorithm for Lagrange multipliers.

    ``bisect_tol`` (default 1e-4) is the per-coordinate bisection bracket
    width delta; shrinking it (e.g. o(N^{-1/2})) targets the idealised
    centred-limit regime of Supplementary Theorem S1.11(b).

    Draws one labeled batch per configuration gamma up front, then cycles
    through gamma = 0, 1, ..., K-1, updating each mu[gamma] via
    ``bisection_update`` on that configuration's fixed batch while holding
    the others fixed.  After each sweep the empirical complementarity
    residual is computed at the current iterate; the loop stops when it
    falls below ``res_tol`` (primary criterion), when the L2-norm of the
    change between successive sweeps falls below ``tol`` (secondary), or
    after ``max_iter`` sweeps.

    Parameters
    ----------
    g_func    : callable
        Likelihood-ratio density on [0, 1] with a callable
        ``sample(n_samples, rng)`` method.
    K         : int
        Total number of hypotheses.
    alpha     : float
        Target FWER level.
    n_samples : int, optional
        Monte Carlo samples per configuration (default 500000).
    tol       : float, optional
        Secondary tolerance on ||mu^(t) - mu^(t-1)||_2 (default 1e-3).
    max_iter  : int, optional
        Maximum number of full coordinate sweeps (default 50).
    u_max     : float, optional
        Initial upper bound for the bisection bracket (default 50.0).
    seed      : int, optional
        RNG seed for reproducibility (default 42).
    res_tol   : float, optional
        Residual tolerance; default sqrt(alpha*(1-alpha)/n_samples).
    return_diagnostics : bool, optional
        If True, additionally return a diagnostics dict.

    Returns
    -------
    mu      : np.ndarray, shape (K,)
        Converged Lagrange multiplier vector.
    history : list of np.ndarray
        List of mu vectors at the end of each full sweep.
    diagnostics : dict (only if return_diagnostics=True)
        Keys: converged, reason, sweeps, final_residual, res_tol,
        residuals (per sweep), deltas (per sweep), and floor (None when the
        run certified; otherwise the diagnose_bracket() report).
    """
    rng = np.random.default_rng(seed)
    if res_tol is None:
        res_tol = default_res_tol(alpha, n_samples)

    # One fixed batch per configuration, reused across all sweeps.
    batches = [
        prepare_samples(gamma, g_func, K, n_samples, rng) for gamma in range(K)
    ]

    def fwer_at(gamma, mu):
        samples_sorted, sort_indices, null_label = batches[gamma]
        return alpha + target_function(
            gamma, mu[gamma], mu, g_func, K, alpha,
            samples_sorted, sort_indices, null_label,
        )

    mu = np.zeros(K)
    history, residuals, deltas = [], [], []
    reason = "max_iter"

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
                tol=bisect_tol,
                samples=batches[gamma],
                f_tol=f_tol,
            )

        history.append(mu.copy())
        delta = float(np.linalg.norm(mu - mu_prev))
        res = _residual(mu, alpha, lambda g: fwer_at(g, mu))
        deltas.append(delta)
        residuals.append(res)

        if res <= res_tol:
            reason = "residual"
            break
        if delta < tol:
            reason = "delta"
            break
        if len(history) >= 3 and float(
                np.linalg.norm(history[-1] - history[-3])) < tol:
            # Period-two orbit: the empirical target jumps across the
            # constraint level, so the residual has a positive floor.
            reason = "cycle"
            break

    # Genuine convergence is certified only by the residual criterion;
    # delta/cycle/max_iter exits are terminations short of that certificate.
    converged = (reason == "residual")
    floor = None if converged else diagnose_bracket(
        mu, alpha, fwer_at, bisect_tol, res_tol
    )
    diagnostics = {
        "converged": converged,
        "reason": reason,
        "floor": floor,
        "sweeps": len(history),
        "final_residual": residuals[-1] if residuals else float("nan"),
        "res_tol": res_tol,
        "residuals": residuals,
        "deltas": deltas,
        "bisect_tol": bisect_tol,
        "f_tol": f_tol,
        "max_iter": max_iter,
    }
    if return_diagnostics:
        return mu, history, diagnostics
    return mu, history


# ---------------------------------------------------------------------------
# Fast coordinate descent using precomputed coefficients
# ---------------------------------------------------------------------------

def bisection_update_fast(
    gamma, mu_current, g_func, K, alpha, n_samples, rng,
    tol=1e-4, u_max=50.0, precomputed=None, f_tol=None,
):
    """
    Like bisection_update, but uses a_batch/b_coeff precomputed once per
    batch and reused for all bisection iterations. ~10-30x faster for K>=5.

    ``precomputed`` is the tuple returned by ``prepare_samples_fast``; when
    None, a fresh batch is drawn and precomputed (one-shot use).
    """
    if precomputed is None:
        precomputed = prepare_samples_fast(gamma, g_func, K, n_samples, rng)
    samples_sorted, sort_indices, null_label, a_batch, b_coeff = precomputed

    mu_rest = mu_current.copy()

    def F(mu_gamma):
        return target_function_fast(
            gamma, mu_gamma, mu_rest, K, alpha,
            sort_indices, null_label, a_batch, b_coeff,
        )

    return _bracket_and_bisect(F, u_max, tol, f_tol)


# Cache precomputed coefficient tensors across sweeps only while their total
# footprint (K configurations x b_coeff of shape (K, n, K) in float64) stays
# below this many bytes; beyond it, coefficients are recomputed at each visit
# from the fixed per-configuration sample batches (identical results).
_PRECOMPUTE_CACHE_BYTES = 3.5e9


def coordinate_descent_fast(
    g_func, K, alpha, n_samples=500000, tol=1e-3,
    max_iter=50, u_max=50.0, seed=42, cache_bytes=None,
    res_tol=None, bisect_tol=1e-4, f_tol=None, return_diagnostics=False,
    power_coeff_fn=None,
):
    """
    Fast coordinate descent using precomputed coefficients.
    Same interface, fixed-batch design, and residual-based stopping rule as
    coordinate_descent.

    cache_bytes overrides the precompute-cache budget (bytes); pass a larger
    value on high-memory machines to cache all K coefficient tensors for
    large K (e.g. the K=21 application).

    power_coeff_fn replaces the objective's power coefficients: a callable
    (u_sorted_batch, g_func, K) -> a_batch of shape (n_samples, K).  The
    default is the manuscript's Pi_K coefficient, constant in the position;
    general_k.prior_objective supplies coefficients for an average power taken
    against a prior on the number of non-nulls.  The family-wise error
    constraints, and hence b_coeff, are unaffected either way.
    """
    rng = np.random.default_rng(seed)
    if res_tol is None:
        res_tol = default_res_tol(alpha, n_samples)

    # One fixed batch per configuration, reused across all sweeps.
    batches = [
        prepare_samples(gamma, g_func, K, n_samples, rng) for gamma in range(K)
    ]

    budget = _PRECOMPUTE_CACHE_BYTES if cache_bytes is None else cache_bytes
    cache_precompute = (K * K * K * n_samples * 8) < budget
    precomputed = [None] * K

    def get_precomputed(gamma):
        if precomputed[gamma] is not None:
            return precomputed[gamma]
        samples_sorted, sort_indices, null_label = batches[gamma]
        a_batch, b_coeff = precompute_coefficients(samples_sorted, g_func, K)
        if power_coeff_fn is not None:
            a_batch = power_coeff_fn(samples_sorted, g_func, K)
        item = (samples_sorted, sort_indices, null_label, a_batch, b_coeff)
        if cache_precompute:
            precomputed[gamma] = item
        return item

    def fwer_at(gamma, mu):
        _, sort_indices, null_label, a_batch, b_coeff = get_precomputed(gamma)
        return alpha + target_function_fast(
            gamma, mu[gamma], mu, K, alpha,
            sort_indices, null_label, a_batch, b_coeff,
        )

    mu = np.zeros(K)
    history, residuals, deltas = [], [], []
    reason = "max_iter"

    for iteration in range(max_iter):
        mu_prev = mu.copy()

        for gamma in range(K):
            mu[gamma] = bisection_update_fast(
                gamma=gamma, mu_current=mu, g_func=g_func,
                K=K, alpha=alpha, n_samples=n_samples, rng=rng,
                u_max=u_max, tol=bisect_tol, precomputed=get_precomputed(gamma),
                f_tol=f_tol,
            )

        history.append(mu.copy())
        delta = float(np.linalg.norm(mu - mu_prev))
        res = _residual(mu, alpha, lambda g: fwer_at(g, mu))
        deltas.append(delta)
        residuals.append(res)

        if res <= res_tol:
            reason = "residual"
            break
        if delta < tol:
            reason = "delta"
            break
        if len(history) >= 3 and float(
                np.linalg.norm(history[-1] - history[-3])) < tol:
            # Period-two orbit: the empirical target jumps across the
            # constraint level, so the residual has a positive floor.
            reason = "cycle"
            break

    # Genuine convergence is certified only by the residual criterion;
    # delta/cycle/max_iter exits are terminations short of that certificate.
    converged = (reason == "residual")
    floor = None if converged else diagnose_bracket(
        mu, alpha, fwer_at, bisect_tol, res_tol
    )
    diagnostics = {
        "converged": converged,
        "reason": reason,
        "floor": floor,
        "sweeps": len(history),
        "final_residual": residuals[-1] if residuals else float("nan"),
        "res_tol": res_tol,
        "residuals": residuals,
        "deltas": deltas,
        "bisect_tol": bisect_tol,
        "f_tol": f_tol,
        "max_iter": max_iter,
    }
    if return_diagnostics:
        return mu, history, diagnostics
    return mu, history
