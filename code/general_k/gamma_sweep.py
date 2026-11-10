"""
Power as a function of the number of non-null hypotheses.

Motivation
----------
SPOT maximises the average power ``Pi_K`` under the all-alternatives
configuration ``h_K``.  A natural question is how quickly that advantage
decays once only ``gamma < K`` of the
``K`` hypotheses are false.  This module evaluates, for a policy that is
computed once and then held fixed, the average power

    Pi_gamma(D) = gamma^{-1} E_{h_gamma} ( sum_{k=1}^{gamma} D_k ),

for every gamma = 1, ..., K, alongside FWER_gamma for gamma = 0, ..., K-1.

The multipliers are *not* recomputed per gamma: SPOT is tuned at gamma = K and
then evaluated off-design, which is exactly the question asked.

Baselines
---------
Bonferroni, Holm, Hochberg, Hommel and Romano-Wolf all reject a leading block
of the sorted p-values, so each is summarised by the number rejected.  The
batch functions here return that count for a whole sample matrix at once;
``tests/`` checks them against the per-sample reference implementations in
``baselines.py``.
"""

import numpy as np

from general_k.closed_testing import CLOSED_TESTS
from general_k.coefficients import precompute_coefficients
from general_k.decision import l_star_batch, l_star_from_precomputed
from general_k.alternatives import sample_from_g


# ---------------------------------------------------------------------------
# Batch baselines: number of rejections per sample
# ---------------------------------------------------------------------------

def _leading_run(ok: np.ndarray) -> np.ndarray:
    """Length of the leading run of True in each row of a 2-D boolean array."""
    n_rows, n_cols = ok.shape
    all_true = np.all(ok, axis=1)
    # argmax of ~ok gives the first False; rows that are all True get 0 there.
    first_false = np.argmax(~ok, axis=1)
    return np.where(all_true, n_cols, first_false)


def bonferroni_n_rej(sorted_p: np.ndarray, alpha: float) -> np.ndarray:
    """Rejections per row for Bonferroni: p_(i) <= alpha / K."""
    K = sorted_p.shape[1]
    return np.sum(sorted_p <= alpha / K, axis=1)


def holm_n_rej(sorted_p: np.ndarray, alpha: float) -> np.ndarray:
    """Rejections per row for Holm: leading run of p_(i) <= alpha / (K-i+1)."""
    K = sorted_p.shape[1]
    thresholds = alpha / (K - np.arange(K))          # i = 1..K, 1-indexed
    return _leading_run(sorted_p <= thresholds[None, :])


def hochberg_n_rej(sorted_p: np.ndarray, alpha: float) -> np.ndarray:
    """Rejections per row for Hochberg: largest i with p_(i) <= alpha/(K-i+1)."""
    K = sorted_p.shape[1]
    thresholds = alpha / (K - np.arange(K))
    ok = sorted_p <= thresholds[None, :]             # (n, K)
    # Largest 1-indexed i that satisfies the criterion; 0 if none does.
    any_ok = np.any(ok, axis=1)
    last_ok = K - 1 - np.argmax(ok[:, ::-1], axis=1)
    return np.where(any_ok, last_ok + 1, 0)


def hommel_n_rej(sorted_p: np.ndarray, alpha: float) -> np.ndarray:
    """
    Rejections per row for Hommel.

    ``j_star`` is the largest j in {1, ..., K} such that
    ``p_(K-j+k) > k*alpha/j`` for every k = 1, ..., j; the rejection threshold
    is then ``alpha / j_star``, with ``j_star = 1`` as the fallback.  The loop
    runs over j only (at most K passes), each pass vectorised over samples.
    """
    n, K = sorted_p.shape
    j_star = np.ones(n, dtype=int)

    for j in range(1, K + 1):
        k = np.arange(1, j + 1)
        idx = K - j + k - 1                          # 0-indexed positions
        holds = np.all(sorted_p[:, idx] > k[None, :] * alpha / j, axis=1)
        j_star = np.where(holds, j, j_star)          # keep the largest valid j

    return np.sum(sorted_p <= (alpha / j_star)[:, None], axis=1)


def romano_wolf_n_rej(sorted_p: np.ndarray, alpha: float) -> np.ndarray:
    """
    Rejections per row for Romano-Wolf stepdown, min-P under independence.

    Step ``i`` leaves ``m = K - i + 1`` hypotheses under test, and the minimum
    of ``m`` independent uniforms has c.d.f. ``1 - (1 - p)^m``, so the stepdown
    threshold is ``1 - (1 - alpha)^(1/m)`` and the rejected set is the leading
    run below it.  Same shape as ``holm_n_rej``, whose thresholds ``alpha / m``
    this dominates.
    """
    K = sorted_p.shape[1]
    m = K - np.arange(K)                             # i = 1..K, 1-indexed
    thresholds = 1.0 - (1.0 - alpha) ** (1.0 / m)
    return _leading_run(sorted_p <= thresholds[None, :])


BATCH_BASELINES = {
    "Bonferroni": bonferroni_n_rej,
    "Holm": holm_n_rej,
    "Hochberg": hochberg_n_rej,
    "Hommel": hommel_n_rej,
    "Romano-Wolf": romano_wolf_n_rej,
}

# Every procedure SPOT is measured against: the five stepwise ones above, and
# closed testing on a combination statistic, which is the right benchmark when
# many hypotheses are false.
COMPARATORS = list(BATCH_BASELINES) + list(CLOSED_TESTS)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_configuration(gamma, g_func, K, n_mc, rng):
    """
    Draw ``n_mc`` samples under configuration ``h_gamma``.

    Columns ``0 .. gamma-1`` are alternatives drawn from ``g``; columns
    ``gamma .. K-1`` are nulls drawn from U(0, 1).

    Returns
    -------
    sorted_p : (n_mc, K) ascending p-values
    sort_indices : (n_mc, K) original column of each sorted position
    """
    samples = np.empty((n_mc, K), dtype=float)

    if gamma > 0:
        alt = sample_from_g(g_func, n_mc * gamma, rng)
        samples[:, :gamma] = alt.reshape(n_mc, gamma)
    if K - gamma > 0:
        samples[:, gamma:] = rng.uniform(0.0, 1.0, size=(n_mc, K - gamma))

    sort_indices = np.argsort(samples, axis=1)
    sorted_p = np.take_along_axis(samples, sort_indices, axis=1)
    return sorted_p, sort_indices


def _mask_from_count(n_rej, K):
    """Leading-block rejection count -> boolean mask in sorted order."""
    return np.arange(K)[None, :] < n_rej[:, None]


def _power_and_fwer(rejected, sort_indices, gamma, K):
    """
    Average power over the ``gamma`` alternatives and the false-rejection rate.

    ``rejected`` is a boolean mask in *sorted* order, so the rejected original
    columns are read off through ``sort_indices``.  Original column j is an
    alternative iff j < gamma.  Taking a mask rather than a count lets closed
    testing be scored by the same code as the rank-based procedures.
    """
    orig_is_alt = sort_indices < gamma                        # (n, K)

    n_true_rej = np.sum(rejected & orig_is_alt, axis=1)
    n_false_rej = np.sum(rejected & ~orig_is_alt, axis=1)

    power = float(np.mean(n_true_rej / gamma)) if gamma > 0 else None
    fwer = float(np.mean(n_false_rej > 0)) if gamma < K else 0.0
    return power, fwer


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def gamma_sweep(mu, g_func, K, alpha, n_mc=50000, seed=2026, methods=None,
                power_coeff_fn=None, closed=None):
    """
    Average power and FWER at every configuration ``h_gamma``, gamma = 0..K.

    All methods see the *same* p-value matrix at each gamma (common random
    numbers), so differences between curves are not Monte Carlo noise between
    independent draws.

    Parameters
    ----------
    mu     : (K,) multipliers from coordinate descent, tuned at gamma = K
    g_func : alternative density / likelihood ratio on [0, 1] with a callable
        ``sample(n_samples, rng)`` method
    K      : number of hypotheses
    alpha  : FWER level
    n_mc   : Monte Carlo replications per gamma
    seed   : RNG seed
    methods: rank-based baseline names to include (default: all four)
    closed : closed-testing comparator names to include (default: all three)
    power_coeff_fn : optional objective power coefficients, as passed to
        ``coordinate_descent_fast``.  Must match whatever ``mu`` was solved
        with, otherwise the policy being evaluated is not the one computed.

    Returns
    -------
    dict
        ``{"SPOT": {"power": [...], "fwer": [...]}, "Hommel": {...}, ...}``
        with entries indexed by gamma = 0, ..., K.  ``power[0]`` is ``None``
        (no alternatives exist) and ``fwer[K]`` is 0 (no nulls exist).
    """
    names = list(BATCH_BASELINES) if methods is None else list(methods)
    closed_names = list(CLOSED_TESTS) if closed is None else list(closed)
    out = {name: {"power": [], "fwer": []}
           for name in ["SPOT"] + names + closed_names}

    rng = np.random.default_rng(seed)

    for gamma in range(K + 1):
        sorted_p, sort_indices = sample_configuration(gamma, g_func, K, n_mc, rng)

        # SPOT: l* is itself the size of the rejected leading block.
        if power_coeff_fn is None:
            n_rej_spot = l_star_batch(mu, sorted_p, g_func, K)
        else:
            _, b_coeff = precompute_coefficients(sorted_p, g_func, K)
            a_batch = power_coeff_fn(sorted_p, g_func, K)
            n_rej_spot = l_star_from_precomputed(mu, a_batch, b_coeff)
        power, fwer = _power_and_fwer(
            _mask_from_count(n_rej_spot, K), sort_indices, gamma, K)
        out["SPOT"]["power"].append(power)
        out["SPOT"]["fwer"].append(fwer)

        for name in names:
            n_rej = BATCH_BASELINES[name](sorted_p, alpha)
            power, fwer = _power_and_fwer(
                _mask_from_count(n_rej, K), sort_indices, gamma, K)
            out[name]["power"].append(power)
            out[name]["fwer"].append(fwer)

        for name in closed_names:
            mask = CLOSED_TESTS[name](sorted_p, alpha)
            power, fwer = _power_and_fwer(mask, sort_indices, gamma, K)
            out[name]["power"].append(power)
            out[name]["fwer"].append(fwer)

    return out
