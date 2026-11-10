"""
Average power under a prior on the number of non-null hypotheses.

The manuscript maximises ``Pi_K``, the average power at the all-alternatives
configuration, whose power coefficient is constant in the position ``k``:

    a(u) = (K-1)! * prod_j g(u_j).

At a general configuration ``h_gamma`` the same derivation gives a coefficient
that *does* depend on the position.  Writing the average power as an integral
over the ordered simplex,

    Pi_gamma(D) = int_{u_1 <= ... <= u_K}  sum_{i=1}^{l(u)} a_{gamma,i}(u)  du,

    a_{gamma,i}(u) = (gamma-1)! (K-gamma)! * g(u_i) * e_{gamma-1}( g(u_{-i}) ),

where ``e_m`` is the m-th elementary symmetric polynomial and ``g(u_{-i})``
drops coordinate ``i``.  The reading is the same as for ``b_{l,i}``: position
``i`` is one of the ``gamma`` alternatives, and the remaining ``gamma-1``
alternatives are distributed over the other ``K-1`` positions in every possible
way.  At ``gamma = K`` this collapses to the manuscript's constant ``a(u)``,
since ``e_{K-1}(g(u_{-i})) = prod_{j != i} g(u_j)``.

A prior ``pi`` on ``gamma in {1, ..., K}`` therefore gives

    Pi_pi = sum_gamma pi(gamma) Pi_gamma,
    a_{pi,i}(u) = sum_gamma pi(gamma) a_{gamma,i}(u),

which is the same elementary-symmetric-polynomial form the solver already
handles.  The family-wise error constraints are untouched: they do not involve
the objective.

Provides
--------
- gamma_power_coefficients : a_{gamma,i} for every gamma, as a batch tensor
- prior_power_coefficients : the a_{pi,i} callable consumed by the solver
- uniform_prior / point_prior / binomial_prior : priors on gamma
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Priors on the number of non-nulls
# ---------------------------------------------------------------------------

def point_prior(K, gamma):
    """All mass at a single configuration.  ``point_prior(K, K)`` is the
    manuscript's objective."""
    pi = np.zeros(K + 1)
    pi[gamma] = 1.0
    return pi


def uniform_prior(K):
    """Equal mass on gamma = 1, ..., K."""
    pi = np.zeros(K + 1)
    pi[1:] = 1.0 / K
    return pi


def binomial_prior(K, p):
    """
    Binomial(K, p) on the number of non-nulls, conditioned on gamma >= 1.

    ``p`` is the prior probability that any one hypothesis is false, so small
    ``p`` describes a sparse family and ``p`` near one a signal-rich family.
    """
    pi = np.zeros(K + 1)
    for gamma in range(1, K + 1):
        pi[gamma] = math.comb(K, gamma) * p**gamma * (1 - p)**(K - gamma)
    pi /= pi.sum()
    return pi


def beta_binomial_prior(K, a, b):
    """
    Beta-binomial on the number of non-nulls, conditioned on gamma >= 1.

    Puts a Beta(a, b) prior on the per-hypothesis non-null probability and
    mixes the binomial over it, which is the hierarchical model behind a prior
    on gamma rather than an arbitrary weighting.  Notable members:

    - ``a = b = 1``  : the flat prior on p, which induces a *uniform* prior on
      gamma over {0, ..., K} -- so ``uniform_prior`` is not an arbitrary
      default but the Bayes prior under a flat prior on the non-null rate;
    - ``a = b = 0.5``: Jeffreys' prior for the binomial rate, U-shaped, placing
      more mass on the sparse and dense extremes than the flat prior;
    - ``a > b``      : tilted toward signal-rich families, and ``a < b`` toward
      sparse ones.

    Uses the Beta-binomial pmf via log-gamma functions so large K is safe.
    """
    gammas = np.arange(K + 1)
    log_pmf = (
        np.array([math.lgamma(K + 1) - math.lgamma(g + 1) - math.lgamma(K - g + 1)
                  for g in gammas])
        + np.array([math.lgamma(g + a) + math.lgamma(K - g + b) - math.lgamma(K + a + b)
                    for g in gammas])
        + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    )
    pi = np.exp(log_pmf - log_pmf.max())
    pi[0] = 0.0                                   # no power at the global null
    pi /= pi.sum()
    return pi


# ---------------------------------------------------------------------------
# Power coefficients
# ---------------------------------------------------------------------------

def _esp_all_orders_leave_one_out(g_vals, K):
    """
    ``esp[i][:, m] = e_m( {g(u_j)}_{j != i} )`` for m = 0, ..., K-1.

    One O(K^2) dynamic program per omitted coordinate, each pass vectorised
    over samples; O(K^3 n) overall, the same order as the b_{l,k} precompute.
    Deflating a single full DP would be faster but loses accuracy when the
    density is large, so the direct recomputation is used.
    """
    n = g_vals.shape[0]
    out = []
    for i in range(K):
        others = np.delete(g_vals, i, axis=1)          # (n, K-1)
        dp = np.zeros((n, K))
        dp[:, 0] = 1.0
        for j in range(K - 1):
            v = others[:, j]
            for m in range(min(j + 1, K - 1), 0, -1):
                dp[:, m] += v * dp[:, m - 1]
        out.append(dp)
    return out


def gamma_power_coefficients(u_sorted_batch, g_func, K):
    """
    Power coefficients at every configuration.

    Returns
    -------
    np.ndarray, shape (K + 1, n_samples, K)
        ``A[gamma, :, i]`` is ``a_{gamma,i+1}`` (i is 0-indexed).  Row 0 is
        zero: the global null has no alternatives and hence no power.
    """
    g_vals = g_func(u_sorted_batch)                    # (n, K)
    n = g_vals.shape[0]
    esp = _esp_all_orders_leave_one_out(g_vals, K)

    A = np.zeros((K + 1, n, K))
    for gamma in range(1, K + 1):
        coeff = float(math.factorial(gamma - 1) * math.factorial(K - gamma))
        for i in range(K):
            A[gamma, :, i] = coeff * g_vals[:, i] * esp[i][:, gamma - 1]
    return A


def prior_power_coefficients(prior):
    """
    Build the ``power_coeff_fn`` the solver consumes from a prior on gamma.

    Parameters
    ----------
    prior : array of length K + 1, indexed by gamma; ``prior[0]`` is ignored.

    Returns
    -------
    callable (u_sorted_batch, g_func, K) -> a_batch of shape (n_samples, K)
    """
    prior = np.asarray(prior, dtype=float)

    def power_coeff_fn(u_sorted_batch, g_func, K):
        if prior.shape[0] != K + 1:
            raise ValueError(
                f"prior has length {prior.shape[0]}, expected K + 1 = {K + 1}"
            )
        A = gamma_power_coefficients(u_sorted_batch, g_func, K)
        return np.einsum("g,gnk->nk", prior, A)

    return power_coeff_fn


# ---------------------------------------------------------------------------
# Objective value, from the coefficients themselves
# ---------------------------------------------------------------------------

def objective_from_coefficients(mu, g_func, K, prior, n_samples, seed):
    """
    Monte Carlo estimate of ``Pi_pi`` from the coefficient representation.

    The representation is an integral over the ordered simplex, so drawing
    ``u`` uniformly on the cube and sorting gives density ``K!`` there:

        Pi_pi = E_{u ~ U[0,1]^K} [ sum_{i <= l(u_sorted)} a_{pi,i}(u_sorted) ] / K!

    Comparing this against direct simulation checks the coefficients
    independently of the solver.
    """
    from general_k.coefficients import precompute_coefficients
    from general_k.decision import l_star_from_precomputed

    rng = np.random.default_rng(seed)
    u_sorted = np.sort(rng.uniform(0.0, 1.0, size=(n_samples, K)), axis=1)

    _, b_coeff = precompute_coefficients(u_sorted, g_func, K)
    a_batch = prior_power_coefficients(prior)(u_sorted, g_func, K)

    l_star = l_star_from_precomputed(mu, a_batch, b_coeff)
    prefix = np.cumsum(a_batch, axis=1)
    totals = np.where(l_star > 0, prefix[np.arange(n_samples),
                                         np.clip(l_star - 1, 0, K - 1)], 0.0)
    values = totals / math.factorial(K)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n_samples))
