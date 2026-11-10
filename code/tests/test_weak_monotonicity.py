"""The optimality-conditions section reports an empirical weak-monotonicity
audit: no violations among the checked truncated-normal and beta policies, and
violations under the nonmonotone Student-t stress test. Representative cases
from both findings are asserted here.

Weak monotonicity, in the sense Rosset et al. (2022) settle on: if the p-values
decrease coordinatewise, the number of rejections cannot decrease.

The sample sizes are small enough to keep the suite fast. The manuscript's
6e6 non-Student-t comparisons come from check_weak_monotonicity.py, which runs
three 100,000-pair schemes for each of the twenty corresponding stored policies
and separately audits the four Student-t policies.
"""

import numpy as np
import pytest

from general_k.algorithm import coordinate_descent_fast
from general_k.alternatives import beta_g, student_t_g, truncated_normal_g
from general_k.decision import l_star_batch


def _pairs(rng, K, n):
    """(u, u') with u <= u' elementwise, sorted, one coordinate decreased.

    The one-coordinate scheme is the sharpest of the three the script uses and
    the only one that produced violations at this sample size.
    """
    up = rng.random((n, K))
    u = up.copy()
    j = rng.integers(0, K, size=n)
    rows = np.arange(n)
    u[rows, j] = up[rows, j] * rng.random(n)
    return np.sort(u, axis=1), np.sort(up, axis=1)


def _violations(mu, g_func, K, n=20000, seed=7):
    rng = np.random.default_rng(seed)
    u, up = _pairs(rng, K, n)
    return int((l_star_batch(np.asarray(mu), u, g_func, K)
                < l_star_batch(np.asarray(mu), up, g_func, K)).sum())


@pytest.mark.parametrize("factory, theta, K", [
    (truncated_normal_g, -2.0, 3),
    (truncated_normal_g, -2.0, 6),
    (beta_g, 0.4, 6),
])
def test_no_violation_in_representative_monotone_policies(factory, theta, K):
    """No coordinatewise decrease in the p-values reduces the rejection count."""
    g = factory(theta)
    mu, _, _ = coordinate_descent_fast(
        g_func=g, K=K, alpha=0.05, n_samples=20000, tol=1e-6,
        max_iter=400, seed=42, return_diagnostics=True)
    assert _violations(mu, g, K) == 0


def test_student_t_working_ratio_is_not_injective():
    """The stated mechanism: g is not monotone, so Assumption 3 fails.

    Asserted directly rather than inferred from the violations, so that a
    change in the alternative shows up here and not only downstream.
    """
    g = student_t_g(6)
    u = np.linspace(1e-6, 1.0, 200000)
    v = g(u)
    assert (np.diff(v) > 0).any(), "expected a rising stretch in the t ratio"
    trough = u[int(np.argmin(v))]
    assert 0.25 < trough < 0.40, trough
    assert g(np.array([0.99]))[0] > v.min()


def test_weak_monotonicity_fails_for_student_t():
    """The manuscript reports the failure; if it ever stops failing, the
    sentence reporting it has to change, so the suite should say so."""
    g = student_t_g(6)
    K = 6
    mu, _, _ = coordinate_descent_fast(
        g_func=g, K=K, alpha=0.05, n_samples=20000, tol=1e-6,
        max_iter=400, seed=42, return_diagnostics=True)
    assert _violations(mu, g, K) > 0
