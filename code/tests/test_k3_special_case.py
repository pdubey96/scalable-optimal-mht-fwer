"""
The general-K coefficients reduce to the submitted K=3 paper's explicit ones.

Section 3.6 of the manuscript presents K = 3 as the worked special case and
claims that the elementary-symmetric-polynomial formulae of Lemma 3.4 collapse
onto the coefficients the K = 3 paper derived by hand.  That is a checkable
claim, so it is checked here rather than asserted.

Reading the dual decomposition of Dubey and Huo (2025) -- the term in mu_0, the
term in mu_1, and the term in mu_2 -- against
``L = sum_gamma mu_gamma alpha + int sum_i D_i R_i``  with
``R_i = a_i - sum_gamma mu_gamma b_{gamma,i}``  gives, at the point prior on
gamma = K:

    a_i      = 2 v1 v2 v3            for every i
    b_{0,1}  = 6,        b_{0,2} = b_{0,3} = 0
    b_{1,1}  = 2(v2+v3), b_{1,2} = 2 v1,      b_{1,3} = 0
    b_{2,1}  = 2 v2 v3,  b_{2,2} = 2 v1 v3,   b_{2,3} = 2 v1 v2

where ``v_j = g(u_j)`` at the jth smallest p-value.
"""

import numpy as np
import pytest

from general_k.coefficients import precompute_coefficients


def _explicit_k3(v1, v2, v3):
    """The nine coefficients as the submitted K=3 paper writes them."""
    zero = np.zeros_like(v1)
    a = 2.0 * v1 * v2 * v3
    b = {
        (0, 1): 6.0 * np.ones_like(v1), (0, 2): zero,          (0, 3): zero,
        (1, 1): 2.0 * (v2 + v3),        (1, 2): 2.0 * v1,      (1, 3): zero,
        (2, 1): 2.0 * v2 * v3,          (2, 2): 2.0 * v1 * v3, (2, 3): 2.0 * v1 * v2,
    }
    return a, b


class TestK3ReducesToTheSubmittedCoefficients:
    """Lemma 3.4 at K = 3 is the K = 3 paper's hand derivation."""

    @pytest.mark.parametrize(
        "g_func,label",
        [
            (lambda x: 1.7 - 1.2 * x, "linear"),
            (lambda x: 2.0 * (1.0 - x) + 0.3, "steeper linear"),
            (lambda x: np.exp(-1.5 * x) + 0.2, "exponential"),
            (lambda x: np.full_like(x, 1.0), "uniform, the degenerate case"),
        ],
    )
    def test_all_nine_coefficients_agree_exactly(self, g_func, label):
        rng = np.random.default_rng(5)
        u = np.sort(rng.uniform(0.0, 1.0, size=(2000, 3)), axis=1)
        a, b = precompute_coefficients(u, g_func, 3)
        v1, v2, v3 = g_func(u[:, 0]), g_func(u[:, 1]), g_func(u[:, 2])

        want_a, want_b = _explicit_k3(v1, v2, v3)
        # a_k is the same for every position at the point prior on gamma = K.
        assert np.array_equal(a, np.tile(want_a[:, None], (1, 3))), label
        for (gamma, k), want in want_b.items():
            assert np.array_equal(b[gamma, :, k - 1], want), (label, gamma, k)

    def test_the_zero_coefficients_are_the_support_condition(self):
        """
        b_{gamma,k} vanishes when gamma < k-1: rejecting position k cannot add
        family-wise error under a configuration with too few true nulls above
        it.  At K = 3 that kills exactly b_{0,2}, b_{0,3} and b_{1,3}.
        """
        rng = np.random.default_rng(11)
        u = np.sort(rng.uniform(0.0, 1.0, size=(500, 3)), axis=1)
        _, b = precompute_coefficients(u, lambda x: 1.7 - 1.2 * x, 3)

        vanishing = {(0, 2), (0, 3), (1, 3)}
        for gamma in range(3):
            for k in (1, 2, 3):
                col = b[gamma, :, k - 1]
                if (gamma, k) in vanishing:
                    assert np.all(col == 0.0), (gamma, k)
                else:
                    assert np.all(col > 0.0), (gamma, k)
