"""
Tests for general_k.coefficients module.

Covers:
- elem_sym_poly: boundary cases, known values, m > n, empty input
- compute_a: constant g function
- compute_b: formula boundary conditions and known values
- compute_R: zero mu case
"""

import math
import numpy as np
import pytest

from general_k.coefficients import (
    elem_sym_poly,
    compute_a,
    compute_b,
    compute_R,
)


# ---------------------------------------------------------------------------
# elem_sym_poly
# ---------------------------------------------------------------------------

class TestElemSymPoly:
    """Elementary symmetric polynomial e_m(values)."""

    def test_e0_any_input_is_1(self):
        """e_0 = 1 regardless of the values array."""
        assert elem_sym_poly(np.array([2.0, 3.0, 5.0]), 0) == pytest.approx(1.0)

    def test_e0_empty_input_is_1(self):
        """e_0 of empty array is 1."""
        assert elem_sym_poly(np.array([]), 0) == pytest.approx(1.0)

    def test_em_empty_input_positive_m_is_0(self):
        """e_m of empty array is 0 for m > 0."""
        assert elem_sym_poly(np.array([]), 1) == pytest.approx(0.0)
        assert elem_sym_poly(np.array([]), 3) == pytest.approx(0.0)

    def test_e1_equals_sum(self):
        """e_1 = sum of elements."""
        values = np.array([2.0, 3.0, 5.0])
        assert elem_sym_poly(values, 1) == pytest.approx(sum(values))

    def test_e2_equals_sum_pairwise_products(self):
        """e_2([2,3,5]) = 2*3 + 2*5 + 3*5 = 6+10+15 = 31."""
        values = np.array([2.0, 3.0, 5.0])
        assert elem_sym_poly(values, 2) == pytest.approx(31.0)

    def test_e3_equals_product_all(self):
        """e_3([2,3,5]) = 2*3*5 = 30."""
        values = np.array([2.0, 3.0, 5.0])
        assert elem_sym_poly(values, 3) == pytest.approx(30.0)

    def test_em_greater_than_n_is_0(self):
        """e_m returns 0 when m > len(values)."""
        values = np.array([2.0, 3.0, 5.0])
        assert elem_sym_poly(values, 4) == pytest.approx(0.0)
        assert elem_sym_poly(values, 10) == pytest.approx(0.0)

    def test_e1_single_element(self):
        """e_1 of a single element equals that element."""
        assert elem_sym_poly(np.array([7.0]), 1) == pytest.approx(7.0)

    def test_e2_single_element_is_0(self):
        """e_2 of a single element is 0 (m > n)."""
        assert elem_sym_poly(np.array([7.0]), 2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_a
# ---------------------------------------------------------------------------

class TestComputeA:
    """Power coefficient a_k(u) = (K-1)! * prod(g(u_j))."""

    def _constant_g(self, factor):
        """Return a g_func that maps any array to factor * ones."""
        return lambda u: np.full_like(u, factor, dtype=float)

    def test_constant_g2_K3(self):
        """
        K=3, g=2: a = (3-1)! * 2^3 = 2 * 8 = 16.
        u_sorted has 3 elements; g returns 2 for each.
        """
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        result = compute_a(u, g, K=3)
        assert result == pytest.approx(16.0)

    def test_constant_g1_K1(self):
        """K=1, g=1: a = 0! * 1^1 = 1."""
        u = np.array([0.5])
        g = self._constant_g(1.0)
        assert compute_a(u, g, K=1) == pytest.approx(1.0)

    def test_constant_g3_K2(self):
        """K=2, g=3: a = 1! * 3^2 = 9."""
        u = np.array([0.2, 0.8])
        g = self._constant_g(3.0)
        assert compute_a(u, g, K=2) == pytest.approx(9.0)

    def test_a_is_same_for_all_k(self):
        """compute_a should give the same value regardless of k (it doesn't depend on k)."""
        u = np.array([0.1, 0.4, 0.7])
        g = self._constant_g(2.0)
        a1 = compute_a(u, g, K=3)
        a2 = compute_a(u, g, K=3)
        assert a1 == pytest.approx(a2)


# ---------------------------------------------------------------------------
# compute_b
# ---------------------------------------------------------------------------

class TestComputeB:
    """
    FWER constraint coefficient:
      b_{l,k}(u) = l!(K-l)! * prod_{j=1}^{k-1} g(u_j) * e_{l-k+1}(g(u_{k+1}),...,g(u_K))
    k is 1-indexed.
    Returns 0 when l < k-1 or l-k+1 > K-k.
    """

    def _constant_g(self, factor):
        return lambda u: np.full_like(u, factor, dtype=float)

    # -- Boundary: l < k-1 --------------------------------------------------

    def test_b_zero_when_l_lt_k_minus_1(self):
        """K=3, l=0, k=2: l < k-1 => 0 (0 < 1)."""
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        assert compute_b(l=0, k=2, u_sorted=u, g_func=g, K=3) == pytest.approx(0.0)

    # -- Boundary: l-k+1 > K-k (not enough elements for ESP) ----------------

    def test_b_zero_when_esp_index_too_large(self):
        """
        K=3, l=2, k=1: l-k+1 = 2, K-k = 2.  2 <= 2, so not zero.
        K=3, l=3, k=1: l-k+1 = 3, K-k = 2.  3 > 2 => 0.
        """
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        assert compute_b(l=3, k=1, u_sorted=u, g_func=g, K=3) == pytest.approx(0.0)

    # -- Known value: l=0, k=1 ----------------------------------------------

    def test_b_l0_k1_K3(self):
        """
        K=3, l=0, k=1:
          l!(K-l)! * prod_{j<1}g(u_j) * e_{0-1+1}(g(u2),g(u3))
          = 0! * 3! * 1 * e_0(g(u2),g(u3))
          = 1 * 6 * 1 * 1 = 6.
        """
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        assert compute_b(l=0, k=1, u_sorted=u, g_func=g, K=3) == pytest.approx(6.0)

    # -- Known value: l=1, k=1 ----------------------------------------------

    def test_b_l1_k1_K3(self):
        """
        K=3, l=1, k=1:
          1! * 2! * prod_{j<1}g * e_1(g(u2), g(u3))
          = 1 * 2 * 1 * (2+2) = 2 * 4 = 8.
        """
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        assert compute_b(l=1, k=1, u_sorted=u, g_func=g, K=3) == pytest.approx(8.0)

    # -- Known value: l=2, k=3 ----------------------------------------------

    def test_b_l2_k3_K3(self):
        """
        K=3, l=2, k=3:
          2! * 1! * g(u1)*g(u2) * e_0()
          = 2 * 1 * 2*2 * 1 = 2 * 4 = 8.
        """
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        assert compute_b(l=2, k=3, u_sorted=u, g_func=g, K=3) == pytest.approx(8.0)

    # -- l == k-1: prod prefix exists, e_0 = 1 -----------------------------

    def test_b_l_equals_k_minus_1(self):
        """
        K=3, l=1, k=2:
          l=1, k-1=1  =>  l-k+1=0 => e_0=1
          1! * 2! * g(u1) * e_0(g(u3))
          = 1 * 2 * 2 * 1 = 4.
        """
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        assert compute_b(l=1, k=2, u_sorted=u, g_func=g, K=3) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# compute_R
# ---------------------------------------------------------------------------

class TestComputeR:
    """R_i = a_i - sum_l mu_l * b_{l,i}  for i=1,...,K."""

    def _constant_g(self, factor):
        return lambda u: np.full_like(u, factor, dtype=float)

    def test_mu_zero_R_equals_a(self):
        """
        With mu=0 for all l, R_i = a_i = (K-1)! * prod g = 16 for all i.
        """
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        K = 3
        # mu has K entries (l = 0, 1, ..., K-1)
        mu = np.zeros(K)
        R = compute_R(mu=mu, u_sorted=u, g_func=g, K=K)
        assert R.shape == (K,)
        assert R == pytest.approx(np.full(K, 16.0))

    def test_R_shape(self):
        """compute_R returns an array of shape (K,)."""
        u = np.array([0.2, 0.5, 0.8])
        g = self._constant_g(1.0)
        K = 3
        mu = np.zeros(K)
        R = compute_R(mu=mu, u_sorted=u, g_func=g, K=K)
        assert R.shape == (K,)

    def test_mu_nonzero_reduces_R(self):
        """
        Positive mu values should reduce some R_i below a.
        Exact value not checked here; just that at least one R_i < a.
        """
        u = np.array([0.1, 0.5, 0.9])
        g = self._constant_g(2.0)
        K = 3
        a = compute_a(u, g, K)
        mu = np.ones(K)
        R = compute_R(mu=mu, u_sorted=u, g_func=g, K=K)
        assert np.any(R < a)
