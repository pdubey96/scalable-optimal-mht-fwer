"""
Tests for the polynomial facts behind the generalised tie-set lemma
(Supplementary Lemma S1.1).

The lemma shows the tie set is Lebesgue-null by exhibiting a monomial of
S_l - S_{l'} whose coefficient cannot vanish.  These tests pin the three facts
the proof rests on:

- a_{pi,i} and b_{gamma,i} are multilinear in the likelihood ratios, so a
  monomial coefficient is exactly recoverable by inclusion-exclusion;
- the coefficient of v_1 ... v_K in a_{pi,i} is pi(K) (K-1)!, for every i;
- the coefficient of v_1 ... v_K in b_{gamma,i} is zero for every gamma <= K-1.

They also pin the boundary case: with a point prior at gamma = 1 the coefficient
vanishes, which is exactly why the lemma carries the hypothesis pi(K) > 0.
"""

import math

import numpy as np
import pytest

from general_k.coefficients import precompute_coefficients
from general_k.prior_objective import (
    beta_binomial_prior,
    point_prior,
    prior_power_coefficients,
    uniform_prior,
)
from verify_tie_set_polynomial import (
    a_pi,
    b_coef,
    is_multilinear,
    top_monomial_coefficient,
)


def priors_for(K):
    return [
        ("point gamma=K", point_prior(K, K)),
        ("uniform", uniform_prior(K)),
        ("Jeffreys", beta_binomial_prior(K, 0.5, 0.5)),
        ("Beta(2,1)", beta_binomial_prior(K, 2.0, 1.0)),
    ]


class TestMultilinearity:
    """Both coefficient families are degree <= 1 in each likelihood ratio."""

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_power_coefficient_is_multilinear(self, K):
        rng = np.random.default_rng(3 + K)
        for _, prior in priors_for(K):
            for i in range(1, K + 1):
                assert is_multilinear(
                    lambda v, i=i, prior=prior: a_pi(v, i, prior, K), K, rng
                )

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_constraint_coefficient_is_multilinear(self, K):
        rng = np.random.default_rng(7 + K)
        for gamma in range(K):
            for i in range(1, K + 1):
                assert is_multilinear(
                    lambda v, gamma=gamma, i=i: b_coef(v, gamma, i, K), K, rng
                )


class TestTopMonomialCoefficient:
    """The coefficient of v_1 ... v_K, which drives the tie-set lemma."""

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_power_coefficient_is_prior_mass_at_K(self, K):
        for _, prior in priors_for(K):
            expected = prior[K] * math.factorial(K - 1)
            assert expected > 0.0
            for i in range(1, K + 1):
                got = top_monomial_coefficient(
                    lambda v, i=i, prior=prior: a_pi(v, i, prior, K), K
                )
                assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_constraint_coefficients_never_reach_it(self, K):
        """b_{gamma,i} has total degree gamma <= K-1, so it cannot contribute."""
        for gamma in range(K):
            for i in range(1, K + 1):
                got = top_monomial_coefficient(
                    lambda v, gamma=gamma, i=i: b_coef(v, gamma, i, K), K
                )
                assert got == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("K", [3, 5])
    def test_boundary_case_a_prior_excluding_gamma_K(self, K):
        """
        With pi(K) = 0 the coefficient vanishes identically.  This is why the
        lemma assumes pi(K) > 0 rather than proving it away.
        """
        prior = point_prior(K, 1)
        for i in range(1, K + 1):
            got = top_monomial_coefficient(
                lambda v, i=i: a_pi(v, i, prior, K), K
            )
            assert got == pytest.approx(0.0, abs=1e-9)


class TestAgreesWithImplementation:
    """The standalone formula used in the proof matches the solver's code."""

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_matches_gamma_power_coefficients(self, K):
        rng = np.random.default_rng(11 + K)
        u = np.sort(rng.uniform(0.1, 1.0, size=(20, K)), axis=1)
        identity_g = lambda arr: arr          # so that v_j = u_j exactly

        for _, prior in priors_for(K):
            from_code = prior_power_coefficients(prior)(u, identity_g, K)
            for n in range(u.shape[0]):
                for i in range(1, K + 1):
                    assert from_code[n, i - 1] == pytest.approx(
                        a_pi(u[n], i, prior, K), rel=1e-10
                    )

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_b_formula_matches_precompute(self, K):
        rng = np.random.default_rng(17 + K)
        u = np.sort(rng.uniform(0.1, 1.0, size=(20, K)), axis=1)
        identity_g = lambda arr: arr

        _, b_code = precompute_coefficients(u, identity_g, K)
        for n in range(u.shape[0]):
            for gamma in range(K):
                for i in range(1, K + 1):
                    assert b_code[gamma, n, i - 1] == pytest.approx(
                        b_coef(u[n], gamma, i, K), rel=1e-10, abs=1e-12
                    )
