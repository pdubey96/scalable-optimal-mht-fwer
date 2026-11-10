"""
Alternative p-value density functions for Monte Carlo FWER estimation.

Each "alternative model" provides two things via a returned object:
- g(u): the likelihood-ratio density on [0,1]
- sample(n, rng): fast direct sampling of p-values under the alternative

Densities are evaluated EXACTLY from closed forms (no lookup-table
interpolation), so the density used by the optimizer is the density of the
sampled p-values.  Evaluations go through log-space where needed for
numerical stability.  The only exception is the heavy-tailed Student-t
model, whose likelihood ratio diverges so fast as u -> 0 that exact linear-
space evaluation overflows double precision; there the log-likelihood ratio
is truncated at LOG_CAP (documented on the class, and disclosed in the
paper, which treats the t model as an out-of-model stress test).

p-values are clipped to [P_FLOOR, 1]: double-precision underflow can round
extreme two-sided p-values to exactly 0, where the likelihood ratio is
undefined.  P_FLOOR = 1e-300 never binds for the normal-tailed models at the
parameter values used in the paper.
"""

import numpy as np
from scipy.stats import truncnorm, norm, t as tdist

P_FLOOR = 1e-300


class AlternativeModel:
    """Base class wrapping g(u) and a fast sampler."""

    def __call__(self, u):
        """Evaluate g(u). u can be scalar or array."""
        raise NotImplementedError

    def sample(self, n_samples, rng):
        """Draw n_samples p-values under the alternative."""
        raise NotImplementedError


class ConstantG(AlternativeModel):
    """g(u) = c (constant). For testing only."""

    def __init__(self, c):
        self.c = c

    def __call__(self, u):
        return np.full_like(np.asarray(u, dtype=float), self.c)

    def sample(self, n_samples, rng):
        # ConstantG is used only in tests; draw its samples directly.
        return rng.uniform(0, 1, size=n_samples)


class TruncatedNormalG(AlternativeModel):
    """Alternative density for one-sided truncated normal.

    Under H0: X ~ N(0,1) truncated to [-M, M]
    Under H1: X ~ N(theta, 1) truncated to [-M, M]
    p-values: u = F_0(X) (null CDF), so x = F_0^{-1}(u)

    g(u) = alt_pdf(x) / null_pdf(x) = exp{theta*x - theta^2/2} * Z_0/Z_theta,
    evaluated exactly via the log-pdf difference.  The truncation to [-M, M]
    (M = 6 throughout the paper) is what makes g bounded, satisfying the
    paper's Assumption 3.5.

    Direct sampling: draw X ~ alt_dist, compute u = null_cdf(X).
    """

    def __init__(self, theta, M=6.0):
        self.theta = theta
        self.M = M
        self.null_dist = truncnorm(-M, M, loc=0, scale=1)
        self.alt_dist = truncnorm(-M - theta, M - theta, loc=theta, scale=1)

    def __call__(self, u):
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        x = self.null_dist.ppf(u)
        log_g = self.alt_dist.logpdf(x) - self.null_dist.logpdf(x)
        return np.exp(log_g)

    def sample(self, n_samples, rng):
        """Draw p-values by sampling X ~ alt_dist, then u = null_cdf(X)."""
        x = self.alt_dist.rvs(size=n_samples, random_state=rng)
        u = self.null_dist.cdf(x)
        return np.clip(u, 0.0, 1.0)


class NormalG(AlternativeModel):
    """Alternative density for the classical one-sided normal, untruncated.

    Under H0: X ~ N(0,1)
    Under H1: X ~ N(theta,1)
    p-values: u = Phi(X), so x = Phi^{-1}(u) and

        g(u) = exp{theta * Phi^{-1}(u) - theta^2 / 2},

    decreasing in u for theta < 0, which is the sign convention used
    throughout (smaller p-value, stronger evidence).

    This is the M -> infinity limit of TruncatedNormalG and is deliberately
    outside the paper's regularity conditions: g is unbounded as u -> 0 and
    vanishes as u -> 1, so Assumptions 3.4 and 3.5 both fail.  The monotone
    likelihood ratio survives, so the population optimality interpretation
    does too.

    It exists to show what the classical, non-truncated statistic does, the
    standard procedures being expected to do well there.
    The two models are not a hair apart.  Truncating at M removes the
    alternative's mass outside [-M, M] and renormalises its distribution.  At
    the paper's M = 6 the largest relative gap between the two densities on the
    diagnostic grid u in [1e-8, 1 - 1e-8] is 1.6% at theta = -1, 3.3% at -2,
    5.2% at -3 and 9.2% at -4, the last because 2.3% of that alternative's mass
    lies below -6.  At M = 12 the gap
    is at machine precision, which is the check that this class really is the
    limit (see tests/test_untruncated_normal.py).
    """

    def __init__(self, theta):
        self.theta = theta

    def __call__(self, u):
        u = np.clip(np.asarray(u, dtype=float), P_FLOOR, 1.0)
        x = norm.ppf(u)
        return np.exp(self.theta * x - self.theta ** 2 / 2.0)

    def sample(self, n_samples, rng):
        """Draw X ~ N(theta, 1), then u = Phi(X)."""
        x = rng.normal(loc=self.theta, scale=1.0, size=n_samples)
        return np.clip(norm.cdf(x), P_FLOOR, 1.0)


class MixtureNormalG(AlternativeModel):
    """Alternative density for two-sided mixture normal.

    Under H0: X ~ N(0,1)
    Under H1: X ~ 0.5*N(theta,1) + 0.5*N(-theta,1)
    p-values: u = 2*Phi(-|X|)
    g(u) = exp(-theta^2/2) * cosh(theta * x_p), x_p = Phi^{-1}(1 - u/2),
    computed exactly as 0.5*[exp(theta*x_p - theta^2/2) +
    exp(-theta*x_p - theta^2/2)] with x_p = isf(u/2) for accuracy at small u.
    """

    def __init__(self, theta):
        self.theta = theta

    def __call__(self, u):
        u = np.clip(np.asarray(u, dtype=float), P_FLOOR, 1.0)
        x_p = norm.isf(u / 2.0)
        t = self.theta
        return 0.5 * (np.exp(t * x_p - 0.5 * t * t)
                      + np.exp(-t * x_p - 0.5 * t * t))

    def sample(self, n_samples, rng):
        """Draw from mixture, convert to two-sided p-values."""
        # Choose component: half from N(theta,1), half from N(-theta,1)
        signs = rng.choice([-1, 1], size=n_samples)
        x = rng.normal(loc=signs * self.theta, scale=1.0)
        u = 2.0 * norm.sf(np.abs(x))
        return np.clip(u, P_FLOOR, 1.0)


class StudentTG(AlternativeModel):
    """Alternative density for heavy-tailed t-distribution.

    Under H0: X ~ N(0,1)
    Under H1: X ~ t_df
    p-values: u = 2*Phi(-|X|)
    g(u) = f_t(x_p; df) / phi(x_p), x_p = Phi^{-1}(1 - u/2).

    NOTE (out-of-model stress test): this likelihood ratio is U-shaped in u
    (decreasing only for u < 2*(1 - Phi(1)) ~ 0.32), so the model violates
    the paper's non-increasing-g (MLR) assumption; the paper discusses it as
    an out-of-model robustness check.  Numerically, g(u) grows like
    exp(x_p^2/2) as u -> 0, overflowing double precision for the extreme
    p-values that heavy-tailed sampling produces; the log-likelihood ratio is
    therefore truncated at LOG_CAP.  The truncation binds for a small
    fraction of alternative samples at low df (measured and disclosed in the
    paper) and keeps all K-fold coefficient products finite.
    """

    LOG_CAP = 45.0  # e^45 ~ 3.5e19; K * LOG_CAP stays far below overflow

    def __init__(self, df):
        self.df = df

    def __call__(self, u):
        u = np.clip(np.asarray(u, dtype=float), P_FLOOR, 1.0)
        x_p = norm.isf(u / 2.0)
        log_g = tdist.logpdf(x_p, self.df) - norm.logpdf(x_p)
        return np.exp(np.minimum(log_g, self.LOG_CAP))

    def sample(self, n_samples, rng):
        """Draw from t_df, convert to two-sided p-values under normal null."""
        x = tdist.rvs(self.df, size=n_samples, random_state=rng)
        u = 2.0 * norm.sf(np.abs(x))
        return np.clip(u, P_FLOOR, 1.0)

    def cap_fraction(self, n_samples=1_000_000, seed=0):
        """Fraction of alternative draws at which the LOG_CAP truncation
        binds (used for the disclosure in the paper)."""
        rng = np.random.default_rng(seed)
        u = self.sample(n_samples, rng)
        x_p = norm.isf(u / 2.0)
        log_g = tdist.logpdf(x_p, self.df) - norm.logpdf(x_p)
        return float(np.mean(log_g > self.LOG_CAP))


class BetaG(AlternativeModel):
    """Alternative density for Beta model on p-values.

    Under H0: u ~ Beta(1,1) = U(0,1)
    Under H1: u ~ Beta(theta, 1)
    g(u) = theta * u^(theta - 1), evaluated exactly.
    """

    def __init__(self, theta):
        self.theta = theta

    def __call__(self, u):
        u = np.clip(np.asarray(u, dtype=float), P_FLOOR, 1.0)
        return self.theta * u ** (self.theta - 1.0)

    def sample(self, n_samples, rng):
        """Draw from Beta(theta, 1)."""
        return np.clip(rng.beta(self.theta, 1.0, size=n_samples), P_FLOOR, 1.0)


# Convenience factory functions (match the API expected by existing code)
def constant_g(c):
    """Return the constant-density test model ``g(u) = c``."""
    return ConstantG(c)

def truncated_normal_g(theta, M=6.0):
    """Return the one-sided truncated-normal alternative model."""
    return TruncatedNormalG(theta, M)

def normal_g(theta):
    """Return the classical untruncated one-sided normal alternative model."""
    return NormalG(theta)

def mixture_normal_g(theta):
    """Return the two-sided symmetric mixture-normal alternative model."""
    return MixtureNormalG(theta)

def student_t_g(df):
    """Return the heavy-tailed Student-t stress-test model."""
    return StudentTG(df)

def beta_g(theta):
    """Return the Beta(theta, 1) alternative model on p-values."""
    return BetaG(theta)


def sample_from_g(g_func, n_samples, rng):
    """Draw p-values from the alternative distribution.

    ``g_func`` must expose a callable ``sample(n_samples, rng)`` method.  The
    built-in ``AlternativeModel`` implementations all provide one.  A density
    callable alone is insufficient for reliable generic sampling because it
    does not supply a certified rejection envelope.

    Raises
    ------
    TypeError
        If ``g_func`` does not provide a callable ``sample`` method.
    """
    sample = getattr(g_func, "sample", None)
    if not callable(sample):
        raise TypeError(
            "Monte Carlo sampling requires g_func to provide a callable "
            ".sample(n_samples, rng) method"
        )
    return sample(n_samples, rng)
