"""
Alternative p-value density functions for Monte Carlo FWER estimation.

Each "alternative model" provides two things via a returned object:
- g(u): the likelihood-ratio density on [0,1]
- sample(n, rng): fast direct sampling of p-values under the alternative

Using direct sampling (via the CDF of the null) avoids the slow rejection
sampler that would need repeated calls to scipy's truncnorm.ppf.
"""

import numpy as np
from scipy.stats import truncnorm, norm, t as tdist


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
        # For constant g, rejection sampling is easy: accept rate = c/c = 1
        # Just return uniform samples (this is only used for testing)
        return rng.uniform(0, 1, size=n_samples)


class TruncatedNormalG(AlternativeModel):
    """Alternative density for one-sided truncated normal.

    Under H0: X ~ N(0,1) truncated to [-M, M]
    Under H1: X ~ N(theta, 1) truncated to [-M, M]

    g(u) = alt_pdf(null_ppf(1-u)) / null_pdf(null_ppf(1-u))

    Direct sampling: draw X ~ alt_dist, compute u = 1 - null_cdf(X).
    """

    def __init__(self, theta, M=6.0):
        self.theta = theta
        self.M = M
        self.null_dist = truncnorm(-M, M, loc=0, scale=1)
        self.alt_dist = truncnorm(-M - theta, M - theta, loc=theta, scale=1)
        # Pre-compute g on a grid for the LUT used in __call__
        self._grid_n = 10000
        self._grid_u = np.linspace(1e-8, 1.0 - 1e-8, self._grid_n)
        self._grid_g = self._compute_g_raw(self._grid_u)

    def _compute_g_raw(self, u):
        # p-value via CDF: u = F_0(x), so x = F_0^{-1}(u) = null_ppf(u)
        x = self.null_dist.ppf(u)
        null_pdf = self.null_dist.pdf(x)
        alt_pdf = self.alt_dist.pdf(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(null_pdf > 0, alt_pdf / null_pdf, 0.0)

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        return np.interp(u, self._grid_u, self._grid_g)

    def sample(self, n_samples, rng):
        """Draw p-values by sampling X ~ alt_dist, then u = null_cdf(X)."""
        x = self.alt_dist.rvs(size=n_samples, random_state=rng)
        u = self.null_dist.cdf(x)
        return np.clip(u, 0.0, 1.0)


class MixtureNormalG(AlternativeModel):
    """Alternative density for two-sided mixture normal.

    Under H0: X ~ N(0,1)
    Under H1: X ~ 0.5*N(theta,1) + 0.5*N(-theta,1)
    p-values: u = 2*Phi(-|X|)
    g(u) = exp(-theta^2/2) * cosh(theta * x_p), x_p = Phi^{-1}(1 - u/2)
    """

    def __init__(self, theta):
        self.theta = theta
        self._grid_n = 10000
        self._grid_u = np.linspace(1e-8, 1.0 - 1e-8, self._grid_n)
        self._grid_g = self._compute_g(self._grid_u)

    def _compute_g(self, u):
        x_p = norm.ppf(1.0 - u / 2.0)
        return np.exp(-0.5 * self.theta**2) * np.cosh(self.theta * x_p)

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        return np.interp(u, self._grid_u, self._grid_g)

    def sample(self, n_samples, rng):
        """Draw from mixture, convert to two-sided p-values."""
        # Choose component: half from N(theta,1), half from N(-theta,1)
        signs = rng.choice([-1, 1], size=n_samples)
        x = rng.normal(loc=signs * self.theta, scale=1.0)
        u = 2.0 * norm.cdf(-np.abs(x))
        return np.clip(u, 0.0, 1.0)


class StudentTG(AlternativeModel):
    """Alternative density for heavy-tailed t-distribution.

    Under H0: X ~ N(0,1)
    Under H1: X ~ t_df
    p-values: u = 2*Phi(-|X|)
    g(u) = dt(x_p, df) / phi(x_p), x_p = Phi^{-1}(1 - u/2)
    """

    def __init__(self, df):
        self.df = df
        self._grid_n = 10000
        self._grid_u = np.linspace(1e-8, 1.0 - 1e-8, self._grid_n)
        self._grid_g = self._compute_g(self._grid_u)

    def _compute_g(self, u):
        x_p = norm.ppf(1.0 - u / 2.0)
        t_pdf = tdist.pdf(x_p, self.df)
        n_pdf = norm.pdf(x_p)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_pdf > 0, t_pdf / n_pdf, 0.0)

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        return np.interp(u, self._grid_u, self._grid_g)

    def sample(self, n_samples, rng):
        """Draw from t_df, convert to two-sided p-values under normal null."""
        x = tdist.rvs(self.df, size=n_samples, random_state=rng)
        u = 2.0 * norm.cdf(-np.abs(x))
        return np.clip(u, 0.0, 1.0)


class BetaG(AlternativeModel):
    """Alternative density for Beta model on p-values.

    Under H0: u ~ Beta(1,1) = U(0,1)
    Under H1: u ~ Beta(theta, 1)
    g(u) = theta * u^(theta - 1)
    """

    def __init__(self, theta):
        self.theta = theta

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(u > 0, self.theta * u ** (self.theta - 1), 0.0)

    def sample(self, n_samples, rng):
        """Draw from Beta(theta, 1)."""
        return rng.beta(self.theta, 1.0, size=n_samples)


# Convenience factory functions (match the API expected by existing code)
def constant_g(c):
    return ConstantG(c)

def truncated_normal_g(theta, M=6.0):
    return TruncatedNormalG(theta, M)

def mixture_normal_g(theta):
    return MixtureNormalG(theta)

def student_t_g(df):
    return StudentTG(df)

def beta_g(theta):
    return BetaG(theta)


def sample_from_g(g_func, n_samples, rng):
    """Draw p-values from the alternative distribution.

    If g_func has a .sample method (AlternativeModel), use direct sampling.
    Otherwise fall back to rejection sampling.
    """
    if hasattr(g_func, 'sample'):
        return g_func.sample(n_samples, rng)

    # Fallback: rejection sampling
    grid = np.linspace(0.0, 1.0, 1000)
    g_max = float(np.max(g_func(grid))) * 1.1

    samples = []
    while len(samples) < n_samples:
        batch = max(n_samples - len(samples), 1000)
        candidates = rng.uniform(0.0, 1.0, size=batch * 3)
        g_vals = g_func(candidates)
        u_accept = rng.uniform(0.0, g_max, size=len(candidates))
        accepted = candidates[u_accept <= g_vals]
        samples.extend(accepted.tolist())

    return np.array(samples[:n_samples])
