"""
Flood Adaptation ABM - Flood Generation Module (v10)

Generates flood levels using GEV distribution fitted to user-defined
return periods and flood levels.

References:
-----------
1. GEV Distribution: Coles (2001), An Introduction to Statistical Modeling
   of Extreme Values, Springer, Chapter 3.
2. Block Maxima Method: Gumbel (1958), Statistics of Extremes.
3. Flood Frequency Analysis: Hosking & Wallis (1997), Regional Frequency
   Analysis, Cambridge University Press.
"""

import numpy as np
from scipy.stats import genextreme
from scipy.optimize import minimize


# ============================================================================
# GEV FUNCTIONS
# ============================================================================

def return_period_to_probability(T):
    """Convert return period to non-exceedance probability (Stedinger et al., 1993)."""
    return 1.0 - 1.0 / np.asarray(T)


def gev_quantile(p, loc, scale, shape):
    """
    GEV quantile function (Coles, 2001, Eq. 3.4).

    Q(p) = mu + (sigma/xi) * [(-ln(p))^(-xi) - 1]  for xi != 0
    Q(p) = mu - sigma * ln(-ln(p))                   for xi = 0
    """
    p = np.asarray(p)
    y = -np.log(p)
    if np.abs(shape) < 1e-8:
        return loc - scale * np.log(y)
    return loc + (scale / shape) * (y**(-shape) - 1)


def fit_gev_to_return_periods(return_periods, flood_levels):
    """
    Fit GEV to return period-flood level pairs (Hosking & Wallis, 1997).

    Uses constrained optimization to ensure non-negative flood values.
    """
    rp = np.asarray(return_periods)
    fl = np.asarray(flood_levels)
    probs = return_period_to_probability(rp)

    loc_init = np.mean(fl)
    scale_init = np.std(fl) * np.sqrt(6) / np.pi
    shape_init = 0.1

    def objective(params):
        loc, scale, shape = params
        if scale <= 0:
            return 1e10
        predicted = gev_quantile(probs, loc, scale, shape)
        sse = np.sum((predicted - fl)**2)
        p_low = gev_quantile(0.01, loc, scale, shape)
        if p_low < 0:
            sse += 1000 * (p_low ** 2)
        return sse

    result = minimize(objective, x0=[loc_init, scale_init, shape_init],
                      method='Nelder-Mead', options={'maxiter': 10000})
    loc, scale, shape = result.x
    return loc, max(scale, 1e-6), shape


# ============================================================================
# FLOOD GENERATOR CLASS
# ============================================================================

class FloodGenerator:
    """
    Stateless GEV flood sampler (Coles, 2001).

    Fits GEV to return periods and provides pure sampling.
    """

    def __init__(self, return_periods, flood_levels, rng=None):
        self.return_periods = np.asarray(return_periods)
        self.flood_levels_input = np.asarray(flood_levels)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.loc, self.scale, self.shape = fit_gev_to_return_periods(
            return_periods, flood_levels
        )

    def sample(self):
        """Sample a single flood level clipped to [0, 1] (normalized)."""
        value = genextreme.rvs(c=-self.shape, loc=self.loc, scale=self.scale,
                               random_state=self.rng)
        return float(np.clip(value, 0.0, 1.0))

    def sample_series(self, n):
        """Sample n flood levels clipped to [0, 1] (normalized)."""
        values = genextreme.rvs(c=-self.shape, loc=self.loc, scale=self.scale,
                                size=n, random_state=self.rng)
        return np.clip(values, 0.0, 1.0)

    def get_parameters(self):
        """Return fitted GEV parameters."""
        return {'loc': self.loc, 'scale': self.scale, 'shape': self.shape}


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("FLOOD MODULE TEST - GEV (Coles, 2001)")
    print("=" * 50)

    rng = np.random.default_rng(42)
    gen = FloodGenerator(
        return_periods=[10, 20, 50, 100],
        flood_levels=[0.10, 0.25, 0.55, 0.95],
        rng=rng)

    params = gen.get_parameters()
    print(f"\nGEV Parameters:")
    print(f"  Location (mu): {params['loc']:.4f}")
    print(f"  Scale (sigma):  {params['scale']:.4f}")
    print(f"  Shape (xi):     {params['shape']:.4f}")

    samples = gen.sample_series(1000)
    print(f"\nSample Statistics (n=1000):")
    print(f"  Min:  {samples.min():.4f}")
    print(f"  Mean: {samples.mean():.4f}")
    print(f"  Max:  {samples.max():.4f}")
