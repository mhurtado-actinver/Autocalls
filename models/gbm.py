import numpy as np
from tqdm import tqdm

def monte_carlo_paths_gbm(S0, q, r, expected_return, sigma, T, steps, n_paths,
                          measure='Risk-Neutral', seed=None):
    """
    Simulate GBM paths without regime switching (constant volatility).

    Parameters:
    - S0: initial price
    - q: dividend yield
    - r: risk-free rate
    - expected_return: expected return under the real-world measure
    - sigma: volatility (float)
    - T: total time in years
    - steps: number of time steps
    - n_paths: number of Monte Carlo paths
    - measure: 'Risk-Neutral' or 'Real', determines drift
    - seed: int, for reproducibility

    Returns:
    - paths: ndarray of shape (steps + 1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    if measure == 'Risk-Neutral':
        mu = r - q
    else:
        mu = expected_return

    paths = np.zeros((steps + 1, n_paths))
    paths[0, :] = S0

    for t in tqdm(range(1, steps + 1), desc="Simulating GBM Paths"):
        z = np.random.standard_normal(n_paths)
        diffusion = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        paths[t, :] = paths[t - 1, :] * diffusion

    return paths