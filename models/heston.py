import numpy as np
import pandas as pd
from tqdm import tqdm


def monte_carlo_paths_heston(S0, q, r, expected_return, T, steps, n_paths,
                              v0, kappa, theta, sigma_v, rho,
                              measure='Risk-Neutral', seed=None):
    """
    Simulate GBM paths under the Heston stochastic volatility model.

    Parameters:
    - S0: initial price
    - q: dividend yield
    - r: risk-free rate
    - expected_return: expected return under the real-world measure
    - T: total time in years
    - steps: number of time steps
    - n_paths: number of Monte Carlo paths
    - v0: initial variance
    - kappa: speed of mean reversion of variance
    - theta: long-term mean of variance
    - sigma_v: volatility of variance ("vol of vol")
    - rho: correlation between stock and variance shocks
    - measure: 'Risk-Neutral' or 'Real', determines drift
    - seed: int, for reproducibility

    Returns:
    - S_paths: ndarray of shape (steps + 1, n_paths), simulated price paths
    - v_paths: ndarray of shape (steps + 1, n_paths), simulated variance paths
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps

    if measure == 'Risk-Neutral':
        mu = r - q
    else:
        mu = expected_return - q

    S_paths = np.zeros((steps + 1, n_paths))
    v_paths = np.zeros((steps + 1, n_paths))
    S_paths[0, :] = S0
    v_paths[0, :] = v0

    for t in tqdm(range(1, steps + 1), desc="Simulating Heston Paths"):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        dW_S = Z1
        dW_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        v_prev = np.maximum(v_paths[t - 1], 0)
        v_paths[t] = (
            v_prev
            + kappa * (theta - v_prev) * dt
            + sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * dW_v
        )
        v_paths[t] = np.maximum(v_paths[t], 0)  # ensure non-negative variance

        S_paths[t] = S_paths[t - 1] * np.exp(
            (mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * np.sqrt(dt) * dW_S
        )

    return S_paths, v_paths




def calibrate_heston_model(price_series, dt=1/252, verbose=True):
    """
    Estimate basic Heston model parameters from historical price series.

    Parameters:
    - price_series: pandas Series or numpy array of prices
    - dt: time step (default: 1/252 for daily data)
    - verbose: bool, whether to print diagnostics

    Returns:
    - v0: initial variance
    - kappa: speed of mean reversion
    - theta: long-term mean of variance
    - sigma_v: volatility of volatility
    - rho: correlation between returns and variance changes
    """
    if isinstance(price_series, np.ndarray):
        price_series = pd.Series(price_series)

    # Log returns
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    returns = log_returns.reset_index(drop=True)

    # Realized variance (squared returns)
    realized_var = returns**2

    # Estimate initial variance
    v0 = realized_var.iloc[0]

    # Mean and variance of realized variance
    theta = realized_var.mean()
    sigma_v = realized_var.std() / np.sqrt(dt)  # crude vol of variance
    kappa = 1 / (realized_var.autocorr(lag=1) * dt + 1e-6)  # approximate mean reversion

    # Estimate correlation between returns and variance changes (leverage effect)
    var_changes = realized_var.diff().dropna()
    rho = np.corrcoef(returns[1:], var_changes)[0, 1]

    if verbose:
        print(f"📈 v0 (initial variance): {v0:.6f}")
        print(f"📉 theta (long-run variance): {theta:.6f}")
        print(f"🌀 sigma_v (vol of vol): {sigma_v:.6f}")
        print(f"🔁 kappa (mean reversion): {kappa:.6f}")
        print(f"🔗 rho (corr with vol changes): {rho:.4f}")

    return v0, kappa, theta, sigma_v, rho



