import numpy as np
import pandas as pd
from tqdm import tqdm


def monte_carlo_paths_jumps(S0, q, r, expected_return, T, steps, n_paths,
                            sigma, lambda_j, m_j, s_j,
                            measure='Risk-Neutral',
                            seed=None):
    """
    Simulate GBM paths with jumps under Merton's Jump-Diffusion model.

    Parameters:
    - S0: initial price
    - q: dividend yield
    - r: risk-free rate
    - expected_return: expected return under real-world measure
    - T: total time in years
    - steps: number of time steps
    - n_paths: number of Monte Carlo paths
    - sigma: volatility of the diffusion component
    - lambda_j: jump intensity (expected jumps per year)
    - m_j: mean of log jump size
    - s_j: standard deviation of log jump size
    - measure: 'Risk-Neutral' or 'Real', determines drift
    - seed: int, for reproducibility

    Returns:
    - paths: ndarray of shape (steps + 1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps

    if measure == 'Risk-Neutral':
        mu = r - q - lambda_j * (np.exp(m_j + 0.5 * s_j**2) - 1)
    else:
        mu = expected_return - lambda_j * (np.exp(m_j + 0.5 * s_j**2) - 1)

    paths = np.zeros((steps + 1, n_paths))
    paths[0, :] = S0

    for t in tqdm(range(1, steps + 1), desc="Simulating Jump Paths"):
        z = np.random.standard_normal(n_paths)
        N_jumps = np.random.poisson(lambda_j * dt, n_paths)
        jump_component = np.exp(m_j * N_jumps + s_j * np.sqrt(N_jumps) * np.random.standard_normal(n_paths))
        diffusion = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        paths[t, :] = paths[t - 1, :] * diffusion * jump_component

    return paths





def calibrate_jump_model(price_series, threshold_multiplier=2, annualize=True, 
                         annualization_factor=252, verbose=True, export_excel=False):
    """
    Calibrate Merton's jump-diffusion model parameters from historical log returns.

    Parameters:
    - price_series: pandas Series or numpy array of prices
    - threshold_multiplier: float, jump threshold = multiplier × daily std
    - annualize: bool, whether to return annualized diffusion volatility
    - annualization_factor: int, number of trading days per year (default: 252)
    - verbose: bool, whether to print diagnostic outputs
    - export_excel: bool, whether to export detected jumps to Excel

    Returns:
    - sigma: float, estimated daily or annualized diffusion volatility
    - lambda_j: float, estimated jump intensity (expected jumps per year)
    - m_j: float, mean of log jump sizes
    - s_j: float, std dev of log jump sizes
    - jumps: pandas Series of detected jump returns
    """
    if isinstance(price_series, np.ndarray):
        price_series = pd.Series(price_series)

    # Compute log returns
    returns = np.log(price_series / price_series.shift(1)).dropna()
    returns = returns.reset_index(drop=True)

    # Estimate daily volatility
    sigma = returns.std()

    # Jump threshold
    threshold = threshold_multiplier * sigma
    jumps = returns[returns.abs() > threshold]

    # Jump parameters
    lambda_j = len(jumps) / len(returns) * annualization_factor
    m_j = jumps.mean()
    s_j = jumps.std()

    # Annualize if requested
    if annualize:
        sigma *= np.sqrt(annualization_factor)

    if verbose:
        print(f"📈 Estimated σ (diffusion volatility): {sigma:.4f} ({'annualized' if annualize else 'daily'})")
        print(f"⚡ Estimated λ (jumps/year): {lambda_j:.2f}")
        print(f"📊 Estimated μ_J (mean log jump): {m_j:.4f}")
        print(f"📊 Estimated σ_J (std dev log jump): {s_j:.4f}")

    if export_excel:
        jumps.to_excel("Data/jump_series.xlsx", index_label="Date", header=["Jump Return"])

    return sigma, lambda_j, m_j, s_j, jumps