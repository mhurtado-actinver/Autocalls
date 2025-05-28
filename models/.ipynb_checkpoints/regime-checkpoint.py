import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def calibrate_volatility_regime_model(price_series, n_regimes=2, annualize=True, verbose=True, export_excel=False):
    """
    Calibrate a regime-switching volatility model using Gaussian Mixture on absolute log returns.

    Parameters:
    - price_series: pandas Series or numpy array of prices
    - n_regimes: int, number of volatility regimes to identify
    - annualize: bool, whether to annualize volatility estimates
    - verbose: bool, whether to print diagnostic outputs
    - export_excel: bool, whether to save the regime series as an Excel file

    Returns:
    - sigmas: list of estimated annualized std per regime (ordered low → high)
    - transition_matrix_df: pandas DataFrame of regime-to-regime transition probabilities
    - regimes_series: pandas Series of regime assignments per date
    """
    if isinstance(price_series, np.ndarray):
        price_series = pd.Series(price_series)

    # Compute log returns
    returns = np.log(price_series / price_series.shift(1)).dropna()
    returns = returns.reset_index(drop=True)
    abs_returns = np.abs(returns.values).reshape(-1, 1)

    # GMM fit on absolute returns
    gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=0, n_init=10) 
    gmm.fit(abs_returns)
    regime_labels = gmm.predict(abs_returns)

    if verbose:
        unique, counts = np.unique(regime_labels, return_counts=True)
        print("🔍 GMM label counts:")
        for label, count in zip(unique, counts):
            print(f"   Regime {label}: {count} points")

    # Order regimes by increasing volatility
    regime_vols = [returns.values[regime_labels == i].std() for i in range(n_regimes)]
    ordered_idx = np.argsort(regime_vols)
    regime_map = {old: new for new, old in enumerate(ordered_idx)}
    regimes_ordered = np.vectorize(regime_map.get)(regime_labels)

    if verbose:
        unannualised_vols = np.sort(regime_vols)
        annualised_vols = unannualised_vols*np.sqrt(252)
        print(f"📊 Regime volatilities (unannualized): {unannualised_vols}")
        print(f"📊 Regime volatilities (annualized): {annualised_vols}")

    # Build regime time series
    regimes_series = pd.Series(regimes_ordered, index=returns.index)

    # Compute volatilities per regime
    sigmas = []
    for i in range(n_regimes):
        sigma = returns[regimes_series == i].std()
        if annualize:
            sigma *= np.sqrt(252)
        sigmas.append(sigma)

    # Transition matrix
    transition_df = compute_transition_matrix(regimes_ordered)

    if verbose:
        print("🧮 Transition matrix:\n", transition_df)
        for i, sigma in enumerate(sigmas):
            print(f"σ_{i} (Regime {i}): {sigma:.4f}")

    if export_excel:
        regimes_series.to_excel("Data/regime_map.xlsx", index_label="Date", header=["Regime"])

    return sigmas, transition_df, regimes_series




    

def monte_carlo_paths_regimes(S0, q, r, expected_return, T, steps, n_paths,
                                          sigmas, transition_matrix,
                                          measure='Risk-Neutral',
                                          init_probs=None,
                                          seed=None):
    """
    Simulate GBM paths with N-state volatility regime switching.

    Parameters:
    - S0: initial price
    - q: dividend yield
    - r: risk-free rate
    - T: total time in years
    - steps: number of time steps
    - n_paths: number of Monte Carlo paths
    - sigmas: list/array of volatilities for each regime (length n)
    - transition_matrix: (n x n) matrix of transition probabilities
    - init_probs: optional initial regime probabilities (length n)
    - seed: int, for reproducibility

    Returns:
    - paths: ndarray of shape (steps + 1, n_paths)
    - regimes: ndarray of shape (steps, n_paths), regime index at each step
    """
    if seed is not None:
        np.random.seed(seed)

    sigmas = np.array(sigmas)
    n_regimes = len(sigmas)
    transition_matrix = np.array(transition_matrix)

    if init_probs is None:
        init_probs = np.ones(n_regimes) / n_regimes
    init_probs = np.array(init_probs)

    dt = T / steps
    if (measure == 'Risk-Neutral'):
        mu = r - q
    else:
        mu = expected_return

    paths = np.zeros((steps + 1, n_paths))
    paths[0, :] = S0

    regimes = np.zeros((steps, n_paths), dtype=int)
    current_regime = np.random.choice(n_regimes, size=n_paths, p=init_probs)

    for t in tqdm(range(1, steps + 1), desc="Simulating N-Regime Paths"):
        # Transition to next regime based on transition_matrix
        next_regime = np.zeros_like(current_regime)
        for i in range(n_paths):
            probs = transition_matrix[current_regime[i]]
            next_regime[i] = np.random.choice(n_regimes, p=probs)
        current_regime = next_regime
        regimes[t - 1, :] = current_regime

        # Apply appropriate volatility
        sigma_t = sigmas[current_regime]
        z = np.random.standard_normal(n_paths)
        diffusion = np.exp((mu - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * z)
        paths[t, :] = paths[t - 1, :] * diffusion

    return paths, regimes




def compute_transition_matrix(regime_labels):
    """
    Compute a Markov transition matrix from a sequence of regime labels.

    Parameters:
    - regime_labels (array-like): Sequence of regime assignments (e.g., [0, 1, 1, 2, ...]).

    Returns:
    - pd.DataFrame: Square matrix of transition probabilities, where entry (i, j)
      is the probability of transitioning from regime i to regime j.
    """
    regime_labels = np.asarray(regime_labels)
    unique_states = np.unique(regime_labels)
    state_to_index = {s: i for i, s in enumerate(unique_states)}
    n = len(unique_states)
    matrix = np.zeros((n, n), dtype=int)

    for from_state, to_state in zip(regime_labels[:-1], regime_labels[1:]):
        i, j = state_to_index[from_state], state_to_index[to_state]
        matrix[i, j] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = matrix / row_sums

    return pd.DataFrame(norm_matrix,
                        index=[f"from_{s}" for s in unique_states],
                        columns=[f"to_{s}" for s in unique_states])





