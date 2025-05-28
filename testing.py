import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from scipy.stats import ks_2samp, anderson_ksamp
from models.jump import calibrate_jump_model
from models.regime import calibrate_volatility_regime_model
from models.heston import calibrate_heston_model


def split_train_test_calendar_window(df, 
                                      test_start_date, 
                                      train_years=5, 
                                      date_column='Date'):
    """
    Splits a DataFrame into train and test sets using calendar-aware logic.
    Handles both regular DataFrames and DatetimeIndex DataFrames.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        test_start_date (str or datetime): Start date of test window.
        train_years (int): Number of years for the training window.
        date_column (str): Column containing dates, used only if DataFrame is not indexed by dates.

    Returns:
        train_df (pd.DataFrame), test_df (pd.DataFrame)
    """
    df = df.copy()

    # Handle input as string or datetime
    test_start = pd.to_datetime(test_start_date)

    # Determine whether to use index or a date column
    if isinstance(df.index, pd.DatetimeIndex):
        # Already indexed by dates
        df = df.sort_index()
        dates = df.index
    else:
        # Use a date column
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        dates = df[date_column]

    # Calculate test end date
    try:
        test_end = test_start.replace(year=test_start.year + 1)
    except ValueError:
        # Handles leap day edge cases
        test_end = test_start.replace(month=2, day=28, year=test_start.year + 1)

    # Calculate training start date
    try:
        train_start = test_start.replace(year=test_start.year - train_years)
    except ValueError:
        train_start = test_start.replace(month=2, day=28, year=test_start.year - train_years)

    # Clip train start to available data
    min_date = dates.min()
    if train_start < min_date:
        train_start = min_date

    # Build train and test sets
    if isinstance(df.index, pd.DatetimeIndex):
        train_df = df[(df.index >= train_start) & (df.index < test_start)]
        test_df = df[(df.index >= test_start) & (df.index < test_end)]
    else:
        train_df = df[(df[date_column] >= train_start) & (df[date_column] < test_start)]
        test_df = df[(df[date_column] >= test_start) & (df[date_column] < test_end)]

    return train_df, test_df



def compare_simulated_to_real_returns(
    real_prices,
    simulated_paths,
    asset_name="Asset",
    verbose=True,
    seed=None
):
    """
    Compare real vs. simulated daily log returns using the Kolmogorov–Smirnov test.

    Parameters:
    ----------
    real_prices : array-like (1D)
        Actual historical prices (e.g., from test_data['Price']).
    simulated_paths : ndarray (2D)
        Simulated paths of shape (steps + 1, n_paths).
    asset_name : str
        Name of the asset for display purposes.
    verbose : bool
        If True, print the KS test results.

    Returns:
    -------
    ks_stat : float
        KS test statistic.
    ks_pval : float
        p-value from the KS test.
    log_returns_sim_flat : ndarray
        Flattened simulated log returns (for diagnostic use).
    """

    # Ensure inputs are NumPy arrays
    real_prices = np.asarray(real_prices)
    simulated_paths = np.asarray(simulated_paths)

    # Remove the first row because it contains, by design, the intial price, and that is known, e only want to include uknown (simulated) data
    simulated_paths = simulated_paths[1:]

    # Check dimensions
    assert simulated_paths.ndim == 2, "simulated_paths must be a 2D array"
    assert real_prices.ndim == 1, "real_prices must be a 1D array"

    # Compute log returns
    returns_real = np.log(real_prices[1:] / real_prices[:-1])
    log_returns_sim = np.log(simulated_paths[1:] / simulated_paths[:-1])  # shape (steps, n_paths)
    log_returns_sim_flat = log_returns_sim.flatten()

    
    if seed:
        np.random.seed(seed)

    # Select a sample of the returns to make a fair comparison - First Approach
    # log_returns_sim_sample = np.random.choice(log_returns_sim_flat, size=len(returns_real), replace=False)

    # Sample one return per day across simulated paths (axis 1) - Second Approach, more robust
    timesteps, n_paths = log_returns_sim.shape
    assert len(returns_real) == timesteps, "Real and simulated returns must have same number of steps"
    
    # Select a random simulated path per day
    sampled_idx = np.random.randint(0, n_paths, size=timesteps)
    log_returns_sim_sample = log_returns_sim[np.arange(timesteps), sampled_idx]

    # Kolmogorov–Smirnov test
    ks_stat, ks_pval = ks_2samp(returns_real, log_returns_sim_sample)

    # Anderson–Darling k-sample test
    ad_result = anderson_ksamp([returns_real, log_returns_sim_sample])
    ad_stat = ad_result.statistic
    ad_sig_level = ad_result.significance_level / 100.0  # Convert from % to decimal

    # Verbose output
    if verbose:
        print(f"📈 Comparison for {asset_name}")
        print(f"— KS Test —")
        print(f"  KS Statistic: {ks_stat:.4f}")
        print(f"  p-value:      {ks_pval:.4f}")
        if ks_pval < 0.05:
            print("  ❌ KS: Distributions differ significantly.")
        else:
            print("  ✅ KS: No significant difference detected.")
        print(f"— Anderson–Darling Test —")
        print(f"  AD Statistic: {ad_stat:.4f}")
        print(f"  Significance Level: {ad_sig_level:.4f}")
        if ad_sig_level < 0.05:
            print("  ❌ AD: Distributions differ significantly.")
        else:
            print("  ✅ AD: No significant difference detected.")

    return ks_stat, ks_pval, ad_stat, ad_sig_level, log_returns_sim_flat





def run_rolling_volatility_regime_model(price_series, 
                                        window_years=5, 
                                        n_regimes=4, 
                                        annualize=True, 
                                        export_excel=True, 
                                        verbose=False, 
                                        save_path="rolling_volatility_results.xlsx"):
    """
    Runs rolling calibration of the volatility regime model using a fixed rolling window.

    Parameters:
    - price_series: pd.Series of historical prices (DatetimeIndex)
    - window_years: Rolling window size in years (default 5 years)
    - n_regimes: Number of regimes for the GMM (default 4)
    - annualize: Whether to annualize the volatility estimates (default True)
    - export_excel: Whether to export the final result as Excel (default True)
    - verbose: Whether to print diagnostics for each window (default False)
    - save_path: Output path for Excel file (default 'rolling_volatility_results.xlsx')

    Returns:
    - results_df: DataFrame with sigma estimates, regime transitions, and assigned regime per day
    """
    price_series = price_series.dropna()
    n_days = window_years * 252  # Approximate number of trading days in the window

    results = []

    for i in tqdm(range(n_days, len(price_series)), desc="Rolling Calibration"):
        window_prices = price_series.iloc[i - n_days:i]
        current_date = price_series.index[i]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore sklearn multithreading warnings
                sigmas, transition_matrix, regimes_series = calibrate_volatility_regime_model(
                    window_prices,
                    n_regimes=n_regimes,
                    annualize=annualize,
                    verbose=verbose,
                    export_excel=False
                )

            row = {"Date": current_date}

            # Add volatility per regime
            for r in range(n_regimes):
                row[f"Sigma_{r}"] = sigmas[r]

            # Add full transition matrix entries
            for r_from in range(n_regimes):
                for r_to in range(n_regimes):
                    row[f"Transition_{r_from}{r_to}"] = transition_matrix.iloc[r_from, r_to]

            # Add last regime classification
            row["Regime"] = regimes_series.iloc[-1]

            results.append(row)

        except Exception as e:
            print(f"⚠️ Skipping {current_date} due to error: {e}")
            continue

    # Final DataFrame
    results_df = pd.DataFrame(results).set_index("Date")

    # Export to Excel
    if export_excel:
        results_df.to_excel(save_path)

    return results_df