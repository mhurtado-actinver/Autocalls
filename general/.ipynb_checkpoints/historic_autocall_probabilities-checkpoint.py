"""
historic_autocall_probabilities.py — Simulate historical autocall outcomes

TODO:
- Replace average volatility with per-issuance volatility, example as below
- Add CLI flag for custom observation intervals
- Add scatter plots: outcome vs. market cap / volatility
- Improve error logging per ticker
- Consider caching volatility calculations per issuance date

Instead of:
vol_annualized = ... .mean() ...

Try:
vol_at_issue = (
    filtered_price_df.pct_change()
    .rolling(252)
    .std()
    * np.sqrt(252)
)

# Take the mean only at issuance dates
vol_mean_by_ticker = {
    ticker: vol_at_issue[ticker].loc[vol_at_issue.index.intersection(results_df["Issuance Date"])].mean()
    for ticker, results_df in zip(tickers_to_use, all_simulated_results)
}
"""





import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

EARLIEST_DATE = '1993-01-01'

# === Simulation Logic ===

def simulate_autocalls(df, ticker='SP500', maturity_days=252, obs_interval=62, barrier_level=0.8):
    """
    Simulates autocallable outcomes for a single ticker based on historical price paths.

    Parameters:
    - df : DataFrame containing historical prices (indexed by date).
    - ticker : The asset for which to simulate outcomes.
    - maturity_days : Days until maturity (typically 252 for 1Y).
    - obs_interval : Interval in business days between autocall observations.
    - barrier_level : Knock-in barrier level as a percentage of initial price.

    Returns:
    - DataFrame summarizing outcome and time to outcome for each issuance date.
    """
    df = df[[ticker]].dropna().copy()
    issuance_dates = df.index
    results = []

    for issue_date in issuance_dates:
        if issue_date + pd.tseries.offsets.BDay(maturity_days) > df.index[-1]:
            continue  # Skip if not enough future data

        S0 = df.loc[issue_date, ticker]
        barrier = S0 * barrier_level
        observation_offsets = [obs_interval * i for i in range(1, 5)]  # Quarterly checks
        outcome, outcome_day = None, None

        # Check for autocall on observation dates
        for i, offset in enumerate(observation_offsets, start=1):
            obs_date = issue_date + pd.tseries.offsets.BDay(offset)
            try:
                if df.loc[obs_date, ticker] >= S0:
                    outcome = f"Autocalled Q{i}"
                    outcome_day = offset
                    break
            except KeyError:
                continue  # Skip if price data missing for this observation

        # If not autocalled, check final payoff at maturity
        if outcome is None:
            maturity_date = issue_date + pd.tseries.offsets.BDay(maturity_days)
            try:
                final_price = df.loc[maturity_date, ticker]
                outcome = "Survived Above KI" if final_price >= barrier else "Breached KI"
                outcome_day = maturity_days
            except KeyError:
                continue  # Skip if final data is missing

        results.append({
            "Issuance Date": issue_date,
            "Initial Price": S0,
            "Outcome": outcome,
            "Days to Outcome": outcome_day
        })

    return pd.DataFrame(results)


# === Main Execution Flow ===

def main():
    """
    Orchestrates loading data, running simulations, and saving results to Excel.
    """
    # --- Parse CLI arguments ---
    parser = argparse.ArgumentParser(description="Simulate autocall outcome distributions.")
    parser.add_argument('--all', action='store_true', help='Use all tickers instead of just the top 100')
    args = parser.parse_args()
    use_all_tickers = args.all

    # --- Load metadata and filter tickers ---
    muestra_df = pd.read_excel("../data/actinver_team/muestra.xlsx")
    filtered_df = muestra_df[muestra_df['Ticker'] != 'GOOG']
    sorted_df = filtered_df.sort_values(by='Market Cap:D-1', ascending=False)
    top_100_tickers = sorted_df['Ticker'].head(100).tolist()
    all_tickers = sorted_df['Ticker'].tolist()
    tickers_to_use = all_tickers if use_all_tickers else top_100_tickers
    tickers_to_use = list(dict.fromkeys(tickers_to_use + ['SP500']))  # Ensure benchmark included

    # --- Load historical price data ---
    df = pd.read_csv("../data/actinver_team/daly_data.csv")
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df.set_index('Fecha', inplace=True)
    matched_columns = [col for col in df.columns if col in tickers_to_use]
    filtered_price_df = df[matched_columns]
    filtered_price_df = filtered_price_df[filtered_price_df.index >= EARLIEST_DATE]

    # --- Calculate Realised Vols i.e. the average historical 1-year rolling volatility across the full dataset per ticker ---
    vol_annualized = (
    filtered_price_df.pct_change()
    .rolling(252)
    .std()
    .mean()
    * np.sqrt(252) 
    )


    # --- Run simulations for each ticker ---
    all_distributions = []

    for ticker in tqdm(tickers_to_use, desc="Simulating autocalls"):
        try:
            if ticker not in filtered_price_df.columns:
                continue

            results_df = simulate_autocalls(filtered_price_df, ticker=ticker)
            dist = results_df["Outcome"].value_counts(normalize=True).sort_index() * 100
            dist_df = dist.round(2).reset_index()
            dist_df.columns = ['Outcome', 'Percentage']
            dist_df.insert(0, 'Ticker', ticker)
            all_distributions.append(dist_df)

        except Exception as e:
            print(f"❌ Failed for {ticker}: {e}")

    # --- Build final DataFrame and merge metadata ---
    raw_df = pd.concat(all_distributions, ignore_index=True)

    dupes = raw_df[raw_df.duplicated(subset=['Ticker', 'Outcome'], keep=False)]
    print(dupes.sort_values(['Ticker', 'Outcome']))


    
    final_df = raw_df.pivot(index='Ticker', columns='Outcome', values='Percentage').fillna(0)

    metadata = muestra_df[['Ticker', 'Nombre', 'GICS Sector ES', 'Market Cap:D-1']]
    metadata = metadata[metadata['Ticker'].isin(final_df.index)]
    final_df_with_meta = final_df.merge(metadata.set_index('Ticker'), left_index=True, right_index=True)
    final_df_with_meta = final_df_with_meta.sort_values(by='Market Cap:D-1', ascending=False)
    final_df_with_meta['Volatility'] = final_df_with_meta.index.map(vol_annualized)

    # --- Reorder columns for clarity ---
    meta_cols = ['Nombre', 'GICS Sector ES', 'Market Cap:D-1']
    outcome_cols = [col for col in final_df.columns]
    final_df_with_meta = final_df_with_meta[meta_cols + outcome_cols + ['Volatility']]

    # --- Save results ---
    output_path = "../data/historic_autocall_distributions_all.xlsx"
    final_df_with_meta.to_excel(output_path, index=True)
    print(f"✅ Saved final table with metadata to: {output_path}")


# === Visualization Functions ===

def plot_autocall_stacked_bar(pivot_df, title="Autocall Outcome Distribution by Ticker"):
    """
    Plots a stacked bar chart of outcome distributions by ticker.
    """
    pivot_df.plot(kind='bar', stacked=True, figsize=(16, 8), colormap='tab20')
    plt.title(title)
    plt.ylabel("Percentage")
    plt.xlabel("Ticker")
    plt.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


def plot_autocall_heatmap(pivot_df, title="Autocall Outcome Heatmap per Ticker"):
    """
    Plots a heatmap showing outcome percentages across tickers.
    """
    plt.figure(figsize=(18, 6))
    sns.heatmap(pivot_df, cmap="coolwarm", cbar_kws={'label': 'Percentage'})
    plt.title(title)
    plt.ylabel("Ticker")
    plt.xlabel("Outcome")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_outcome_vs_volatility(final_df, outcome="Breached KI"):
    """
    Plots a scatter plot of a given autocall outcome vs. historical volatility.
    
    Parameters:
    - final_df: DataFrame with outcome percentages, volatility, market cap, etc.
    - outcome: Name of the outcome column to plot on the Y-axis (e.g., "Breached KI")
    """
    df = final_df.copy()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Volatility",
        y=outcome,
        size="Market Cap:D-1",
        hue="GICS Sector ES",
        sizes=(30, 300),
        alpha=0.7
    )
    plt.title(f"{outcome} vs. Historical Volatility")
    plt.xlabel("Annualized Volatility")
    plt.ylabel(f"% {outcome}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Entry Point ===

if __name__ == "__main__":
    main()