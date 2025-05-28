import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
muestra_df = pd.read_excel("../data/actinver_team/muestra.xlsx")

# Drop 'GOOG' from the 'Ticker' column
filtered_df = muestra_df[muestra_df['Ticker'] != 'GOOG']

# Sort by 'Market Cap:D-1' in descending order
sorted_df = filtered_df.sort_values(by='Market Cap:D-1', ascending=False)

# Select top 100 tickers
top_100_tickers = sorted_df['Ticker'].head(100).tolist()

# Add 'S&P500' to the list
top_100_tickers.append('SP500')

df = pd.read_csv("../data/actinver_team/daly_data.csv")
df['Fecha'] = pd.to_datetime(df['Fecha'])
df.set_index('Fecha', inplace=True)

# Filter columns that match top 100 tickers
matched_columns = [col for col in df.columns if col in top_100_tickers]
filtered_price_df = df[matched_columns]
filtered_price_df = filtered_price_df[filtered_price_df.index >= '1993-01-01']

def simulate_autocalls(df, ticker='SP500', maturity_days=252, obs_interval=62, barrier_level=0.8):
    df = df.copy()
    df = df[[ticker]].dropna()
    df = df[df.index <= pd.Timestamp("2025-05-28")]  # cutoff

    issuance_dates = df.index
    results = []

    for issue_date in issuance_dates:
        if issue_date + pd.tseries.offsets.BDay(maturity_days) > df.index[-1]:
            continue  # not enough data to evaluate this issuance

        S0 = df.loc[issue_date, ticker]
        barrier = S0 * barrier_level
        observation_offsets = [obs_interval * i for i in range(1, 5)]
        outcome = None
        outcome_day = None

        for i, offset in enumerate(observation_offsets, start=1):
            obs_date = issue_date + pd.tseries.offsets.BDay(offset)
            try:
                price_on_obs = df.loc[obs_date, ticker]
                if price_on_obs >= S0:
                    outcome = f"Autocalled Q{i}"
                    outcome_day = offset
                    break
            except KeyError:
                continue  # missing data on that date

        if outcome is None:
            maturity_date = issue_date + pd.tseries.offsets.BDay(maturity_days)
            try:
                final_price = df.loc[maturity_date, ticker]
                if final_price >= barrier:
                    outcome = "Survived Above KI"
                else:
                    outcome = "Breached KI"
                outcome_day = maturity_days
            except KeyError:
                continue  # skip if final maturity data is missing

        results.append({
            "Issuance Date": issue_date,
            "Initial Price": S0,
            "Outcome": outcome,
            "Days to Outcome": outcome_day
        })

    return pd.DataFrame(results)


# Collect all rows into a list of DataFrames
all_distributions = []

for ticker in tqdm(top_100_tickers, desc="Simulating autocalls"):
    try:
        if ticker not in filtered_price_df.columns:
            continue

        # Run the simulation
        results_df = simulate_autocalls(filtered_price_df, ticker=ticker)

        # Compute the outcome distribution
        dist = (
            results_df["Outcome"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)

        # Create a DataFrame with Ticker column
        dist_df = dist.reset_index()
        dist_df.columns = ['Outcome', 'Percentage']
        dist_df.insert(0, 'Ticker', ticker)

        all_distributions.append(dist_df)

    except Exception as e:
        print(f"❌ Failed for {ticker}: {e}")

# Combine all into a single DataFrame
raw_df = pd.concat(all_distributions, ignore_index=True)

# Pivot the table to make it easier to perform data analysis
final_df = raw_df.pivot(index='Ticker', columns='Outcome', values='Percentage').fillna(0)



# Prepare metadata (e.g. 'Company Name', 'Sector') from muestra_df
metadata = muestra_df[['Ticker', 'Nombre', 'GICS Sector ES']]
metadata = metadata[metadata['Ticker'].isin(final_df.index)]

# Merge with pivoted table
final_df_with_meta = final_df.merge(metadata.set_index('Ticker'), left_index=True, right_index=True)

# Optional: reorder columns so metadata comes first
meta_cols = ['Nombre', 'GICS Sector ES']
outcome_cols = [col for col in final_df.columns]
final_df_with_meta = final_df_with_meta[meta_cols + outcome_cols]

# Save to Excel
output_path = "../data/historic_autocall_distributions.xlsx"
final_df_with_meta.to_excel(output_path, index=True)

print(f"✅ Saved final table with metadata to: {output_path}")


# Plotting functions

def plot_autocall_stacked_bar(pivot_df, title="Autocall Outcome Distribution by Ticker"):
    """
    Plots a stacked bar chart of autocall outcome distributions.
    
    Parameters:
    - pivot_df: DataFrame with tickers as index and outcome percentages as columns.
    - title: Title of the plot.
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
    Plots a heatmap of autocall outcome distributions.
    
    Parameters:
    - pivot_df: DataFrame with tickers as index and outcome percentages as columns.
    - title: Title of the heatmap.
    """
    plt.figure(figsize=(18, 6))
    sns.heatmap(pivot_df, cmap="coolwarm", cbar_kws={'label': 'Percentage'})
    plt.title(title)
    plt.ylabel("Ticker")
    plt.xlabel("Outcome")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()