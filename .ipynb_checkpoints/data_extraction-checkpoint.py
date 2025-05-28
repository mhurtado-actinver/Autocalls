import numpy as np
import pandas as pd
import eikon as ek
from datetime import datetime





def get_continuous_risk_free_rate(fred_client, series_id='DGS1'):
    """
    Fetch the latest 1-year T-note yield (proxy for risk-free rate) from FRED,
    convert it from percentage to decimal, and transform it from
    annual compounding to continuous compounding.
    
    Parameters:
        fred_client: An initialized FRED API client
        series_id (str): FRED series ID, default is 'DGS1' (1-year T-notes)
    
    Returns:
        R (float): Continuously compounded annual risk-free rate
        r_quoted (float): Quoted annual rate in decimal form
        rly_series (pd.Series): Full historical series fetched from FRED
    """
    rly_series = fred_client.get_series_latest_release(series_id)
    r_quoted = rly_series.iloc[-1] / 100  # Convert from percent to decimal
    R = np.log(1 + r_quoted)              # Convert to continuous compounding
    return (R, r_quoted, rly_series)





def get_live_ric_data(ric, option_ric):
    """
    Fetches live pricing, volatility, and dividend yield for a given RIC using Refinitiv Eikon.
    Falls back to hardcoded values if data cannot be retrieved.

    Parameters:
        ric (str): RIC for the underlying asset (e.g., "AAPL.O")
        option_ric (str): RIC for the option contract (e.g., "AAPL.MI")

    Returns:
        S0 (float): Spot price of the underlying
        sigma (float): Annualized implied volatility (in decimal)
        dividend (float): Continuously compounded dividend yield
    """
    try:
        # Fetch latest mid price
        price_data, err = ek.get_data(ric, ['CF_BID', 'CF_ASK'])
        if err or price_data.empty or pd.isna(price_data.iloc[0, 1]) or pd.isna(price_data.iloc[0, 2]):
            raise ValueError(f"Error fetching price data: {err}")
        mid_price = (price_data['CF_BID'] + price_data['CF_ASK']) / 2
        S0 = mid_price[0]

        # Fetch the volatility - WARNING This is the longest current data we have for implied vol, which  means our simulation of the future is only valid for the next 90 days.
        option_fields = 'TR.90DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORCALLOPTIONS'
        vol_data, err = ek.get_data([option_ric], option_fields)
        if err or vol_data.empty or pd.isna(vol_data.iloc[0, 1]):
            raise ValueError(f"Error fetching volatility data: {err}")
        sigma = vol_data.iloc[0, 1] / 100  # Convert % to decimal

        # Fetch dividend amount
        dividend_data, err = ek.get_data(ric, ['CF_YIELD'])
        if err or dividend_data.empty or pd.isna(dividend_data.iloc[0, 1]):
            raise ValueError(f"Error fetching dividend data: {err}")
        dividend_quoted = dividend_data['CF_YIELD'].iloc[0] / 100
        dividend = np.log(1 + dividend_quoted)

        print("✅ Live data fetched successfully from Refinitiv Eikon.")

    except Exception as e:
        # Fallback to hardcoded values (these need to be defined globally or passed in)
        print("⚠️  Live fetch failed. Using hardcoded fallback values.")
        print(f"Error: {e}")

        S0 = PRICE
        sigma = SIGMA
        dividend = Q

    return S0, sigma, dividend