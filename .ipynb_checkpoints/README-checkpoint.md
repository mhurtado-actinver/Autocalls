# Data
- `Data/historic_data_spx.xlsx`: Cleaned and aligned dataset with price, IV, risk-free and Actinver forecasts for the returns of the S&P500. All except the Actinver expected returns were extracted from Bloomberg. Dates span Early 2006 to early/mid 2025 ~ 20 years. The dividend and the risk free rate are already in continuously compunded terms. All percentage figures have been turned to decimals. Each column has the same number of rows, hence all dates are aligned. We assume Actinver forecasts and those of the market, include the dividend effect.



# Limits:
- The get current data method is only valid for 90 days
- The drift is calibrated to real world measure. It should be the average expected return of the underlying according to the market. However, we use Actinvers forecasts until we can get a hold of the actual database. The amrket expecteds are under the label 'spx_bottom_up'
- There are memory problems in the calibration stage of the model
- I don't know if the function compute_transition_matrix is correctly coded
- The imports could be cleaner, not need to import everything from every module
- We make a key assumption, especially when calibrating regimes (due to GMM), that absolute log returns are distributed normally
- The Heston calibration is very crude, and it requires historical vol surfaces of which we currently have no access to

# Design Principles
- Insight is a function not only of what is observed, but of the resolution at which it is observed. Granularity is perspective. And perspective is power.
- SOLID, mostly single responsibility and testability. Simplicity over Complexity. It must be very clear what comes in and what comes out, but not necessarily the inner workings

# Potential improvements
- Include in the GMM not only the magnitude, but the direction of the volatility
- Dynamically adjust the number of regimes in calibration
- Optimise with pomegranate library the calibration of the regimes model

# Notes
- The regime model uses a probabilistic regime-switching framework (Markovian), where each path evolves through volatility regimes according to a transition matrix. At every time step, each path has a stochastic chance of transitioning to another regime, independent of price or volatility thresholds. An alternative approach — suggested by Gustavo — is to assign regimes deterministically based on thresholds of observed daily returns or realized volatility. For example, if a path exhibits daily volatility greater than a fixed threshold, it is classified into a high-volatility regime (e.g., regime 3). However, this approach presents a circular inconsistency when regimes are inherently tied to different volatility levels. Specifically:

    - Suppose regimes are ordered such that regime 0 always has the lowest σ, and regime N (e.g., regime 3) has the highest σ.

    - If a path is in regime 3, its volatility is structurally high by construction (σ₃ is large)

    - But if regime assignments are based on observed realized volatility, then being in regime 3 should require observing high volatility (e.g., a 3% daily move).
      
    - Ironically, once you're in regime 3 (a high-volatility regime), the random returns you generate — while drawn from a high-σ distribution — won't always look volatile in the short term. Some of those returns might be small just by chance, which causes the measured realized volatility (e.g., daily standard deviation) to drop below the threshold used to define regime 3.
      
    - The analogy is, even if you're rolling a 20-sided die (high variance), you might get several "10s" or "11s" in a row. So it doesn’t look wild — but that doesn’t mean you’re not using a high-vol die.
      
    - → In other words, regime 3 can kill the very signal needed to justify being in regime 3 in an inconsistent manner.
      
    - Other implications of a deterministic approach is that:
      
    - You may never enter regime 3 directly from regime 0 or 1, because those regimes don’t generate high enough volatility (and  therefore, returns and prices), which is what you need to get to higher ordered regimes
      
    - You’re only likely to reach regime 3 if you first pass through regime 2, or if a highly improbable sequence of shocks in regime 0 accumulates enough volatility.
      
    - This creates a biased and constrained regime structure, where some transitions are effectively blocked or require "wild" paths, defeating the purpose of a flexible, realistic volatility model.