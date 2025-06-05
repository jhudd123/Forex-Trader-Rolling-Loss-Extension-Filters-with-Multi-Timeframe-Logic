import numpy as np
import pandas as pd
import talib as ta
from hurst import compute_Hc
from fbm import FBM
from statsmodels.tsa.stattools import grangercausalitytests


# Calculate trend strength and direction (using EMAs, ADX, and other indicators)
def calculate_trend_strength(df):
    # Short, mid, and long term EMAs
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()  # Short-term
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()  # Mid-term
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()  # Long-term

    # ADX for different timeframes
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)  # Short-term
    df['adx_50'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=50)  # Mid-term
    df['adx_200'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=200)  # Long-term

    # Calculate the trend direction based on EMAs and ADX
    df = calculate_trend_direction(df)

    return df

def calculate_volatility(df):
    # Bollinger Bands for short, mid, and long-term windows
    df['bollinger_upper_short'], df['bollinger_middle_short'], df['bollinger_lower_short'] = ta.BBANDS(df['close'], timeperiod=14)  # Short-term
    df['bollinger_upper_mid'], df['bollinger_middle_mid'], df['bollinger_lower_mid'] = ta.BBANDS(df['close'], timeperiod=50)  # Mid-term
    df['bollinger_upper_long'], df['bollinger_middle_long'], df['bollinger_lower_long'] = ta.BBANDS(df['close'], timeperiod=200)  # Long-term

    # Calculate Bollinger Band widths for each timeframe
    df['bollinger_width_short'] = df['bollinger_upper_short'] - df['bollinger_lower_short']
    df['bollinger_width_mid'] = df['bollinger_upper_mid'] - df['bollinger_lower_mid']
    df['bollinger_width_long'] = df['bollinger_upper_long'] - df['bollinger_lower_long']

    return df

def calculate_mean_reversion(df):
    # Z-Score calculation for different windows
    df['price_zscore_short'] = (df['close'] - df['close'].rolling(window=14).mean()) / df['close'].rolling(window=14).std()  # Short-term
    df['price_zscore_mid'] = (df['close'] - df['close'].rolling(window=50).mean()) / df['close'].rolling(window=50).std()  # Mid-term
    df['price_zscore_long'] = (df['close'] - df['close'].rolling(window=200).mean()) / df['close'].rolling(window=200).std()  # Long-term

    # Bollinger Bands for short, mid, and long term
    df['bollinger_upper_short'], df['bollinger_middle_short'], df['bollinger_lower_short'] = ta.BBANDS(df['close'], timeperiod=14)
    df['bollinger_upper_mid'], df['bollinger_middle_mid'], df['bollinger_lower_mid'] = ta.BBANDS(df['close'], timeperiod=50)
    df['bollinger_upper_long'], df['bollinger_middle_long'], df['bollinger_lower_long'] = ta.BBANDS(df['close'], timeperiod=200)

    # RSI for short, mid, and long term
    df['rsi_short'] = ta.RSI(df['close'], timeperiod=14)  # Short-term RSI
    df['rsi_mid'] = ta.RSI(df['close'], timeperiod=50)  # Mid-term RSI
    df['rsi_long'] = ta.RSI(df['close'], timeperiod=200)  # Long-term RSI

    # RSI difference for short, mid, and long term
    df['rsi_diff_short'] = df['rsi_short'].diff()
    df['rsi_diff_mid'] = df['rsi_mid'].diff()
    df['rsi_diff_long'] = df['rsi_long'].diff()

    return df

from hurst import compute_Hc  # Ensure this import is present

def calculate_hurst_exponent(price_series):
    if len(price_series) < 50 or np.ptp(price_series) == 0:  # Skip flat or small windows
        return np.nan
    try:
        H, _, _ = compute_Hc(price_series, kind='price')
        return H
    except Exception as e:
        print(f"Hurst calculation failed: {e}")
        return np.nan
       
# Lead-Lag Analysis Feature with Rolling Window
def rolling_lead_lag_features(df1, df2, max_lag=5, window_size=100):
    rolling_lead_lag_features = []

    # Apply rolling window
    for start in range(len(df1) - window_size + 1):
        end = start + window_size
        df1_window = df1.iloc[start:end]
        df2_window = df2.iloc[start:end]

        # Calculate lead-lag correlations for the window
        lead_lag_features = generate_lead_lag_features(df1_window, df2_window, max_lag)
        rolling_lead_lag_features.append(lead_lag_features.iloc[-1])  # Take the last result from each window

    # Align the resulting features with the original index
    rolling_lead_lag_df = pd.DataFrame(rolling_lead_lag_features, index=df1.index[window_size - 1:])
    rolling_lead_lag_df = rolling_lead_lag_df.reindex(df1.index)

    return rolling_lead_lag_df
   
def rolling_granger_causality(df1, df2, max_lag=15, window_size=50):
    rolling_granger_features = []

    # Apply rolling window
    for start in range(len(df1) - window_size + 1):
        end = start + window_size
        df1_window = df1.iloc[start:end]
        df2_window = df2.iloc[start:end]

        # Perform Granger causality on the window
        granger_features = generate_granger_causality_features(df1_window, df2_window, max_lag)
        rolling_granger_features.append(granger_features.iloc[-1])  # Take the last result from each window

    # Align the resulting features with the original index
    rolling_granger_df = pd.DataFrame(rolling_granger_features, index=df1.index[window_size - 1:])
    rolling_granger_df = rolling_granger_df.reindex(df1.index)

    return rolling_granger_df

       
# Granger Causality Test Feature
def generate_granger_causality_features(df1, df2, max_lag=15):
    # Ensure both dataframes have the same length
    min_length = min(len(df1), len(df2))
    df1 = df1.tail(min_length)
    df2 = df2.tail(min_length)

    # Concatenate the 'close' columns
    combined_df = pd.concat([df1['close'], df2['close']], axis=1)
    combined_df.columns = ['df1_close', 'df2_close']
   
    # Drop missing values and check if enough data is available
    combined_df = combined_df.dropna()
    if combined_df.shape[0] < max_lag:
        print("Not enough data for Granger Causality test.")
        return pd.DataFrame(index=df1.index)  # Return an empty DataFrame with original index
   
    # Perform the Granger Causality test with error handling
    try:
        causality_results = grangercausalitytests(combined_df, max_lag, verbose=False)
    except Exception as e:
        print(f"Granger Causality test failed: {e}")
        return pd.DataFrame(index=df1.index)  # Return an empty DataFrame with original index

    # Extract p-values and generate features for each lag
    granger_features = {}
    for lag in range(1, max_lag + 1):
        try:
            p_value = causality_results[lag][0]['ssr_ftest'][1]
            granger_features[f'granger_causality_lag_{lag}'] = 1 if p_value < 0.05 else 0
        except KeyError as e:
            granger_features[f'granger_causality_lag_{lag}'] = None  # Mark as None if test fails

    # Return features aligned with df1's index
    return pd.DataFrame([granger_features] * len(df1), index=df1.index)



# Lead-Lag Analysis Feature
def generate_lead_lag_features(df1, df2, max_lag=5):
    df1['pct_change'] = df1['close'].pct_change()
    df2['pct_change'] = df2['close'].pct_change()
   
    cross_corr = [df1['pct_change'].shift(lag).corr(df2['pct_change']) for lag in range(-max_lag, max_lag+1)]

    lead_lag_features = {f'lead_lag_corr_lag_{lag}': cross_corr[lag + max_lag] for lag in range(-max_lag, max_lag+1)}

    # Return features aligned with df1's index
    return pd.DataFrame([lead_lag_features] * len(df1), index=df1.index)


# Prepare features by comparing three currency pairs and incorporating fractal features
def prepare_features_with_fractal(df1, df2, df3):
    df1['price_diff1'] = df1['close'] - df2['close']
    df1['price_diff2'] = df1['close'] - df3['close']
    df1['volume_diff1'] = df1['tick_volume'] - df2['tick_volume']
    df1['volume_diff2'] = df1['tick_volume'] - df3['tick_volume']
    df1['volatility_diff1'] = df1['ATR'] - df2['ATR']
    df1['volatility_diff2'] = df1['ATR'] - df3['ATR']
    df1['rel_price1'] = df1['close'] / df2['close']
    df1['rel_price2'] = df1['close'] / df3['close']
    df1['rel_volume1'] = df1['tick_volume'] / df2['tick_volume']
    df1['rel_volume2'] = df1['tick_volume'] / df3['tick_volume']
    df1['rel_volatility1'] = df1['ATR'] / df2['ATR']
    df1['rel_volatility2'] = df1['ATR'] / df3['ATR']
   
    # New Features
    df1['RV'] = df1['tick_volume'] / df1['tick_volume'].rolling(window=14).mean()
    df1['RP'] = df1['close'] / df1['close'].rolling(window=14).mean()
    df1['RM'] = df1['close'].diff() / df1['close'].shift(1)

    # Bollinger Bands
    df1['bollinger_upper'], df1['bollinger_middle'], df1['bollinger_lower'] = ta.BBANDS(df1['close'], timeperiod=14)
   
    # MACD
    df1['macd'], df1['macd_signal'], df1['macd_hist'] = ta.MACD(df1['close'], fastperiod=12, slowperiod=26, signalperiod=9)
   
    # On-Balance Volume (OBV)
    df1['obv'] = ta.OBV(df1['close'], df1['tick_volume'])
   
    # RSI
    df1['rsi'] = ta.RSI(df1['close'], timeperiod=14)
   
    # Calculate Fractal Dimensions (Hurst Exponent)
    df1['hurst'] = calculate_hurst_exponent(df1['close'])

    # Ensure both dataframes have the same length before processing
    df1, df2 = df1.align(df2, join='inner', axis=0)
   
    # Apply rolling Granger causality
    granger_features_df = rolling_granger_causality(df1, df2)
    df1 = pd.concat([df1, granger_features_df], axis=1)

    # Apply rolling lead-lag features
    lead_lag_features_df = rolling_lead_lag_features(df1, df2)
    df1 = pd.concat([df1, lead_lag_features_df], axis=1)

   
    # Ensure the rows are aligned again after all transformations
    df1.dropna(inplace=True)

    # Convert all features to numeric types to avoid any data type issues
    df1 = df1.apply(pd.to_numeric, errors='coerce')
   
    # Ensure all columns are numeric types
    df1 = df1.astype(float)

    print(len(df1))

    return df1



def calculate_trend_direction(df):
    # Handle any missing values in the ema columns
    df[['ema_50', 'ema_200', 'adx']] = df[['ema_50', 'ema_200', 'adx']].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate the slope of the EMAs
    df['ema_50_slope'] = df['ema_50'].diff()
    df['ema_200_slope'] = df['ema_200'].diff()

    # Determine if the ADX indicates a strong trend (threshold of 25)
    df['adx_strong_trend'] = (df['adx'] > 25).astype(int)

    # Calculate the difference between the EMAs
    df['ema_diff'] = df['ema_50'] - df['ema_200']

    # Initialize trend_direction column with "no trend" (0)
    df['trend_direction'] = 0

    # Define conditions for trend direction
    strong_uptrend = (df['ema_50_slope'] > 0) & (df['ema_200_slope'] > 0) & (df['ema_diff'] > 0) & (df['adx_strong_trend'] == 1)
    strong_downtrend = (df['ema_50_slope'] < 0) & (df['ema_200_slope'] < 0) & (df['ema_diff'] < 0) & (df['adx_strong_trend'] == 1)
    weak_uptrend = (df['ema_50_slope'] > 0) & (df['ema_diff'] > 0)
    weak_downtrend = (df['ema_50_slope'] < 0) & (df['ema_diff'] < 0)

    # Apply the trend direction based on conditions
    df.loc[strong_uptrend, 'trend_direction'] = 1    # Strong Uptrend
    df.loc[strong_downtrend, 'trend_direction'] = -1  # Strong Downtrend
    df.loc[weak_uptrend, 'trend_direction'] = 0.5     # Weak Uptrend
    df.loc[weak_downtrend, 'trend_direction'] = -0.5  # Weak Downtrend

    # Round the trend direction to ensure only the intended categories
    df['trend_direction'] = df['trend_direction'].round(1)

    # Calculate average trend over the past 10 periods
    df['average_trend_past'] = df['trend_direction'].rolling(window=10).mean()

    return df



def assign_targets_based_on_trend(df, window=10):
    # Ensure the trend_direction has been calculated
    df = calculate_trend_direction(df)
   
    # Calculate the rolling average trend over the next window periods for target creation
    df['average_trend'] = df['trend_direction'].rolling(window=window).mean().shift(-window)
   
    # Assign targets based on the average trend (future information is used here only for target)
    df['target'] = np.where(df['average_trend'] > 0.2, 1,  # Long
                            np.where(df['average_trend'] < -0.2, 0,  # Short
                                     2))  # Hold
   
    # Drop rows with NaN values introduced by the rolling shift
    df.dropna(subset=['average_trend', 'target'], inplace=True)
   
    # Drop 'average_trend' column to avoid using it as a feature
    df = df.drop(columns=['average_trend'])

    # Debugging the output size after dropping NaNs
    print(f"Length of DataFrame after dropping NaN: {len(df)}")

    return df




# Simulate fGBM with Historical Volatility Context
def simulate_fgbm_with_volatility_context(initial_price, hurst_exponent, num_steps, historical_volatility):
    f = FBM(n=num_steps - 1, hurst=hurst_exponent, length=1, method='daviesharte')
    fbm_series = f.fbm()
    volatility_adjusted_series = fbm_series * historical_volatility
    # Simulate the entire price series, not just the last price
    simulated_prices = initial_price * np.exp(volatility_adjusted_series)
    return simulated_prices  # Return the full series instead of just the last price


def fgbm_based_strategy(df, hurst_exponent):
    # Simulate fGBM prices for the length of the df based on the initial price and Hurst exponent
    simulated_prices = simulate_fgbm_with_volatility_context(df['close'].iloc[0], hurst_exponent, len(df), df['historical_volatility'].mean())

    # Ensure the simulated prices match the length of the dataframe
    if len(simulated_prices) < len(df):
        simulated_prices = np.append(simulated_prices, [simulated_prices[-1]] * (len(df) - len(simulated_prices)))
    elif len(simulated_prices) > len(df):
        simulated_prices = simulated_prices[:len(df)]

    # Add simulated prices to the dataframe
    df['simulated_prices'] = simulated_prices
    df['simulated_return'] = df['simulated_prices'].pct_change()

    # Calculate the difference between actual prices and simulated prices
    df['price_diff'] = df['close'] - df['simulated_prices']
    df['price_diff_pct'] = (df['close'] - df['simulated_prices']) / df['close']

    # Add directional and absolute differences
    df['price_diff_direction'] = np.sign(df['price_diff'])
    df['price_diff_abs'] = np.abs(df['price_diff'])

    return df



# Step 1: Calculate Rolling Features Without Market Regime
def rolling_features_without_regime(df, short_window=20, mid_window=50, long_window=200, window_size=100):

    # Short-term features
    df['pct_change'] = df['close'].pct_change()
   
    # Calculate rolling statistics for short-term window
    df['short_rolling_skew'] = df['pct_change'].rolling(window=short_window, min_periods=5).skew()
    df['short_rolling_kurtosis'] = df['pct_change'].rolling(window=short_window, min_periods=5).kurt()

    # Separate positive and negative returns for short-term
    positive_returns = df['pct_change'].apply(lambda x: x if x > 0 else np.nan)
    negative_returns = df['pct_change'].apply(lambda x: x if x < 0 else np.nan)

    # Calculate skew for positive and negative returns in short-term window
    df['short_positive_skew'] = positive_returns.rolling(window=short_window, min_periods=5).skew()
    df['short_negative_skew'] = negative_returns.rolling(window=short_window, min_periods=5).skew()

    # Mid-term features
    df['mid_rolling_skew'] = df['pct_change'].rolling(window=mid_window, min_periods=5).skew()
    df['mid_rolling_kurtosis'] = df['pct_change'].rolling(window=mid_window, min_periods=5).kurt()

    # Calculate skew for positive and negative returns in mid-term window
    df['mid_positive_skew'] = positive_returns.rolling(window=mid_window, min_periods=5).skew()
    df['mid_negative_skew'] = negative_returns.rolling(window=mid_window, min_periods=5).skew()

    # Long-term features
    df['long_rolling_skew'] = df['pct_change'].rolling(window=long_window, min_periods=5).skew()
    df['long_rolling_kurtosis'] = df['pct_change'].rolling(window=long_window, min_periods=5).kurt()

    # Calculate skew for positive and negative returns in long-term window
    df['long_positive_skew'] = positive_returns.rolling(window=long_window, min_periods=5).skew()
    df['long_negative_skew'] = negative_returns.rolling(window=long_window, min_periods=5).skew()

    # Flag extreme skewness and kurtosis in the short, mid, and long term
    df['short_skew_extreme'] = np.where(df['short_rolling_skew'] > 2, 1, 0)
    df['short_kurt_extreme'] = np.where(df['short_rolling_kurtosis'] > 5, 1, 0)

    df['mid_skew_extreme'] = np.where(df['mid_rolling_skew'] > 2, 1, 0)
    df['mid_kurt_extreme'] = np.where(df['mid_rolling_kurtosis'] > 5, 1, 0)

    df['long_skew_extreme'] = np.where(df['long_rolling_skew'] > 2, 1, 0)
    df['long_kurt_extreme'] = np.where(df['long_rolling_kurtosis'] > 5, 1, 0)

    # Fill remaining NaNs in skew/kurtosis with 0 or another appropriate value
    df.fillna({
        'short_rolling_skew': 0, 'short_rolling_kurtosis': 0, 
        'short_positive_skew': 0, 'short_negative_skew': 0, 
        'mid_rolling_skew': 0, 'mid_rolling_kurtosis': 0, 
        'mid_positive_skew': 0, 'mid_negative_skew': 0, 
        'long_rolling_skew': 0, 'long_rolling_kurtosis': 0, 
        'long_positive_skew': 0, 'long_negative_skew': 0
    }, inplace=True)

    # Calculate historical volatility for each timeframe
    df['historical_volatility_short'] = df['close'].pct_change().rolling(window=short_window).std()
    df['historical_volatility_mid'] = df['close'].pct_change().rolling(window=mid_window).std()
    df['historical_volatility_long'] = df['close'].pct_change().rolling(window=long_window).std()
    df['historical_volatility'] = df['historical_volatility_mid'].fillna(method='bfill').fillna(method='ffill')
    
    # Apply the updated Hurst Exponent calculation for different timeframes
    df['hurst_short'] = df['close'].rolling(window=100).apply(calculate_hurst_exponent, raw=False)
    df['hurst_mid'] = df['close'].rolling(window=150).apply(calculate_hurst_exponent, raw=False)
    df['hurst_long'] = df['close'].rolling(window=200).apply(calculate_hurst_exponent, raw=False)

    # Fill or handle NaN values in Hurst exponent calculations
    df[['hurst_short', 'hurst_mid', 'hurst_long']] = df[['hurst_short', 'hurst_mid', 'hurst_long']].fillna(method='bfill').fillna(method='ffill')

    # Simulate fGBM for different timeframes with additional checks for Hurst exponent
    def safe_fgbm_simulation(window):
        hurst_exponent = calculate_hurst_exponent(window)
        if hurst_exponent <= 0 or hurst_exponent >= 1 or np.isnan(hurst_exponent):
            return np.nan
        # Simulate the price series
        simulated_prices = simulate_fgbm_with_volatility_context(window.iloc[0], hurst_exponent, len(window), df['historical_volatility_mid'].mean())
        return simulated_prices[-1]  # Return the last price in the series

    # Now apply this in the rolling function
    df['fgbm'] = df['close'].rolling(window=window_size).apply(safe_fgbm_simulation, raw=False)

    # Fill remaining NaNs for simulated prices
    df['fgbm'].fillna(method='bfill', inplace=True)

    # Other feature calculations
    df['relative_change'] = df['close'].pct_change()

    # Apply trend strength, volatility, and mean reversion calculations
    df = calculate_trend_strength(df)  # This includes calculating the trend direction
    df = calculate_volatility(df)
    df = calculate_mean_reversion(df)
   
    # Adding new features from fGBM-based strategy
    df = fgbm_based_strategy(df, df['hurst_mid'].mean())

    # Drop rows with any remaining NaN values after all feature creation is done
    df = df.dropna()

    return df

def classify_market_regime(hurst, low_threshold, high_threshold):
    if hurst <= low_threshold:
        return 0  # Low Hurst (stable, low volatility)
    elif low_threshold < hurst <= high_threshold:
        return 1  # Mid Hurst (transitional)
    else:
        return 2  # High Hurst (volatile)


def apply_market_regime_classification(df):
    # Calculate percentiles for Hurst exponent
    low_threshold = np.percentile(df['hurst'], 25)
    high_threshold = np.percentile(df['hurst'], 75)

    # Apply relative thresholds to classify market regimes as 0, 1, or 2
    df['market_regime'] = df['hurst'].apply(classify_market_regime, args=(low_threshold, high_threshold))

    return df


# Step 3: Recalculate Rolling Features with Market Regime
def rolling_features_with_regime(df, short_window=20, mid_window=50, long_window=200, window_size=100):
    # New Feature: Volatility Ratio for short, mid, and long term
    df['volatility_ratio_short'] = df['ATR'] / df['historical_volatility_short'].replace(0, 1e-10)
    df['volatility_ratio_mid'] = df['ATR'] / df['historical_volatility_mid'].replace(0, 1e-10)
    df['volatility_ratio_long'] = df['ATR'] / df['historical_volatility_long'].replace(0, 1e-10)

    # New Feature: MACD-Bollinger Deviation for short, mid, and long term
    df['macd_bollinger_dev_short'] = df['macd'] - df['bollinger_middle'].rolling(window=short_window).mean().replace(0, 1e-10)
    df['macd_bollinger_dev_mid'] = df['macd'] - df['bollinger_middle'].rolling(window=mid_window).mean().replace(0, 1e-10)
    df['macd_bollinger_dev_long'] = df['macd'] - df['bollinger_middle'].rolling(window=long_window).mean().replace(0, 1e-10)

    # New Feature: Volatility-Adjusted Market Regime for short, mid, and long term
    df['volatility_adjusted_regime_short'] = df['market_regime'] * df['volatility_ratio_short'].replace(np.inf, 1e10)
    df['volatility_adjusted_regime_mid'] = df['market_regime'] * df['volatility_ratio_mid'].replace(np.inf, 1e10)
    df['volatility_adjusted_regime_long'] = df['market_regime'] * df['volatility_ratio_long'].replace(np.inf, 1e10)

    # New Feature: RSI-Volatility Interaction for short, mid, and long term
    df['rsi_volatility_interaction_short'] = df['rsi'] * df['historical_volatility_short'].replace(np.inf, 1e10)
    df['rsi_volatility_interaction_mid'] = df['rsi'] * df['historical_volatility_mid'].replace(np.inf, 1e10)
    df['rsi_volatility_interaction_long'] = df['rsi'] * df['historical_volatility_long'].replace(np.inf, 1e10)

    # New Feature: Relative Price Change with Momentum for short, mid, and long term
    df['rel_price_momentum_short'] = (df['close'] - df['bollinger_middle'].replace(0, 1e-10)) * df['rsi'].rolling(window=short_window).mean()
    df['rel_price_momentum_mid'] = (df['close'] - df['bollinger_middle'].replace(0, 1e-10)) * df['rsi'].rolling(window=mid_window).mean()
    df['rel_price_momentum_long'] = (df['close'] - df['bollinger_middle'].replace(0, 1e-10)) * df['rsi'].rolling(window=long_window).mean()

    # New Feature: Spread-Adjusted ATR for short, mid, and long term
    df['spread_adjusted_atr_short'] = df['ATR'] * df['spread'].rolling(window=short_window).mean().replace(np.inf, 1e10)
    df['spread_adjusted_atr_mid'] = df['ATR'] * df['spread'].rolling(window=mid_window).mean().replace(np.inf, 1e10)
    df['spread_adjusted_atr_long'] = df['ATR'] * df['spread'].rolling(window=long_window).mean().replace(np.inf, 1e10)

    # New Feature: Composite Feature Interaction for short, mid, and long term
    df['composite_feature_short'] = df['rsi'] * df['ATR'] / df['market_regime'].replace(0, 1e-10).rolling(window=short_window).mean()
    df['composite_feature_mid'] = df['rsi'] * df['ATR'] / df['market_regime'].replace(0, 1e-10).rolling(window=mid_window).mean()
    df['composite_feature_long'] = df['rsi'] * df['ATR'] / df['market_regime'].replace(0, 1e-10).rolling(window=long_window).mean()

    return df

def calculate_extension_score(df):
    # Normalize RSI to a 0-1 scale
    df['rsi_normalized'] = (df['rsi'] - 30) / (70 - 30)
    df['rsi_normalized'] = np.clip(df['rsi_normalized'], 0, 1)  # Ensure it's between 0 and 1

    # Normalize MACD divergence
    df['macd_normalized'] = (df['macd_hist'] - df['macd_hist'].min()) / (df['macd_hist'].max() - df['macd_hist'].min())

    # Use Hurst exponent as-is (between 0 and 1)
    df['hurst_normalized'] = np.clip(df['hurst'], 0, 1)

    # Calculate composite score (weighing the indicators)
    df['extension_score'] = (0.4 * df['rsi_normalized']) + (0.3 * df['macd_normalized']) + (0.3 * df['hurst_normalized'])
    df['adjusted_extension_score'] = np.where(df['hurst'] < 0.5, df['extension_score'] * 1.2, df['extension_score'] * 0.8)

    return df

