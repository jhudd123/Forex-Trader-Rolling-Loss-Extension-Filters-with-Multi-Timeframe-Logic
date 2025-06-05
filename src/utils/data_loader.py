import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib as ta


# Initialize MT5
def initialize_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    print("MT5 initialized successfully")
    return True

# Fetch historical data
def get_data(symbol, num_rows, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_rows)
    if rates is None:
        print(f"Failed to retrieve data for {symbol}.")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Dynamically determine the precision of the 'close' prices
    precision = len(str(df['close'].iloc[0]).split('.')[1])
    factor = 10 ** precision

    # Adjust spread based on detected precision and fill missing values
    df['spread'] = df['spread'] / factor
    # Replace 0s in 'spread' column with NaN
    df['spread'] = df['spread'].replace(0, np.nan)
   
    # Forward fill missing values in 'spread' column
    df['spread'] = df['spread'].ffill()
   
    # Backward fill missing values in 'spread' column
    df['spread'] = df['spread'].bfill()

   
    # Calculate ATR (volatility) based on high, low, and close
    df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)


    return df