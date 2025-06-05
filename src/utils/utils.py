import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_detailed_performance(equity_curve, date_index, return_output=False):
    """
    Print and optionally return daily and overall performance statistics.
    
    Parameters:
    - equity_curve: list or np.array of equity values
    - date_index: pd.DatetimeIndex aligned with equity_curve
    - return_output: if True, returns the performance metrics as a list of strings
    """
    output = []

    # Ensure matching length
    min_length = min(len(equity_curve), len(date_index))
    equity_curve = equity_curve[:min_length]
    date_index = date_index[:min_length]

    max_drawdown, longest_drawdown_duration = calculate_drawdown(equity_curve)
    daily_wins, daily_losses, daily_win_streak, daily_loss_streak = calculate_daily_performance(equity_curve, date_index)

    output.append(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
    output.append(f"Longest Drawdown Duration: {longest_drawdown_duration} periods")
    output.append(f"Daily Wins: {daily_wins}, Daily Losses: {daily_losses}")
    output.append(f"Longest Daily Win Streak: {daily_win_streak}, Longest Daily Loss Streak: {daily_loss_streak}")

    if return_output:
        return output
    else:
        print("\n".join(output))

# Check for constant columns
def check_constant_columns(df):
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_columns.append(col)
   
    if constant_columns:
        print(f"The following columns have constant values:\n{constant_columns}")
    else:
        print("No constant columns found.")
   
    return constant_columns
def calculate_drawdown(equity_curve):
    """
    Calculate drawdown sizes and durations.
    """
    max_equity = np.maximum.accumulate(equity_curve)
    drawdowns = max_equity - equity_curve
    drawdown_percentage = drawdowns / max_equity
    max_drawdown = np.max(drawdown_percentage)

    # Calculate drawdown durations
    drawdown_durations = np.where(drawdowns > 0, 1, 0)
    drawdown_streaks = np.diff(np.flatnonzero(np.concatenate(([drawdown_durations[0]],
                                                              drawdown_durations[:-1] != drawdown_durations[1:], [True]))))
    longest_drawdown_duration = max(drawdown_streaks) if len(drawdown_streaks) > 0 else 0

    return max_drawdown, longest_drawdown_duration


def calculate_win_loss_streaks(equity_curve, date_index):
    """
    Calculate winning and losing streaks.
    """
    results = np.diff(equity_curve)
    win_streaks, lose_streaks = [], []
    current_win_streak, current_loss_streak = 0, 0

    for i, result in enumerate(results):
        if result > 0:
            current_win_streak += 1
            if current_loss_streak > 0:
                lose_streaks.append(current_loss_streak)
                current_loss_streak = 0
        elif result < 0:
            current_loss_streak += 1
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
                current_win_streak = 0
    
    win_streaks.append(current_win_streak)
    lose_streaks.append(current_loss_streak)

    return win_streaks, lose_streaks


def calculate_weekly_performance(equity_curve, date_index):
    """
    Calculate weekly performance: win/loss streaks, winning/losing days.
    """
    df = pd.DataFrame({'equity': equity_curve}, index=date_index)

    # Resample to daily to ensure proper weekly aggregation
    df = df.resample('D').ffill()  # Forward fill missing data for daily frequency
    
    # Now resample to weekly returns
    df['weekly_returns'] = df['equity'].pct_change().resample('W').sum()
    
    # Debugging: print out weekly returns
    print("Weekly Returns After Daily Resample:")
    print(df['weekly_returns'])

    weekly_wins = (df['weekly_returns'] > 0).sum()
    weekly_losses = (df['weekly_returns'] < 0).sum()
    weekly_win_streak = (df['weekly_returns'] > 0).astype(int).groupby((df['weekly_returns'] <= 0).cumsum()).cumsum().max()
    weekly_loss_streak = (df['weekly_returns'] < 0).astype(int).groupby((df['weekly_returns'] >= 0).cumsum()).cumsum().max()

    return weekly_wins, weekly_losses, weekly_win_streak, weekly_loss_streak

def check_data_duration(date_index):
    start_date = date_index.min()
    end_date = date_index.max()
    total_days = (end_date - start_date).days
    total_periods = len(date_index)
    
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Total Days: {total_days}")
    print(f"Total Periods: {total_periods}")

    return total_days, total_periods


def calculate_daily_performance(equity_curve, date_index):
    """
    Calculate daily performance metrics: win/loss streaks, winning/losing days, 
    based on actual daily returns, ensuring each day is considered once.
    """
    # Ensure equity_curve is aligned with the date_index and create a DataFrame
    df = pd.DataFrame({'equity': equity_curve}, index=date_index)
    
    # Resample to daily frequency using the last value of each day (to reflect end-of-day equity)
    df_daily = df.resample('D').last().dropna()  # Drop NaN to avoid gaps

    # Calculate daily returns (based on closing equity of each day)
    df_daily['daily_returns'] = df_daily['equity'].pct_change()

    # Print the equity curve at daily intervals to check for activity
    print("Daily Equity Values:")
    print(df_daily['equity'])
    
    # Identify the winning and losing days based on daily returns
    winning_days = df_daily[df_daily['daily_returns'] > 0].index
    losing_days = df_daily[df_daily['daily_returns'] < 0].index

    # Print the unique dates for winning and losing days
    print("Winning Days:")
    print(winning_days)
    
    print("Losing Days:")
    print(losing_days)

    # Count the number of winning and losing days
    daily_wins = len(winning_days)
    daily_losses = len(losing_days)

    # Calculate the longest win and loss streaks based on consecutive daily returns
    daily_win_streak = (df_daily['daily_returns'] > 0).astype(int).groupby((df_daily['daily_returns'] <= 0).cumsum()).cumsum().max()
    daily_loss_streak = (df_daily['daily_returns'] < 0).astype(int).groupby((df_daily['daily_returns'] >= 0).cumsum()).cumsum().max()

    return daily_wins, daily_losses, daily_win_streak, daily_loss_streak


def get_positive_combinations(performance_df):
    # Replace inf profit factors with 0
    performance_df['Profit Factor'].replace([np.inf, -np.inf], 0, inplace=True)
   
    # Filter for combinations with positive profit factor and ensure Number of Trades > 0
    positive_combinations = performance_df[(performance_df['Profit Factor'] > 1.2) &
                                           (performance_df['Number of Trades'] > 0)][['Trend Direction', 'Market Regime']]
    return set([tuple(x) for x in positive_combinations.values])

def simulate_trade(df_test, index, trade_type):
    """ Simulates a trade result based on the historical data without actually executing it. """
    
    if trade_type == 2:  # Handle "hold" prediction
        return 0  # No trade executed for a hold prediction

    # Proceed with trade simulation logic for long/short trades
    if index + 1 >= len(df_test):
        return 0  # No further data, so no trade

    entry_price = df_test['close'].iloc[index]
    atr = df_test['ATR'].iloc[index]
    spread = df_test['spread'].iloc[index]

    if trade_type == 1:  # Simulate a long trade
        sl = entry_price - (atr * 2) - spread
        tp = entry_price + (atr * 2) + spread
        if df_test['high'].iloc[index + 1] >= tp:  # Take Profit hit
            return (tp - entry_price) / entry_price
        elif df_test['low'].iloc[index + 1] <= sl:  # Stop Loss hit
            return (sl - entry_price) / entry_price
        else:  # Neither SL nor TP hit, assume no change
            return 0

    elif trade_type == 0:  # Simulate a short trade
        sl = entry_price + (atr * 2) + spread
        tp = entry_price - (atr * 2) - spread
        if df_test['low'].iloc[index + 1] <= tp:  # Take Profit hit
            return (entry_price - tp) / entry_price
        elif df_test['high'].iloc[index + 1] >= sl:  # Stop Loss hit
            return (entry_price - sl) / entry_price
        else:  # Neither SL nor TP hit, assume no change
            return 0

    return 0  # Default if no valid trade type



# Plot the final equity curve
def plot_equity_curve(equity_curve, show_plot=True):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve)
    plt.title('Test Set Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Returns')
    
    if show_plot:
        plt.show()




def calculate_profit_factor(y_true, y_pred, df):
    results = []
    grouped = df.groupby(['trend_direction', 'market_regime'])

    for (trend_dir, regime), group in grouped:
        indices = group.index  # get the indices of the group
        for i in range(len(indices) - 1):  # Adjusted to ensure i + 1 is within bounds
            idx = indices[i]

            entry_price = group['close'].iloc[i]
            atr = group['ATR'].iloc[i]
            spread = group['spread'].iloc[i]

            if y_pred[i] == 1:  # Long trade
                tp = entry_price + (atr * 2) + spread
                sl = entry_price - (atr * 2) - spread

                if group['high'].iloc[i + 1] >= tp:
                    results.append((tp - entry_price) / entry_price)
                elif group['low'].iloc[i + 1] <= sl:
                    results.append((sl - entry_price) / entry_price)
                else:
                    results.append(-spread / entry_price)

            elif y_pred[i] == 0:  # Short trade
                tp = entry_price - (atr * 2) - spread
                sl = entry_price + (atr * 2) + spread

                if group['low'].iloc[i + 1] <= tp:
                    results.append((entry_price - tp) / entry_price)
                elif group['high'].iloc[i + 1] >= sl:
                    results.append((entry_price - sl) / entry_price)
                else:
                    results.append(-spread / entry_price)

    gross_profit = sum([r for r in results if r > 0])
    gross_loss = abs(sum([r for r in results if r < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    return profit_factor


def analyze_loss_clusters(df, loss_clusters):
    """
    Analyze periods marked as loss clusters to extract insights.
    """
    cluster_analysis = []

    for i in range(len(loss_clusters)):
        if not np.isnan(loss_clusters[i]):
            # Extract market conditions, volatility, trend direction during loss clusters
            cluster_period = df.iloc[i]
            cluster_analysis.append({
                'Index': i,
                'ATR': cluster_period['ATR'],
                'Hurst': cluster_period['hurst'],
                'Trend Direction': cluster_period['trend_direction'],
                'Volatility': cluster_period['historical_volatility'],
                'Market Regime': cluster_period['market_regime'],
            })

    cluster_analysis_df = pd.DataFrame(cluster_analysis)
    return cluster_analysis_df




def summarize_performance_by_trend_and_regime(df_executed_trades, results, return_output=False):
    """
    Summarize performance by trend direction and market regime.

    Parameters:
    - df_executed_trades: DataFrame with executed trades
    - results: list of trade returns
    - return_output: if True, returns printed summary as a list of strings

    Returns:
    - performance_df: DataFrame summary
    - output (optional): List of string outputs
    """
    output = []

    if len(results) < len(df_executed_trades):
        results.extend([0] * (len(df_executed_trades) - len(results)))
    elif len(results) > len(df_executed_trades):
        results = results[:len(df_executed_trades)]

    df_executed_trades = df_executed_trades.copy()
    df_executed_trades.loc[:, 'returns'] = results

    grouped = df_executed_trades.groupby(['trend_direction', 'market_regime'])
    performance_summary = []

    for (trend_dir, regime), group in grouped:
        valid_trades = group.dropna(subset=['returns'])
        total_wins = valid_trades[valid_trades['returns'] > 0]['returns'].sum()
        total_losses = valid_trades[valid_trades['returns'] < 0]['returns'].sum()
        win_rate = len(valid_trades[valid_trades['returns'] > 0]) / len(valid_trades) if len(valid_trades) > 0 else 0
        profit_factor = total_wins / abs(total_losses) if total_losses != 0 else np.inf

        performance_summary.append({
            'Trend Direction': trend_dir,
            'Market Regime': regime,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Number of Trades': len(valid_trades)
        })

    performance_df = pd.DataFrame(performance_summary)

    output.append("Performance by Trend Direction and Market Regime:")
    output.append(performance_df.to_string(index=False))

    if return_output:
        return performance_df, output
    else:
        print("\n".join(output))
        return performance_df





