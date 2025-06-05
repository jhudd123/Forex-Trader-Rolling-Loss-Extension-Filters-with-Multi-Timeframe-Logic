import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from utils.utils import (
    calculate_profit_factor,
    get_positive_combinations,
    print_detailed_performance,
    check_data_duration,
    simulate_trade,
    summarize_performance_by_trend_and_regime
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import RANDOM_SEED



# Check for class imbalance prior to model training
def check_class_imbalance(df):
    print("Class distribution in the target variable:")
    print(df['target'].value_counts(normalize=True))


# Balance classes using SMOTE (Synthetic Minority Over-sampling Technique)
def balance_classes(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# Train_ensemble_with_folds to use optimized hyperparameters
def train_ensemble_with_folds(df, n_splits=5, optimized_params=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    positive_combinations_set = set()  # To collect positive combinations from validation results

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(df)):
        X_train, X_valid = df.iloc[train_idx].drop(columns=['target']), df.iloc[valid_idx].drop(columns=['target'])
        y_train, y_valid = df.iloc[train_idx]['target'], df.iloc[valid_idx]['target']

        # Print trend direction for fold debugging
        print(f"Fold {fold}: Trend Direction - {df.iloc[valid_idx]['trend_direction'].unique()}")

        # Train the model
        model = lgb.LGBMClassifier(**optimized_params, random_state=RANDOM_SEED, objective='multiclass', num_class=3)
        model.fit(X_train, y_train)

        # Simulate trading on the validation set
        valid_results, valid_performance = simulate_trading_on_validation(df.iloc[valid_idx], model, X_train.columns)

        # Summarize performance by trend direction and market regime
        performance_df = summarize_performance_by_trend_and_regime(df.iloc[valid_idx], valid_results)

        # Extract positive combinations
        positive_combinations = get_positive_combinations(performance_df)
        positive_combinations_set.update(positive_combinations)  # Aggregate positive combinations across all folds

        # Store the model and selected features
        models.append((model, X_train.columns))

        print(f"Fold {fold} - Positive Combinations: {positive_combinations}")

    # Save aggregated positive combinations
    joblib.dump(positive_combinations_set, "positive_combinationsAUDUSD.joblib")  # Save to a file

    # Print final list of positive combinations
    print(f"Final List of Positive Combinations: {positive_combinations_set}")

    return models, positive_combinations_set



def apply_ensemble_to_test_with_filter(df_test, models, positive_combinations, extension_threshold=0.5):
    balance = 1000
    equity_curve = [balance]
    results = []
    open_trades = []
    trade_durations = []
    combined_predictions = np.zeros((len(df_test), len(models)))
    extension_threshold = np.percentile(df_test['adjusted_extension_score'], 75)

    # Initialize counters
    total_trades_count = 0
    blocked_trades_count = 0
    resumed_trades_count = 0
    blocked_trade_wins = 0
    blocked_trade_losses = 0
    long_trades_count = 0
    short_trades_count = 0

    trend_regime_loss_tracker = {}
    window_size = 8
    loss_threshold = 0.5

    for model_idx, (model, selected_features) in enumerate(models):
        predictions = model.predict(df_test[selected_features])
        combined_predictions[:, model_idx] = predictions

    majority_long = np.sum(combined_predictions == 1, axis=1)
    majority_short = np.sum(combined_predictions == 0, axis=1)
    majority_hold = np.sum(combined_predictions == 2, axis=1)

    final_predictions = np.where(majority_hold > (len(models) / 2), 2,
                         np.where(majority_long > (len(models) / 2), 1,
                         np.where(majority_short > (len(models) / 2), 0, -1)))

    for i in range(1, len(df_test)):
        total_trades_count += 1

        if df_test['adjusted_extension_score'].iloc[i] > extension_threshold:
            blocked_trades_count += 1
            continue

        trend_regime_combination = (df_test['trend_direction'].iloc[i], df_test['market_regime'].iloc[i])
        if trend_regime_combination not in trend_regime_loss_tracker:
            trend_regime_loss_tracker[trend_regime_combination] = []

        if final_predictions[i] == 2:
            continue

        if len(trend_regime_loss_tracker[trend_regime_combination]) >= window_size:
            recent_losses = trend_regime_loss_tracker[trend_regime_combination][-window_size:]
            if sum(recent_losses) / window_size >= loss_threshold:
                blocked_trades_count += 1
                trade_result = simulate_trade(df_test, i, final_predictions[i])
                if trade_result > 0:
                    blocked_trade_wins += 1
                else:
                    blocked_trade_losses += 1
                trend_regime_loss_tracker[trend_regime_combination].append(1 if trade_result < 0 else 0)
                continue
        else:
            resumed_trades_count += 1

        entry_price = df_test['close'].iloc[i]
        atr = df_test['ATR'].iloc[i]
        spread = df_test['spread'].iloc[i]
        trade_result = 0

        if final_predictions[i] == 1:
            sl = entry_price - (atr * 2) - spread
            tp = entry_price + (atr * 2) + spread
            open_trades.append({'entry_price': entry_price, 'sl': sl, 'tp': tp, 'open_period': i, 'trade_type': 'long'})
            long_trades_count += 1

        elif final_predictions[i] == 0:
            sl = entry_price + (atr * 2) + spread
            tp = entry_price - (atr * 2) - spread
            open_trades.append({'entry_price': entry_price, 'sl': sl, 'tp': tp, 'open_period': i, 'trade_type': 'short'})
            short_trades_count += 1

        for trade in open_trades[:]:
            if trade['trade_type'] == 'long':
                if df_test['high'].iloc[i] >= trade['tp']:
                    trade_result = (trade['tp'] - trade['entry_price']) / trade['entry_price']
                    results.append(trade_result)
                    balance += balance * trade_result
                    equity_curve.append(balance)
                    trend_regime_loss_tracker[trend_regime_combination].append(0 if trade_result > 0 else 1)
                    trade_durations.append(i - trade['open_period'])
                    open_trades.remove(trade)
                elif df_test['low'].iloc[i] <= trade['sl']:
                    trade_result = (trade['sl'] - trade['entry_price']) / trade['entry_price']
                    results.append(trade_result)
                    balance += balance * trade_result
                    equity_curve.append(balance)
                    trend_regime_loss_tracker[trend_regime_combination].append(1)
                    trade_durations.append(i - trade['open_period'])
                    open_trades.remove(trade)
            elif trade['trade_type'] == 'short':
                if df_test['low'].iloc[i] <= trade['tp']:
                    trade_result = (trade['entry_price'] - trade['tp']) / trade['entry_price']
                    results.append(trade_result)
                    balance += balance * trade_result
                    equity_curve.append(balance)
                    trend_regime_loss_tracker[trend_regime_combination].append(0 if trade_result > 0 else 1)
                    trade_durations.append(i - trade['open_period'])
                    open_trades.remove(trade)
                elif df_test['high'].iloc[i] >= trade['sl']:
                    trade_result = (trade['entry_price'] - trade['sl']) / trade['entry_price']
                    results.append(trade_result)
                    balance += balance * trade_result
                    equity_curve.append(balance)
                    trend_regime_loss_tracker[trend_regime_combination].append(1)
                    trade_durations.append(i - trade['open_period'])
                    open_trades.remove(trade)

    gross_profit = sum(r for r in results if r > 0)
    gross_loss = abs(sum(r for r in results if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.max(np.maximum.accumulate(equity_curve))
    win_rate = len([r for r in results if r > 0]) / len(results) if results else 0
    avg_trade_duration = np.mean(trade_durations) if trade_durations else np.nan

    metrics_output = []
    metrics_output.append(f"Total Trades: {total_trades_count}")
    metrics_output.append(f"Blocked Trades by Rolling Loss Filter: {blocked_trades_count}")
    metrics_output.append(f"Resumed Trades After Blocking: {resumed_trades_count}")
    metrics_output.append(f"Blocked Trade Wins: {blocked_trade_wins}")
    metrics_output.append(f"Blocked Trade Losses: {blocked_trade_losses}")
    metrics_output.append(f"Executed Long Trades: {long_trades_count}")
    metrics_output.append(f"Executed Short Trades: {short_trades_count}")
    metrics_output.append(f"Profit Factor: {profit_factor:.2f}")
    metrics_output.append(f"Win Rate: {win_rate:.2%}")
    metrics_output.append(f"Max Drawdown: {max_drawdown:.4f}")
    metrics_output.append(f"Average Trade Duration: {avg_trade_duration:.2f} periods")

    # Still print for real-time visibility
    print("\n".join(metrics_output))
    print_detailed_performance(equity_curve, df_test.index)
    check_data_duration(df_test.index)

    return results, trend_regime_loss_tracker, equity_curve, avg_trade_duration, metrics_output



def simulate_trading_on_validation(df_valid, model, selected_features):
    """
    Simulates trading on validation data using the given model and selected features.
    Returns the trading results and performance summary.
    """
    balance = 1000
    equity_curve = [balance]
    results = []
    open_trades = []
    combined_predictions = np.zeros(len(df_valid))

    # Predict using the model
    predictions = model.predict(df_valid[selected_features])
    combined_predictions = predictions

    for i in range(1, len(df_valid)):
        entry_price = df_valid['close'].iloc[i]
        atr = df_valid['ATR'].iloc[i]
        spread = df_valid['spread'].iloc[i]

        # Skip trades when the prediction is "hold" (2)
        if combined_predictions[i] == 2:  # Hold trade
            continue

        if combined_predictions[i] == 1:  # Long trade
            sl = entry_price - (atr * 2) - spread
            tp = entry_price + (atr * 2) + spread
            open_trades.append({'entry_price': entry_price, 'sl': sl, 'tp': tp, 'open_period': i, 'trade_type': 'long'})

        elif combined_predictions[i] == 0:  # Short trade
            sl = entry_price + (atr * 2) + spread
            tp = entry_price - (atr * 2) - spread
            open_trades.append({'entry_price': entry_price, 'sl': sl, 'tp': tp, 'open_period': i, 'trade_type': 'short'})

        # Manage open trades (check if SL/TP hit)
        for trade in open_trades[:]:
            if trade['trade_type'] == 'long':
                if df_valid['high'].iloc[i] >= trade['tp']:
                    trade_result = (trade['tp'] - trade['entry_price']) / trade['entry_price']
                    results.append(trade_result)
                    balance += balance * trade_result
                    equity_curve.append(balance)
                    open_trades.remove(trade)
                elif df_valid['low'].iloc[i] <= trade['sl']:
                    trade_result = (trade['sl'] - trade['entry_price']) / trade['entry_price']
                    results.append(trade_result)
                    balance += balance * trade_result
                    equity_curve.append(balance)
                    open_trades.remove(trade)

            elif trade['trade_type'] == 'short':
                if df_valid['low'].iloc[i] <= trade['tp']:
                    trade_result = (trade['entry_price'] - trade['tp']) / trade['entry_price']
                    results.append(trade_result)
                    balance += balance * trade_result
                    equity_curve.append(balance)
                    open_trades.remove(trade)
                elif df_valid['high'].iloc[i] >= trade['sl']:
                    trade_result = (trade['entry_price'] - trade['sl']) / trade['entry_price']
                    results.append(trade_result)
                    balance += balance * trade_result
                    equity_curve.append(balance)
                    open_trades.remove(trade)

    # Summarize performance
    gross_profit = sum([r for r in results if r > 0])
    gross_loss = abs(sum([r for r in results if r < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = len([r for r in results if r > 0]) / len(results) if results else 0

    performance_summary = pd.DataFrame({
        'Profit Factor': [profit_factor],
        'Win Rate': [win_rate],
        'Number of Trades': [len(results)]
    })

    return results, performance_summary

# Save models and parameters without saving the feature names
def save_models_and_params(models, optimized_params, model_path="final_models_AUDUSD.joblib", params_path="optimized_params_AUDUSD.joblib"):
    # Save the models
    joblib.dump(models, model_path)

    # Save the optimized hyperparameters
    joblib.dump(optimized_params, params_path)


# Now use this scorer in your cross-validation
def objective(trial, df_train):
    param = {
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),  # Tune learning rate
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # Adjusting num_leaves
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),  # Tuning min_data_in_leaf
        'max_depth': trial.suggest_int('max_depth', 3, 10),  # Added max_depth to control tree depth
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1.0),
        'objective': 'multiclass',
        'num_class': 3
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, valid_idx in tscv.split(df_train):
        X_train, X_valid = df_train.iloc[train_idx].drop(columns=['target']), df_train.iloc[valid_idx].drop(columns=['target'])
        y_train, y_valid = df_train.iloc[train_idx]['target'], df_train.iloc[valid_idx]['target']

        model = lgb.LGBMClassifier(**param, random_state=RANDOM_SEED)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='multi_logloss')

        y_pred = model.predict(X_valid)
        score = calculate_profit_factor(y_valid, y_pred, df_train.iloc[valid_idx])
        scores.append(score)

    return np.mean(scores)



def optimize_hyperparameters(df_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df_train), n_trials=100)  # Adjust the number of trials as needed

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


