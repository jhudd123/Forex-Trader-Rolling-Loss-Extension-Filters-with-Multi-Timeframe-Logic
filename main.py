import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from utils.data_loader import get_data, initialize_mt5

from features import (
    prepare_features_with_fractal,
    rolling_features_without_regime,
    apply_market_regime_classification,
    rolling_features_with_regime,
    calculate_trend_strength,
    calculate_extension_score,
    assign_targets_based_on_trend
)
from models import (
    check_class_imbalance,
    balance_classes,
    optimize_hyperparameters,
    train_ensemble_with_folds,
    apply_ensemble_to_test_with_filter,
    save_models_and_params
)
from utils import (
    check_constant_columns,
    plot_equity_curve,
    summarize_performance_by_trend_and_regime
)


def main():
    if not initialize_mt5():
        return

    symbols = ['AUDUSD', 'GBPUSD', 'CADJPY']
    timeframe = mt5.TIMEFRAME_M30
    num_rows = 5000

    df1 = get_data(symbols[0], num_rows, timeframe)
    df2 = get_data(symbols[1], num_rows, timeframe)
    df3 = get_data(symbols[2], num_rows, timeframe)

    if df1.empty or df2.empty or df3.empty:
        print("No data available to process.")
        mt5.shutdown()
        return

    df = prepare_features_with_fractal(df1, df2, df3)
    df = rolling_features_without_regime(df)
    df = apply_market_regime_classification(df)
    df = rolling_features_with_regime(df)
    df = calculate_trend_strength(df)
    df = calculate_extension_score(df)

    constant_cols = check_constant_columns(df)

    print(f"Length before assignment: {len(df)}")
    df = assign_targets_based_on_trend(df)
    print(f"Length after assignment: {len(df)}")

    check_class_imbalance(df)

    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(df_train.drop(columns=['target']))
    y_train = df_train['target']

    X_train_resampled, y_train_resampled = balance_classes(X_train_imputed, y_train)

    df_train_resampled = pd.DataFrame(X_train_resampled, columns=df_train.drop(columns=['target']).columns)
    df_train_resampled['target'] = y_train_resampled

    optimized_params = optimize_hyperparameters(df_train_resampled)

    models, positive_combinations = train_ensemble_with_folds(df_train_resampled, optimized_params=optimized_params)

    save_models_and_params(models, optimized_params)

    # FIXED: Now correctly unpack metrics_output
    combined_results, rolling_loss_percent, final_equity_curve, avg_trade_duration, metrics_output = apply_ensemble_to_test_with_filter(
        df_test, models, positive_combinations
    )

    # Save equity curve plot
    plt.figure()
    plot_equity_curve(final_equity_curve, show_plot=False)
    plt.savefig("test_equity_curve.png")

    # Summarize by trend & regime with output
    performance_df, summary_output = summarize_performance_by_trend_and_regime(
        df_test, combined_results, return_output=True
    )

    # Save metrics to file
    all_metrics = metrics_output + summary_output
    with open("test_metrics_summary.txt", "w") as f:
        f.write("\n".join(all_metrics))

    print("✅ Saved metrics to test_metrics_summary.txt and equity curve to test_equity_curve.png")

    mt5.shutdown()


if __name__ == "__main__":
    main()
