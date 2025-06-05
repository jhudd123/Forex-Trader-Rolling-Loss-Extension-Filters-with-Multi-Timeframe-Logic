Forex Trader: Rolling Loss & Extension Filters with Multi-Timeframe Logic

This project implements a modular machine learning-based trading strategy designed for the foreign exchange (forex) market using MetaTrader 5 (MT5) price data. The approach combines robust feature engineering, dynamic regime filtering, class balancing (via SMOTE), and ensemble-based forecasting using LightGBM classifiers.

🧠 Key Features

📊 Feature Engineering

Multi-timeframe Analysis: Uses one primary and one augmenting timeframe (e.g., 1H with daily) to extract richer signal context.

Custom Indicators: Generates features such as:

Fractal alignment

Market regime classification

Trend strength scoring

Extension scoring (normalized positioning within volatility range)

Rolling feature transformations are applied both with and without trend regime conditioning.

📉 Target Construction

Targets are assigned based on future direction classification using custom logic tied to market context (trend + extension zones).

🔁 Model Architecture

Model Type: LightGBM (gradient boosting decision trees)

Objective: Multiclass classification (Long, Short, Hold)

Training Strategy:

Time-series-aware cross-validation (TimeSeriesSplit)

Hyperparameter tuning via Optuna

Class balancing using SMOTE to mitigate directional imbalance

Feature selection handled dynamically per fold

🔍 Evaluation Enhancements

📊 Performance Metrics

Profit Factor, Win Rate, and Drawdown are used alongside standard model metrics.

Custom Trade Simulation Engine is embedded during validation and test phases to replicate realistic forward execution.

🧠 Rolling Loss Filter (Novel)

The strategy uses a trend-regime-specific rolling loss tracker:

Identifies poor-performing market states

Temporarily blocks trades when local conditions degrade

Resumes trading once conditions stabilize

Tracks blocked vs. resumed trade outcomes for research

🔧 Customizability

You can easily adjust the following:

⚙️ Currency pairs (primary + augmenting symbols)

🕒 Timeframes (any supported by MT5)

📈 Feature combinations or targets

🧪 Backtest windows and filter thresholds

📂 Project Structure

Forex Trader: Rolling Loss & Extension Filters with Multi-Timeframe Logic/
│
├── main.py                                # Entry point for the strategy pipeline
├── config.py                              # Global config values (e.g., RANDOM_SEED)
├── requirements.txt                       # Core dependencies
├── README.md                              # You're reading this
│
├── final_models_AUDUSD.joblib             # Trained ensemble model (LightGBM)
├── optimized_params_AUDUSD.joblib         # Saved Optuna hyperparameters
├── positive_combinationsAUDUSD.joblib     # Positive trend/regime combos
│
├── test_equity_curve.png                  # Output: test set equity curve plot
├── test_metrics_summary.txt               # Output: strategy performance metrics
│
├── data/                                  # Optional: raw CSV input or staging folder
├── src/                                   # Source code
│   ├── features/                          # Feature engineering logic
│   ├── models/                            # Training, simulation, optimization
│   └── utils/                             # Data loaders, metrics, drawdown functions
│
└── .venv/                                 # Virtual environment (excluded via .gitignore)

🚀 How to Run

Step 1: Setup virtual environment (recommended)

python -m venv .venv
.venv\Scripts\activate  # Windows

Step 2: Install required packages

pip install -r requirements.txt

Step 3: Install TA-Lib manually (required for indicators on Windows)

Due to compilation issues, use a precompiled wheel:

Visit: https://github.com/cgohlke/talib-build/releases

Download the wheel matching your Python version (e.g., TA_Lib‐0.4.0‐cp311‐cp311‐win_amd64.whl)

Install it:

pip install path\to\TA_Lib‐0.4.0‐cp311‐cp311‐win_amd64.whl

Step 4: Run the model

python main.py

📡 Data Source: MetaTrader 5 (MT5)

This project pulls live or historical data via the MT5 API:

Requires MetaTrader 5 installed on your machine

You must have an MT5 brokerage account (even if demo)

Ensure you are logged in with the desired symbols added to your Market Watch

✅ Dependencies (Highlights)

lightgbm: Gradient boosting classifier

optuna: Hyperparameter optimization

imbalanced-learn: For SMOTE class balancing

MetaTrader5: For real-time and historical market data access

ta-lib: Technical indicators

joblib: For model persistence

pandas, numpy, matplotlib: Core scientific stack

📤 Output Files

After running the strategy, two key outputs will be generated in the project root:

- **`test_equity_curve.png`** — A visual plot of the equity curve over the test window, showing cumulative returns per trade.
- **`test_metrics_summary.txt`** — Text summary of key performance metrics, including:
  - Total trades, win rate, profit factor, and max drawdown
  - Blocked/resumed trade statistics via the rolling loss filter
  - Performance by trend direction and market regime

These files are automatically saved when running `main.py` and can be used to validate strategy robustness or visualize results.


🧪 Portfolio Value

This project demonstrates:

Practical experience with time-series ML

Domain-specific feature engineering

Custom pipeline design for financial forecasting

Use of real-world data via a broker API

Model risk management techniques (filters, imbalance correction)



📄 License

Educational use only. Not financial advice.

