# Forex Trader: Rolling Loss & Extension Filters with Multi-Timeframe Logic

This project demonstrates a simple machine learning-based trading strategy using multi-timeframe logic and custom technical indicators. It is implemented in a modular format for clarity and scalability.

## 📌 Project Overview

The strategy focuses on the AUD/USD forex pair and utilizes a series of engineered features such as:
- Rolling loss indicator
- Extension filter
- RSI and MFI proxies
- Binary classification of bullish/bearish movements

The model used is LightGBM, trained on processed features to predict short-term price direction.

## 📂 Directory Structure

```
forex_modular_project/
│
├── config.py                 # Central configuration
├── main.py                   # Entry point to run the pipeline
├── requirements.txt          # 
│
├── utils/
│   ├── data_loader.py        # Load and preprocess raw data
│   ├── features.py           # Feature engineering functions
│   ├── model.py              # Model training and evaluation
│   └── utils.py              # Miscellaneous helpers (placeholder)
│
└── data/
    └── AUDUSD.csv            # Example CSV (not included in this repo)

```

## 🚀 How to Run

1. Place your historical AUD/USD data as `AUDUSD.csv` in the `data/` folder. It must contain columns: `date`, `open`, `high`, `low`, `close`, `volume`.
2. Install the requirements:
```bash
pip install pandas numpy lightgbm scikit-learn
```
3. Run the pipeline:
```bash
python main.py
```

## 📊 Output

After training, the script will print a classification report and metrics on test data, showing model performance.

## 📈 Features Used

- `rolling_loss`: Custom indicator based on recent price dips.
- `extension_filter`: Measures price extension vs moving average.
- `mfi_proxy`: Simplified Money Flow Index.
- `rsi_proxy`: Simplified Relative Strength Index.

## 🛠️ Future Work

- Additional parameter tuning
- Integrate additional currency pairs or timeframes

## 📄 License

This project is for educational and illustrative purposes only. Not financial advice.

---

Crafted for a modular, professional portfolio. Contributions and forks welcome.
