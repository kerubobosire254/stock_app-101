# 📈 Stock Price Prediction Intelligence Terminal

A Streamlit dashboard for Netflix stock analysis combining LSTM deep learning predictions, Monte Carlo simulations, and technical indicators — all in a clean, white terminal-style UI.

---

## Features

- **Price History** — Closing price with MA10/MA50/MA200 overlays, daily returns, volume, and candlestick view
- **Technical Analysis** — Bollinger Bands (20, ±2σ), RSI (14), MACD (12/26/9), Bollinger %B, and ATR (14)
- **Monte Carlo Forecast** — Geometric Brownian Motion simulation with configurable horizon (10–90 days) and simulation count (100–2,000)
- **LSTM Predictions** — Trained deep learning model showing actual vs predicted prices, loss curves, scatter plot, and residual analysis

---

## Project Structure

```
stock_app/
│
├── app.py                    # Main Streamlit dashboard
├── train_and_save.py         # Model training script
├── Netflix Stock Prices.csv  # Source data
│
├── netflix_lstm_model.keras  # Generated after training
├── netflix_scaler.pkl        # Generated after training
└── netflix_model_meta.json   # Generated after training
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn tensorflow yfinance
```

### 2. Train the LSTM model

Place `Netflix Stock Prices.csv` in the project folder, then run:

```bash
python train_and_save.py
```

This will generate three files the dashboard needs:
- `netflix_lstm_model.keras`
- `netflix_scaler.pkl`
- `netflix_model_meta.json`

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model Architecture

The LSTM model is trained on 5 features engineered from raw OHLCV data:

| Feature | Description |
|---|---|
| `Close` | Raw closing price |
| `Volume` | Log-transformed volume (`log1p`) |
| `MA10` | 10-day moving average |
| `MA50` | 50-day moving average |
| `Return` | Daily percentage change |

**Architecture:**
```
Input (60 timesteps × 5 features)
  → LSTM(256, return_sequences=True)
  → Dropout(0.2)
  → LSTM(128)
  → Dropout(0.2)
  → Dense(1)
```

**Training config:**
- Optimizer: Adam (lr=0.001, clipnorm=1.0)
- Loss: MSE
- Early stopping: patience=10 on val_loss
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Train/test split: 80/20 (no shuffle — temporal order preserved)
- Sequence length: 60 trading days

---

## Data Sources

The dashboard supports three data modes selectable from the sidebar:

| Mode | Description |
|---|---|
| 📊 Demo data | Synthetic NFLX-like history (2019–2024), generated on the fly — no file needed |
| 📂 Upload CSV | Upload your own OHLCV CSV with columns: `Date, Open, High, Low, Close, Volume` |
| 🌐 Fetch live | Pull real-time data via `yfinance` for any ticker and time period |

---

## Monte Carlo Simulation

Simulations follow **Geometric Brownian Motion** using historical log-return statistics (μ, σ) derived from the loaded dataset. The forecast fan shows 50% and 90% confidence intervals across all simulated paths.

> ⚠️ Simulations do not account for regime changes, macro events, or mean reversion. Not financial advice.

---

## Requirements

| Package | Purpose |
|---|---|
| `streamlit` | Dashboard framework |
| `pandas` / `numpy` | Data processing |
| `plotly` | Interactive charts |
| `scikit-learn` | MinMaxScaler |
| `tensorflow` | LSTM model training & inference |
| `yfinance` *(optional)* | Live data fetching |

> **Windows note:** TensorFlow ≥ 2.11 does not support native Windows GPU. Use WSL2 or the TensorFlow-DirectML plugin for GPU acceleration.
