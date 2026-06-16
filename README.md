# Stock Price Prediction Intelligence 📈

> Built by [Kerubo Bosire](https://linkedin.com/in/kerubo-bosire-364523283) · [GitHub](https://github.com/kerubobosire254)

---
***Live Demo*** https://stockapp-101.streamlit.app/

### Snippet of the App

<img width="1358" height="570" alt="image" src="https://github.com/user-attachments/assets/8bf0b20d-5f72-4368-bab3-daf3f81af675" />



## The Problem — And Why It Matters

Imagine you have some savings. Maybe it's KES 50,000. Maybe more. You've heard that investing in stocks — companies like Netflix — can grow that money over time. But when you open a financial website, you're met with a wall of numbers, unfamiliar abbreviations, and charts that look like they were designed for someone who went to business school.

**Most people give up at this point.** Not because they aren't smart — but because the tools weren't built for them.

The people who do understand those tools? They're typically sitting in trading firms with Bloomberg terminals that cost **$24,000 a year** to access. They have data scientists running models in the background. They make decisions based on patterns in data that the average person simply cannot see.

That gap — between what professional investors know and what everyone else has access to — is real, and it's costly. When ordinary people invest without understanding what the data is telling them, they often buy high and sell low, panic at the wrong moment, or miss opportunities entirely.

**This app was built to close that gap.**

---

## The Solution — What This App Does

The **Stock Price Intelligence Terminal** is a data-driven dashboard that takes years of Netflix stock price data and turns it into clear, visual insights — the kind that used to be locked behind expensive professional software.

You don't need to know what "MACD" means. You don't need to understand neural networks. The app does the heavy lifting and shows you what the numbers are actually saying.

Here's what it does, in plain language:

### 📈 Price History
Shows you how Netflix's stock price has moved over time — not just a flat line, but layered with **moving averages** (think of these as smoothed-out trend lines that filter out the day-to-day noise). You can also see daily returns and trading volume. At a glance, you can tell when Netflix was on a tear and when it was struggling.

### 📐 Technical Analysis
This is where it gets interesting. The app calculates three widely-used signals that professional traders rely on:

- **Bollinger Bands** — A kind of price "envelope." When the price touches the top of the envelope, the stock may be overheated. When it touches the bottom, it may be undervalued. The app draws this for you automatically.

- **RSI (Relative Strength Index)** — A number between 0 and 100 that tells you whether a stock is being *overbought* (everyone is piling in and the price is inflated) or *oversold* (everyone is panicking and the price may be lower than it should be). The app flags this with a clear label: OVERBOUGHT, OVERSOLD, or NEUTRAL.

- **MACD** — A signal used to spot momentum shifts — basically, is the stock picking up speed going up, or slowing down and about to turn? The app shows this as a chart with a clear colour-coded histogram.

### 🔮 Monte Carlo Forecast
This is the most powerful section. The app runs **600 simulated futures** for Netflix's stock price — each one a mathematically plausible path based on how the stock has behaved historically.

Think of it like this: imagine you want to know if it will rain next week. A meteorologist doesn't just give you one answer — they run hundreds of models and say "there's a 70% chance of rain." This app does the same thing for Netflix stock.

The result is a **fan chart** — a wide band showing the range of likely outcomes, with a median (middle) path highlighted. You can see:
- The **bull case** — the optimistic scenario (top 5% of simulations)
- The **bear case** — the pessimistic scenario (bottom 5%)
- The **probability that the price goes up** from today

### 🤖 LSTM Predictions *(requires model files)*
This section uses a **Long Short-Term Memory neural network** — a type of deep learning model specifically designed to learn from sequences over time, making it well-suited for stock price patterns.

The model was trained on historical Netflix data and tested on data it had never seen before. The dashboard shows you how closely its predictions matched reality, along with honest performance metrics so you can judge how much to trust it.

---

## How to Run It Yourself

### What you need
- Python 3.9 or newer installed on your computer
- A terminal / command prompt

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Launch the app
```bash
streamlit run app.py
```

Your browser will open automatically. If it doesn't, go to `http://localhost:8501`.

### Step 3 — Choose your data
When the app opens, look at the **left sidebar**. You have three options:

| Option | What it means |
|---|---|
| 📊 Demo data | Loads instantly — synthetic but realistic Netflix data 2019–2024. Great for exploring. |
| 📂 Upload CSV | Upload your own Netflix stock CSV (e.g. downloaded from Yahoo Finance) |
| 🌐 Fetch live | Pulls real-time data directly (requires `yfinance` installed) |

### Step 4 — For LSTM predictions *(optional)*
If you want the AI prediction tab to work, you'll need to run the training script first:
```bash
python train_and_save.py
```
This generates three files: `netflix_lstm_model.keras`, `netflix_scaler.pkl`, and `netflix_model_meta.json`. Place them in the same folder as `app.py`.

---

## Project Structure

```
📁 your-project-folder/
├── app.py                     ← Main application (this is what you run)
├── train_and_save.py          ← Script to train the LSTM model
├── netflix_lstm_model.keras   ← Trained model (generated by training script)
├── netflix_scaler.pkl         ← Data scaler (generated by training script)
├── netflix_model_meta.json    ← Model metrics (generated by training script)
├── requirements.txt           ← Python packages needed
└── README.md                  ← You are here
```

---

## Limitations — What This App Cannot Do

Being honest about what a tool *can't* do is just as important as showing what it can. Here are the real limitations:

**1. It cannot predict the future with certainty.**
No model can. The Monte Carlo simulation and LSTM predictions are based on historical patterns. If Netflix announces something completely unexpected tomorrow — a merger, a scandal, a pandemic — no model trained on the past will see it coming.

**2. The LSTM model uses only price data.**
It does not read news. It does not know about interest rates, competitor moves, or what Netflix's quarterly earnings report says. A professional analyst would combine all of these. This model only sees numbers.

**3. The demo data is synthetic.**
It was generated to look like real Netflix stock behaviour, but it is not real historical data. For real analysis, use the CSV upload or live fetch option.

**4. Stock markets are not fully predictable by design.**
If a model could reliably predict prices, everyone would use it, prices would adjust instantly, and the prediction would stop working. This is a research and learning tool — not a trading bot.

**5. Past performance does not guarantee future results.**
This applies to every financial model ever built. The RSI, MACD, and Bollinger Band signals are indicators, not certainties.

---

## What Could Make This Better — Future Improvements

If this project continued to grow, here's what would take it to the next level:

- **Fundamental data integration** — pulling in Netflix's actual revenue, subscriber numbers, and profit margins alongside price data. Price alone tells only half the story.

- **Sentiment analysis layer** — scanning financial news headlines and social media to detect public mood around Netflix. Markets are partly driven by emotion.

- **Multi-stock comparison** — letting you compare Netflix against competitors like Disney+, Amazon Prime, and Apple TV+ on the same dashboard.

- **Dockerised deployment** — packaging the entire app into a container so anyone can run it with one command, no Python setup required.

- **MLflow model tracking** — logging every training run with its metrics so you can compare model versions and track improvement over time.

- **Alert system** — notifying you when the RSI crosses into overbought/oversold territory or when the Monte Carlo simulation shifts significantly.

- **Backtesting engine** — showing what would have happened if you had bought or sold based on the signals this app generates. Putting the strategy to the test against real historical outcomes.

---

## Tech Stack

| Layer | Tools Used |
|---|---|
| Frontend / Dashboard | Streamlit, Plotly |
| Data Processing | Pandas, NumPy |
| Machine Learning | TensorFlow / Keras (LSTM) |
| Statistical Modelling | NumPy (Monte Carlo / GBM) |
| Technical Indicators | Custom NumPy/Pandas implementation |
| Live Data | yfinance |

---

## About the Builder

This app was built by **Kerubo Bosire**, an actuarial scientist transitioning into AI Engineering and machine learning. The goal was to build something that demonstrates real-world ML engineering skills — not just a textbook example, but a tool that solves a genuine problem with clean, professional output.

- 🔗 [LinkedIn](https://linkedin.com/in/kerubo-bosire-364523283)
- 💻 [GitHub](https://github.com/kerubobosire254)

---

*This project is for educational and portfolio purposes. Nothing in this app constitutes financial advice.*
