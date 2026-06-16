python-3.11

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, json, os
from datetime import datetime, timedelta

# ── imports ──────────────────────────────────
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Prediction Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM 
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #ffffff !important; color: #0f172a; }
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stSidebar"] { background: #f8fafc !important; border-right: 1px solid #e2e8f0; }
[data-testid="stHeader"] { background: #ffffff !important; }
section[data-testid="stSidebar"] > div { background: #f8fafc !important; }

/* ── Terminal header ── */
.terminal-header {
    border-bottom: 2px solid #e11d48;
    padding-bottom: 16px;
    margin-bottom: 24px;
}
.ticker-badge {
    display: inline-block;
    background: #e11d48;
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 3px;
    margin-bottom: 6px;
}
.terminal-title {
    font-family: 'Inter', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.8px;
    line-height: 1.1;
}
.terminal-sub {
    font-size: 12px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ── KPI cards ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #e2e8f0;
    border-radius: 6px;
    padding: 14px 16px;
}
.kpi-card.accent { border-top-color: #e11d48; }
.kpi-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #0f172a;
    letter-spacing: -0.5px;
}
.kpi-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    margin-top: 3px;
    font-weight: 500;
}
.delta-up   { color: #16a34a; }
.delta-down { color: #dc2626; }
.delta-neutral { color: #64748b; }

/* ── Section labels ── */
.section-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #e11d48;
    margin-bottom: 2px;
}
.section-heading {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.3px;
    margin-bottom: 4px;
}
.section-caption {
    font-size: 12px;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 16px;
}

/* ── Metric chip ── */
.metric-chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #475569;
    margin-right: 6px;
    margin-bottom: 4px;
}
.metric-chip strong { color: #0f172a; }

/* ── Info box ── */
.info-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 13px;
    color: #475569;
}
.warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 13px;
    color: #92400e;
}

/* ── Tab styling ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #e2e8f0;
    gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
    color: #94a3b8;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #e11d48 !important;
    border-bottom: 2px solid #e11d48 !important;
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    padding: 10px 22px !important;
    letter-spacing: 0.3px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #e11d48 !important; }

/* ── Misc ── */
hr { border-color: #e2e8f0 !important; margin: 20px 0 !important; }
[data-testid="stExpander"] { border: 1px solid #e2e8f0 !important; border-radius: 6px !important; }
.stSelectbox > div, .stFileUploader > div { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, sans-serif", color="#475569", size=11),
    xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", zeroline=False, showgrid=True),
    yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", zeroline=False, showgrid=True),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=11)),
    hoverlabel=dict(bgcolor="#0f172a", bordercolor="#0f172a",
                    font=dict(color="#ffffff", size=12, family="JetBrains Mono")),
)
RED   = "#e11d48"
BLUE  = "#3b82f6"
GREEN = "#16a34a"
AMBER = "#f59e0b"
SLATE = "#64748b"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(col, label, value, delta=None, accent=False):
    delta_html = ""
    if delta is not None:
        cls   = "delta-up" if delta >= 0 else "delta-down"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {abs(delta):.2f}%</div>'
    accent_cls = "accent" if accent else ""
    col.markdown(f"""
    <div class="kpi-card {accent_cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

def section(eyebrow, heading, caption=""):
    st.markdown(f'<div class="section-label">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-heading">{heading}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA  
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_demo_data():
    """Synthetic stock price history with realistic regimes."""
    np.random.seed(42)
    dates  = pd.date_range("2019-01-02", "2024-12-31", freq="B")
    n      = len(dates)
    regime_drift = np.zeros(n)
    regime_drift[:260]       =  0.0008
    regime_drift[260:300]    = -0.035
    regime_drift[300:520]    =  0.0018
    regime_drift[520:700]    = -0.0025
    regime_drift[700:850]    = -0.008
    regime_drift[850:]       =  0.0012

    vol = np.where(np.abs(regime_drift) > 0.003, 0.032, 0.018)
    daily_ret   = np.random.normal(regime_drift, vol)
    close       = 330.0 * np.exp(np.cumsum(daily_ret))
    close       = np.clip(close, 150, 700)

    volume      = np.random.lognormal(14.8, 0.4, n).astype(int)
    high        = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low         = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_       = close * (1 + np.random.normal(0, 0.005, n))

    return pd.DataFrame({
        "Date":   dates,
        "Open":   np.round(open_,  2),
        "High":   np.round(high,   2),
        "Low":    np.round(low,    2),
        "Close":  np.round(close,  2),
        "Volume": volume,
    })

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def prepare_data(df_raw):
    df = df_raw.copy()
    df['Date']   = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
    df           = df.sort_values('Date').reset_index(drop=True)
    df['Volume'] = np.log1p(df['Volume'])
    df['Return'] = df['Close'].pct_change()
    df['MA10']   = df['Close'].rolling(10).mean()
    df['MA50']   = df['Close'].rolling(50).mean()
    df['MA200']  = df['Close'].rolling(200).mean()
    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_up']  = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lo']  = df['BB_mid'] - 2 * df['BB_std']
    df['BB_pct'] = (df['Close'] - df['BB_lo']) / (df['BB_up'] - df['BB_lo'])
    # RSI
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - 100 / (1 + rs)
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']
    # ATR proxy
    df['ATR14'] = df['Close'].rolling(14).std()
    return df.dropna().reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO FORECAST
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo(close_series, days=30, simulations=600, seed=0):
    np.random.seed(seed)
    log_ret = np.log(close_series / close_series.shift(1)).dropna()
    mu      = log_ret.mean()
    sigma   = log_ret.std()
    last    = close_series.iloc[-1]
    sims    = np.zeros((simulations, days))
    for i in range(simulations):
        shocks      = np.random.normal(mu, sigma, days)
        sims[i, :]  = last * np.exp(np.cumsum(shocks))
    return sims

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    files = ['netflix_lstm_model.keras', 'netflix_scaler.pkl', 'netflix_model_meta.json']
    if all(os.path.exists(f) for f in files):
        model = load_model('netflix_lstm_model.keras')
        with open('netflix_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('netflix_model_meta.json') as f:
            meta = json.load(f)
        return model, scaler, meta
    return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data Source")
   
    data_source = st.radio(
        "Select data source",
        ["📊 Demo data", "📂 Upload CSV",
         "🌐 Fetch live" if HAS_YF else "🌐 Fetch live (install yfinance)"],
        index=0,
        label_visibility="collapsed",
    )

    raw_df = None

    if "Demo" in data_source:
        raw_df = generate_demo_data()
        st.success("Demo data loaded (2019–2024)")
        st.caption("Synthetic NFLX-like history · 1,500+ trading days")

    elif "Upload" in data_source:
        uploaded = st.file_uploader("Upload Netflix CSV", type=["csv"])
        if uploaded:
            raw_df = pd.read_csv(uploaded)
            st.success(f"{len(raw_df):,} rows loaded")

    elif "Fetch live" in data_source and HAS_YF:
        ticker = st.text_input("Ticker", value="NFLX")
        period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
        if st.button("Fetch"):
            with st.spinner("Fetching…"):
                try:
                    t    = yf.Ticker(ticker)
                    data = t.history(period=period)
                    data = data.reset_index()[['Date','Open','High','Low','Close','Volume']]
                    data['Date'] = data['Date'].dt.tz_localize(None)
                    raw_df = data
                    st.session_state['live_df'] = raw_df
                    st.success(f"{len(raw_df):,} rows · {ticker}")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
        if 'live_df' in st.session_state and raw_df is None:
            raw_df = st.session_state['live_df']

    st.markdown("---")
    st.markdown("### Model")
    if HAS_TF:
        model, scaler, meta = load_artifacts()
        if model:
            st.success("LSTM model loaded ✓")
            st.caption(f"RMSE ${meta['rmse']:.2f} · MAPE {meta['mape']:.2f}%")
        else:
            st.info("No model files found.\nRun `train_and_save.py` to generate\n`netflix_lstm_model.keras`,\n`netflix_scaler.pkl`,\n`netflix_model_meta.json`")
            model, scaler, meta = None, None, None
    else:
        st.warning("TensorFlow not installed — predictions tab disabled.")
        model, scaler, meta = None, None, None

    st.markdown("---")
    st.markdown("### Forecast Settings")
    mc_days = st.slider("Forecast horizon (days)", 10, 90, 30)
    mc_sims = st.slider("MC simulations", 100, 2000, 600, step=100)

# ─────────────────────────────────────────────────────────────────────────────
# GATE
# ─────────────────────────────────────────────────────────────────────────────
if raw_df is None:
    st.markdown("""
    <div class="terminal-header">
      <div class="ticker-badge">NFLX</div>
      <div class="terminal-title">Price Intelligence Terminal</div>
      <div class="terminal-sub">LSTM · MONTE CARLO · TECHNICAL ANALYSIS · SIGNAL DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="info-box">⬅️ Select a data source in the sidebar to launch the terminal.</div>',
                unsafe_allow_html=True)
    st.stop()

df = prepare_data(raw_df)

FEATURES        = ['Close', 'Volume', 'MA10', 'MA50', 'Return']
SEQUENCE_LENGTH = 60
SPLIT_RATIO     = 0.80
latest          = df['Close'].iloc[-1]
prev            = df['Close'].iloc[-2]
first           = df['Close'].iloc[0]
pct_1d          = (latest - prev)  / prev  * 100
pct_all         = (latest - first) / first * 100
high52          = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=365)]['Close'].max()
low52           = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=365)]['Close'].min()
rsi_now         = df['RSI'].iloc[-1]
atr_now         = df['ATR14'].iloc[-1]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER + KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="terminal-header">
  <div class="ticker-badge">NFLX · NASDAQ</div>
  <div class="terminal-title">Price Intelligence Terminal</div>
  <div class="terminal-sub">LSTM · MONTE CARLO · TECHNICAL ANALYSIS · {len(df):,} TRADING DAYS</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpi_card(c1, "Last Close",   f"${latest:,.2f}",        pct_1d, accent=True)
kpi_card(c2, "1-Day Chg",    f"{pct_1d:+.2f}%")
kpi_card(c3, "52W High",     f"${high52:,.2f}")
kpi_card(c4, "52W Low",      f"${low52:,.2f}")
kpi_card(c5, "RSI (14)",     f"{rsi_now:.1f}")
kpi_card(c6, "ATR (14)",     f"${atr_now:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

rsi_lbl = "OVERBOUGHT" if rsi_now > 70 else ("OVERSOLD" if rsi_now < 30 else "NEUTRAL")
rsi_col = "delta-down" if rsi_now > 70 else ("delta-up" if rsi_now < 30 else "delta-neutral")
st.markdown(
    f'<span class="metric-chip">RSI signal: <strong class="{rsi_col}">{rsi_lbl}</strong></span>'
    f'<span class="metric-chip">Total return: <strong>{pct_all:+.1f}%</strong></span>'
    f'<span class="metric-chip">Vol (log): <strong>{df["Volume"].iloc[-1]:.2f}</strong></span>'
    f'<span class="metric-chip">MACD hist: <strong>{df["MACD_hist"].iloc[-1]:+.2f}</strong></span>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Price History",
    "📐  Technical Analysis",
    "🔮  Forecast",
    "🤖  LSTM Predictions",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    section("Historical Data", "Closing Price + Moving Averages",
            f"Showing {len(df):,} trading days · MA10 · MA50 · MA200")

    fig_h = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          row_heights=[0.60, 0.20, 0.20], vertical_spacing=0.02)
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
        name="Close", line=dict(color=RED, width=1.8),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: $%{y:,.2f}<extra></extra>"),
        row=1, col=1)
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['MA10'],
        name="MA10", line=dict(color=BLUE, width=1, dash='dot')), row=1, col=1)
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['MA50'],
        name="MA50", line=dict(color=AMBER, width=1, dash='dot')), row=1, col=1)
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['MA200'],
        name="MA200", line=dict(color=SLATE, width=1, dash='dash')), row=1, col=1)
    colors_ret = np.where(df['Return'] >= 0, GREEN, "#dc2626")
    fig_h.add_trace(go.Bar(x=df['Date'], y=df['Return'],
        name="Daily Ret", marker_color=colors_ret, showlegend=False), row=2, col=1)
    fig_h.add_trace(go.Bar(x=df['Date'], y=df['Volume'],
        name="Log Vol", marker_color="#cbd5e1", showlegend=False), row=3, col=1)

    fig_h.update_layout(**PL, height=500,
        title=dict(text="Netflix Close · Moving Averages · Returns · Volume",
                   font=dict(size=13, color="#0f172a")))
    fig_h.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_h.update_yaxes(title_text="Return",    row=2, col=1)
    fig_h.update_yaxes(title_text="Log Vol",   row=3, col=1)
  
    st.plotly_chart(fig_h, width='stretch')

    with st.expander("🕯️  Candlestick — last 120 trading days"):
        df_candle = df.tail(120)
        fig_c = go.Figure(go.Candlestick(
            x=df_candle['Date'], open=df_candle['Open'],
            high=df_candle['High'], low=df_candle['Low'], close=df_candle['Close'],
            increasing=dict(line=dict(color=GREEN), fillcolor=GREEN),
            decreasing=dict(line=dict(color="#dc2626"), fillcolor="#dc2626"),
            name="OHLC"))
        fig_c.update_layout(**PL, height=340,
            title=dict(text="OHLC Candlestick · Last 120 Days",
                       font=dict(size=13, color="#0f172a")),
            xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_c, width='stretch')

    with st.expander("📊  Daily Returns Distribution"):
        fig_d = make_subplots(rows=1, cols=2,
                    subplot_titles=["Return Distribution", "Return QQ-like (sorted)"])
        fig_d.add_trace(go.Histogram(x=df['Return'], nbinsx=60,
            marker_color=RED, opacity=0.75, name="Returns"), row=1, col=1)
        sorted_ret = np.sort(df['Return'].values)
        n_ret      = len(sorted_ret)
        fig_d.add_trace(go.Scatter(
            x=np.linspace(0, 100, n_ret), y=sorted_ret,
            mode='lines', line=dict(color=BLUE, width=1.5),
            name="Sorted returns"), row=1, col=2)
        fig_d.add_hline(y=0, line=dict(color=SLATE, dash='dash', width=1), row=1, col=2)
        fig_d.update_layout(**PL, height=300, showlegend=False)
        skew = df['Return'].skew()
        kurt = df['Return'].kurt()
        st.plotly_chart(fig_d, width='stretch')
        st.markdown(
            f'<span class="metric-chip">Mean: <strong>{df["Return"].mean()*100:.3f}%</strong></span>'
            f'<span class="metric-chip">Std Dev: <strong>{df["Return"].std()*100:.3f}%</strong></span>'
            f'<span class="metric-chip">Skewness: <strong>{skew:.3f}</strong></span>'
            f'<span class="metric-chip">Kurtosis: <strong>{kurt:.3f}</strong></span>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    section("Technical Analysis", "Bollinger Bands · RSI · MACD",
            "Industry-standard momentum & volatility indicators")

    st.markdown("**Bollinger Bands (20, ±2σ)**")
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(
        x=pd.concat([df['Date'], df['Date'][::-1]]),
        y=pd.concat([df['BB_up'], df['BB_lo'][::-1]]),
        fill='toself', fillcolor='rgba(59,130,246,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name="Band"))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
        name="Close", line=dict(color=RED, width=1.6),
        hovertemplate="<b>%{x|%b %d %Y}</b><br>$%{y:,.2f}<extra></extra>"))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_mid'],
        name="Mid (MA20)", line=dict(color=BLUE, width=1, dash='dot')))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_up'],
        name="Upper (2σ)", line=dict(color="#94a3b8", width=0.8)))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_lo'],
        name="Lower (2σ)", line=dict(color="#94a3b8", width=0.8)))
    fig_bb.update_layout(**PL, height=340,
        title=dict(text="Bollinger Bands — Price Volatility Envelope",
                   font=dict(size=13, color="#0f172a")))
    st.plotly_chart(fig_bb, width='stretch')

    col_rsi, col_macd = st.columns(2)

    # ── RSI ────────────────────────────────────────────────────────────────────
    with col_rsi:
        st.markdown("**RSI (14-period)**")
        fig_rsi = go.Figure()
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.06)",
                          line_width=0, annotation_text="Overbought",
                          annotation_position="top left",
                          annotation_font=dict(color="#dc2626", size=10))
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(22,163,74,0.06)",
                          line_width=0, annotation_text="Oversold",
                          annotation_position="bottom left",
                          annotation_font=dict(color="#16a34a", size=10))
        fig_rsi.add_hline(y=70, line=dict(color="#dc2626", dash='dash', width=0.8))
        fig_rsi.add_hline(y=30, line=dict(color="#16a34a", dash='dash', width=0.8))
        fig_rsi.add_hline(y=50, line=dict(color=SLATE, dash='dot', width=0.8))
        fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'],
            name="RSI", line=dict(color=AMBER, width=1.6),
            hovertemplate="RSI: %{y:.1f}<extra></extra>"))

        fig_rsi.update_layout(**PL, height=280,
            title=dict(text=f"RSI (14) · Current: {rsi_now:.1f} — {rsi_lbl}",
                       font=dict(size=12, color="#0f172a")))
        fig_rsi.update_yaxes(range=[0, 100], gridcolor="#e2e8f0")
        st.plotly_chart(fig_rsi, width='stretch')

    # ── MACD ───────────────────────────────────────────────────────────────────
    with col_macd:
        st.markdown("**MACD (12, 26, 9)**")
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.55, 0.45], vertical_spacing=0.04)
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'],
            name="MACD", line=dict(color=BLUE, width=1.4)), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_signal'],
            name="Signal", line=dict(color=RED, width=1.2, dash='dot')), row=1, col=1)
        hist_colors = np.where(df['MACD_hist'] >= 0, GREEN, "#dc2626")
        fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_hist'],
            marker_color=hist_colors, name="Histogram", showlegend=False), row=2, col=1)
        fig_macd.update_layout(**PL, height=280,
            title=dict(text=f"MACD · Histogram: {df['MACD_hist'].iloc[-1]:+.2f}",
                       font=dict(size=12, color="#0f172a")))
        st.plotly_chart(fig_macd, width='stretch')

    with st.expander("📏  Bollinger %B + ATR (14)"):
        col_bb, col_atr = st.columns(2)
        with col_bb:
            fig_bbp = go.Figure()
            fig_bbp.add_hrect(y0=1, y1=1.4, fillcolor="rgba(220,38,38,0.07)", line_width=0)
            fig_bbp.add_hrect(y0=-0.4, y1=0, fillcolor="rgba(22,163,74,0.07)", line_width=0)
            fig_bbp.add_hline(y=0.5, line=dict(color=SLATE, dash='dot', width=0.8))
            fig_bbp.add_trace(go.Scatter(x=df['Date'], y=df['BB_pct'],
                name="%B", line=dict(color=BLUE, width=1.4)))
            fig_bbp.update_layout(**PL, height=240,
                title=dict(text="Bollinger %B", font=dict(size=12, color="#0f172a")))
            st.plotly_chart(fig_bbp, width='stretch')
        with col_atr:
            fig_atr = go.Figure()
            fig_atr.add_trace(go.Scatter(x=df['Date'], y=df['ATR14'],
                name="ATR14", line=dict(color=AMBER, width=1.4),
                fill='tozeroy', fillcolor='rgba(245,158,11,0.08)'))
            fig_atr.update_layout(**PL, height=240,
                title=dict(text="ATR (14) — Volatility",
                           font=dict(size=12, color="#0f172a")))
            st.plotly_chart(fig_atr, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MONTE CARLO FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    section("Probabilistic Forecast", f"{mc_days}-Day Monte Carlo Simulation",
            f"{mc_sims:,} simulated paths · GBM with historical μ & σ")

    with st.spinner(f"Running {mc_sims:,} simulations…"):
        sims = monte_carlo(df['Close'], days=mc_days, simulations=mc_sims)

    last_date  = df['Date'].iloc[-1]
    fut_dates  = pd.bdate_range(last_date + timedelta(days=1), periods=mc_days)

    pct5   = np.percentile(sims, 5,  axis=0)
    pct25  = np.percentile(sims, 25, axis=0)
    pct50  = np.percentile(sims, 50, axis=0)
    pct75  = np.percentile(sims, 75, axis=0)
    pct95  = np.percentile(sims, 95, axis=0)
    final  = sims[:, -1]

    f1, f2, f3, f4, f5 = st.columns(5)
    kpi_card(f1, "Median Target",   f"${pct50[-1]:,.2f}", (pct50[-1]-latest)/latest*100)
    kpi_card(f2, "Bull Case (95%)", f"${pct95[-1]:,.2f}", (pct95[-1]-latest)/latest*100)
    kpi_card(f3, "Bear Case (5%)",  f"${pct5[-1]:,.2f}",  (pct5[-1]-latest)/latest*100)
    kpi_card(f4, "Prob Upside",     f"{(final > latest).mean()*100:.1f}%", accent=True)
    kpi_card(f5, "Exp. Range",      f"${pct5[-1]:,.0f}–{pct95[-1]:,.0f}")
    st.markdown("<br>", unsafe_allow_html=True)

    fig_mc = go.Figure()
    for lo, hi, alpha, label in [
        (pct5, pct95, 0.07, "90% CI"),
        (pct25, pct75, 0.12, "50% CI"),
    ]:
        fig_mc.add_trace(go.Scatter(
            x=list(fut_dates) + list(fut_dates[::-1]),
            y=list(hi) + list(lo[::-1]),
            fill='toself', fillcolor=f'rgba(225,29,72,{alpha})',
            line=dict(color='rgba(0,0,0,0)'), name=label, showlegend=True))
    fig_mc.add_trace(go.Scatter(x=fut_dates, y=pct50,
        name="Median path", line=dict(color=RED, width=2.2, dash='dot')))
    hist_tail = df.tail(90)
    fig_mc.add_trace(go.Scatter(x=hist_tail['Date'], y=hist_tail['Close'],
        name="Historical", line=dict(color="#0f172a", width=1.8)))
    fig_mc.add_vline(x=last_date.timestamp() * 1000,
                     line=dict(color=SLATE, dash='dash', width=1),
                     annotation_text="Today", annotation_font_color=SLATE)
    fig_mc.update_layout(**PL, height=420,
        title=dict(text=f"Monte Carlo Forecast — {mc_days} Business Days",
                   font=dict(size=13, color="#0f172a")))
    st.plotly_chart(fig_mc, width='stretch')

    col_dist, col_path = st.columns(2)
    with col_dist:
        fig_fd = go.Figure()
        fig_fd.add_trace(go.Histogram(x=final, nbinsx=60,
            marker_color=RED, opacity=0.75, name="Final price"))
        fig_fd.add_vline(x=latest,    line=dict(color="#0f172a", dash='dash', width=1.5),
                         annotation_text="Today",  annotation_font_color="#0f172a")
        fig_fd.add_vline(x=pct50[-1], line=dict(color=BLUE,    dash='dot',   width=1.5),
                         annotation_text="Median", annotation_font_color=BLUE)
        fig_fd.update_layout(**PL, height=280,
            title=dict(text="Final Price Distribution",
                       font=dict(size=12, color="#0f172a")),
            showlegend=False, xaxis_title="Price ($)", yaxis_title="Count")
        st.plotly_chart(fig_fd, width='stretch')

    with col_path:
        sample_idx = np.random.choice(mc_sims, 80, replace=False)
        fig_paths  = go.Figure()
        for i in sample_idx:
            fig_paths.add_trace(go.Scatter(x=fut_dates, y=sims[i],
                mode='lines', line=dict(color='rgba(225,29,72,0.06)', width=0.8),
                showlegend=False))
        fig_paths.add_trace(go.Scatter(x=fut_dates, y=pct50,
            name="Median", line=dict(color=RED, width=2)))
        fig_paths.update_layout(**PL, height=280,
            title=dict(text="Sample Simulation Paths",
                       font=dict(size=12, color="#0f172a")))
        st.plotly_chart(fig_paths, width='stretch')

    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Disclaimer:</strong> Monte Carlo simulations use historical return statistics (μ, σ) under a Geometric Brownian Motion assumption. They do not account for regime changes, macro events, or mean reversion. Not financial advice.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LSTM PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    section("Deep Learning Predictions", "LSTM — Actual vs Predicted",
            "Sequence length: 60 · Features: Close · Log Volume · MA10 · MA50 · Daily Return")

    if not HAS_TF:
        st.markdown('<div class="warn-box">TensorFlow is not installed in this environment. Install with <code>pip install tensorflow</code>.</div>',
                    unsafe_allow_html=True)
    elif model is None:
        st.markdown("""<div class="info-box">
        <strong>No model files detected.</strong><br>
        Run <code>train_and_save.py</code> to generate:
        <ul style="margin:6px 0 0 16px; padding:0; font-family: monospace; font-size:12px;">
          <li>netflix_lstm_model.keras</li>
          <li>netflix_scaler.pkl</li>
          <li>netflix_model_meta.json</li>
        </ul>
        Place them next to <code>app.py</code> and relaunch.
        </div>""", unsafe_allow_html=True)
    else:
        data_arr  = df[FEATURES].values
        split_idx = int(len(data_arr) * SPLIT_RATIO)
        test_scaled = scaler.transform(data_arr[split_idx:])
        test_gen    = TimeseriesGenerator(
            test_scaled, test_scaled[:, 0],
            length=SEQUENCE_LENGTH, batch_size=32)

        with st.spinner("Running inference…"):
            preds_scaled     = model.predict(test_gen, verbose=0)
        dummy_pred           = np.zeros((len(preds_scaled), len(FEATURES)))
        dummy_pred[:, 0]     = preds_scaled[:, 0]
        predicted_prices     = scaler.inverse_transform(dummy_pred)[:, 0]
        dummy_actual         = np.zeros((len(test_scaled) - SEQUENCE_LENGTH, len(FEATURES)))
        dummy_actual[:, 0]   = test_scaled[SEQUENCE_LENGTH:, 0]
        actual_prices        = scaler.inverse_transform(dummy_actual)[:, 0]
        test_dates           = df['Date'].values[split_idx + SEQUENCE_LENGTH:]

        rmse    = meta['rmse']
        mae     = meta['mae']
        mape    = meta['mape']
        dir_acc = meta['dir_acc']

        m1, m2, m3, m4 = st.columns(4)
        kpi_card(m1, "RMSE",             f"${rmse:.2f}", accent=True)
        kpi_card(m2, "MAE",              f"${mae:.2f}")
        kpi_card(m3, "MAPE",             f"{mape:.2f}%")
        kpi_card(m4, "Directional Acc.", f"{dir_acc:.1f}%")
        st.markdown("<br>", unsafe_allow_html=True)

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=pd.concat([pd.Series(test_dates), pd.Series(test_dates[::-1])]),
            y=np.concatenate([predicted_prices + rmse, (predicted_prices - rmse)[::-1]]),
            fill='toself', fillcolor='rgba(225,29,72,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name="±1 RMSE"))
        fig_p.add_trace(go.Scatter(x=test_dates, y=actual_prices,
            name="Actual", line=dict(color="#0f172a", width=1.6),
            hovertemplate="<b>%{x|%b %d %Y}</b><br>Actual: $%{y:,.2f}<extra></extra>"))
        fig_p.add_trace(go.Scatter(x=test_dates, y=predicted_prices,
            name="Predicted", line=dict(color=RED, width=2),
            hovertemplate="<b>%{x|%b %d %Y}</b><br>Predicted: $%{y:,.2f}<extra></extra>"))
        fig_p.update_layout(**PL, height=400,
            title=dict(text="Test Set · Actual vs Predicted Close Price",
                       font=dict(size=13, color="#0f172a")))
        st.plotly_chart(fig_p, width='stretch')

        col_l, col_s = st.columns(2)
        with col_l:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=meta['train_loss'],
                name="Train", line=dict(color=BLUE, width=1.5)))
            fig_loss.add_trace(go.Scatter(y=meta['val_loss'],
                name="Val", line=dict(color=RED, width=1.5)))
            fig_loss.add_vline(x=meta['best_epoch'] - 1,
                line=dict(color=GREEN, dash='dash', width=1),
                annotation_text=f"Best epoch {meta['best_epoch']}",
                annotation_font_color=GREEN)
            fig_loss.update_layout(**PL, height=310,
                title=dict(text=f"Training Loss Curve · {meta['total_epochs']} epochs",
                           font=dict(size=12, color="#0f172a")))
            st.plotly_chart(fig_loss, width='stretch')

        with col_s:
            lim = [min(actual_prices.min(), predicted_prices.min()),
                   max(actual_prices.max(), predicted_prices.max())]
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=actual_prices, y=predicted_prices,
                mode='markers', marker=dict(color=RED, size=3.5, opacity=0.4),
                name="Predictions"))
            fig_sc.add_trace(go.Scatter(x=lim, y=lim, mode='lines',
                line=dict(color=SLATE, dash='dash', width=1), name="Perfect fit"))
            fig_sc.update_layout(**PL, height=310,
                title=dict(text="Actual vs Predicted Scatter",
                           font=dict(size=12, color="#0f172a")),
                xaxis_title="Actual ($)", yaxis_title="Predicted ($)")
            st.plotly_chart(fig_sc, width='stretch')

        with st.expander("📐  Residual Analysis"):
            residuals = actual_prices - predicted_prices
            fig_r = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuals over time", "Error distribution"])
            fig_r.add_trace(go.Scatter(x=list(range(len(residuals))), y=residuals,
                mode='lines', line=dict(color=AMBER, width=1)), row=1, col=1)
            fig_r.add_hline(y=0, line=dict(color=SLATE, dash='dash'), row=1, col=1)
            fig_r.add_trace(go.Histogram(x=residuals, nbinsx=40,
                marker_color=RED, opacity=0.75), row=1, col=2)
            fig_r.update_layout(**PL, height=290, showlegend=False)
            st.plotly_chart(fig_r, width='stretch')
            st.markdown(
                f'<span class="metric-chip">Mean error: <strong>${residuals.mean():.2f}</strong></span>'
                f'<span class="metric-chip">Std: <strong>${residuals.std():.2f}</strong></span>'
                f'<span class="metric-chip">Max over: <strong>${residuals.max():.2f}</strong></span>'
                f'<span class="metric-chip">Max under: <strong>${residuals.min():.2f}</strong></span>',
                unsafe_allow_html=True)

python-3.11
