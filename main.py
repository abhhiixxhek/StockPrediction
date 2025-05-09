import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Streamlit Config ---
st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")

# --- Gemini Setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = GenerativeModel("gemini-2.0-flash")

# --- Indicator Computations ---
@st.cache_data
def calculate_technical_indicators(df, params):
    df = df.copy().sort_values('timestamp')
    # Multiple SMAs
    for window in params.get('sma_windows', []):
        df[f'SMA_{window}'] = df['close'].rolling(window).mean()
    # Multiple EMAs
    for span in params.get('ema_spans', []):
        df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    # Single RSI
    if params['enable_rsi']:
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(params['rsi_period']).mean()
        loss = -delta.clip(upper=0).rolling(params['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    # Single MACD
    if params['enable_macd']:
        exp_fast = df['close'].ewm(span=params['macd_fast'], adjust=False).mean()
        exp_slow = df['close'].ewm(span=params['macd_slow'], adjust=False).mean()
        df['MACD'] = exp_fast - exp_slow
        df['Signal_Line'] = df['MACD'].ewm(span=params['macd_signal'], adjust=False).mean()
    # Bollinger Bands
    if params['enable_bb']:
        mid = df['close'].rolling(params['bb_window']).mean()
        std = df['close'].rolling(params['bb_window']).std()
        df['Upper_Band'] = mid + params['bb_multiplier'] * std
        df['Lower_Band'] = mid - params['bb_multiplier'] * std
    return df.dropna()

# --- Summary Stats ---
@st.cache_data
def compute_summary(df, start_date, end_date):
    sub = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    return {
        'Period High': sub['high'].max(),
        'Period Low': sub['low'].min(),
        'Average Close': sub['close'].mean(),
        'Total Volume': sub['volume'].sum(),
        'Volatility (%)': ((sub['close'].std()/sub['close'].mean())*100)
    }

# --- Plot-Based Summary ---
def build_plot_summary(df, ti, params, lookback=5):
    latest = ti.iloc[-1]
    prev = ti.iloc[-lookback]
    lines = []
    # BB
    if params['enable_bb']:
        width = latest['Upper_Band'] - latest['Lower_Band']
        pct = (latest['close'] - latest['Lower_Band']) / width * 100
        lines.append(f"Price is at {pct:.1f}% of BB range (Lower {latest['Lower_Band']:.2f}, Upper {latest['Upper_Band']:.2f}).")
    # SMAs
    for window in params.get('sma_windows', []):
        col = f"SMA_{window}"
        slope = (latest[col] - prev[col]) / prev[col] * 100
        lines.append(f"{col} slope over last {lookback} periods: {slope:.2f}% (from {prev[col]:.2f} to {latest[col]:.2f}).")
    # EMAs
    for span in params.get('ema_spans', []):
        col = f"EMA_{span}"
        slope = (latest[col] - prev[col]) / prev[col] * 100
        lines.append(f"{col} slope over last {lookback} periods: {slope:.2f}% (from {prev[col]:.2f} to {latest[col]:.2f}).")
    # RSI
    if params['enable_rsi']:
        change = latest['RSI'] - prev['RSI']
        lines.append(f"RSI change over last {lookback} periods: {change:.2f} points (from {prev['RSI']:.2f} to {latest['RSI']:.2f}).")
    # MACD
    if params['enable_macd']:
        hist_now = latest['MACD'] - latest['Signal_Line']
        hist_prev = prev['MACD'] - prev['Signal_Line']
        lines.append(f"MACD histogram change: {hist_now-hist_prev:.2f} (from {hist_prev:.2f} to {hist_now:.2f}).")
    # Volume
    vol_now = df['volume'].iloc[-1]
    vol_prev = df['volume'].iloc[-lookback]
    vol_pct = (vol_now - vol_prev) / vol_prev * 100
    lines.append(f"Volume change over last {lookback} periods: {vol_pct:.1f}% (from {vol_prev:.0f} to {vol_now:.0f}).")
    return "\n".join(lines)

# --- LLM Feedback ---
def get_indicator_feedback(indicator_summary: str, plot_summary: str) -> str:
    prompt = (
        "You are a senior quantitative financial analyst specializing in technical indicators and market timing. "
        "Analyze the following technical indicator values and chart behavior in detail and provide a clear, structured insight. "
        "Your response must include:\n"
        "1. Current Price Trend Analysis (based on price, moving averages, and slope)\n"
        "2. RSI and MACD Interpretation (including momentum and possible reversal signals)\n"
        "3. Bollinger Bands and Volume Context (volatility and confirmation)\n"    
        "4. Overall Technical Sentiment\n"
        "5. Final Buy/Hold/Sell Recommendation (with reasoning and forward outlook)\n"
        "Base your reasoning strictly on the data below, emphasizing chart-based observations. "
        "Avoid generic explanations—give precise, market-level insights.\n\n"
        "---\n"
        "Technical Indicator Summary:\n" + indicator_summary + "\n"
        "Chart-Based Summary:\n" + plot_summary
    )
    response = MODEL.generate_content(prompt)
    return response.text.strip()

# --- Load Data ---
@st.cache_data
def load_stock_data(path="stockdata"):
    data = {}
    for fn in os.listdir(path):
        if fn.endswith('.csv'):
            name = fn[:-4]
            df = pd.read_csv(os.path.join(path, fn), parse_dates=['timestamp'])
            data[name] = df.sort_values('timestamp')
    return data

# --- Main App ---
def main():
    st.sidebar.title("Navigation")
    stock_data = load_stock_data()
    stock = st.sidebar.selectbox("Select Stock", list(stock_data.keys()))

    # Date Range
    df_full = stock_data[stock]
    dates = df_full['timestamp']
    start_date, end_date = st.sidebar.date_input("Select Date Range", [dates.min(), dates.max()])

    # Multi-select for SMAs and EMAs
    sma_windows = st.sidebar.multiselect(
        "SMA Windows (up to 3)", options=[5,10,20,50,100,200], default=[20],
        help="Select up to three SMA periods"
    )
    if len(sma_windows) > 3:
        st.sidebar.error("Please select at most 3 SMA windows.")
        sma_windows = sma_windows[:3]

    ema_spans = st.sidebar.multiselect(
        "EMA Spans (up to 3)", options=[5,10,20,50,100,200], default=[20],
        help="Select up to three EMA periods"
    )
    if len(ema_spans) > 3:
        st.sidebar.error("Please select at most 3 EMA spans.")
        ema_spans = ema_spans[:3]

    # Other toggles & params
    enable_rsi = st.sidebar.checkbox("RSI", value=True)
    enable_macd = st.sidebar.checkbox("MACD", value=True)
    enable_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

    params = {
        'sma_windows': sma_windows,
        'ema_spans': ema_spans,
        'enable_rsi': enable_rsi,
        'enable_macd': enable_macd,
        'enable_bb': enable_bb,
        'rsi_period': st.sidebar.number_input("RSI Period", 5, 50, 14),
        'macd_fast': st.sidebar.number_input("MACD Fast Span", 5, 50, 12),
        'macd_slow': st.sidebar.number_input("MACD Slow Span", 10, 100, 26),
        'macd_signal': st.sidebar.number_input("MACD Signal Span", 5, 50, 9),
        'bb_window': st.sidebar.number_input("BB Window", 5, 100, 20),
        'bb_multiplier': st.sidebar.number_input("BB Multiplier", 1.0, 3.0, 2.0, step=0.1)
    }

    # Filter Data
    mask = (df_full['timestamp'] >= pd.to_datetime(start_date)) & (df_full['timestamp'] <= pd.to_datetime(end_date))
    df = df_full.loc[mask]

    st.title(f"{stock} Analysis & Prediction")

    # Summary Metrics
    summary = compute_summary(df, pd.to_datetime(start_date), pd.to_datetime(end_date))
    cols = st.columns(5)
    for i,(k,v) in enumerate(summary.items()):
        cols[i].metric(label=k, value=f"{v:.2f}" if isinstance(v, float) else int(v))

    st.subheader("Historical Data Preview")
    st.dataframe(df.tail(), use_container_width=True)

    ti = calculate_technical_indicators(df, params)

    # Build hover text
    hover_text = [
        f"Time: {ts:%Y-%m-%d %H:%M:%S}<br>"
        f"Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
        for ts,o,h,l,c in zip(df['timestamp'], df['open'], df['high'], df['low'], df['close'])
    ]

    # Plot Price & Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3], subplot_titles=("Price & MAs","Volume"))
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price", hoverinfo='text', hovertext=hover_text
    ), row=1, col=1)

    # Plot multiple SMAs
    for window in sma_windows:
        fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti[f"SMA_{window}"], name=f"SMA {window}",
            hovertemplate=f"SMA {window}: %{{y:.2f}}<extra></extra>"
        ), row=1, col=1)

    # Plot multiple EMAs
    for span in ema_spans:
        fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti[f"EMA_{span}"], name=f"EMA {span}",
            hovertemplate=f"EMA {span}: %{{y:.2f}}<extra></extra>"
        ), row=1, col=1)

    # BB bands
    if enable_bb:
        fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti['Upper_Band'], name="Upper BB",
            line=dict(dash='dash'), hovertemplate="Upper BB: %{y:.2f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti['Lower_Band'], name="Lower BB",
            line=dict(dash='dash'), hovertemplate="Lower BB: %{y:.2f}<extra></extra>"
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['volume'], name="Volume",
        showlegend=False, hovertemplate="Volume: %{y}<extra></extra>"
    ), row=2, col=1)

    fig.update_layout(height=750, template="plotly_white",
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=40,r=40,t=60,b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Plot Indicators
    ind_fig = go.Figure()
    if enable_rsi:
        ind_fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti['RSI'], name="RSI",
            hovertemplate="RSI: %{y:.2f}<extra></extra>"
        ))
        ind_fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought", opacity=0.5)
        ind_fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold", opacity=0.5)
    if params['enable_macd']:
        ind_fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti['MACD'], name="MACD",
            hovertemplate="MACD: %{y:.2f}<extra></extra>"
        ))
        ind_fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti['Signal_Line'], name="Signal Line",
            hovertemplate="Signal: %{y:.2f}<extra></extra>"
        ))
    ind_fig.update_layout(height=450, template="plotly_white",
                          margin=dict(l=40,r=40,t=40,b=40))
    st.plotly_chart(ind_fig, use_container_width=True)

    # LLM Insights with spinner
    st.sidebar.subheader("Get Insights from Indicators")
    if st.sidebar.button("Run Indicator Feedback"):
        with st.spinner("Generating insights... please wait"):
            parts = []
            # Build indicator summary
            for window in sma_windows:
                parts.append(f"Latest SMA {window}: {ti[f'SMA_{window}'].iloc[-1]:.2f}")
            for span in ema_spans:
                parts.append(f"Latest EMA {span}: {ti[f'EMA_{span}'].iloc[-1]:.2f}")
            if enable_rsi:
                parts.append(f"Latest RSI: {ti['RSI'].iloc[-1]:.2f}")
            if params['enable_macd']:
                parts.append(f"Latest MACD: {ti['MACD'].iloc[-1]:.2f}, Signal: {ti['Signal_Line'].iloc[-1]:.2f}")
            if enable_bb:
                parts.append(f"Latest Bollinger Bands: Upper {ti['Upper_Band'].iloc[-1]:.2f}, Lower {ti['Lower_Band'].iloc[-1]:.2f}")
            indicator_summary = "\n".join(parts)
            plot_summary = build_plot_summary(df, ti, params)
            feedback = get_indicator_feedback(indicator_summary, plot_summary)
        st.subheader("LLM Insights & Recommendations")
        st.write(feedback)
        st.balloons()

    # Prediction Page
    if st.sidebar.button("Go to Prediction Page 📈"):
        st.switch_page("pages/predict.py")

if __name__ == "__main__":
    main()