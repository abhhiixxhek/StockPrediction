import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


# --- Streamlit Config ---
st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")

# --- Gemini Setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = GenerativeModel("gemini-2.0-flash")

# --- Indicator Computations ---
@st.cache_data
def calculate_technical_indicators(df, params):
    df = df.copy()
    if params['enable_sma']:
        df[f'SMA_{params["sma_window"]}'] = df['close'].rolling(params["sma_window"]).mean()
    if params['enable_ema']:
        df[f'EMA_{params["ema_span"]}'] = df['close'].ewm(span=params["ema_span"], adjust=False).mean()
    if params['enable_rsi']:
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(params["rsi_period"]).mean()
        loss = -delta.clip(upper=0).rolling(params["rsi_period"]).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    if params['enable_macd']:
        exp_fast = df['close'].ewm(span=params["macd_fast"], adjust=False).mean()
        exp_slow = df['close'].ewm(span=params["macd_slow"], adjust=False).mean()
        df['MACD'] = exp_fast - exp_slow
        df['Signal_Line'] = df['MACD'].ewm(span=params["macd_signal"], adjust=False).mean()
    if params['enable_bb']:
        mid = df['close'].rolling(params["bb_window"]).mean()
        std = df['close'].rolling(params["bb_window"]).std()
        df['Upper_Band'] = mid + params["bb_multiplier"] * std
        df['Lower_Band'] = mid - params["bb_multiplier"] * std
    return df.dropna()

# --- Load Data ---
@st.cache_data
def load_stock_data(path="stock_data"):
    data = {}
    for fn in os.listdir(path):
        if fn.endswith('.csv'):
            name = fn[:-4]
            df = pd.read_csv(os.path.join(path, fn), parse_dates=['timestamp'])
            data[name] = df.sort_values('timestamp')
    return data

# --- LLM Feedback ---
def get_indicator_feedback(indicator_summary: str) -> str:
    prompt = (
    "You are a senior quantitative financial analyst specializing in technical indicators and market timing. "
    "Analyze the following technical indicator values in detail and provide a clear, structured insight. "
    "Your response must include:\n"
    "1. Current Price Trend Analysis (based on price and moving averages)\n"
    "2. RSI and MACD Interpretation (including momentum and possible reversal signals)\n"
    "3. Bollinger Bands and Volume Context (volatility and confirmation)\n"
    "4. Overall Technical Sentiment\n"
    "5. Final Buy/Hold/Sell Recommendation (with reasoning and forward outlook)\n"
    "Base your reasoning strictly on technical indicator behavior and recent price action. "
    "Avoid generic explanations—give precise, market-level insights.\n\n"
    "Technical Indicator Summary:\n" + indicator_summary
)

    response = MODEL.generate_content(prompt)
    return response.text.strip()

# --- Main App ---
def main():
    st.sidebar.title("Navigation")
    stock_data = load_stock_data()
    stock = st.sidebar.selectbox("Select Stock", list(stock_data.keys()))

    st.sidebar.markdown("### Select Indicators")
    enable_sma = st.sidebar.checkbox("SMA", value=True)
    enable_ema = st.sidebar.checkbox("EMA", value=True)
    enable_rsi = st.sidebar.checkbox("RSI", value=True)
    enable_macd = st.sidebar.checkbox("MACD", value=True)
    enable_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

    # Indicator parameter inputs
    st.sidebar.markdown("### Indicator Parameters")
    params = {
        "enable_sma": enable_sma,
        "enable_ema": enable_ema,
        "enable_rsi": enable_rsi,
        "enable_macd": enable_macd,
        "enable_bb": enable_bb,
        "sma_window": st.sidebar.number_input("SMA Window", min_value=5, max_value=200, value=20),
        "ema_span": st.sidebar.number_input("EMA Span", min_value=5, max_value=200, value=20),
        "rsi_period": st.sidebar.number_input("RSI Period", min_value=5, max_value=50, value=14),
        "macd_fast": st.sidebar.number_input("MACD Fast Span", min_value=5, max_value=50, value=12),
        "macd_slow": st.sidebar.number_input("MACD Slow Span", min_value=10, max_value=100, value=26),
        "macd_signal": st.sidebar.number_input("MACD Signal Span", min_value=5, max_value=50, value=9),
        "bb_window": st.sidebar.number_input("BB Window", min_value=5, max_value=100, value=20),
        "bb_multiplier": st.sidebar.number_input("BB Multiplier", min_value=1.0, max_value=3.0, step=0.1, value=2.0)
    }

    df = stock_data[stock]
    st.title(f"{stock} Analysis & Prediction")

    st.subheader("Historical Data Preview")
    st.dataframe(df.tail(5), use_container_width=True)

    ti = calculate_technical_indicators(df, params)

    st.subheader("Price & Volume Chart")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], subplot_titles=("Price", "Volume"))
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Price"
    ), row=1, col=1)

    if enable_sma:
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti[f"SMA_{params['sma_window']}"] , name="SMA"), row=1, col=1)
    if enable_ema:
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti[f"EMA_{params['ema_span']}"] , name="EMA"), row=1, col=1)
    if enable_bb:
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['Upper_Band'], name="Upper BB", line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['Lower_Band'], name="Lower BB", line=dict(dash='dash')), row=1, col=1)

    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name="Volume", marker_color='rgba(0,0,255,0.3)'), row=2, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Indicator Charts")
    ind_fig = go.Figure()
    if enable_rsi:
        ind_fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['RSI'], name="RSI"))
    if enable_macd:
        ind_fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['MACD'], name="MACD"))
        ind_fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['Signal_Line'], name="Signal Line"))
    st.plotly_chart(ind_fig, use_container_width=True)

    st.sidebar.subheader("Get Insights from Indicators")
    if st.sidebar.button("Run Indicator Feedback"):
        summary = ""
        if enable_rsi:
            summary += f"Latest RSI: {ti['RSI'].iloc[-1]:.2f}\n"
        if enable_macd:
            summary += f"Latest MACD: {ti['MACD'].iloc[-1]:.2f}, Signal: {ti['Signal_Line'].iloc[-1]:.2f}\n"
        if enable_sma:
            sma_col = f"SMA_{params['sma_window']}"
            summary += f"Latest SMA: {ti[sma_col].iloc[-1]:.2f}\n"


        if enable_ema:
            ema_col = f"EMA_{params['ema_span']}"
            summary += f"Latest EMA: {ti[ema_col].iloc[-1]:.2f}\n"

        if enable_bb:
            summary += f"Latest Bollinger Bands: Upper {ti['Upper_Band'].iloc[-1]:.2f}, Lower {ti['Lower_Band'].iloc[-1]:.2f}"
        feedback = get_indicator_feedback(summary)
        st.subheader("LLM Insights & Recommendations")
        st.write(feedback)

if __name__ == "__main__":
    main()