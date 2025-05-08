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

# --- Caching Decorator ---
def cache_data(func):
    return st.cache_data(func)

# --- Indicator Computations ---
@cache_data
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
@cache_data
def load_stock_data(path="stockdata"):
    data = {}
    for fn in os.listdir(path):
        if fn.endswith('.csv'):
            name = fn[:-4]
            df = pd.read_csv(os.path.join(path, fn), parse_dates=['timestamp'])
            data[name] = df.sort_values('timestamp')
    return data

# --- LLM Helpers ---
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
        "Technical Indicator Summary:\n"
        + indicator_summary
    )
    response = MODEL.generate_content(prompt)
    return response.text.strip()


def get_param_suggestions(stock: str, df: pd.DataFrame) -> dict:
    rec = df.tail(100)[['open','high','low','close','volume']].describe().to_dict()
    summary = f"Stock: {stock}\nRecent 100-day summary:\n" + \
              "".join([f"{k}: {v['mean']:.2f}, std: {v['std']:.2f}\n" for k,v in rec.items()])
    prompt = ("""You are a quantitative financial analyst with deep expertise in technical indicators and market timing. 
Based on the following summary of the most recent 100 trading days’ price and volume statistics, recommend optimal parameter values within the given ranges for each of these indicators:
  • sma_window: integer between 5 and 200  
  • ema_span: integer between 5 and 200  
  • rsi_period: integer between 5 and 50  
  • macd_fast: integer between 5 and 50  
  • macd_slow: integer between 10 and 100  
  • macd_signal: integer between 5 and 50  
  • bb_window: integer between 5 and 100  
  • bb_multiplier: float between 1.0 and 3.0  

Return **only** a single valid JSON object matching this schema (no extra text before or after):

{  
  "sma_window": <int>,  
  "rationale_sma": <string>,  
  "ema_span": <int>,  
  "rationale_ema": <string>,  
  "rsi_period": <int>,  
  "rationale_rsi": <string>,  
  "macd_fast": <int>,  
  "macd_slow": <int>,  
  "macd_signal": <int>,  
  "rationale_macd": <string>,  
  "bb_window": <int>,  
  "bb_multiplier": <float>,  
  "rationale_bb": <string>
}

Here is the summary to analyze:""" + summary)
    response = MODEL.generate_content(prompt)
    try:
        return eval(response.text.strip())
    except:
        st.warning("Could not parse LLM response for parameter suggestions.\n" + response.text)
        return {}

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

    if st.sidebar.button("Suggest Best Params"):
        suggestions = get_param_suggestions(stock, stock_data[stock])
        if suggestions:
            st.sidebar.markdown("### Suggested Parameters:")
            for k, v in suggestions.items():
                st.sidebar.write(f"**{k}**: {v}")

    df = stock_data[stock]
    st.title(f"{stock} Analysis & Prediction")
    st.subheader("Historical Data Preview")
    st.dataframe(df.tail(5), use_container_width=True)

    ti = calculate_technical_indicators(df, params)

    # --- Enhanced Chart Styles: Price & Volume Chart ---
    st.subheader("Price & Volume Chart")
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price & Moving Averages", "Volume"),
    )

    # 1) Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name="Price",
            increasing_line_color='green', decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # 2) Moving Averages & Bands
    if enable_sma:
        fig.add_trace(
            go.Scatter(
                x=ti['timestamp'],
                y=ti[f"SMA_{params['sma_window']}"],
                name=f"SMA ({params['sma_window']})",
                line=dict(width=2)
            ),
            row=1, col=1
        )
    if enable_ema:
        fig.add_trace(
            go.Scatter(
                x=ti['timestamp'],
                y=ti[f"EMA_{params['ema_span']}"],
                name=f"EMA ({params['ema_span']})",
                line=dict(width=2, dash='dot')
            ),
            row=1, col=1
        )
    if enable_bb:
        fig.add_trace(
            go.Scatter(
                x=ti['timestamp'], y=ti['Upper_Band'],
                name="Upper BB", line=dict(dash='dash'),
                opacity=0.6
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=ti['timestamp'], y=ti['Lower_Band'],
                name="Lower BB", line=dict(dash='dash'),
                opacity=0.6, fill='tonexty', fillcolor='rgba(0,0,255,0.1)'
            ),
            row=1, col=1
        )

    # 3) Volume Bars (clean, colored, with its own axis title + rangeslider)
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name="Volume",
            marker_color="blue",
            opacity=0.8
        ),
        row=2, col=1
    )

    # 4) Layout tweaks
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
    fig.update_layout(
        template='plotly_white',
        height=800,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Indicator Charts: split RSI & MACD into two clean panes ---
    st.subheader("Indicator Charts")
    ind_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.6],
        subplot_titles=("RSI", "MACD"),
    )

    # RSI pane
    if enable_rsi:
        ind_fig.add_trace(
            go.Scatter(
                x=ti['timestamp'], y=ti['RSI'],
                name="RSI", line=dict(width=2)
            ),
            row=1, col=1
        )
        ind_fig.add_hline(y=70, line_dash='dash', annotation_text='Overbought', row=1, col=1)
        ind_fig.add_hline(y=30, line_dash='dash', annotation_text='Oversold', row=1, col=1)
    ind_fig.update_yaxes(range=[0, 100], title_text="RSI", row=1, col=1)

    # MACD pane
    if enable_macd:
        ind_fig.add_trace(
            go.Scatter(
                x=ti['timestamp'], y=ti['MACD'],
                name="MACD", line=dict(width=2)
            ),
            row=2, col=1
        )
        ind_fig.add_trace(
            go.Scatter(
                x=ti['timestamp'], y=ti['Signal_Line'],
                name="Signal Line", line=dict(width=2, dash='dot')
            ),
            row=2, col=1
        )
    ind_fig.update_yaxes(title_text="MACD", row=2, col=1)

    ind_fig.update_layout(
        template='plotly_white',
        height=500,
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode='x'
    )

    st.plotly_chart(ind_fig, use_container_width=True)

    # --- LLM Insights ---
    st.sidebar.subheader("Get Insights from Indicators")
    if st.sidebar.button("Run Indicator Feedback"):
        summary = []
        if enable_rsi:
            summary.append(f"Latest RSI: {ti['RSI'].iloc[-1]:.2f}")
        if enable_macd:
            summary.append(f"Latest MACD: {ti['MACD'].iloc[-1]:.2f}, Signal: {ti['Signal_Line'].iloc[-1]:.2f}")
        if enable_sma:
            sma_col = f"SMA_{params['sma_window']}"
            summary.append(f"Latest SMA: {ti[sma_col].iloc[-1]:.2f}")
        if enable_ema:
            ema_col = f"EMA_{params['ema_span']}"
            summary.append(f"Latest EMA: {ti[ema_col].iloc[-1]:.2f}")
        if enable_bb:
            summary.append(f"Latest Bollinger Bands: Upper {ti['Upper_Band'].iloc[-1]:.2f}, Lower {ti['Lower_Band'].iloc[-1]:.2f}")
        feedback = get_indicator_feedback("\n".join(summary))
        st.subheader("LLM Insights & Recommendations")
        st.write(feedback)

if __name__ == "__main__":
    main()