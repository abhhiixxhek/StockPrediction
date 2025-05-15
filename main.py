import os
import time
import requests
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel

# --- Load environment ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = GenerativeModel("gemini-2.0-flash")

st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")

# --- Datafetcher with yfinance ---
def get_three_years_ago_date() -> str:
    return (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

def fetch_top50_nse() -> list[str]:
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json().get('data', [])
    return [item['symbol'] + '.NS' for item in data]

def fetch_and_store(symbols: list[str], start_date: str, output_dir: str = "stockdata"):
    os.makedirs(output_dir, exist_ok=True)
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(start=start_date)
            if df.empty:
                print(f"No data for {sym}, skipping.")
            else:
                df = df.reset_index().rename(columns={"Date": "timestamp"})
                out = os.path.join(output_dir, f"{sym.replace('.NS','')}.csv")
                df.to_csv(out, index=False)
                print(f"Saved {sym} → {len(df)} rows")
        except Exception as e:
            print(f"Error {sym}: {e}")
        time.sleep(1)

def refresh_stock_data():
    start = get_three_years_ago_date()
    syms = fetch_top50_nse()
    st.sidebar.write(f"Fetching {len(syms)} symbols since {start}…")
    fetch_and_store(syms, start)
    st.sidebar.success("Stock data refreshed.")

# --- Load & compute ---
@st.cache_data
def load_stock_data(start_date: str):
    data = {}
    cutoff = pd.to_datetime(start_date)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    for fn in os.listdir("stockdata"):
        if not fn.lower().endswith(".csv"):
            continue

        symbol = fn.replace(".csv", "")
        path = os.path.join("stockdata", fn)
        try:
            df = pd.read_csv(path)
        except:
            continue

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        elif "Date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["Date"], errors='coerce')
        else:
            continue

        df = df.dropna(subset=["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df = df[df["timestamp"] >= cutoff].copy()
        data[symbol] = df

    return data

@st.cache_data
def calculate_technical_indicators(df, params):
    df = df.sort_values("timestamp").copy()
    for w in params.get("sma_windows", []):
        df[f"SMA_{w}"] = df["Close"].rolling(w).mean()
    for e in params.get("ema_spans", []):
        df[f"EMA_{e}"] = df["Close"].ewm(span=e, adjust=False).mean()
    if params["enable_rsi"]:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(params["rsi_period"]).mean()
        loss = -delta.clip(upper=0).rolling(params["rsi_period"]).mean()
        df["RSI"] = 100 - (100 / (1 + gain / loss))
    if params["enable_macd"]:
        fast = df["Close"].ewm(span=params["macd_fast"], adjust=False).mean()
        slow = df["Close"].ewm(span=params["macd_slow"], adjust=False).mean()
        df["MACD"] = fast - slow
        df["Signal_Line"] = df["MACD"].ewm(span=params["macd_signal"], adjust=False).mean()
    if params["enable_bb"]:
        mid = df["Close"].rolling(params["bb_window"]).mean()
        std = df["Close"].rolling(params["bb_window"]).std()
        df["Upper_Band"] = mid + params["bb_multiplier"] * std
        df["Lower_Band"] = mid - params["bb_multiplier"] * std
    return df.dropna()

@st.cache_data
def compute_summary(df, start_dt, end_dt):
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    sub = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
    if sub.empty:
        return {
            "Period High": None,
            "Period Low": None,
            "Average Close": None,
            "Total Volume": 0,
            "Volatility (%)": 0,
        }
    return {
        "Period High": sub["High"].max(),
        "Period Low": sub["Low"].min(),
        "Average Close": sub["Close"].mean(),
        "Total Volume": sub["Volume"].sum(),
        "Volatility (%)": (sub["Close"].std() / sub["Close"].mean()) * 100 if sub["Close"].mean() != 0 else 0,
    }

def build_plot_summary(df, ti, params, lb=5):
    latest, prev = ti.iloc[-1], ti.iloc[-lb]
    lines = []
    if params["enable_bb"]:
        width = latest["Upper_Band"] - latest["Lower_Band"]
        pct = (latest["Close"] - latest["Lower_Band"]) / width * 100
        lines.append(f"Price at {pct:.1f}% of BB range.")
    for w in params["sma_windows"]:
        c1, c0 = latest[f"SMA_{w}"], prev[f"SMA_{w}"]
        lines.append(f"SMA_{w} slope: {(c1-c0)/c0*100:.2f}%")
    for e in params["ema_spans"]:
        c1, c0 = latest[f"EMA_{e}"], prev[f"EMA_{e}"]
        lines.append(f"EMA_{e} slope: {(c1-c0)/c0*100:.2f}%")
    if params["enable_rsi"]:
        lines.append(f"RSI change: {latest['RSI']-prev['RSI']:.2f}")
    if params["enable_macd"]:
        hist1 = latest["MACD"] - latest["Signal_Line"]
        hist0 = prev["MACD"] - prev["Signal_Line"]
        lines.append(f"MACD hist change: {hist1-hist0:.2f}")
    v1, v0 = df["Volume"].iloc[-1], df["Volume"].iloc[-lb]
    lines.append(f"Volume change: {(v1-v0)/v0*100:.1f}%")
    return "\n".join(lines)

def get_indicator_feedback(ind_sum, plot_sum):
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
        "Technical Indicator Summary:\n" + ind_sum + "\n"
        "Chart-Based Summary:\n" + plot_sum
    )
    return MODEL.generate_content(prompt).text.strip()

# --- App ---
def main():
    st.sidebar.title("Navigation")

    # Load existing data
    start = get_three_years_ago_date()
    all_data = load_stock_data(start)

    # Display latest date in sidebar
    if all_data:
        latest_dates = [df["timestamp"].max() for df in all_data.values()]
        overall_latest = max(latest_dates)
        st.sidebar.markdown(f"**Latest data in DB:** {overall_latest.date()}")
    else:
        st.sidebar.markdown("**No local data yet.**")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Stock Data"):
        with st.spinner("Fetching data..."):
            refresh_stock_data()
        all_data = load_stock_data(start)
        latest_dates = [df["timestamp"].max() for df in all_data.values()]
        overall_latest = max(latest_dates)
        st.sidebar.markdown(f"**Latest data in DB:** {overall_latest.date()}")

    if not all_data:
        st.error("No data available—please refresh.")
        return

    # Select symbol & date range
    sym = st.sidebar.selectbox("Select Stock", sorted(all_data))
    df_full = all_data[sym]
    min_d, max_d = df_full["timestamp"].min(), df_full["timestamp"].max()
    dr = st.sidebar.date_input("Date Range", [min_d, max_d])

    # Filter data
    mask = (df_full["timestamp"] >= pd.to_datetime(dr[0])) & (df_full["timestamp"] <= pd.to_datetime(dr[1]))
    df = df_full.loc[mask]

    # Indicator parameters
    params = {
        "sma_windows": st.sidebar.multiselect("SMA (max 3)", [5,10,20,50,100,200], [20])[:3],
        "ema_spans":  st.sidebar.multiselect("EMA (max 3)", [5,10,20,50,100,200], [20])[:3],
        "enable_rsi": st.sidebar.checkbox("RSI", True),
        "enable_macd": st.sidebar.checkbox("MACD", True),
        "enable_bb":  st.sidebar.checkbox("BB", True),
        "rsi_period": st.sidebar.number_input("RSI Period",5,50,14),
        "macd_fast":  st.sidebar.number_input("MACD Fast",5,50,12),
        "macd_slow":  st.sidebar.number_input("MACD Slow",10,100,26),
        "macd_signal": st.sidebar.number_input("MACD Signal",5,50,9),
        "bb_window":  st.sidebar.number_input("BB Window",5,100,20),
        "bb_multiplier": st.sidebar.number_input("BB Multiplier",1.0,3.0,2.0,0.1),
    }

    st.title(f"{sym} Analysis & Prediction")

    # Summary metrics
    summary = compute_summary(df, pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))
    cols = st.columns(len(summary))
    for c, (k, v) in zip(cols, summary.items()):
        c.metric(k, f"{v:.2f}" if v is not None else "N/A")

    # Historical Data Preview
    st.subheader("Historical Data Preview")
    df_preview = df.drop(columns=["Dividends", "Stock Splits", "timestamp"], errors="ignore")
    st.dataframe(df_preview.tail(10), use_container_width=True, height=300)

    # Technical indicators
    ti = calculate_technical_indicators(df, params)

    # Price & volume chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3], subplot_titles=("Price & MAs","Volume"))
    hover = [f"{ts:%Y-%m-%d}<br>O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f}"
             for ts,o,h,l,c in zip(df['timestamp'], df['Open'], df['High'], df['Low'], df['Close'])]
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price", hovertext=hover, hoverinfo='text'
    ), row=1, col=1)
    for w in params['sma_windows']:
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti[f"SMA_{w}"], name=f"SMA {w}"), row=1, col=1)
    for e in params['ema_spans']:
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti[f"EMA_{e}"], name=f"EMA {e}"), row=1, col=1)
    if params['enable_bb']:
        fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti['Upper_Band'], name="Upper BB", line=dict(dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ti['timestamp'], y=ti['Lower_Band'], name="Lower BB", line=dict(dash='dash')
        ), row=1, col=1)
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['Volume'], showlegend=False), row=2, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # RSI & MACD chart
    ind_fig = go.Figure()
    if params['enable_rsi']:
        ind_fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['RSI'], name="RSI"))
        ind_fig.add_hline(y=70, line_dash='dash', annotation_text='Overbought')
        ind_fig.add_hline(y=30, line_dash='dash', annotation_text='Oversold')
    if params['enable_macd']:
        ind_fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['MACD'], name="MACD"))
        ind_fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['Signal_Line'], name="Signal Line"))
    ind_fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(ind_fig, use_container_width=True)

    # LLM Insights
    if st.sidebar.button("Run Indicator Feedback"):
        ind_sum = []
        ind_sum += [f"Latest SMA {w}: {ti[f'SMA_{w}'].iloc[-1]:.2f}" for w in params['sma_windows']]
        ind_sum += [f"Latest EMA {e}: {ti[f'EMA_{e}'].iloc[-1]:.2f}" for e in params['ema_spans']]
        if params['enable_rsi']:
            ind_sum.append(f"Latest RSI: {ti['RSI'].iloc[-1]:.2f}")
        if params['enable_macd']:
            ind_sum.append(f"Latest MACD: {ti['MACD'].iloc[-1]:.2f}, Signal: {ti['Signal_Line'].iloc[-1]:.2f}")
        if params['enable_bb']:
            ind_sum.append(f"BB Upper: {ti['Upper_Band'].iloc[-1]:.2f}, Lower: {ti['Lower_Band'].iloc[-1]:.2f}")
        plot_sum = build_plot_summary(df, ti, params)
        feedback = get_indicator_feedback("\n".join(ind_sum), plot_sum)
        st.subheader("LLM Insights & Recommendation")
        st.write(feedback)
        st.balloons()

if __name__ == "__main__":
    main()
