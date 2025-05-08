import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import load
from statsmodels.tsa.arima.model import ARIMAResults

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# --- Technical Indicators ---
@st.cache_data
def calculate_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    mid = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['Upper_Band'] = mid + 2 * std
    df['Lower_Band'] = mid - 2 * std
    return df.dropna()

def prepare_indicator_data(df):
    df2 = calculate_technical_indicators(df)
    df2['Price_Up'] = (df2['close'].shift(-1) > df2['close']).astype(int)
    X = df2[['SMA_20','SMA_50','EMA_20','RSI','MACD','Signal_Line']]
    y = df2['Price_Up']
    return X, y

# --- Advanced LSTM Definition ---
class AdvancedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.drop(last))

# --- Load Pretrained Models ---
@st.cache_resource
def load_models():
    # Pure LSTM
    scaler_lstm   = load('models/scaler.joblib')
    lstm_model    = AdvancedLSTM().to('cpu')
    lstm_model.load_state_dict(torch.load('models/lstm.pth', map_location='cpu'))
    lstm_model.eval()
    # Hybrid LSTM + ARIMA
    scaler_hybrid   = load('models/scalerhybrid.joblib')
    hybrid_model    = AdvancedLSTM().to('cpu')
    hybrid_model.load_state_dict(torch.load('models/lstmhybrid.pth', map_location='cpu'))
    hybrid_model.eval()
    arima_mod       = load('models/arima.joblib')  # ARIMAResults instance
    return scaler_lstm, lstm_model, scaler_hybrid, hybrid_model, arima_mod

# --- Sequence Builder ---
def create_sequences(data, seq_len=60):
    xs = []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
    return np.array(xs)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")
    st.sidebar.title("Navigation")

    # — Load data —
    stock_data = {}
    for fn in os.listdir("stock_data"):
        if fn.endswith(".csv"):
            name = fn[:-4]
            df = pd.read_csv(f"stock_data/{fn}", parse_dates=['timestamp'])
            stock_data[name] = df

    stock = st.sidebar.selectbox("Select Stock", list(stock_data))
    inds = st.sidebar.multiselect(
        "Select Technical Indicators",
        ['SMA','EMA','RSI','MACD','Bollinger Bands']
    )

    df = stock_data[stock]
    st.title(f"{stock} Analysis & Prediction")
    st.subheader("Historical Data")
    st.dataframe(df.tail(5), use_container_width=True)

    # — Price + Volume plot —
    st.subheader("Price & Volume")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3],
                        subplot_titles=("Price","Volume"))

    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Price"), row=1, col=1)

    ti = calculate_technical_indicators(df)
    if 'SMA' in inds:
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['SMA_20'],
                       name="SMA 20", line=dict(width=1)), row=1,col=1)
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['SMA_50'],
                       name="SMA 50", line=dict(width=1)), row=1,col=1)
    if 'Bollinger Bands' in inds:
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['Upper_Band'],
                       name="Upper BB", line=dict(width=1)), row=1,col=1)
        fig.add_trace(go.Scatter(x=ti['timestamp'], y=ti['Lower_Band'],
                       name="Lower BB", fill='tonexty', line=dict(width=1)),
                       row=1,col=1)

    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'],
                        name="Volume", marker_color='rgba(0,0,255,0.3)'),
                  row=2,col=1)
    fig.update_layout(showlegend=True, height=800,
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # — Indicator subplots —
    if inds:
        st.subheader("Selected Indicators")
        ind_fig = go.Figure()
        for kind in inds:
            if kind=='RSI':
                ind_fig.add_trace(go.Scatter(
                    x=ti['timestamp'], y=ti['RSI'], name="RSI"))
            if kind=='MACD':
                ind_fig.add_trace(go.Scatter(
                    x=ti['timestamp'], y=ti['MACD'], name="MACD"))
                ind_fig.add_trace(go.Scatter(
                    x=ti['timestamp'], y=ti['Signal_Line'], name="Signal"))
            if kind=='EMA':
                ind_fig.add_trace(go.Scatter(
                    x=ti['timestamp'], y=ti['EMA_20'], name="EMA"))
        ind_fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(ind_fig, use_container_width=True)

    # — Sidebar: Prediction —
    st.sidebar.subheader("Prediction Settings")
    if st.sidebar.button("Run Indicator Model"):
        if not inds:
            st.sidebar.error("Select ≥1 indicator")
        else:
            X, y = prepare_indicator_data(df)

    if st.sidebar.button("Run LSTM + ARIMA Hybrid"):
        scaler_lstm, lstm_model, scaler_hybrid, hybrid_model, arima_mod = load_models()

        series = df[['timestamp','close']].set_index('timestamp')
        data = scaler_lstm.transform(series)

        Xs, ys = create_sequences(data, seq_len=60)

        # Take last 60 → predict next LSTM
        last_seq = torch.tensor(Xs[-1:]).float()
        lstm_out = lstm(last_seq).detach().numpy().flatten()  # shape (1,)
        lstm_price = scaler.inverse_transform(lstm_out.reshape(-1,1)).item()

        # ARIMA correction on residuals
        # (ARIMA was trained on residuals of train set)
        arima_fore = arima_mod.forecast(steps=1)[0]
        corrected = lstm_price + arima_fore

        st.subheader("Hybrid Forecast")
        st.metric("LSTM Only", f"{lstm_price:.2f}")
        st.metric("ARIMA Residual Corr.", f"{arima_fore:.2f}")
        st.metric("Final Forecast", f"{corrected:.2f}")

if __name__ == "__main__":
    main()