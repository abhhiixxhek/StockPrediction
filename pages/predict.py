import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import google.generativeai as genai
from datetime import timedelta
import ta  # Technical Analysis library

# --- Streamlit Config ---
st.set_page_config(page_title="📈 Advanced Stock Predictor", layout="wide")

# --- Gemini AI Setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = genai.GenerativeModel("gemini-2.0-flash")

# --- Device Setup ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- LSTM with Attention Model ---
class AttnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, dropout=0.4, seq_length=60):
        super().__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.attn = nn.Linear(hidden_size, seq_length)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        scores = self.attn(out[:, -1, :])
        weights = torch.softmax(scores, dim=1).unsqueeze(1)
        context = torch.bmm(weights, out).squeeze(1)
        return self.fc(self.dropout(context))


# --- Utility Functions ---
@st.cache_data
def load_stock_data(path="yahoostockdata"):
    data = {}
    for fn in os.listdir(path):
        if fn.endswith('.csv'):
            name = fn[:-4]
            df = pd.read_csv(os.path.join(path, fn), parse_dates=[0])
            df.columns = [c.lower() for c in df.columns]
            # df.rename(columns={'date': 'date_col'}, inplace=True)
            df.sort_values('timestamp', inplace=True)
            df.set_index('timestamp', inplace=True)
            data[name] = df
    return data

@st.cache_data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length, 0])
    return np.array(xs), np.array(ys)

@st.cache_data
def generate_insight(actual, forecast):
    trend = "up" if forecast[-1] > actual[-1] else "down"
    prompt = (
        f"Predicted stock closing price is moving {trend}. "
        f"Last actual price: ₹{actual[-1]:.2f}, "
        f"Predicted price: ₹{forecast[-1]:.2f}. "
        "Give a short, practical trading insight for a retail investor based on this movement."
    )
    return MODEL.generate_content(prompt).text


# --- Streamlit UI ---
st.title("📈 Advanced Stock Prediction with Indicators, LSTM+ARIMA and AI Insights")

# Load data
data_dict = load_stock_data()
stock_name = st.sidebar.selectbox("Select Stock", list(data_dict.keys()))

# Sidebar: model settings
with st.sidebar.expander("Model & Training Settings"):
    SEQ_LEN    = st.number_input("Sequence Length", 10, 200, 60)
    BATCH_SIZE = st.number_input("Batch Size", 8, 128, 32)
    EPOCHS     = st.number_input("Epochs", 10, 500, 50)
    LR         = st.number_input("Learning Rate", format="%.5f", value=0.001)
    TRAIN_SPLIT= st.slider("Train/Test Split (%)", 50, 90, 80)

# Prepare dataframe and indicators
df = data_dict[stock_name].copy()
df = df[df.index >= df.index.max() - timedelta(days=365)]   # ← train on last 1 year only
df['EMA_50']  = df['close'].ewm(span=50).mean()
df['EMA_200'] = df['close'].ewm(span=200).mean()
df['RSI']     = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

# Display full data + indicators
st.subheader("📊 Full Data & Indicators")
st.line_chart(df[['close', 'EMA_50', 'EMA_200']].dropna())
st.line_chart(df[['RSI']].dropna())
st.markdown("**RSI thresholds:** 30 = Oversold, 70 = Overbought")

# # Visual context: last 2 years
# st.subheader("🕒 Last 2 Years Overview")
# recent_df = df.last('730D')
# st.line_chart(recent_df[['close', 'EMA_50', 'EMA_200']])

# Prediction button
if st.button("Run Prediction 🚀"):
    progress = st.progress(0)

    # Use full data for model training
    raw_close = df[['close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw_close)

    X, y = create_sequences(scaled, SEQ_LEN)
    split = int(len(X) * TRAIN_SPLIT / 100)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # Prepare DataLoader
    X_train_t = torch.tensor(X_train).float().to(DEVICE)
    y_train_t = torch.tensor(y_train).float().unsqueeze(1).to(DEVICE)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss, optimizer
    model = AttnLSTM(input_size=1, seq_length=SEQ_LEN).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min', patience=5)

    # Training loop
    best_loss = float('inf')
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        scheduler.step(avg)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), 'best_model.pt')
        if epoch % 10 == 0:
            st.write(f"Epoch {epoch}/{EPOCHS} — Loss: {avg:.6f}")
        progress.progress(int(epoch / EPOCHS * 100))

    # Load best model and predict on all sequences
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X).float().to(DEVICE)
        lstm_preds = model(X_all).cpu().numpy().flatten()

    # Rescale predictions & actuals
    lstm_rescaled = scaler.inverse_transform(lstm_preds.reshape(-1,1)).flatten()
    actual = scaler.inverse_transform(y.reshape(-1,1)).flatten()

    # ARIMA on residuals
    residuals = actual - lstm_rescaled
    train_res, test_res = residuals[:split], residuals[split:]
    arima_fit = ARIMA(train_res, order=(2,0,2)).fit()
    arima_forecast = arima_fit.forecast(steps=len(test_res))
    hybrid = lstm_rescaled[-len(test_res):] + arima_forecast

    # Prepare results DataFrame
    idx = df.index[-len(test_res):]
    result = pd.DataFrame({'Actual': actual[-len(test_res):],
                           'Hybrid': hybrid},
                           index=idx)

    # Metrics & plots
    rmse = np.sqrt(mean_squared_error(result['Actual'], result['Hybrid']))
    mae  = mean_absolute_error(result['Actual'], result['Hybrid'])

    st.subheader("📉 Actual vs Hybrid Forecast")
    st.line_chart(result)
    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{rmse:.2f}")
    c2.metric("MAE",  f"{mae:.2f}")

    # Download CSV
    csv = result.to_csv().encode()
    st.download_button("📥 Download Forecast", csv,
                       "predictions.csv", "text/csv")

    # AI-generated insight
    st.subheader("💡AI Trading Insight")
    st.markdown(generate_insight(result['Actual'].values,
                                result['Hybrid'].values))

# Back button
if st.sidebar.button("🔙 Back to Analysis"):
    st.switch_page("main.py")
