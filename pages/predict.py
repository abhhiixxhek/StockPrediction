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

# --- Streamlit Config ---
st.set_page_config(page_title="📈 Advanced Stock Predictor", layout="wide")

# --- Gemini AI Setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = genai.GenerativeModel("gemini-2.0-flash")

# --- Device Setup ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- LSTM Model with Attention ---
class AttnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, dropout=0.3, seq_length=60):
        super(AttnLSTM, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.attn = nn.Linear(hidden_size, seq_length)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        scores = self.attn(out[:, -1, :])
        weights = torch.softmax(scores, dim=1).unsqueeze(1)
        context = torch.bmm(weights, out).squeeze(1)
        context = self.dropout(context)
        return self.fc(context)

# --- Utility Functions ---
@st.cache_data
def load_stock_data(path="yahoostockdata"):
    data = {}
    for fn in os.listdir(path):
        if fn.endswith('.csv'):
            name = fn[:-4]
            df = pd.read_csv(os.path.join(path, fn), parse_dates=[0])
            # normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            # expected: date, open, high, low, close, volume
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'date_col'})
            else:
                df['date_col'] = df.index
            df = df.rename(columns={
                'open': 'open', 'high': 'high', 'low': 'low',
                'close': 'close', 'volume': 'volume'
            })
            df.sort_values('date_col', inplace=True)
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
st.title("📈 Advanced Stock Prediction with LSTM+ARIMA and AI Insights")

data_dict = load_stock_data()
stock_list = list(data_dict.keys())
stock_name = st.sidebar.selectbox("Select Stock", stock_list)

with st.sidebar.expander("Model & Training Settings"):
    SEQ_LEN    = st.number_input("Sequence Length", 10, 200, 60)
    BATCH_SIZE = st.number_input("Batch Size", 8, 128, 32)
    EPOCHS     = st.number_input("Epochs", 10, 500, 10)
    LR         = st.number_input("Learning Rate", format="%.5f", value=0.001)
    TRAIN_SPLIT= st.slider("Train/Test Split (%)", 50, 90, 80)

if st.button("Run Prediction 🚀"):
    progress = st.progress(0)

    df_raw = data_dict[stock_name].copy()
    # use lowercase names
    df = df_raw[['date_col', 'close']].copy()
    df.set_index('date_col', inplace=True)
    features = df.values  # shape (n,1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = create_sequences(scaled, SEQ_LEN)
    split = int(len(X) * TRAIN_SPLIT / 100)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    X_train_t = torch.tensor(X_train).float().to(DEVICE)
    y_train_t = torch.tensor(y_train).float().unsqueeze(1).to(DEVICE)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    model = AttnLSTM(input_size=1, seq_length=SEQ_LEN).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

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

    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X).float().to(DEVICE)
        lstm_preds = model(X_all).cpu().numpy().flatten()

    lstm_rescaled = scaler.inverse_transform(
        np.concatenate([lstm_preds.reshape(-1,1), np.zeros((len(lstm_preds),0))], axis=1)
    )[:,0]
    actual = scaler.inverse_transform(
        np.concatenate([y.reshape(-1,1), np.zeros((len(y),0))], axis=1)
    )[:,0]

    residuals = actual - lstm_rescaled
    train_res, test_res = residuals[:split], residuals[split:]
    arima_fit = ARIMA(train_res, order=(2,0,2)).fit()
    arima_forecast = arima_fit.forecast(steps=len(test_res))
    hybrid = lstm_rescaled[-len(test_res):] + arima_forecast

    idx = df.index[-len(test_res):]
    result = pd.DataFrame({'Actual': actual[-len(test_res):], 'Hybrid': hybrid}, index=idx)

    rmse = np.sqrt(mean_squared_error(result['Actual'], result['Hybrid']))
    mae  = mean_absolute_error(result['Actual'], result['Hybrid'])

    st.subheader("🔍 Actual vs Hybrid Prediction")
    st.line_chart(result)
    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{rmse:.2f}")
    c2.metric("MAE",  f"{mae:.2f}")

    csv = result.to_csv().encode()
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

    insight = generate_insight(result['Actual'].values, result['Hybrid'].values)
    st.subheader("AI - Insights")
    st.markdown(insight)

if st.sidebar.button("🔙 Back to Analysis"):
    st.switch_page("main.py")
