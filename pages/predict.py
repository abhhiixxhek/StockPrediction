import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# --- Streamlit Config ---
st.set_page_config(page_title="Stock Prediction", layout="wide")

# --- Device Setup ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- LSTM Model Definition ---
class AdvancedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, dropout=0.4):
        super(AdvancedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# --- Utility Functions ---
@st.cache_data
def load_stock_data(path="stockdata"):
    data = {}
    for fn in os.listdir(path):
        if fn.endswith('.csv'):
            name = fn[:-4]
            df = pd.read_csv(os.path.join(path, fn), parse_dates=['timestamp'])
            df.sort_values('timestamp', inplace=True)
            data[name] = df
    return data


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- Page Content ---
st.title("📈 Stock Prediction with LSTM + ARIMA Hybrid")

# Load data
data_dict = load_stock_data()
stock_list = list(data_dict.keys())
stock_name = st.selectbox("Select Stock for Prediction", stock_list)

# Prediction parameters
STOCK_NAME = stock_name
SEQ_LEN = st.sidebar.number_input("Sequence Length", min_value=10, max_value=200, value=60)
BATCH_SIZE = st.sidebar.number_input("Batch Size", min_value=8, max_value=128, value=32)
EPOCHS = st.sidebar.number_input("Epochs", min_value=10, max_value=500, value=100)
LR = st.sidebar.number_input("Learning Rate", format="%.5f", value=0.001)
TRAIN_SPLIT = st.sidebar.slider("Train/Test Split (%)", 50, 90, 80)

if st.button("Run Prediction 🚀"):
    with st.spinner("Training models, please wait... 🌕"):
        # Prepare data
        df = data_dict[STOCK_NAME][['timestamp', 'close']].copy()
        df.set_index('timestamp', inplace=True)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['close']].values)

        # Create sequences
        X, y = create_sequences(scaled, SEQ_LEN)
        split_idx = int(len(X) * TRAIN_SPLIT / 100)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.tensor(X_train).float().to(DEVICE)
        y_train_t = torch.tensor(y_train).float().to(DEVICE)
        X_test_t = torch.tensor(X_test).float().to(DEVICE)
        y_test_t = torch.tensor(y_test).float().to(DEVICE)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

        # Initialize model
        model = AdvancedLSTM(input_size=1).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{EPOCHS} — Loss: {epoch_loss/len(train_loader):.6f}")

        # LSTM Predictions
        model.eval()
        with torch.no_grad():
            X_all = torch.tensor(X).float().to(DEVICE)
            lstm_preds = model(X_all).cpu().numpy().flatten()
        lstm_rescaled = scaler.inverse_transform(lstm_preds.reshape(-1,1)).flatten()
        actual = scaler.inverse_transform(y.reshape(-1,1)).flatten()

        # Residuals for ARIMA
        residuals = actual - lstm_rescaled
        train_res = residuals[:split_idx]
        test_res = residuals[split_idx:]

        # Fit ARIMA
        arima_order = (2,0,2)
        arima_model = ARIMA(train_res, order=arima_order)
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test_res))

        # Hybrid forecast
        hybrid_pred = lstm_rescaled[-len(test_res):] + arima_forecast

    # --- Display Results ---
    st.subheader("🔍 Actual vs Hybrid Predictions")
    result_df = pd.DataFrame({
        'Actual': actual[-len(test_res):],
        'Hybrid_Prediction': hybrid_pred
    }, index=df.index[-len(test_res):])
    st.line_chart(result_df)

    # Metrics
    rmse = np.sqrt(mean_squared_error(result_df['Actual'], result_df['Hybrid_Prediction']))
    mae = mean_absolute_error(result_df['Actual'], result_df['Hybrid_Prediction'])
    st.metric("Hybrid RMSE", f"{rmse:.2f}")
    st.metric("Hybrid MAE", f"{mae:.2f}")

# Navigation back to analysis
if st.sidebar.button("🔙 Go to Analysis"):
    st.switch_page("main.py")
