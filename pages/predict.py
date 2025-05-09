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

# --- OpenAI / Gemini Setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = genai.GenerativeModel("gemini-2.0-flash")
# --- Device Setup ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Advanced LSTM with Attention ---
class AttnLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=3, dropout=0.3, seq_length=60):
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
        out, _ = self.lstm(x)               # shape: (batch, seq_len, hidden)
        # attention weights on last LSTM hidden state
        scores = self.attn(out[:, -1, :])   # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(1)  # (batch,1,seq_len)
        context = torch.bmm(weights, out).squeeze(1)         # (batch, hidden)
        context = self.dropout(context)
        return self.fc(context)

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



@st.cache_data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length, 0])  # predict close price
    return np.array(xs), np.array(ys)

# Generate LLM-based insight
@st.cache_data
@st.cache_data
def generate_insight(actual, forecast):
    trend = "up" if forecast[-1] > actual[-1] else "down"
    prompt = (
        f"Predicted stock closing price is moving {trend}. "
        f"Last actual price: ₹{actual[-1]:.2f}, "
        f"Predicted price: ₹{forecast[-1]:.2f}. "
        "Give a short, practical trading insight for a retail investor based on this movement."
    )
    response = MODEL.generate_content(prompt)
    return response.text

# --- Streamlit App UI ---
st.title("📈 Advanced Stock Prediction with LSTM+ARIMA with AI Insights")

# Load data & select
data_dict = load_stock_data()
stock_list = list(data_dict.keys())
stock_name = st.sidebar.selectbox("Select Stock", stock_list)

# Advanced settings
with st.sidebar.expander("Model & Training Settings"):
    SEQ_LEN   = st.number_input("Sequence Length", min_value=10, max_value=200, value=60)
    BATCH_SIZE= st.number_input("Batch Size",    min_value=8,  max_value=128, value=32)
    EPOCHS    = st.number_input("Epochs",        min_value=10, max_value=500, value=10)
    LR        = st.number_input("Learning Rate", format="%.5f", value=0.001)
    TRAIN_SPLIT = st.slider("Train/Test Split (%)", 50, 90, 80)
    APPLY_INC = st.checkbox("Use Technical Indicators (RSI, ATR, OBV)", value=True)

if st.button("Run Prediction 🚀"):
    progress = st.progress(0)
    # 1. Prepare data
    df_raw = data_dict[stock_name].copy()
    df = df_raw[['timestamp','open','high','low','close','volume']].copy()
    df.set_index('timestamp', inplace=True)
    if APPLY_INC:
        df = compute_indicators(df)
    features = df[['close','RSI','ATR','OBV']].values if APPLY_INC else df[['close']].values

    # scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # sequences
    X, y = create_sequences(scaled, SEQ_LEN)
    split = int(len(X)*TRAIN_SPLIT/100)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # tensors & loader
    X_train_t = torch.tensor(X_train).float().to(DEVICE)
    y_train_t = torch.tensor(y_train).float().unsqueeze(1).to(DEVICE)
    train_loader = DataLoader(TensorDataset(X_train_t,y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize model
    model = AttnLSTM(input_size=X.shape[2], seq_length=SEQ_LEN).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # 3. Train
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
        avg = total_loss/len(train_loader)
        scheduler.step(avg)
        # early stop
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), 'best_model.pt')
        if epoch % 10 == 0:
            st.write(f"Epoch {epoch}/{EPOCHS} — Loss: {avg:.6f}")
        progress.progress(int(epoch/EPOCHS*100))

    # load best
    model.load_state_dict(torch.load('best_model.pt'))

    # 4. Predict
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X).float().to(DEVICE)
        lstm_preds = model(X_all).cpu().numpy().flatten()
    lstm_rescaled = scaler.inverse_transform(
        np.concatenate([lstm_preds.reshape(-1,1),
                        np.zeros((len(lstm_preds), X.shape[2]-1))], axis=1)
    )[:,0]
    actual = scaler.inverse_transform(
        np.concatenate([y.reshape(-1,1),
                        np.zeros((len(y), X.shape[2]-1))],axis=1)
    )[:,0]

    # 5. ARIMA on residuals
    residuals = actual - lstm_rescaled
    train_res, test_res = residuals[:split], residuals[split:]
    arima_fit = ARIMA(train_res, order=(2,0,2)).fit()
    arima_forecast = arima_fit.forecast(steps=len(test_res))
    hybrid = lstm_rescaled[-len(test_res):] + arima_forecast

    # Results DF
    idx = df.index[-len(test_res):]
    result = pd.DataFrame({'Actual': actual[-len(test_res):],
                           'Hybrid': hybrid}, index=idx)

    # 6. Metrics & Plot
    rmse = np.sqrt(mean_squared_error(result['Actual'], result['Hybrid']))
    mae  = mean_absolute_error(result['Actual'], result['Hybrid'])

    st.subheader("🔍 Actual vs Hybrid Prediction")
    st.line_chart(result)
    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{rmse:.2f}")
    c2.metric("MAE",  f"{mae:.2f}")

    # 7. Download
    csv = result.to_csv().encode()
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

    # 8. LLM Insight
    insight = generate_insight(result['Actual'].values, result['Hybrid'].values)
    st.subheader("AI - Insights")
    st.markdown(insight)

    # 9. SHAP


# Sidebar back
if st.sidebar.button("🔙 Back to Analysis"):
    st.switch_page("main.py")
