# Project 2: Time Series Prediction with LSTM

## โจทย์โปรเจกต์

สร้างระบบทำนายข้อมูล Time Series ด้วย LSTM/GRU พร้อม API สำหรับการ Prediction และ Dashboard แสดงผลการวิเคราะห์

## วัตถุประสงค์

1. เข้าใจการทำงานกับ Sequential Data
2. สร้างและเทรน LSTM/GRU Model
3. วิเคราะห์และ Visualize ผลลัพธ์
4. สร้าง API สำหรับ Real-time Prediction
5. Deploy Dashboard แบบ Interactive

## Dataset Options

### Option 1: Stock Price Prediction (แนะนำ)
- ทำนายราคาหุ้น
- ข้อมูล: Yahoo Finance API
- ตัวอย่าง: SET50, AAPL, TSLA

### Option 2: Weather Forecasting
- ทำนายอุณหภูมิ, ความชื้น
- ข้อมูล: OpenWeatherMap API
- Historical weather data

### Option 3: Energy Consumption
- ทำนายการใช้ไฟฟ้า
- Dataset: [UCI Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

### Option 4: Custom Data
- Sales Forecasting
- Traffic Prediction
- COVID-19 Cases
- Cryptocurrency Price

## Requirements

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn \
            yfinance fastapi uvicorn plotly dash
```

## ขั้นตอนการทำโปรเจกต์

### Phase 1: Data Collection & Exploration (1-2 ชั่วโมง)

1. **Collect Data**

```python
import yfinance as yf
import pandas as pd

# Download stock data
ticker = "AAPL"
df = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Save
df.to_csv('stock_data.csv')
```

2. **Exploratory Data Analysis**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot time series
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['Close'])
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

# Statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Trend analysis
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()
```

3. **Stationarity Check**

```python
from statsmodels.tsa.stattools import adfuller

# ADF Test
result = adfuller(df['Close'].dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

if result[1] > 0.05:
    print("Data is non-stationary. Consider differencing.")
```

### Phase 2: Data Preprocessing (1-2 ชั่วโมง)

1. **Normalization**

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
```

2. **Create Sequences**

```python
def create_sequences(data, seq_length=60):
    """
    สร้าง sequences สำหรับ LSTM
    Args:
        data: ข้อมูล
        seq_length: จำนวนวันย้อนหลัง
    Returns:
        X: Input sequences
        y: Target values
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create sequences
sequence_length = 60  # ใช้ข้อมูล 60 วันย้อนหลัง
X, y = create_sequences(scaled_data, sequence_length)

# Reshape for LSTM [samples, timesteps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
```

3. **Train/Test Split**

```python
# Split (80% train, 20% test)
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

### Phase 3: Model Development (2-3 ชั่วโมง)

1. **Baseline LSTM Model**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
```

2. **Train Model**

```python
# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_lstm_model.h5', save_best_only=True)
]

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

3. **Alternative: GRU Model**

```python
from tensorflow.keras.layers import GRU

gru_model = Sequential([
    GRU(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    GRU(50, return_sequences=True),
    Dropout(0.2),
    GRU(50),
    Dropout(0.2),
    Dense(1)
])

gru_model.compile(optimizer='adam', loss='mean_squared_error')
```

4. **Bidirectional LSTM**

```python
from tensorflow.keras.layers import Bidirectional

bi_model = Sequential([
    Bidirectional(LSTM(50, return_sequences=True), input_shape=(sequence_length, 1)),
    Dropout(0.2),
    Bidirectional(LSTM(50)),
    Dropout(0.2),
    Dense(1)
])

bi_model.compile(optimizer='adam', loss='mean_squared_error')
```

### Phase 4: Evaluation & Visualization (1-2 ชั่วโมง)

1. **Predictions**

```python
# Predict
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
```

2. **Metrics**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

train_mae = mean_absolute_error(y_train_actual, train_predict)
test_mae = mean_absolute_error(y_test_actual, test_predict)

test_r2 = r2_score(y_test_actual, test_predict)

# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

test_mape = mean_absolute_percentage_error(y_test_actual, test_predict)

print(f"Train RMSE: ${train_rmse:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")
print(f"Test MAE: ${test_mae:.2f}")
print(f"Test MAPE: {test_mape:.2f}%")
print(f"Test R²: {test_r2:.4f}")
```

3. **Visualization**

```python
import plotly.graph_objects as go

# Plot predictions
fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(
    y=y_test_actual.flatten(),
    mode='lines',
    name='Actual',
    line=dict(color='blue')
))

# Predicted
fig.add_trace(go.Scatter(
    y=test_predict.flatten(),
    mode='lines',
    name='Predicted',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='Stock Price Prediction',
    xaxis_title='Time',
    yaxis_title='Price (USD)',
    hovermode='x unified'
)

fig.show()
```

4. **Training History**

```python
# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### Phase 5: Advanced Features (2-3 ชั่วโมง)

1. **Multi-step Prediction**

```python
def predict_future(model, last_sequence, steps=30):
    """ทำนายอนาคต n วัน"""
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(steps):
        # Predict next value
        next_pred = model.predict(current_sequence.reshape(1, sequence_length, 1))
        predictions.append(next_pred[0, 0])

        # Update sequence
        current_sequence = np.append(current_sequence[1:], next_pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Predict next 30 days
future_predictions = predict_future(model, scaled_data[-sequence_length:], steps=30)
```

2. **Confidence Intervals**

```python
# Monte Carlo Dropout
def predict_with_uncertainty(model, X, n_iterations=100):
    predictions = []
    for _ in range(n_iterations):
        pred = model(X, training=True)  # Enable dropout
        predictions.append(pred.numpy())

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    return mean_pred, std_pred

# Get predictions with uncertainty
mean_pred, std_pred = predict_with_uncertainty(model, X_test)
```

3. **Feature Engineering**

```python
# Add technical indicators
def add_features(df):
    df = df.copy()

    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    # Bollinger Bands
    df['BB_upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()

    return df.dropna()
```

### Phase 6: API Development (2-3 ชั่วโมง)

```python
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI(title="Stock Price Prediction API")

# Load model and scaler
model = tf.keras.models.load_model('best_lstm_model.h5')
scaler = joblib.load('scaler.pkl')

class PredictionRequest(BaseModel):
    sequence: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence_interval: tuple[float, float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Prepare data
        sequence = np.array(request.sequence).reshape(-1, 1)
        scaled_sequence = scaler.transform(sequence)
        X = scaled_sequence.reshape(1, -1, 1)

        # Predict
        prediction = model.predict(X)
        prediction = scaler.inverse_transform(prediction)[0, 0]

        # Confidence interval (simplified)
        ci_lower = prediction * 0.95
        ci_upper = prediction * 1.05

        return PredictionResponse(
            prediction=float(prediction),
            confidence_interval=(float(ci_lower), float(ci_upper))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Stock Price Prediction API"}

# Run: uvicorn api:app --reload
```

### Phase 7: Dashboard (2-3 ชั่วโมง)

```python
# dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import yfinance as yf

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

st.title("📈 Stock Price Prediction Dashboard")

# Sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker Symbol", "AAPL")
    days_back = st.slider("Days of Historical Data", 30, 365, 180)
    forecast_days = st.slider("Days to Forecast", 7, 60, 30)

# Load data
@st.cache_data
def load_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d")
    return df

df = load_data(ticker, days_back)

# Display current price
col1, col2, col3, col4 = st.columns(4)
current_price = df['Close'].iloc[-1]
price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
price_change_pct = (price_change / df['Close'].iloc[-2]) * 100

with col1:
    st.metric("Current Price", f"${current_price:.2f}",
              f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
with col2:
    st.metric("High (Today)", f"${df['High'].iloc[-1]:.2f}")
with col3:
    st.metric("Low (Today)", f"${df['Low'].iloc[-1]:.2f}")
with col4:
    st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")

# Plot
tab1, tab2, tab3 = st.tabs(["📊 Historical Data", "🔮 Predictions", "📉 Metrics"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))
    fig.update_layout(title=f"{ticker} Stock Price", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if st.button("Generate Forecast"):
        with st.spinner("Predicting..."):
            # Load model and predict
            model = tf.keras.models.load_model('best_lstm_model.h5')
            # ... prediction logic ...
            st.success("Forecast generated!")

with tab3:
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", "$2.45")
        st.metric("MAE", "$1.89")
    with col2:
        st.metric("MAPE", "2.34%")
        st.metric("R²", "0.9823")

# Run: streamlit run dashboard.py
```

## โครงสร้างโปรเจกต์

```
Project_2_Time_Series_Prediction/
├── data/
│   ├── raw/
│   │   └── stock_data.csv
│   └── processed/
│       ├── scaled_data.npy
│       └── scaler.pkl
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_evaluation.ipynb
├── models/
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   └── best_model.h5
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── api.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## Evaluation Criteria

1. **Model Performance:**
   - RMSE < 5% of mean price
   - MAPE < 10%
   - R² > 0.85

2. **API:**
   - Response time < 200ms
   - Proper error handling
   - API documentation

3. **Dashboard:**
   - Interactive และใช้งานง่าย
   - Real-time data update
   - Clear visualizations

## ส่วนขยาย

1. **Attention Mechanism**
2. **Transformer Model**
3. **Ensemble Methods**
4. **Real-time Streaming Data**
5. **Alert System** (Email/LINE Notify)

## Tips

- เริ่มจาก simple univariate prediction
- ทดสอบหลาย sequence lengths
- ใช้ early stopping เพื่อป้องกัน overfitting
- เปรียบเทียบ LSTM vs GRU vs Bidirectional
- ทำ walk-forward validation

## Resources

- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)

Good luck! 🚀📈
