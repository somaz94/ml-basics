import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Download stock price data
print("\n[1] Downloading stock price data from Yahoo Finance")
ticker = 'AAPL'  # Example: Apple
stock = yf.download(ticker, start='2018-01-01', end='2023-01-01')
print(stock.head())

# 2. Data preprocessing and time series feature creation
print("\n[2] Data preprocessing and time series feature creation")
stock = stock[['Close']].dropna()
scaler = MinMaxScaler()
stock['Close_scaled'] = scaler.fit_transform(stock[['Close']])

def create_sequences(data, seq_length=20):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
X, y = create_sequences(stock['Close_scaled'].values, seq_length)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Linear Regression model
print("\n[3] LinearRegression model training and prediction")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, pred_lr)
print(f"LinearRegression MSE: {mse_lr:.5f}")

# 4. LSTM model
print("\n[4] LSTM deep learning model training and prediction")
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

X_train_torch = torch.FloatTensor(X_train).unsqueeze(-1)
y_train_torch = torch.FloatTensor(y_train).unsqueeze(-1)
X_test_torch = torch.FloatTensor(X_test).unsqueeze(-1)
y_test_torch = torch.FloatTensor(y_test).unsqueeze(-1)

model = StockLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_torch)
    loss = criterion(output, y_train_torch)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")

model.eval()
with torch.no_grad():
    pred_lstm = model(X_test_torch).squeeze().numpy()
mse_lstm = mean_squared_error(y_test, pred_lstm)
print(f"LSTM MSE: {mse_lstm:.5f}")

# 5. Visualization
print("\n[5] Visualization of prediction results")
plt.figure(figsize=(14,6))
plt.plot(range(len(stock)), stock['Close_scaled'], label='Actual')
plt.plot(range(split+seq_length, split+seq_length+len(pred_lr)), pred_lr, label='LinearRegression Prediction')
plt.plot(range(split+seq_length, split+seq_length+len(pred_lstm)), pred_lstm, label='LSTM Prediction')
plt.legend()
plt.title('Stock Price Prediction Comparison')
plt.xlabel('Time')
plt.ylabel('Scaled Price')
plt.show()

# 6. Practical tips
print("\n[6] Practical tip: Beware of overfitting in time series forecasting, try various window/features") 