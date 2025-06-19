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

# 1. 데이터 다운로드
print("\n[1] Yahoo Finance에서 주가 데이터 다운로드")
ticker = 'AAPL'  # 예시: 애플
stock = yf.download(ticker, start='2018-01-01', end='2023-01-01')
print(stock.head())

# 2. 데이터 전처리 및 시계열 특성 생성
print("\n[2] 데이터 전처리 및 시계열 특성 생성")
stock = stock[['Close']].dropna()
scaler = MinMaxScaler()
stock['Close_scaled'] = scaler.fit_transform(stock[['Close']])

# 시계열 데이터셋 생성 함수
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

# 학습/테스트 분할
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. 선형회귀 모델
print("\n[3] 선형회귀(LinearRegression) 모델 학습 및 예측")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, pred_lr)
print(f"선형회귀 MSE: {mse_lr:.5f}")

# 4. LSTM 모델
print("\n[4] LSTM 딥러닝 모델 학습 및 예측")
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
        print(f"에포크 {epoch+1}, 손실: {loss.item():.5f}")

model.eval()
with torch.no_grad():
    pred_lstm = model(X_test_torch).squeeze().numpy()
mse_lstm = mean_squared_error(y_test, pred_lstm)
print(f"LSTM MSE: {mse_lstm:.5f}")

# 5. 예측 결과 시각화
print("\n[5] 예측 결과 시각화")
plt.figure(figsize=(14,6))
plt.plot(range(len(stock)), stock['Close_scaled'], label='실제값')
plt.plot(range(split+seq_length, split+seq_length+len(pred_lr)), pred_lr, label='선형회귀 예측')
plt.plot(range(split+seq_length, split+seq_length+len(pred_lstm)), pred_lstm, label='LSTM 예측')
plt.legend()
plt.title('주가 예측 결과 비교')
plt.xlabel('Time')
plt.ylabel('Scaled Price')
plt.show()

# 6. 실무 팁
print("\n[6] 실무 팁: 시계열 데이터는 미래 예측에 과적합 주의, 데이터 윈도우/특성 다양화 필요") 