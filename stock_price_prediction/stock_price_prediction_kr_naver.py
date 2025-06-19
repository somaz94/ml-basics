import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import time
import io
import matplotlib
matplotlib.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 네이버 금융에서 코스피 지수 데이터 크롤링
print("\n[1] 네이버 금융에서 코스피 지수 데이터 다운로드 (최근 3년)")

def get_kospi_data(pages=60):
    url = 'https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI'
    dfs = []
    for page in range(1, pages+1):
        pg_url = f'{url}&page={page}'
        res = requests.get(pg_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        })
        if res.status_code != 200:
            print(f"{page}페이지 응답 오류: {res.status_code}")
            continue
        tables = pd.read_html(io.StringIO(res.text), header=0)
        found = False
        for df in tables:
            if any('날짜' in col for col in df.columns) and any('체결가' in col for col in df.columns):
                dfs.append(df)
                found = True
                break
        if not found:
            print(f"{page}페이지: 유효 테이블 없음, 컬럼들: {[list(df.columns) for df in tables]}")
        if page % 10 == 0:
            print(f"{page}페이지 수집 완료")
        time.sleep(0.1)
    if not dfs:
        raise ValueError("유효한 데이터를 찾지 못했습니다.")
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna()
    data = data.rename(columns={'날짜':'Date', '체결가':'Close'})
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    return data[['Date', 'Close']]

kospi = get_kospi_data(pages=60)
print(kospi.tail())

# 2. 데이터 전처리 및 시계열 특성 생성
print("\n[2] 데이터 전처리 및 시계열 특성 생성")
kospi['Close'] = pd.to_numeric(kospi['Close'], errors='coerce')
scaler = MinMaxScaler()
kospi['Close_scaled'] = scaler.fit_transform(kospi[['Close']])

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
X, y = create_sequences(kospi['Close_scaled'].values, seq_length)

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
plt.plot(range(len(kospi)), kospi['Close_scaled'], label='실제값')
plt.plot(range(split+seq_length, split+seq_length+len(pred_lr)), pred_lr, label='선형회귀 예측')
plt.plot(range(split+seq_length, split+seq_length+len(pred_lstm)), pred_lstm, label='LSTM 예측')
plt.legend()
plt.title('코스피 지수 예측 결과 비교')
plt.xlabel('Time')
plt.ylabel('Scaled KOSPI')
plt.show()

# 6. 예측 결과 표로 출력
print("\n[6] 예측 결과(마지막 10개)")
result_df = pd.DataFrame({
    '날짜': kospi['Date'].iloc[split+seq_length:split+seq_length+len(pred_lr)].values,
    '실제값': kospi['Close'].iloc[split+seq_length:split+seq_length+len(pred_lr)].values,
    '선형회귀 예측': scaler.inverse_transform(pred_lr.reshape(-1,1)).flatten(),
    'LSTM 예측': scaler.inverse_transform(pred_lstm.reshape(-1,1)).flatten()
})
print(result_df.tail(10))

# 7. 실무 팁
print("\n[7] 실무 팁: 네이버 금융 데이터는 페이지별로 수집, 결측치/이상치 주의, 시계열 윈도우 다양화 필요") 