import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (boston housing)
print("\n보스턴 집값 데이터 로드 중...")
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# 데이터 타입 확인 및 수치형 변환
print("데이터 타입 확인:")
print(df.dtypes)
print("\n데이터 전처리 중...")

# 모든 컬럼을 수치형으로 변환
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 결측값 확인 및 처리
print(f"결측값 개수: {df.isnull().sum().sum()}")
if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print("결측값 제거 완료")

X = df.drop('MEDV', axis=1)
y = df['MEDV']

print(f"데이터 shape: {X.shape}, 타겟 shape: {y.shape}")

# -----------------------
# 지도학습: 집값 예측 (LinearRegression, RandomForestRegressor)
# -----------------------
print("\n📘 지도학습 - 집값 예측")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse:.2f}")

# -----------------------
# 비지도학습: KMeans 군집화
# -----------------------
print("\n📘 비지도학습 - KMeans로 지역 군집화")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans 군집 결과(0~2):", np.bincount(clusters))

# -----------------------
# 시각화: 방 개수-집값 산점도
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X['RM'], y, c=y, cmap='viridis', edgecolor='k')
plt.xlabel('방 개수(RM)')
plt.ylabel('집값(MEDV)')
plt.title('실제 집값')

plt.subplot(1,2,2)
plt.scatter(X['RM'], y, c=clusters, cmap='viridis', edgecolor='k')
plt.xlabel('방 개수(RM)')
plt.ylabel('집값(MEDV)')
plt.title('KMeans 군집')
plt.tight_layout()
plt.show() 