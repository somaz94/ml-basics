import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (와인 품질 데이터)
print("\n와인 품질 데이터 로드 중...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
X = df.drop('quality', axis=1)
y = df['quality']

print(f"데이터 shape: {X.shape}, 타겟 shape: {y.shape}")

# -----------------------
# 지도학습: 품질 등급 분류 (RandomForest)
# -----------------------
print("\n📘 지도학습 - RandomForest로 와인 품질 등급 분류")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"RandomForest 분류 정확도: {acc:.3f}")

# -----------------------
# 지도학습: 품질 점수 회귀 (LinearRegression)
# -----------------------
print("\n📘 지도학습 - LinearRegression으로 와인 품질 점수 예측")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_reg = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_reg)
print(f"LinearRegression MSE: {mse:.3f}")

# -----------------------
# 비지도학습: KMeans 군집화
# -----------------------
print("\n📘 비지도학습 - KMeans로 와인 군집화")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans 군집 결과(0~2):", np.bincount(clusters))
print("실제 품질 등급 분포:", np.bincount(y))

# -----------------------
# 시각화: 알코올-산도-품질 산점도
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X['alcohol'], X['fixed acidity'], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('알코올')
plt.ylabel('산도')
plt.title('실제 품질 등급')

plt.subplot(1,2,2)
plt.scatter(X['alcohol'], X['fixed acidity'], c=clusters, cmap='viridis', edgecolor='k')
plt.xlabel('알코올')
plt.ylabel('산도')
plt.title('KMeans 군집')
plt.tight_layout()
plt.show() 