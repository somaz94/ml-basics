import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (seaborn 내장)
print("\n타이타닉 데이터 로드 중...")
df = sns.load_dataset('titanic')
print(df.head())

# 주요 특성만 사용 & 결측치/범주형 처리
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = df[features + ['survived']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[features]
y = df['survived']

print(f"데이터 shape: {X.shape}, 타겟 shape: {y.shape}")

# -----------------------
# 지도학습: 생존 예측 (DecisionTree, RandomForest, LogisticRegression)
# -----------------------
print("\n📘 지도학습 - 생존 예측")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=200)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} 정확도: {acc:.3f}")

# -----------------------
# 비지도학습: KMeans 군집화
# -----------------------
print("\n📘 비지도학습 - KMeans로 승객 그룹 분석")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans 군집 결과(0/1):", np.bincount(clusters))
print("실제 생존자 분포:", np.bincount(y))
print("군집-실제 생존자 혼동 행렬:\n", confusion_matrix(y, clusters))

# -----------------------
# 시각화: 나이-요금-생존 산점도
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X['age'], X['fare'], c=y, cmap='coolwarm', edgecolor='k')
plt.xlabel('나이')
plt.ylabel('요금')
plt.title('실제 생존 여부')

plt.subplot(1,2,2)
plt.scatter(X['age'], X['fare'], c=clusters, cmap='coolwarm', edgecolor='k')
plt.xlabel('나이')
plt.ylabel('요금')
plt.title('KMeans 군집')
plt.tight_layout()
plt.show() 