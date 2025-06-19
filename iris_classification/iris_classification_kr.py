import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# -----------------------
# 지도학습: SVM, KNN, LogisticRegression
# -----------------------
print("\n📘 지도학습 - SVM, KNN, LogisticRegression로 품종 분류")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=200)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} 정확도: {acc:.3f}")

# -----------------------
# 비지도학습: KMeans 군집화 및 실제 품종과 비교
# -----------------------
print("\n📘 비지도학습 - KMeans로 군집화 및 실제 품종과 비교")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans 군집 결과(0~2):", clusters)
print("실제 품종 라벨(0~2):", y)
print("군집-실제 라벨 혼동 행렬:\n", confusion_matrix(y, clusters))

# -----------------------
# 2D 시각화
# -----------------------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('실제 품종 (2D)')

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis', edgecolor='k')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('KMeans 군집 (2D)')
plt.tight_layout()
plt.show()

# -----------------------
# 3D 시각화
# -----------------------
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='viridis', edgecolor='k')
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.set_zlabel(feature_names[2])
ax1.set_title('실제 품종 (3D)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X[:,0], X[:,1], X[:,2], c=clusters, cmap='viridis', edgecolor='k')
ax2.set_xlabel(feature_names[0])
ax2.set_ylabel(feature_names[1])
ax2.set_zlabel(feature_names[2])
ax2.set_title('KMeans 군집 (3D)')
plt.tight_layout()
plt.show() 