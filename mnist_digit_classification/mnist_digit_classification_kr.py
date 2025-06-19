import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (MNIST 손글씨 숫자)
print("\nMNIST 손글씨 숫자 데이터 로드 중...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

# 빠른 처리를 위해 일부 데이터만 사용
X = X[:5000]
y = y[:5000]

print(f"데이터 형태: {X.shape}, 타겟 형태: {y.shape}")

# -----------------------
# 지도학습: RandomForest와 SVM 분류
# -----------------------
print("\n📘 지도학습 - RandomForest와 SVM으로 숫자 분류")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

for name, model in models.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} 정확도: {acc:.3f}")

# -----------------------
# 딥러닝: PyTorch로 간단한 CNN
# -----------------------
print("\n📘 딥러닝 - CNN으로 숫자 분류")
# CNN을 위한 데이터 재구성 (샘플, 채널, 높이, 너비)
X_cnn = X.reshape(-1, 1, 28, 28) / 255.0
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.3, random_state=42)

# PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train_cnn)
y_train_tensor = torch.LongTensor(y_train_cnn)
X_test_tensor = torch.FloatTensor(X_test_cnn)
y_test_tensor = torch.LongTensor(y_test_cnn)

# 간단한 CNN 구축
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 모델, 손실 함수, 옵티마이저 초기화
cnn_model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters())

# CNN 학습
print("CNN 학습 중...")
cnn_model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = cnn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1 == 0:
        print(f'에포크 [{epoch+1}/5], 손실: {loss.item():.4f}')

# CNN 평가
cnn_model.eval()
with torch.no_grad():
    outputs = cnn_model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    cnn_accuracy = accuracy_score(y_test_cnn, predicted.numpy())
    print(f"CNN 테스트 정확도: {cnn_accuracy:.3f}")

# -----------------------
# 비지도학습: KMeans 군집화
# -----------------------
print("\n📘 비지도학습 - KMeans로 숫자 군집화")
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans 군집화 결과 (0~9):", np.bincount(clusters))

# -----------------------
# 시각화: 샘플 이미지와 결과
# -----------------------
print("\n📘 시각화 - 샘플 이미지와 결과")

# 샘플 이미지
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    row = i // 5
    col = i % 5
    # 각 숫자의 첫 번째 발생 찾기
    digit_idx = np.where(y == i)[0][0]
    axes[row, col].imshow(X[digit_idx].reshape(28, 28), cmap='gray')
    axes[row, col].set_title(f'숫자: {i}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()

# RandomForest 혼동행렬
y_pred_rf = models['RandomForest'].predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('RandomForest 혼동행렬')
plt.xlabel('예측값')
plt.ylabel('실제값')

# 군집화 vs 실제 라벨
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=clusters[:len(X_test)], cmap='viridis', alpha=0.6)
plt.title('KMeans 군집화 (첫 2개 특성)')
plt.xlabel('특성 1')
plt.ylabel('특성 2')
plt.tight_layout()
plt.show()

# -----------------------
# 모델 비교
# -----------------------
print("\n📘 모델 비교 요약")
print("=" * 40)
print(f"RandomForest 정확도: {accuracy_score(y_test, models['RandomForest'].predict(X_test)):.3f}")
print(f"SVM 정확도: {accuracy_score(y_test, models['SVM'].predict(X_test)):.3f}")
print(f"CNN 정확도: {cnn_accuracy:.3f}")
print("=" * 40) 