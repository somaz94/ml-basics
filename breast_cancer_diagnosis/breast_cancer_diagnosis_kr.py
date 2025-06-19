import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (유방암 위스콘신 데이터셋)
print("\n유방암 위스콘신 데이터셋 로드 중...")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"데이터 형태: {X.shape}, 타겟 형태: {y.shape}")
print(f"타겟 분포: {np.bincount(y)}")
print(f"타겟 이름: {cancer.target_names}")
print(f"특성 이름: {cancer.feature_names[:5]}...")  # 처음 5개 특성만 표시

# -----------------------
# 데이터 전처리
# -----------------------
print("\n📘 데이터 전처리")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"훈련 세트: {X_train.shape}, 테스트 세트: {X_test.shape}")

# -----------------------
# 지도학습: 로지스틱 회귀와 랜덤 포레스트
# -----------------------
print("\n📘 지도학습 - 유방암 분류")

models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n{name} 학습 중...")
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 평가 지표
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results[name] = {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} 정확도: {acc:.3f}")
    if auc:
        print(f"{name} AUC: {auc:.3f}")

# -----------------------
# 딥러닝: 신경망
# -----------------------
print("\n📘 딥러닝 - 암 분류를 위한 신경망")

# PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 신경망 구축
class CancerNN(nn.Module):
    def __init__(self, input_size=30):
        super(CancerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)  # 2개 클래스: 악성/양성
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 모델 초기화
nn_model = CancerNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# 신경망 학습
print("신경망 학습 중...")
nn_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'에포크 [{epoch+1}/100], 손실: {loss.item():.4f}')

# 신경망 평가
nn_model.eval()
with torch.no_grad():
    outputs = nn_model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    nn_accuracy = accuracy_score(y_test, predicted.numpy())
    nn_probabilities = torch.softmax(outputs, dim=1)[:, 1].numpy()
    nn_auc = roc_auc_score(y_test, nn_probabilities)
    
    results['NeuralNetwork'] = {
        'accuracy': nn_accuracy,
        'auc': nn_auc,
        'predictions': predicted.numpy(),
        'probabilities': nn_probabilities
    }
    
    print(f"신경망 정확도: {nn_accuracy:.3f}")
    print(f"신경망 AUC: {nn_auc:.3f}")

# -----------------------
# 비지도학습: KMeans 군집화
# -----------------------
print("\n📘 비지도학습 - 암 데이터 군집화")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 실제 라벨과 비교하여 군집화 평가
cluster_accuracy = max(
    accuracy_score(y, clusters),
    accuracy_score(y, 1 - clusters)  # 역매핑 시도
)
print(f"KMeans 군집화 정확도: {cluster_accuracy:.3f}")

# -----------------------
# 시각화: 결과 분석
# -----------------------
print("\n📘 시각화 - 결과 분석")

# 1. 특성 중요도 (랜덤 포레스트)
rf_model = models['RandomForest']
feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(15, 10))

# 특성 중요도 그래프
plt.subplot(2, 3, 1)
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('특성 중요도')
plt.title('상위 10개 중요 특성 (랜덤 포레스트)')
plt.gca().invert_yaxis()

# 혼동행렬
for i, (name, result) in enumerate(results.items()):
    plt.subplot(2, 3, i + 2)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} 혼동행렬')
    plt.xlabel('예측값')
    plt.ylabel('실제값')

plt.tight_layout()
plt.show()

# 2. ROC 곡선
plt.figure(figsize=(10, 6))
for name, result in results.items():
    if result['probabilities'] is not None:
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='무작위')
plt.xlabel('거짓 양성률')
plt.ylabel('진짜 양성률')
plt.title('ROC 곡선 비교')
plt.legend()
plt.grid(True)
plt.show()

# 3. 군집화 시각화 (2D를 위한 PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 4))

# 실제 라벨
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('실제 라벨 (악성 vs 양성)')
plt.xlabel('PCA 성분 1')
plt.ylabel('PCA 성분 2')

# 군집화 결과
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('KMeans 군집화 결과')
plt.xlabel('PCA 성분 1')
plt.ylabel('PCA 성분 2')

plt.tight_layout()
plt.show()

# -----------------------
# 모델 비교 및 의료적 통찰
# -----------------------
print("\n📘 모델 비교 및 의료적 통찰")
print("=" * 50)
print("모델 성능 요약:")
print("-" * 30)
for name, result in results.items():
    print(f"{name:15} | 정확도: {result['accuracy']:.3f} | AUC: {result['auc']:.3f}")

print(f"\nKMeans 군집화 | 정확도: {cluster_accuracy:.3f}")
print("=" * 50)

# 의료적 해석
print("\n📘 의료적 해석:")
print("- 양성 (0): 양성 종양 - 일반적으로 위험하지 않음")
print("- 악성 (1): 악성 종양 - 암으로 분류되어 즉시 치료 필요")
print("- 높은 정확도는 의료 진단에 매우 중요")
print("- 거짓 음성(암을 놓친 경우)이 거짓 양성보다 더 위험")

# 특성 통찰
print(f"\n📘 주요 진단 특성 (상위 5개):")
for i, (_, row) in enumerate(feature_importance.head().iterrows()):
    print(f"{i+1}. {row['feature']} (중요도: {row['importance']:.3f})")

print("\n📘 임상적 권장사항:")
print("- 교차 검증을 위해 여러 모델 사용 권장")
print("- 고위험 환자의 정기 검진 및 후속 조치")
print("- 환자 병력 및 추가 검사 고려")
print("- AI는 의료진을 보조하는 도구이지 대체할 수 없음") 