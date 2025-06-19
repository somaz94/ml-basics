import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 합성 신용카드 사기 데이터셋 생성
print("\n합성 신용카드 사기 데이터셋 생성 중...")
np.random.seed(42)

# 특성 생성
n_samples = 10000
n_features = 28  # V1-V28 (익명화된 특성)

# 정상 거래 생성 (클래스 0)
normal_samples = int(n_samples * 0.995)  # 99.5% 정상
fraud_samples = n_samples - normal_samples  # 0.5% 사기

# 정상 거래: 낮은 값, 더 집중된 분포
X_normal = np.random.normal(0, 1, (normal_samples, n_features))
# 사기 거래: 높은 값, 더 분산된 분포
X_fraud = np.random.normal(2, 2, (fraud_samples, n_features))

# 데이터 결합
X = np.vstack([X_normal, X_fraud])
y = np.hstack([np.zeros(normal_samples), np.ones(fraud_samples)])

# 거래 금액 특성 추가 (정상: $10-1000, 사기: $1-10000)
amount_normal = np.random.uniform(10, 1000, normal_samples)
amount_fraud = np.random.uniform(1, 10000, fraud_samples)
amount = np.hstack([amount_normal, amount_fraud])

# 금액을 특성으로 추가
X = np.column_stack([X, amount])

print(f"데이터셋 형태: {X.shape}")
print(f"클래스 분포: {np.bincount(y.astype(int))}")
print(f"사기 비율: {fraud_samples/n_samples*100:.2f}%")

# -----------------------
# 데이터 전처리
# -----------------------
print("\n📘 데이터 전처리")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"훈련 세트: {X_train.shape}, 테스트 세트: {X_test.shape}")
print(f"훈련 사기 비율: {np.mean(y_train)*100:.2f}%")

# -----------------------
# 기준 모델 (불균형 데이터)
# -----------------------
print("\n📘 기준 모델 - 불균형 데이터")

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

baseline_results = {}
for name, model in models.items():
    print(f"\n{name} 학습 중 (기준)...")
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 평가 지표
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    baseline_results[name] = {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} 정확도: {acc:.3f}")
    print(f"{name} AUC: {auc:.3f}")

# -----------------------
# SMOTE로 균형 잡힌 데이터
# -----------------------
print("\n📘 SMOTE - 균형 잡힌 데이터")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"균형 잡힌 훈련 세트: {X_train_balanced.shape}")
print(f"균형 잡힌 사기 비율: {np.mean(y_train_balanced)*100:.2f}%")

# 균형 잡힌 데이터로 모델 학습
balanced_results = {}
for name, model in models.items():
    print(f"\n{name} 학습 중 (SMOTE 균형)...")
    
    # 새로운 모델 인스턴스 생성
    if name == 'RandomForest':
        balanced_model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        balanced_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # 모델 학습
    balanced_model.fit(X_train_balanced, y_train_balanced)
    
    # 예측
    y_pred = balanced_model.predict(X_test)
    y_pred_proba = balanced_model.predict_proba(X_test)[:, 1]
    
    # 평가 지표
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    balanced_results[name] = {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} (SMOTE) 정확도: {acc:.3f}")
    print(f"{name} (SMOTE) AUC: {auc:.3f}")

# -----------------------
# 딥러닝: 신경망
# -----------------------
print("\n📘 딥러닝 - 사기 탐지를 위한 신경망")

# PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train_balanced)
y_train_tensor = torch.LongTensor(y_train_balanced)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 신경망 구축
class FraudNN(nn.Module):
    def __init__(self, input_size=29):
        super(FraudNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)  # 2개 클래스: 정상/사기
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
nn_model = FraudNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# 신경망 학습
print("신경망 학습 중...")
nn_model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'에포크 [{epoch+1}/50], 손실: {loss.item():.4f}')

# 신경망 평가
nn_model.eval()
with torch.no_grad():
    outputs = nn_model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    nn_accuracy = accuracy_score(y_test, predicted.numpy())
    nn_probabilities = torch.softmax(outputs, dim=1)[:, 1].numpy()
    nn_auc = roc_auc_score(y_test, nn_probabilities)
    
    balanced_results['NeuralNetwork'] = {
        'accuracy': nn_accuracy,
        'auc': nn_auc,
        'predictions': predicted.numpy(),
        'probabilities': nn_probabilities
    }
    
    print(f"신경망 정확도: {nn_accuracy:.3f}")
    print(f"신경망 AUC: {nn_auc:.3f}")

# -----------------------
# 시각화: 결과 분석
# -----------------------
print("\n📘 시각화 - 결과 분석")

# 1. 클래스 분포 비교
plt.figure(figsize=(15, 10))

# 원본 vs 균형 잡힌 분포
plt.subplot(2, 3, 1)
original_dist = np.bincount(y_train.astype(int))
balanced_dist = np.bincount(y_train_balanced.astype(int))
x = np.arange(2)
width = 0.35

plt.bar(x - width/2, original_dist, width, label='원본', alpha=0.7)
plt.bar(x + width/2, balanced_dist, width, label='SMOTE 균형', alpha=0.7)
plt.xlabel('클래스')
plt.ylabel('개수')
plt.title('클래스 분포: 원본 vs 균형')
plt.xticks(x, ['정상', '사기'])
plt.legend()

# 2. 혼동행렬 비교
for i, (name, result) in enumerate(balanced_results.items()):
    plt.subplot(2, 3, i + 2)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} 혼동행렬')
    plt.xlabel('예측값')
    plt.ylabel('실제값')

plt.tight_layout()
plt.show()

# 3. ROC 곡선 비교
plt.figure(figsize=(12, 5))

# 기준 vs 균형 ROC 곡선
plt.subplot(1, 2, 1)
for name, result in baseline_results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f'{name} 기준 (AUC = {result["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='무작위')
plt.xlabel('거짓 양성률')
plt.ylabel('진짜 양성률')
plt.title('ROC 곡선: 기준 모델')
plt.legend()
plt.grid(True)

# 균형 모델 ROC 곡선
plt.subplot(1, 2, 2)
for name, result in balanced_results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f'{name} 균형 (AUC = {result["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='무작위')
plt.xlabel('거짓 양성률')
plt.ylabel('진짜 양성률')
plt.title('ROC 곡선: 균형 모델')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. 정밀도-재현율 곡선
plt.figure(figsize=(10, 6))
for name, result in balanced_results.items():
    precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
    plt.plot(recall, precision, label=f'{name}')

plt.xlabel('재현율')
plt.ylabel('정밀도')
plt.title('정밀도-재현율 곡선')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# 모델 비교 및 비즈니스 통찰
# -----------------------
print("\n📘 모델 비교 및 비즈니스 통찰")
print("=" * 60)
print("기준 모델 (불균형 데이터):")
print("-" * 40)
for name, result in baseline_results.items():
    print(f"{name:15} | 정확도: {result['accuracy']:.3f} | AUC: {result['auc']:.3f}")

print(f"\n균형 모델 (SMOTE):")
print("-" * 40)
for name, result in balanced_results.items():
    print(f"{name:15} | 정확도: {result['accuracy']:.3f} | AUC: {result['auc']:.3f}")
print("=" * 60)

# 비즈니스 해석
print("\n📘 비즈니스 해석:")
print("- 정상 (0): 정상 거래 - 승인 처리")
print("- 사기 (1): 사기 거래 - 거부 처리")
print("- 거짓 양성: 정상 거래를 사기로 잘못 분류 (고객 불편)")
print("- 거짓 음성: 사기 거래를 정상으로 잘못 분류 (금융 손실)")

# 비용 분석
print(f"\n📘 비용 분석:")
print("- 사기 탐지는 금융 보안에 매우 중요")
print("- 거짓 음성이 거짓 양성보다 더 비용이 큼")
print("- SMOTE는 소수 클래스(사기) 탐지를 개선")
print("- 높은 AUC는 좋은 판별 능력을 나타냄")

# 특성 중요도 (랜덤 포레스트)
rf_model = models['RandomForest']
feature_names = [f'V{i+1}' for i in range(28)] + ['Amount']
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📘 상위 10개 중요 특성:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"{i+1:2d}. {row['feature']:8s} (중요도: {row['importance']:.3f})")

print("\n📘 비즈니스 권장사항:")
print("- 실시간 사기 탐지 시스템 구현")
print("- 더 나은 정확도를 위해 앙상블 방법 사용")
print("- 고객 경험 유지를 위해 거짓 양성률 모니터링")
print("- 새로운 사기 패턴으로 정기적인 모델 재학습")
print("- 거래 금액을 중요한 특성으로 고려") 