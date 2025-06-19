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

# Generate synthetic credit card fraud dataset
print("\nGenerating synthetic credit card fraud dataset...")
np.random.seed(42)

# Generate features
n_samples = 10000
n_features = 28  # V1-V28 (anonymized features)

# Generate normal transactions (class 0)
normal_samples = int(n_samples * 0.995)  # 99.5% normal
fraud_samples = n_samples - normal_samples  # 0.5% fraud

# Normal transactions: lower values, more clustered
X_normal = np.random.normal(0, 1, (normal_samples, n_features))
# Fraud transactions: higher values, more spread out
X_fraud = np.random.normal(2, 2, (fraud_samples, n_features))

# Combine data
X = np.vstack([X_normal, X_fraud])
y = np.hstack([np.zeros(normal_samples), np.ones(fraud_samples)])

# Add some amount feature (normal: $10-1000, fraud: $1-10000)
amount_normal = np.random.uniform(10, 1000, normal_samples)
amount_fraud = np.random.uniform(1, 10000, fraud_samples)
amount = np.hstack([amount_normal, amount_fraud])

# Add amount as a feature
X = np.column_stack([X, amount])

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y.astype(int))}")
print(f"Fraud rate: {fraud_samples/n_samples*100:.2f}%")

# -----------------------
# Data Preprocessing
# -----------------------
print("\n📘 Data Preprocessing")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Training fraud rate: {np.mean(y_train)*100:.2f}%")

# -----------------------
# Baseline Models (Imbalanced Data)
# -----------------------
print("\n📘 Baseline Models - Imbalanced Data")

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

baseline_results = {}
for name, model in models.items():
    print(f"\nTraining {name} (baseline)...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    baseline_results[name] = {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} Accuracy: {acc:.3f}")
    print(f"{name} AUC: {auc:.3f}")

# -----------------------
# SMOTE for Balanced Data
# -----------------------
print("\n📘 SMOTE - Balanced Data")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Balanced training set: {X_train_balanced.shape}")
print(f"Balanced fraud rate: {np.mean(y_train_balanced)*100:.2f}%")

# Train models on balanced data
balanced_results = {}
for name, model in models.items():
    print(f"\nTraining {name} (SMOTE balanced)...")
    
    # Create new model instance
    if name == 'RandomForest':
        balanced_model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        balanced_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Train model
    balanced_model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_pred = balanced_model.predict(X_test)
    y_pred_proba = balanced_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    balanced_results[name] = {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} (SMOTE) Accuracy: {acc:.3f}")
    print(f"{name} (SMOTE) AUC: {auc:.3f}")

# -----------------------
# Deep Learning: Neural Network
# -----------------------
print("\n📘 Deep Learning - Neural Network for Fraud Detection")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_balanced)
y_train_tensor = torch.LongTensor(y_train_balanced)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Build neural network
class FraudNN(nn.Module):
    def __init__(self, input_size=29):
        super(FraudNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)  # 2 classes: normal/fraud
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model
nn_model = FraudNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Train neural network
print("Training Neural Network...")
nn_model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}')

# Evaluate neural network
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
    
    print(f"Neural Network Accuracy: {nn_accuracy:.3f}")
    print(f"Neural Network AUC: {nn_auc:.3f}")

# -----------------------
# Visualization: Results Analysis
# -----------------------
print("\n📘 Visualization - Results Analysis")

# 1. Class distribution comparison
plt.figure(figsize=(15, 10))

# Original vs balanced distribution
plt.subplot(2, 3, 1)
original_dist = np.bincount(y_train.astype(int))
balanced_dist = np.bincount(y_train_balanced.astype(int))
x = np.arange(2)
width = 0.35

plt.bar(x - width/2, original_dist, width, label='Original', alpha=0.7)
plt.bar(x + width/2, balanced_dist, width, label='SMOTE Balanced', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution: Original vs Balanced')
plt.xticks(x, ['Normal', 'Fraud'])
plt.legend()

# 2. Confusion matrices comparison
for i, (name, result) in enumerate(balanced_results.items()):
    plt.subplot(2, 3, i + 2)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# 3. ROC curves comparison
plt.figure(figsize=(12, 5))

# Baseline vs Balanced ROC curves
plt.subplot(1, 2, 1)
for name, result in baseline_results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f'{name} Baseline (AUC = {result["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Baseline Models')
plt.legend()
plt.grid(True)

# Balanced models ROC curves
plt.subplot(1, 2, 2)
for name, result in balanced_results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f'{name} Balanced (AUC = {result["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Balanced Models')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. Precision-Recall curves
plt.figure(figsize=(10, 6))
for name, result in balanced_results.items():
    precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
    plt.plot(recall, precision, label=f'{name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# Model Comparison and Business Insights
# -----------------------
print("\n📘 Model Comparison and Business Insights")
print("=" * 60)
print("Baseline Models (Imbalanced Data):")
print("-" * 40)
for name, result in baseline_results.items():
    print(f"{name:15} | Accuracy: {result['accuracy']:.3f} | AUC: {result['auc']:.3f}")

print(f"\nBalanced Models (SMOTE):")
print("-" * 40)
for name, result in balanced_results.items():
    print(f"{name:15} | Accuracy: {result['accuracy']:.3f} | AUC: {result['auc']:.3f}")
print("=" * 60)

# Business interpretation
print("\n📘 Business Interpretation:")
print("- Normal (0): 정상 거래 - 승인 처리")
print("- Fraud (1): 사기 거래 - 거부 처리")
print("- False Positive: 정상 거래를 사기로 잘못 분류 (고객 불편)")
print("- False Negative: 사기 거래를 정상으로 잘못 분류 (금융 손실)")

# Cost analysis
print(f"\n📘 Cost Analysis:")
print("- Fraud detection is critical for financial security")
print("- False negatives are more costly than false positives")
print("- SMOTE helps improve detection of minority class (fraud)")
print("- High AUC indicates good discrimination ability")

# Feature importance (Random Forest)
rf_model = models['RandomForest']
feature_names = [f'V{i+1}' for i in range(28)] + ['Amount']
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📘 Top 10 Important Features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"{i+1:2d}. {row['feature']:8s} (Importance: {row['importance']:.3f})")

print("\n📘 Business Recommendations:")
print("- Implement real-time fraud detection system")
print("- Use ensemble methods for better accuracy")
print("- Monitor false positive rates to maintain customer experience")
print("- Regular model retraining with new fraud patterns")
print("- Consider transaction amount as important feature") 