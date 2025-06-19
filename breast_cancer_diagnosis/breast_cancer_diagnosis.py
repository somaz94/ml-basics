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

# Data loading (Breast Cancer Wisconsin dataset)
print("\nLoading Breast Cancer Wisconsin dataset...")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"Data shape: {X.shape}, Target shape: {y.shape}")
print(f"Target distribution: {np.bincount(y)}")
print(f"Target names: {cancer.target_names}")
print(f"Feature names: {cancer.feature_names[:5]}...")  # Show first 5 features

# -----------------------
# Data Preprocessing
# -----------------------
print("\n📘 Data Preprocessing")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# -----------------------
# Supervised Learning: Logistic Regression and Random Forest
# -----------------------
print("\n📘 Supervised Learning - Breast Cancer Classification")

models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results[name] = {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} Accuracy: {acc:.3f}")
    if auc:
        print(f"{name} AUC: {auc:.3f}")

# -----------------------
# Deep Learning: Neural Network
# -----------------------
print("\n📘 Deep Learning - Neural Network for Cancer Classification")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Build neural network
class CancerNN(nn.Module):
    def __init__(self, input_size=30):
        super(CancerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)  # 2 classes: malignant/benign
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
nn_model = CancerNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Train neural network
print("Training Neural Network...")
nn_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Evaluate neural network
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
    
    print(f"Neural Network Accuracy: {nn_accuracy:.3f}")
    print(f"Neural Network AUC: {nn_auc:.3f}")

# -----------------------
# Unsupervised Learning: KMeans Clustering
# -----------------------
print("\n📘 Unsupervised Learning - Cancer Data Clustering")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Evaluate clustering against actual labels
cluster_accuracy = max(
    accuracy_score(y, clusters),
    accuracy_score(y, 1 - clusters)  # Try inverse mapping
)
print(f"KMeans clustering accuracy: {cluster_accuracy:.3f}")

# -----------------------
# Visualization: Results Analysis
# -----------------------
print("\n📘 Visualization - Results Analysis")

# 1. Feature importance (Random Forest)
rf_model = models['RandomForest']
feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(15, 10))

# Feature importance plot
plt.subplot(2, 3, 1)
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 10 Important Features (Random Forest)')
plt.gca().invert_yaxis()

# Confusion matrices
for i, (name, result) in enumerate(results.items()):
    plt.subplot(2, 3, i + 2)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# 2. ROC curves
plt.figure(figsize=(10, 6))
for name, result in results.items():
    if result['probabilities'] is not None:
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 3. Clustering visualization (PCA for 2D)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 4))

# Actual labels
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('Actual Labels (Malignant vs Benign)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Clustering results
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('KMeans Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()

# -----------------------
# Model Comparison and Medical Insights
# -----------------------
print("\n📘 Model Comparison and Medical Insights")
print("=" * 50)
print("Model Performance Summary:")
print("-" * 30)
for name, result in results.items():
    print(f"{name:15} | Accuracy: {result['accuracy']:.3f} | AUC: {result['auc']:.3f}")

print(f"\nKMeans Clustering | Accuracy: {cluster_accuracy:.3f}")
print("=" * 50)

# Medical interpretation
print("\n📘 Medical Interpretation:")
print("- Benign (0): 양성 종양 - 일반적으로 위험하지 않음")
print("- Malignant (1): 악성 종양 - 암으로 분류되어 즉시 치료 필요")
print("- High accuracy is crucial for medical diagnosis")
print("- False negatives (missed cancer) are more dangerous than false positives")

# Feature insights
print(f"\n📘 Key Diagnostic Features (Top 5):")
for i, (_, row) in enumerate(feature_importance.head().iterrows()):
    print(f"{i+1}. {row['feature']} (Importance: {row['importance']:.3f})")

print("\n📘 Clinical Recommendations:")
print("- Multiple models should be used for cross-validation")
print("- Regular screening and follow-up for high-risk patients")
print("- Consider patient history and additional tests")
print("- AI should assist, not replace, medical professionals") 