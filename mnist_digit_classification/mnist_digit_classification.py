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

# Data loading (MNIST digits)
print("\nLoading MNIST digits data...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

# Use a subset for faster processing
X = X[:5000]
y = y[:5000]

print(f"Data shape: {X.shape}, Target shape: {y.shape}")

# -----------------------
# Supervised Learning: RandomForest and SVM Classification
# -----------------------
print("\n📘 Supervised Learning - Digit Classification with RandomForest and SVM")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {acc:.3f}")

# -----------------------
# Deep Learning: Simple CNN with PyTorch
# -----------------------
print("\n📘 Deep Learning - CNN for Digit Classification")
# Reshape data for CNN (samples, channels, height, width)
X_cnn = X.reshape(-1, 1, 28, 28) / 255.0
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_cnn)
y_train_tensor = torch.LongTensor(y_train_cnn)
X_test_tensor = torch.FloatTensor(X_test_cnn)
y_test_tensor = torch.LongTensor(y_test_cnn)

# Build simple CNN
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

# Initialize model, loss function, and optimizer
cnn_model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters())

# Train CNN
print("Training CNN...")
cnn_model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = cnn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

# Evaluate CNN
cnn_model.eval()
with torch.no_grad():
    outputs = cnn_model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    cnn_accuracy = accuracy_score(y_test_cnn, predicted.numpy())
    print(f"CNN test accuracy: {cnn_accuracy:.3f}")

# -----------------------
# Unsupervised Learning: KMeans Clustering
# -----------------------
print("\n📘 Unsupervised Learning - Digit Clustering with KMeans")
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans clustering result (0~9):", np.bincount(clusters))

# -----------------------
# Visualization: Sample Images and Results
# -----------------------
print("\n📘 Visualization - Sample Images and Results")

# Sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    row = i // 5
    col = i % 5
    # Find first occurrence of each digit
    digit_idx = np.where(y == i)[0][0]
    axes[row, col].imshow(X[digit_idx].reshape(28, 28), cmap='gray')
    axes[row, col].set_title(f'Digit: {i}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()

# Confusion matrix for RandomForest
y_pred_rf = models['RandomForest'].predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('RandomForest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Clustering vs actual labels
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=clusters[:len(X_test)], cmap='viridis', alpha=0.6)
plt.title('KMeans Clustering (First 2 features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()

# -----------------------
# Model Comparison
# -----------------------
print("\n📘 Model Comparison Summary")
print("=" * 40)
print(f"RandomForest Accuracy: {accuracy_score(y_test, models['RandomForest'].predict(X_test)):.3f}")
print(f"SVM Accuracy: {accuracy_score(y_test, models['SVM'].predict(X_test)):.3f}")
print(f"CNN Accuracy: {cnn_accuracy:.3f}")
print("=" * 40) 