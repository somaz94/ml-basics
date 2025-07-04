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

# Data loading
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# -----------------------
# Supervised Learning: SVM, KNN, LogisticRegression
# -----------------------
print("\n📘 Supervised Learning - Species Classification with SVM, KNN, LogisticRegression")
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
    print(f"{name} accuracy: {acc:.3f}")

# -----------------------
# Unsupervised Learning: KMeans clustering and comparison with actual species
# -----------------------
print("\n📘 Unsupervised Learning - Clustering with KMeans and comparison with actual species")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans clustering result (0~2):", clusters)
print("Actual species labels (0~2):", y)
print("Clustering-Actual labels confusion matrix:\n", confusion_matrix(y, clusters))

# -----------------------
# 2D Visualization
# -----------------------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Actual Species (2D)')

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis', edgecolor='k')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('KMeans Clustering (2D)')
plt.tight_layout()
plt.show()

# -----------------------
# 3D Visualization
# -----------------------
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='viridis', edgecolor='k')
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.set_zlabel(feature_names[2])
ax1.set_title('Actual Species (3D)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X[:,0], X[:,1], X[:,2], c=clusters, cmap='viridis', edgecolor='k')
ax2.set_xlabel(feature_names[0])
ax2.set_ylabel(feature_names[1])
ax2.set_zlabel(feature_names[2])
ax2.set_title('KMeans Clustering (3D)')
plt.tight_layout()
plt.show() 