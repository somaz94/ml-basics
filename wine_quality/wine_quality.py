import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error

# Data loading (wine quality data)
print("\nLoading wine quality data...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
X = df.drop('quality', axis=1)
y = df['quality']

print(f"Data shape: {X.shape}, Target shape: {y.shape}")

# -----------------------
# Supervised Learning: Quality Grade Classification (RandomForest)
# -----------------------
print("\n📘 Supervised Learning - Wine Quality Grade Classification with RandomForest")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"RandomForest classification accuracy: {acc:.3f}")

# -----------------------
# Supervised Learning: Quality Score Regression (LinearRegression)
# -----------------------
print("\n📘 Supervised Learning - Wine Quality Score Prediction with LinearRegression")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_reg = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_reg)
print(f"LinearRegression MSE: {mse:.3f}")

# -----------------------
# Unsupervised Learning: KMeans Clustering
# -----------------------
print("\n📘 Unsupervised Learning - Wine Clustering with KMeans")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans clustering result (0~2):", np.bincount(clusters))
print("Actual quality grade distribution:", np.bincount(y))

# -----------------------
# Visualization: Alcohol-Acidity-Quality Scatter Plot
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X['alcohol'], X['fixed acidity'], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Alcohol')
plt.ylabel('Acidity')
plt.title('Actual Quality Grade')

plt.subplot(1,2,2)
plt.scatter(X['alcohol'], X['fixed acidity'], c=clusters, cmap='viridis', edgecolor='k')
plt.xlabel('Alcohol')
plt.ylabel('Acidity')
plt.title('KMeans Clustering')
plt.tight_layout()
plt.show() 