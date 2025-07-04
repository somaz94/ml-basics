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

# Data loading (seaborn built-in)
print("\nLoading Titanic data...")
df = sns.load_dataset('titanic')
print(df.head())

# Use only key features & handle missing values/categorical data
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = df[features + ['survived']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[features]
y = df['survived']

print(f"Data shape: {X.shape}, Target shape: {y.shape}")

# -----------------------
# Supervised Learning: Survival Prediction (DecisionTree, RandomForest, LogisticRegression)
# -----------------------
print("\n📘 Supervised Learning - Survival Prediction")
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
    print(f"{name} accuracy: {acc:.3f}")

# -----------------------
# Unsupervised Learning: KMeans Clustering
# -----------------------
print("\n📘 Unsupervised Learning - Passenger Group Analysis with KMeans")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans clustering result (0/1):", np.bincount(clusters))
print("Actual survivor distribution:", np.bincount(y))
print("Clustering-Actual survivors confusion matrix:\n", confusion_matrix(y, clusters))

# -----------------------
# Visualization: Age-Fare-Survival Scatter Plot
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X['age'], X['fare'], c=y, cmap='coolwarm', edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Actual Survival Status')

plt.subplot(1,2,2)
plt.scatter(X['age'], X['fare'], c=clusters, cmap='coolwarm', edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('KMeans Clustering')
plt.tight_layout()
plt.show() 