import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Data loading (boston housing)
print("\nLoading Boston housing data...")
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# Data type checking and numeric conversion
print("Data type checking:")
print(df.dtypes)
print("\nData preprocessing...")

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check and handle missing values
print(f"Missing value count: {df.isnull().sum().sum()}")
if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print("Missing values removed")

X = df.drop('MEDV', axis=1)
y = df['MEDV']

print(f"Data shape: {X.shape}, Target shape: {y.shape}")

# -----------------------
# Supervised Learning: House Price Prediction (LinearRegression, RandomForestRegressor)
# -----------------------
print("\n📘 Supervised Learning - House Price Prediction")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse:.2f}")

# -----------------------
# Unsupervised Learning: KMeans Clustering
# -----------------------
print("\n📘 Unsupervised Learning - Regional Clustering with KMeans")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans clustering result (0~2):", np.bincount(clusters))

# -----------------------
# Visualization: Room Count-House Price Scatter Plot
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X['RM'], y, c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Room Count (RM)')
plt.ylabel('House Price (MEDV)')
plt.title('Actual House Price')

plt.subplot(1,2,2)
plt.scatter(X['RM'], y, c=clusters, cmap='viridis', edgecolor='k')
plt.xlabel('Room Count (RM)')
plt.ylabel('House Price (MEDV)')
plt.title('KMeans Clustering')
plt.tight_layout()
plt.show() 