import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Data loading (Telco Customer Churn dataset)
print("\n[1] Data loading")
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(url)
print(df.head())

# 2. Data preprocessing
print("\n[2] Data preprocessing and feature engineering")
df = df.drop(['customerID'], axis=1)
df = df.replace(' ', np.nan).dropna()
for col in df.select_dtypes('object').columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])
df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model training and prediction
print("\n[3] RandomForest, LogisticRegression model training and prediction")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
logr = LogisticRegression(max_iter=1000)
logr.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
pred_logr = logr.predict(X_test)

# 4. Evaluation and feature importance visualization
print("\n[4] Evaluation and feature importance visualization")
print("\n[RandomForest] Classification report:")
print(classification_report(y_test, pred_rf))
print("\n[LogisticRegression] Classification report:")
print(classification_report(y_test, pred_logr))

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('RandomForest Feature Importance')
plt.show()

# 5. Confusion matrix visualization
plt.figure(figsize=(10,4))
sns.heatmap(confusion_matrix(y_test, pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('RandomForest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 6. Practical tips
print("\n[5] Practical tip: Use feature importance for marketing strategy and churn prediction") 