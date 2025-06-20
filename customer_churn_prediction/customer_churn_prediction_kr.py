import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로딩 (공개 Telco Customer Churn 데이터셋)
print("\n[1] 데이터 로딩")
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(url)
print(df.head())

# 2. 데이터 전처리
print("\n[2] 데이터 전처리 및 특성 엔지니어링")
df = df.drop(['customerID'], axis=1)
df = df.replace(' ', np.nan).dropna()
for col in df.select_dtypes('object').columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])
df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 모델 학습 및 예측
print("\n[3] RandomForest, LogisticRegression 모델 학습 및 예측")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
logr = LogisticRegression(max_iter=1000)
logr.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
pred_logr = logr.predict(X_test)

# 4. 평가 및 특성 중요도 시각화
print("\n[4] 평가 및 특성 중요도 시각화")
print("\n[RandomForest] 분류 리포트:")
print(classification_report(y_test, pred_rf))
print("\n[LogisticRegression] 분류 리포트:")
print(classification_report(y_test, pred_logr))

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('RandomForest 특성 중요도')
plt.show()

# 5. 혼동행렬 시각화
plt.figure(figsize=(10,4))
sns.heatmap(confusion_matrix(y_test, pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('RandomForest 혼동행렬')
plt.xlabel('예측값')
plt.ylabel('실제값')
plt.show()

# 6. 실무 팁
print("\n[5] 실무 팁: 특성 중요도 기반 마케팅 전략, 이탈 고객 예측 활용") 