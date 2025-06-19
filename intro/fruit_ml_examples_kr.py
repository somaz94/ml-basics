from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# 한글 폰트 설정
# --------------------------------
plt.rcParams['font.family'] = 'AppleGothic'  # macOS 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# -----------------------
# 지도학습: 과일 분류기
# -----------------------
print("📘 지도학습 - Decision Tree로 과일 예측")

# 과일의 [색, 크기, 당도]
X = [[1, 1, 9], [1, 2, 10], [3, 4, 3], [2, 4, 4], [4, 1, 8]]
y = [0, 0, 1, 1, 2]  # 0:사과, 1:바나나, 2:포도

clf = DecisionTreeClassifier()
clf.fit(X, y)

pred = clf.predict([[1, 1, 10]])
print("예측 결과 (라벨):", pred)

# -----------------------
# 비지도학습: 과일 군집화
# -----------------------
print("\n📘 비지도학습 - KMeans로 군집화")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
print("군집 결과:", clusters)

# 시각화
plt.scatter([x[0] for x in X], [x[2] for x in X], c=clusters)
plt.xlabel("색상")
plt.ylabel("당도")
plt.title("KMeans 과일 군집화")
plt.show()

# -----------------------
# 유사도 비교 (Cosine)
# -----------------------
print("\n📘 유사도 - Cosine Similarity")

apple = np.array([[1, 1, 9]])
pear = np.array([[2, 1, 9]])
banana = np.array([[3, 4, 3]])

sim_apple_pear = cosine_similarity(apple, pear)
sim_apple_banana = cosine_similarity(apple, banana)

print("사과-배 유사도:", sim_apple_pear)
print("사과-바나나 유사도:", sim_apple_banana)
