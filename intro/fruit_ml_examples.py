from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Supervised Learning: Fruit Classifier
# -----------------------
print("📘 Supervised Learning - Fruit Prediction with Decision Tree")

# Fruit features [color, size, sweetness]
X = [[1, 1, 9], [1, 2, 10], [3, 4, 3], [2, 4, 4], [4, 1, 8]]
y = [0, 0, 1, 1, 2]  # 0:apple, 1:banana, 2:grape

clf = DecisionTreeClassifier()
clf.fit(X, y)

pred = clf.predict([[1, 1, 10]])
print("Prediction result (label):", pred)

# -----------------------
# Unsupervised Learning: Fruit Clustering
# -----------------------
print("\n📘 Unsupervised Learning - Clustering with KMeans")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
print("Clustering result:", clusters)

# Visualization
plt.scatter([x[0] for x in X], [x[2] for x in X], c=clusters)
plt.xlabel("Color")
plt.ylabel("Sweetness")
plt.title("KMeans Fruit Clustering")
plt.show()

# -----------------------
# Similarity Comparison (Cosine)
# -----------------------
print("\n📘 Similarity - Cosine Similarity")

apple = np.array([[1, 1, 9]])
pear = np.array([[2, 1, 9]])
banana = np.array([[3, 4, 3]])

sim_apple_pear = cosine_similarity(apple, pear)
sim_apple_banana = cosine_similarity(apple, banana)

print("Apple-Pear similarity:", sim_apple_pear)
print("Apple-Banana similarity:", sim_apple_banana)
