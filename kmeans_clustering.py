"""
Part 3: Unsupervised learning for clustering (K-means on Iris)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score

# ---------- (1) Cluster Iris with K=3 (labels ignored) ----------
iris = load_iris()
X = iris.data            # 4 features only, no labels used for clustering
y_true = iris.target     # used ONLY afterwards, to check the clusters
names = iris.target_names

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# RMSE of each cluster: typical distance from a point to its centroid
print("RMSE per cluster (distance from points to centroid):")
for k in range(3):
    pts = X[clusters == k]
    sq_dist = np.sum((pts - centroids[k]) ** 2, axis=1)
    rmse = np.sqrt(np.mean(sq_dist))
    print(f"  Cluster {k}: {len(pts):3d} points, RMSE = {rmse:.4f}")

# ---------- (2) Compare clusters to the real classes ----------
# Cluster numbers are arbitrary, so match each cluster to the
# species that appears most often inside it.
mapping = {}
for k in range(3):
    mapping[k] = np.bincount(y_true[clusters == k]).argmax()
y_pred = np.array([mapping[c] for c in clusters])

print("\nCluster -> majority species:")
for k in range(3):
    print(f"  Cluster {k} -> {names[mapping[k]]}")

print("\nConfusion matrix (rows = real species, cols = predicted from cluster):")
print(confusion_matrix(y_true, y_pred))

acc = np.mean(y_pred == y_true) * 100
print(f"\nCluster/class agreement: {acc:.2f}%")
print(f"Adjusted Rand Index:     {adjusted_rand_score(y_true, clusters):.4f}"
      "  (1.0 = perfect match, 0 = random)")
print("MLP classifier (Part 2) test accuracy for comparison: ~96.67%")

# ---------- Plot: K-means clusters vs real species ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(X[:, 2], X[:, 3], c=clusters, cmap="viridis", s=25)
axes[0].scatter(centroids[:, 2], centroids[:, 3], c="red", marker="X", s=200,
                label="centroids")
axes[0].set_title("K-means clusters (no labels used)")
axes[0].legend()
axes[1].scatter(X[:, 2], X[:, 3], c=y_true, cmap="viridis", s=25)
axes[1].set_title("Real species")
for ax in axes:
    ax.set_xlabel("petal length (cm)")
    ax.set_ylabel("petal width (cm)")
plt.tight_layout()
plt.show()
