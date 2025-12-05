import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow method to determine optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply KMeans with optimal clusters (example: 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y = kmeans.fit_predict(X)

# Visualizing clusters
plt.scatter(X.iloc[y == 0, 0], X.iloc[y == 0, 1], s=50, label='Cluster 1')
plt.scatter(X.iloc[y == 1, 0], X.iloc[y == 1, 1], s=50, label='Cluster 2')
plt.scatter(X.iloc[y == 2, 0], X.iloc[y == 2, 1], s=50, label='Cluster 3')
plt.scatter(X.iloc[y == 3, 0], X.iloc[y == 3, 1], s=50, label='Cluster 4')
plt.scatter(X.iloc[y == 4, 0], X.iloc[y == 4, 1], s=50, label='Cluster 5')

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, c='black', marker='X', label='Centroids')

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
