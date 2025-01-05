import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

file_path = r"predata.csv"
df = pd.read_csv(file_path)
numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
data = df[numeric_cols].values

# Chuẩn hóa dữ liệu
def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

X = standardize(data)

# Hàm tính khoảng cách Euclidean
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans(X, k, max_iters=100):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        
        # Gán điểm vào cụm gần nhất
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        
        # Lưu lại tâm cụm cũ để kiểm tra hội tụ
        old_centroids = centroids.copy()
        
        # Tính lại tâm cụm
        centroids = np.array([np.mean(cluster, axis=0) if cluster else old_centroids[i] for i, cluster in enumerate(clusters)])
        
        # Kiểm tra hội tụ
        if np.all(centroids == old_centroids):
            break

    return centroids, clusters

# Elbow Method để xác định số cụm tối ưu
inertia = []
K = range(1, 11)
for k in K:
    centroids, clusters = kmeans(X, k)
    total_inertia = sum([
        sum([euclidean_distance(point, centroids[i])**2 for point in cluster])
        for i, cluster in enumerate(clusters)
    ])
    inertia.append(total_inertia)

# Vẽ Elbow Plot
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

optimal_clusters = 4
centroids, clusters = kmeans(X, optimal_clusters)

# Chuyển đổi lại tâm cụm về giá trị ban đầu
centroids_original = centroids * np.std(data, axis=0) + np.mean(data, axis=0)

df = pd.DataFrame(data, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
df['Cluster Label'] = [i for cluster_idx, cluster in enumerate(clusters) for i in [cluster_idx] * len(cluster)]

fig = px.scatter_3d(
    df,
    x='Age',
    y='Annual Income (k$)',
    z='Spending Score (1-100)',
    color='Cluster Label',
    title='3D Customer Segmentation',
    labels={
        'Age': 'Age',
        'Annual Income (k$)': 'Annual Income (k$)',
        'Spending Score (1-100)': 'Spending Score (1-100)',
        'Cluster Label': 'Cluster Label'
    }
)
fig.update_layout(
    scene=dict(
        xaxis_title='Age',
        yaxis_title='Annual Income (k$)',
        zaxis_title='Spending Score (1-100)'
    ),
    legend_title='Cluster Label'
)
fig.show()

plt.figure(figsize=(8, 5))
colors = ['b', 'g', 'r', 'orange']
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    if cluster.size > 0:
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Age (Standardized)')
plt.ylabel('Annual Income (Standardized)')
plt.title('Age vs Annual Income (with Centroids)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    if cluster.size > 0:
        plt.scatter(cluster[:, 0], cluster[:, 2], c=colors[i], label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 2], s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Age (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.title('Age vs Spending Score (with Centroids)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    if cluster.size > 0:
        plt.scatter(cluster[:, 1], cluster[:, 2], c=colors[i], label=f'Cluster {i}')
plt.scatter(centroids[:, 1], centroids[:, 2], s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.title('Annual Income vs Spending Score (with Centroids)')
plt.legend()
plt.show()