import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score, davies_bouldin_score

class KMeansClustering:
    def __init__(self, file_path, numeric_cols):
        self.file_path = file_path
        self.numeric_cols = numeric_cols
        self.df = pd.read_csv(file_path)
        self.data = self.df[numeric_cols].values
        self.X = self.standardize(self.data)
        self.centroids = None
        self.clusters = None

    def standardize(self, data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def kmeans(self, k, max_iters=100):
        np.random.seed(42)
        centroids = self.X[np.random.choice(self.X.shape[0], k, replace=False)]
        
        for _ in range(max_iters):
            clusters = [[] for _ in range(k)]
            
            for point in self.X:
                distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(point)
            
            old_centroids = centroids.copy()
            centroids = np.array([np.mean(cluster, axis=0) if cluster else old_centroids[i] for i, cluster in enumerate(clusters)])
            
            if np.all(centroids == old_centroids):
                break

        self.centroids = centroids
        self.clusters = clusters

    def elbow_method(self, max_k=10):
        inertia = []
        K = range(1, max_k + 1)
        for k in K:
            self.kmeans(k)
            total_inertia = sum([
                sum([self.euclidean_distance(point, self.centroids[i])**2 for point in cluster])
                for i, cluster in enumerate(self.clusters)
            ])
            inertia.append(total_inertia)

        plt.figure(figsize=(8, 5))
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.show()

    def fit(self, optimal_clusters):
        self.kmeans(optimal_clusters)
        centroids_original = self.centroids * np.std(self.data, axis=0) + np.mean(self.data, axis=0)
        self.df['Cluster Label'] = [i for cluster_idx, cluster in enumerate(self.clusters) for i in [cluster_idx] * len(cluster)]
        return centroids_original

    def evaluate(self):
        sil_score = silhouette_score(self.X, self.df['Cluster Label'])
        db_score = davies_bouldin_score(self.X, self.df['Cluster Label'])
        print(f"Silhouette Score: {sil_score:.4f}")
        print(f"Davies-Bouldin Score: {db_score:.4f}")

    def plot_3d(self):
        colors = ['blue', 'green', 'red', 'orange']
        fig = px.scatter_3d(
            self.df,
            x='Age',
            y='Annual Income (k$)',
            z='Spending Score (1-100)',
            color='Cluster Label',
            title='3D Customer Segmentation',
            labels={'Cluster': 'Cluster'},
            color_discrete_sequence=colors
        )
        fig.update_layout(
            legend_title_text='Cluster',
            coloraxis_colorbar=None
        )
        fig.update_traces(marker=dict(size=6))

    def plot_2d(self):
        plt.figure(figsize=(8, 5))
        colors = ['blue', 'green', 'red', 'orange']
        for i, cluster in enumerate(self.clusters):
            cluster = np.array(cluster)
            if cluster.size > 0:
                plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
        plt.xlabel('Age (Standardized)')
        plt.ylabel('Annual Income (Standardized)')
        plt.title('Age vs Annual Income (with Centroids)')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        for i, cluster in enumerate(self.clusters):
            cluster = np.array(cluster)
            if cluster.size > 0:
                plt.scatter(cluster[:, 0], cluster[:, 2], c=colors[i], label=f'Cluster {i}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 2], s=200, c='black', marker='X', label='Centroids')
        plt.xlabel('Age (Standardized)')
        plt.ylabel('Spending Score (Standardized)')
        plt.title('Age vs Spending Score (with Centroids)')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        for i, cluster in enumerate(self.clusters):
            cluster = np.array(cluster)
            if cluster.size > 0:
                plt.scatter(cluster[:, 1], cluster[:, 2], c=colors[i], label=f'Cluster {i}')
        plt.scatter(self.centroids[:, 1], self.centroids[:, 2], s=200, c='black', marker='X', label='Centroids')
        plt.xlabel('Annual Income (Standardized)')
        plt.ylabel('Spending Score (Standardized)')
        plt.title('Annual Income vs Spending Score (with Centroids)')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # file_path = r"../data/data.csv"
    file_path = r"C:\Users\nduyh\py\customer-persionality-analysis\data\data.csv"
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    kmeans = KMeansClustering(file_path, numeric_cols)
    kmeans.elbow_method()
    optimal_clusters = 4
    kmeans.fit(optimal_clusters)
    kmeans.evaluate()
    kmeans.plot_3d()
    kmeans.plot_2d()