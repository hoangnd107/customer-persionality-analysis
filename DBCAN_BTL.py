import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D

class DBSCANClustering:
    def __init__(self, data, eps_values, min_samples_values):
        self.data = data
        self.eps_values = eps_values
        self.min_samples_values = min_samples_values
        self.best_silhouette_score = -1
        self.best_davies_bouldin_score = float('inf')
        self.best_eps_silhouette = None
        self.best_min_samples_silhouette = None
        self.best_eps_davies = None
        self.best_min_samples_davies = None
        self.labels_silhouette = None
        self.labels_davies = None
        
    def region_query(self, point_idx, eps):
        neighbors = []
        for idx, point in enumerate(self.data):
            if np.linalg.norm(self.data[point_idx] - point) <= eps:
                neighbors.append(idx)
        return neighbors

    def expand_cluster(self, labels, point_idx, cluster_id, eps, min_samples):
        neighbors = self.region_query(point_idx, eps)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Điểm nhiễu
            return False
        else:
            labels[point_idx] = cluster_id
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                if labels[neighbor_idx] == -1:  # Nếu là điểm nhiễu, gán vào cụm
                    labels[neighbor_idx] = cluster_id
                elif labels[neighbor_idx] == 0:  # Nếu chưa được thăm, thêm vào cụm
                    labels[neighbor_idx] = cluster_id
                    new_neighbors = self.region_query(neighbor_idx, eps)
                    if len(new_neighbors) >= min_samples:
                        neighbors.extend(new_neighbors)
                i += 1
            return True

    def dbscan(self, eps, min_samples):
        labels = np.zeros(len(self.data), dtype=int)  # 0: chưa thăm, -1: nhiễu, >0: cụm
        cluster_id = 0

        for point_idx in range(len(self.data)):
            if labels[point_idx] == 0:  # Nếu điểm chưa được thăm
                if self.expand_cluster(labels, point_idx, cluster_id + 1, eps, min_samples):
                    cluster_id += 1
        return labels

    def run(self):
        for eps in self.eps_values:
            for min_samples in self.min_samples_values:
                labels = self.dbscan(eps, min_samples)
                
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(self.data, labels)
                    davies_bouldin = davies_bouldin_score(self.data, labels)
                    print(f"eps={eps}, min_samples={min_samples}, Silhouette Score={silhouette}, Davies-Bouldin Score={davies_bouldin}")
                    
                    if silhouette > self.best_silhouette_score:
                        self.best_silhouette_score = silhouette
                        self.best_eps_silhouette = eps
                        self.best_min_samples_silhouette = min_samples
                        
                    if davies_bouldin < self.best_davies_bouldin_score:
                        self.best_davies_bouldin_score = davies_bouldin
                        self.best_eps_davies = eps
                        self.best_min_samples_davies = min_samples

        print(f"\nBest Silhouette Score: eps={self.best_eps_silhouette}, min_samples={self.best_min_samples_silhouette}, Score={self.best_silhouette_score}")
        print(f"Best Davies-Bouldin Score: eps={self.best_eps_davies}, min_samples={self.best_min_samples_davies}, Score={self.best_davies_bouldin_score}")
        
        self.labels_silhouette = self.dbscan(self.best_eps_silhouette, self.best_min_samples_silhouette)
        self.labels_davies = self.dbscan(self.best_eps_davies, self.best_min_samples_davies)

class DataVisualizer:
    def __init__(self, raw_data, labels_silhouette, labels_davies, numeric_columns):
        self.raw_data = raw_data
        self.labels_silhouette = labels_silhouette
        self.labels_davies = labels_davies
        self.numeric_columns = numeric_columns
    
    def visualize(self):
        unique_clusters_silhouette = np.unique(self.labels_silhouette)
        unique_clusters_davies = np.unique(self.labels_davies)

        if self.raw_data.shape[1] == 2:
            self._plot_2d(unique_clusters_silhouette, unique_clusters_davies)
        elif self.raw_data.shape[1] >= 3:
            self._plot_3d(unique_clusters_silhouette, unique_clusters_davies)
            self._plot_2d_pairs(unique_clusters_silhouette, unique_clusters_davies)
        else:
            print("Dữ liệu có hơn 3 chiều, không thể trực quan hóa dễ dàng.")
    
    def _plot_2d(self, unique_clusters_silhouette, unique_clusters_davies):
        x_label, y_label = self.numeric_columns[:2]
        
        plt.figure(figsize=(8, 6))
        for cluster in unique_clusters_silhouette:
            plt.scatter(
                self.raw_data[self.labels_silhouette == cluster, 0], 
                self.raw_data[self.labels_silhouette == cluster, 1], 
                label=f'Cluster {cluster}' if cluster != -1 else 'Noise'
            )
        plt.title('DBSCAN Clustering (2D) - Silhouette Score')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(8, 6))
        for cluster in unique_clusters_davies:
            plt.scatter(
                self.raw_data[self.labels_davies == cluster, 0], 
                self.raw_data[self.labels_davies == cluster, 1], 
                label=f'Cluster {cluster}' if cluster != -1 else 'Noise'
            )
        plt.title('DBSCAN Clustering (2D) - Davies-Bouldin Score')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    def _plot_3d(self, unique_clusters_silhouette, unique_clusters_davies):
        x_label, y_label, z_label = self.numeric_columns[:3]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for cluster in unique_clusters_silhouette:
            ax.scatter(
                self.raw_data[self.labels_silhouette == cluster, 0], 
                self.raw_data[self.labels_silhouette == cluster, 1], 
                self.raw_data[self.labels_silhouette == cluster, 2], 
                label=f'Cluster {cluster}' if cluster != -1 else 'Noise'
            )
        ax.set_title('DBSCAN Clustering (3D) - Silhouette Score')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.legend()
        plt.show()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for cluster in unique_clusters_davies:
            ax.scatter(
                self.raw_data[self.labels_davies == cluster, 0], 
                self.raw_data[self.labels_davies == cluster, 1], 
                self.raw_data[self.labels_davies == cluster, 2], 
                label=f'Cluster {cluster}' if cluster != -1 else 'Noise'
            )
        ax.set_title('DBSCAN Clustering (3D) - Davies-Bouldin Score')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.legend()
        plt.show()
        
    def _plot_2d_pairs(self, unique_clusters_silhouette, unique_clusters_davies):
        pairs = list(combinations(self.numeric_columns, 2))
        for (x_label, y_label) in pairs:
            plt.figure(figsize=(8, 6))
            for cluster in unique_clusters_silhouette:
                plt.scatter(
                    data[data['Cluster_Silhouette'] == cluster][x_label], 
                    data[data['Cluster_Silhouette'] == cluster][y_label], 
                    label=f'Cluster {cluster}' if cluster != -1 else 'Noise'
                )
            plt.title(f'DBSCAN Clustering ({x_label} vs {y_label}) - Silhouette Score')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.show()
            
            plt.figure(figsize=(8, 6))
            for cluster in unique_clusters_davies:
                plt.scatter(
                    data[data['Cluster_Davies'] == cluster][x_label], 
                    data[data['Cluster_Davies'] == cluster][y_label], 
                    label=f'Cluster {cluster}' if cluster != -1 else 'Noise'
                )
            plt.title(f'DBSCAN Clustering ({x_label} vs {y_label}) - Davies-Bouldin Score')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.show()

# Bước 1: Đọc dữ liệu từ file CSV
file_path = "Mall_Customers.csv"  # Đường dẫn tới file CSV
data = pd.read_csv(file_path)

# Bước 2: Loại bỏ cột không cần thiết và xử lý dữ liệu
columns_to_exclude = ['CustomerID']  # Cột không sử dụng
filtered_data = data.drop(columns=columns_to_exclude, errors='ignore')

# Kiểm tra các giá trị null
filtered_data = filtered_data.dropna()

# Lọc các cột số
numeric_data = filtered_data.select_dtypes(include=[np.number])
numeric_columns = numeric_data.columns  # Lưu lại tên cột gốc để hiển thị

# Sử dụng dữ liệu gốc, không scale
raw_data = numeric_data.values

# Các giá trị eps và min_samples để thử nghiệm
eps_values = np.arange(11, 16, 1)  # Danh sách các giá trị eps để thử nghiệm
min_samples_values = np.arange(5, 13, 1)  # Danh sách các giá trị min_samples để thử nghiệm

# Tạo đối tượng DBSCANClustering và chạy thuật toán
dbscan_clustering = DBSCANClustering(raw_data, eps_values, min_samples_values)
dbscan_clustering.run()

# Thêm nhãn cụm vào DataFrame
data['Cluster_Silhouette'] = dbscan_clustering.labels_silhouette
data['Cluster_Davies'] = dbscan_clustering.labels_davies

# Bước 5: Lưu kết quả ra file CSV
data.to_csv("data_with_clusters.csv", index=False)

# Bước 6: Trực quan hóa cụm (biểu đồ cho Silhouette Score và Davies-Bouldin Score)
visualizer = DataVisualizer(raw_data, dbscan_clustering.labels_silhouette, dbscan_clustering.labels_davies, numeric_columns)
visualizer.visualize()
