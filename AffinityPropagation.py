import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations

class AffinityPropagationClustering:
    def __init__(self, data, damping_values, preference):
        self.data = data
        self.damping_values = damping_values
        self.preference = preference  # Thêm thuộc tính preference
        self.best_silhouette_score = -1
        self.best_damping_silhouette = None
        self.labels_silhouette = None

    def run_affinity_propagation(self, damping):
        ap = AffinityPropagation(damping=damping, preference=self.preference)
        labels = ap.fit_predict(self.data)
        return labels


    def run(self):
        for damping in self.damping_values:
            labels = self.run_affinity_propagation(damping)
            num_clusters = len(np.unique(labels))
            if num_clusters < 2 or num_clusters > len(self.data) // 2:
                continue  # Bỏ qua các trường hợp số cụm không hợp lý
            
            silhouette = silhouette_score(self.data, labels)
            
            print(f"Damping: {damping:.2f}, Silhouette Score: {silhouette:.3f}")
            
            if silhouette > self.best_silhouette_score:
                self.best_silhouette_score = silhouette
                self.best_damping_silhouette = damping
                self.labels_silhouette = labels
                
        print(f"\nBest Silhouette Score: damping={self.best_damping_silhouette if self.best_damping_silhouette is not None else 'None'}, Score={self.best_silhouette_score}")

class DataVisualizer:
    def __init__(self, raw_data, labels_silhouette, numeric_columns):
        self.raw_data = raw_data
        self.labels_silhouette = labels_silhouette
        self.numeric_columns = numeric_columns
    
    def visualize(self):
        unique_clusters_silhouette = np.unique(self.labels_silhouette)

        if self.raw_data.shape[1] == 2:
            self._plot_2d(unique_clusters_silhouette)
        elif self.raw_data.shape[1] >= 3:
            self._plot_3d(unique_clusters_silhouette)
            self._plot_2d_pairs(unique_clusters_silhouette)
        else:
            print("Dữ liệu có hơn 3 chiều, không thể trực quan hóa dễ dàng.")
    
    def _plot_2d(self, unique_clusters_silhouette):
        x_label, y_label = self.numeric_columns[:2]
        
        plt.figure(figsize=(8, 6))
        for cluster in unique_clusters_silhouette:
            plt.scatter(
                self.raw_data[self.labels_silhouette == cluster, 0], 
                self.raw_data[self.labels_silhouette == cluster, 1], 
                label=f'Cluster {cluster}' if cluster != -1 else 'Noise'
            )
        plt.title('Affinity Propagation Clustering (2D)')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
        
    def _plot_3d(self, unique_clusters_silhouette):
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
        ax.set_title('Affinity Propagation Clustering (3D)')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.legend()
        plt.show()
        
    def _plot_2d_pairs(self, unique_clusters_silhouette):
        pairs = list(combinations(self.numeric_columns, 2))
        for (x_label, y_label) in pairs:
            plt.figure(figsize=(8, 6))
            for cluster in unique_clusters_silhouette:
                plt.scatter(
                    data[data['Cluster_Silhouette'] == cluster][x_label], 
                    data[data['Cluster_Silhouette'] == cluster][y_label], 
                    label=f'Cluster {cluster}' if cluster != -1 else 'Noise'
                )
            plt.title(f'Affinity Propagation Clustering ({x_label} vs {y_label}) - Silhouette Score')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.show()
            
file_path = "data.csv"
data = pd.read_csv(file_path)

filtered_data = data.drop(columns=['CustomerID'], errors='ignore')

# Lọc các cột số
numeric_data = filtered_data.select_dtypes(include=[np.number])
numeric_columns = numeric_data.columns  # Lưu lại tên cột gốc để hiển thị

# Sử dụng dữ liệu gốc, chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Các giá trị damping để thử nghiệm
damping_values = np.linspace(0.5, 0.9, 5)  # Danh sách các giá trị damping để thử nghiệm
preference = -50
affinity_propagation_clustering = AffinityPropagationClustering(scaled_data, damping_values, preference)
affinity_propagation_clustering.run()

# Thêm nhãn cụm vào DataFrame
data['Cluster_Silhouette'] = affinity_propagation_clustering.labels_silhouette

visualizer = DataVisualizer(scaled_data, affinity_propagation_clustering.labels_silhouette, numeric_columns)
visualizer.visualize()