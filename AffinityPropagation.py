import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu từ file CSV
file_path = "Mall_Customers.csv"
data = pd.read_csv(file_path)

# Loại bỏ cột không cần thiết và lọc dữ liệu số
columns_to_exclude = ['CustomerID']
numeric_data = data.drop(columns=columns_to_exclude, errors='ignore').select_dtypes(include=[np.number])
numeric_data = numeric_data.dropna()

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Chạy thuật toán với các giá trị damping khác nhau
best_silhouette = -1
best_davies_bouldin = float('inf')
best_damping = None
best_labels = None

damping_values = np.linspace(0.5, 0.9, 5)

for damping in damping_values:
    affinity = AffinityPropagation(damping=damping, random_state=0)
    labels = affinity.fit_predict(scaled_data)
    num_clusters = len(np.unique(labels))
    if num_clusters < 2 or num_clusters > len(scaled_data) // 2:
        continue  # Bỏ qua các trường hợp số cụm không hợp lý
    
    # Tính toán các chỉ số đánh giá
    silhouette = silhouette_score(scaled_data, labels)
    davies_bouldin = davies_bouldin_score(scaled_data, labels)
    
    print(f"Damping: {damping:.2f}, Silhouette Score: {silhouette:.3f}, Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Lưu kết quả tốt nhất
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_davies_bouldin = davies_bouldin
        best_damping = damping
        best_labels = labels

# Kết quả tốt nhất
if best_damping is not None:
    print("\nKết quả tốt nhất:")
    print(f"Damping: {best_damping:.2f}, Silhouette Score: {best_silhouette:.3f}, Davies-Bouldin Score: {best_davies_bouldin:.3f}")

    # Trực quan hóa kết quả
    unique_labels = np.unique(best_labels)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for label in unique_labels:
        cluster_data = scaled_data[best_labels == label]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
                   label=f'Cụm {label}')

    ax.set_title("Affinity Propagation Clustering (3D)")
    ax.set_xlabel("Đặc trưng 1")
    ax.set_ylabel("Đặc trưng 2")
    ax.set_zlabel("Đặc trưng 3")
    ax.legend()
    plt.show()
else:
    print("Không tìm thấy cụm hợp lý cho các giá trị damping đã thử.")
