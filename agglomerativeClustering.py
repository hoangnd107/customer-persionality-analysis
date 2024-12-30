import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px

class AgglomerativeClustering:
    def __init__(self, file_path, n_clusters=5):
        self.file_path = file_path
        self.n_clusters = n_clusters
        self.df = None
        self.X_scaled = None
        self.labels = None
        self.numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        self.colors = ['red', 'blue', 'green', 'purple', 'orange']

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        X = self.df[self.numeric_cols].values
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

    def plot_dendrogram(self):
        """Create and display dendrogram"""
        plt.figure(figsize=(17, 8))
        linkage_matrix = linkage(self.X_scaled, method='ward')
        dendrogram(linkage_matrix)
        plt.title('Dendrogram', fontsize=15)
        plt.xlabel('Number of customers in cluster leaf node')
        plt.ylabel('Cluster distance')
        plt.show()

    def perform_clustering(self):
        """Perform agglomerative clustering"""
        agc = AgglomerativeClustering(
            n_clusters=self.n_clusters, 
            metric='euclidean', 
            linkage='ward'
        )
        self.labels = agc.fit_predict(self.X_scaled)
        self.df['Cluster'] = self.labels.astype(str)

    def calculate_scores(self):
        """Calculate clustering evaluation scores"""
        sil_score = silhouette_score(self.X_scaled, self.labels)
        db_score = davies_bouldin_score(self.X_scaled, self.labels)
        print(f"Silhouette Score: {sil_score:.4f}")
        print(f"Davies-Bouldin Score: {db_score:.4f}")

    def plot_3d_clusters(self):
        """Create and display 3D scatter plot"""
        fig = px.scatter_3d(
            self.df,
            x='Age',
            y='Annual Income (k$)',
            z='Spending Score (1-100)',
            color='Cluster',
            title='3D Customer Segmentation',
            labels={'Cluster': 'Cluster'},
            color_discrete_sequence=self.colors
        )

        fig.update_layout(
            legend_title_text='Cluster',
            coloraxis_colorbar=None
        )
        fig.update_traces(marker=dict(size=6))
        fig.show()

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        self.load_data()
        self.plot_dendrogram()
        self.perform_clustering()
        self.calculate_scores()
        self.plot_3d_clusters()

if __name__ == "__main__":
    segmentation = AgglomerativeClustering("predata.csv")
    segmentation.run_analysis()