import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px

class AgglomerativeCluster:
    def __init__(self, file_path, n_clusters=5):
        self.file_path = file_path
        self.n_clusters = n_clusters
        self.df = pd.read_csv(file_path)
        self.numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        self.X = self.df[self.numeric_cols].values
        self.labels = None

    def plot_dendrogram(self):
        plt.figure(figsize=(17, 8))
        linkage_matrix = linkage(self.X, method='ward')
        dendrogram(linkage_matrix)
        plt.title('Dendrogram', fontsize=15)
        plt.xlabel('Samples')
        plt.ylabel('Euclidean Distance')
        plt.show()

    def fit_predict(self):
        agc = AgglomerativeClustering(n_clusters=self.n_clusters, metric='euclidean', linkage='ward')
        self.labels = agc.fit_predict(self.X)
        self.df['Cluster'] = self.labels.astype(str)

    def evaluate_clustering(self):
        sil_score = silhouette_score(self.X, self.labels)
        db_score = davies_bouldin_score(self.X, self.labels)
        print(f"Silhouette Score: {sil_score:.4f}")
        print(f"Davies-Bouldin Score: {db_score:.4f}")

    def plot_3d_scatter(self):
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        fig = px.scatter_3d(
            self.df,
            x='Age',
            y='Annual Income (k$)',
            z='Spending Score (1-100)',
            color='Cluster',
            title='3D Customer Segmentation',
            labels={'Cluster': 'Cluster'},
            color_discrete_sequence=colors
        )
        fig.update_layout(
            legend_title_text='Cluster',
            coloraxis_colorbar=None
        )
        fig.update_traces(marker=dict(size=6))
        fig.show()

    def plot_2d_scatter(self, x_col, y_col, title, x_label, y_label):
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        plt.figure(figsize=(8, 5))
        plt.scatter(self.df[x_col], self.df[y_col], c=[colors[int(i)] for i in self.labels])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        for i in range(self.n_clusters):
            plt.scatter([], [], c=colors[i], label=f'Cluster {i}')
        plt.legend()
        plt.show()

    def plot_all(self):
        self.plot_2d_scatter('Age', 'Annual Income (k$)', 'Age vs Annual Income', 'Age', 'Annual Income (k$)')
        self.plot_2d_scatter('Age', 'Spending Score (1-100)', 'Age vs Spending Score', 'Age', 'Spending Score (1-100)')
        self.plot_2d_scatter('Annual Income (k$)', 'Spending Score (1-100)', 'Annual Income vs Spending Score', 'Annual Income (k$)', 'Spending Score (1-100)')

if __name__ == "__main__":
    agc = AgglomerativeCluster(r"../data/data.csv")
    agc.plot_dendrogram()
    agc.fit_predict()
    agc.evaluate_clustering()
    agc.plot_3d_scatter()
    agc.plot_all()