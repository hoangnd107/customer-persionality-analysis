import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px

file_path = "predata.csv"
df = pd.read_csv(file_path)

numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[numeric_cols].values

# Vẽ Dendrogram
plt.figure(figsize=(17, 8))
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.title('Dendrogram', fontsize=15)
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.show()

agc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
labels = agc.fit_predict(X)

# Gắn nhãn phân cụm vào DataFrame
df['Cluster'] = labels.astype(str)

sil_score = silhouette_score(X, labels)
db_score = davies_bouldin_score(X, labels)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

colors = ['red', 'blue', 'green', 'purple', 'orange']
fig = px.scatter_3d(
    df,
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

plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Annual Income (k$)'], c=[colors[int(i)] for i in labels])
plt.title('Age vs Annual Income')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
for i in range(5):
    plt.scatter([], [], c=colors[i], label=f'Cluster {i}')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Spending Score (1-100)'], c=[colors[int(i)] for i in labels])
plt.title('Age vs Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
for i in range(5):
    plt.scatter([], [], c=colors[i], label=f'Cluster {i}')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=[colors[int(i)] for i in labels])
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
for i in range(5):
    plt.scatter([], [], c=colors[i], label=f'Cluster {i}')
plt.legend()
plt.show()