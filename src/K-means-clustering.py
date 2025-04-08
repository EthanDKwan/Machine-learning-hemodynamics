import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
#Helpers
from Helpers.plot_silhouette import plot_silhouette
from Helpers.plot_tsne_clusters import plot_tsne_clusters

file_path = os.path.join("..", "sample data", "PAH Project Selected Predictors-preprocessed.xlsx")
#data = pd.read_excel(file_path,sheet_name='Averaged W0, W4, W8, W12')
data = pd.read_excel(file_path, sheet_name = "Processed Averaged W4, W8, W12") #No controls
data = data.drop(['GroundTruth', 'ID'], axis=1)
#data_array = data.to_numpy()
#print(data)


#PCA
pca = PCA(n_components=6)  # Reduce to 6 dimensions = ~70% variance
reduced_data = pca.fit_transform(data)

#K-means clustering
nclusters = 3 #Based on silhouette score, choosing k = 3 and k = 5
kmeans = KMeans(n_clusters=nclusters, n_init = 10) 
kmeans.fit(reduced_data)
cluster_assignments = kmeans.labels_ #Note: .labels_ is used for accessing labels immediately after training)
#To reuse the trained model, use kmeans.predict(X)
data['k3_clusters'] = cluster_assignments

### t-SNE plots for cluster visualization
plot_tsne_clusters(reduced_data, kmeans.labels_,title='t-SNE: Hemodynamics Clusters (k=5)')

#Printing cluster assignments to command window
cluster_indices = [[] for _ in range(nclusters)]
for row_number, cluster_label in enumerate(cluster_assignments):
    cluster_indices[cluster_label].append(row_number)
for i, cluster_indices_list in enumerate(cluster_indices):
    print(f'Cluster {i} Row Numbers:')
    print(cluster_indices_list)
    print()
print("cluster assignments: ", cluster_assignments)

save_dir = os.path.join("..","Sample Data")
save_path = os.path.join(save_dir,"Hemodynamics_with_Kclusters.csv")
data.to_csv(save_path, index = False)

#Plotting cluster assignments in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for cluster_label in set(cluster_assignments):
    cluster_data = reduced_data[cluster_assignments == cluster_label]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster_label}')
ax.set_title('K-means Clustering (3D Projection)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.show()


# Plotting basic cluster feature means and stdeviations
features_to_plot = ['HR','EF','dp/dt max','dp/dt min','PVR','AdjESP','AdjEDP','Ees','Eed','Sex']
cluster_stats = (
    data.groupby('k3_clusters')[features_to_plot]
    .agg(['mean', 'std'])
    .stack(level=0)  # Reshape for easier plotting
    .reset_index()
    .rename(columns={'level_1': 'feature'})
)
n_clusters = len(cluster_stats['k3_clusters'].unique())
n_features = len(features_to_plot)
bar_width = 0.25
index = np.arange(n_features)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
fig, axes = plt.subplots(2,5,figsize=(25,10))
axes = axes.flatten()
for idx, (ax, feature) in enumerate(zip(axes, features_to_plot)):
    stats = data.groupby('k3_clusters')[feature].agg(['mean', 'std']).reset_index()
    bars = ax.bar(
        stats['k3_clusters'],
        stats['mean'],
        yerr=stats['std'],
        color=colors,
        capsize=5,
        width=0.6)
    ax.set_title(feature, fontsize=30)
    ax.set_xlabel('Cluster', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_xticks(stats['k3_clusters'])
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])
plt.tight_layout(pad=2.0)
plt.show()


"""
#Plotting parallel coordinates #Didn't like this visualization
cluster_means = data.groupby('k3_clusters')[['AdjESP', 'EF', 'Ees', 'Treatment Duration']].mean().reset_index()
plt.figure(figsize=(10, 6))
parallel_coordinates(cluster_means, 'k3_clusters', colormap='viridis')
plt.title("Feature Trends Across Clusters")
plt.xticks(rotation=45)
plt.show()
"""

# Silhouette Analyses ####################################
silhouette_scores = []
k_range = range(2, 11)  # Evaluates k=2 through k=10
for k in k_range:
    # Fit K-means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_data)  # Use PCA-reduced or standardized data    
    # Compute silhouette score
    score = silhouette_score(reduced_data, cluster_labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.3f}")
# Plot the silhouette scores against the number of clusters (K)
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for K-means Clustering")
plt.grid(True)
plt.show()
"\Results\K Clustering\silhouette_k3.png"

#Plotting k = 3, k = 5 per-cluster silhouette analysis
save_dir = os.path.join("..", "Results", "K Clustering")
save_path = os.path.join(save_dir, "silhouette_k3.png")
plot_silhouette(reduced_data, k=3, save_path=save_path)
save_path = os.path.join(save_dir, "silhouette_k5.png")
plot_silhouette(reduced_data, k=5, save_path=save_path)