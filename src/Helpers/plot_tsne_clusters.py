# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:41:54 2025

@author: edkwa
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def plot_tsne_clusters(X, labels, title='t-SNE Cluster Visualization', 
                      perplexity=30, random_state=42, save_path=None):
    """
    Generate a t-SNE plot colored by cluster assignments.
    
    Args:
        X (array-like): Scaled feature matrix (n_samples x n_features)
        labels (array): Cluster labels (e.g., from KMeans)
        title (str): Plot title
        perplexity (int): t-SNE perplexity (default=30)
        random_state (int): Reproducibility seed
        save_path (str): Path to save the figure (e.g., 'figures/tsne_k3.png')
    """
    # Compute t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], 
        y=X_tsne[:, 1], 
        hue=labels,
        palette='viridis',
        s=50,
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Formatting
    plt.title(title, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# Assuming you've already run KMeans:
# kmeans_k3 = KMeans(n_clusters=3).fit(X_scaled)
# kmeans_k5 = KMeans(n_clusters=5).fit(X_scaled)

# Generate plots
"""
plot_tsne_clusters(X_scaled, kmeans_k3.labels_, 
                   title='t-SNE: Hemodynamics Clusters (k=3)',
                   save_path='figures/tsne_k3.png')

plot_tsne_clusters(X_scaled, kmeans_k5.labels_,
                   title='t-SNE: Hemodynamics Clusters (k=5)',
                   save_path='figures/tsne_k5.png')
"""