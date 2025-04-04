# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:06:03 2025

@author: edkwa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.cm import Spectral

def plot_silhouette(X, k, save_path=None):
    """
    Generate a silhouette plot for per-cluster analysis.
    
    Args:
        X (array-like): Scaled feature matrix (n_samples x n_features)
        k (int): Number of clusters
        save_path (str): Path to save the figure (optional)
    """
    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Compute silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10  # Space between silhouette "blades"
    
    for i in range(k):
        # Aggregate silhouette scores for cluster i
        ith_cluster_silhouette = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette.sort()
        
        size_cluster_i = ith_cluster_silhouette.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = Spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label clusters
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # Add gap between clusters
    
    # Format plot
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f"Average Score: {silhouette_avg:.2f}")
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_title(f"Silhouette Plot for k={k}", fontweight='bold')
    ax.legend(loc="upper right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# plot_silhouette(X_scaled, k=3, save_path='figures/silhouette_k3.png')
# plot_silhouette(X_scaled, k=5, save_path='figures/silhouette_k5.png')