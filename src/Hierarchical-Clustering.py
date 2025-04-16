# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:05:23 2023

@author: edkwa

This script performs hierarchical clustering using ward's linkage on the 
PAHProjectSelectedPredictors spreadsheet (on the sheet named 'Averaged').
Hierarchical clustering doesn't require the number of clusters to be set 
beforehand. This script contains code for the elbow method and for silhouette
score analysis to try and determine the best number of clusters. 

"""
# import needed python packages
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

"""
READ IN DATA
"""

# Read in the PAH Project Selected Predictors spreadsheet
file_path = os.path.join("..", "sample data", "PAH Project Selected Predictors-preprocessed.xlsx")
#data = pd.read_excel(file_path, sheet_name = "Averaged W0, W4, W8, W12") #With controls
data = pd.read_excel(file_path, sheet_name = "Formatted 0-15")
# In the variable 'data', each row represents a sample (132 total)
# and each column represents a feature of heart function (27 total)

# Drop columns 'GroundTruth' and 'ID'
data = data.drop(['GroundTruth', 'ID'], axis=1)
data_array = data.to_numpy() #pandas dataframe to numpy array

"""
PCA
"""

# Perform z-score standardization on the data. z score standardization standardizes data with different units/scales
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_array)
# Convert the standardized data back to a pandas DataFrame (optional but useful for labeling data points)
df_standardized = pd.DataFrame(data_standardized, columns=data.columns)

#Do PCA AFTER z-score standardization - 
#Note - EK: means of data must be 0 for PCA and z-score standardization is probably better than mean subtraction"
pca = PCA()
data_reduced = pca.fit_transform(data_standardized)
#calculate eigenvalues and cumulative eigenvalues
eigenvalues = pca.explained_variance_
total_eigenvalues = np.sum(eigenvalues)
cumulative_eigenvalues = np.cumsum(eigenvalues)
normalized_cumulative_eigenvalues = cumulative_eigenvalues/total_eigenvalues

#plot cumulative eigenvalues for each component
plt.plot(range(1,len(cumulative_eigenvalues)+1), normalized_cumulative_eigenvalues, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Eigenvalues')
plt.title('Part A: PCA results: eigen cumsum for all data')
plt.show()



"""
PCA Data Vis. (Biplots)
"""

#Biplot of predictors vs Principal components (For data vis and for fun)
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
#Call the function. Use only the first 2 PCs.
myplot(data_reduced[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.ylim(-0.5,0.5)
plt.xlim(-0.5,0.5)
plt.title('Biplot showing contribution of predictors to Principal components 1-2')
plt.show()
#In plot 2, we show the 3-4th PCs
myplot(data_reduced[:,2:4],np.transpose(pca.components_[2:4, :]))
plt.ylim(-0.5,0.5)
plt.xlim(-0.5,0.5)
plt.title('Biplot showing contribution of predictors to Principal components 3-4')
plt.show()

# get the most important independent variables
n_important = 6 # number of most important independent variables
most_important = pca.components_[:n_important, :]
print("most important ", n_important, " PCA components: ",most_important)



"""
Hierarchical Clustering
"""
# Perform hierarchical clustering using linkage function (e.g., 'ward', 'single', 'complete', 'average', etc.)
linkage_method = 'ward'
Z = hierarchy.linkage(df_standardized, method=linkage_method)

# Set the distance threshold to cut the dendrogram and obtain clusters
distance_threshold = 20  # Adjust this value as needed
clusters = hierarchy.fcluster(Z, t=distance_threshold, criterion='distance')
from scipy.cluster.hierarchy import fcluster
clusters3 = fcluster(Z, t=3, criterion='maxclust')
clusters6 = fcluster(Z, t =6, criterion = 'maxclust')


"""
Density based clustering (DBSCAN)
"""
from sklearn.cluster import DBSCAN
# Adjust these parameters!
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Start with these defaults
clustersDBSCAN = dbscan.fit_predict(data_standardized)  # Or use data_scaled
n_clusters = len(np.unique(clustersDBSCAN)) - (1 if -1 in clustersDBSCAN else 0)
print(f"Found {n_clusters} clusters (+ noise)")


"""
DATA VIS using T-SNE
"""
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30)
data_tsne = tsne.fit_transform(data_standardized)

plt.figure(figsize = (12,6))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=clustersDBSCAN, alpha=0.5)  # Color by K=6 clusters
plt.title('tDistributed Stochastic Neighbor Embedding visualization of density-based clusters')
plt.show()


"""
Dendrogram from hierarchical clustering
"""
# Plot the dendrogram to visualize the clustering hierarchy
plt.figure(figsize=(12, 6))
dn = hierarchy.dendrogram(Z)
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
plt.axhline(y=distance_threshold, color='r', linestyle='--', label=f'Distance Threshold = {distance_threshold}')
plt.legend()
plt.show()

# Create a DataFrame to store the clusters along with the data points
cluster_data = pd.DataFrame({'Data Point': range(1, len(clusters) + 1), 'Cluster': clusters})
# The 'clusters' variable now contains the hierarchical cluster assignments for each data point
print('Here are my clusters', clusters)

"""
determing how many clusters is appropriate
Elbow method: quick heuristic
silhouette analysis: rigourous validation -> Best silhouette score @ 2 or 6 clusters
"""

# Elbow Method ##############################################
# The elbow method helps determine the number of clusters
# Calculate the variance for each number of clusters (K)
variances = []
for k in range(1, len(data) + 1):
    Z = hierarchy.linkage(df_standardized, method=linkage_method)
    clusters = hierarchy.fcluster(Z, t=k, criterion='maxclust')
    variances.append(np.var(clusters))

# Create a DataFrame to store the elbow method data
elbow_data = pd.DataFrame({'Number of Clusters (K)': range(1, len(data) + 1), 'Variance within Clusters': variances})

# Plot the variance values against the number of clusters (K) to identify the elbow point
plt.plot(range(1, len(data) + 1), variances, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Variance within Clusters')
plt.title('Elbow Method for Hierarchical Clustering')
plt.show()

# Silhouette Analysis ####################################
silhouette_scores = []
for k in range(2, len(data) + 1):
    Z = hierarchy.linkage(df_standardized, method=linkage_method)
    clusters = hierarchy.fcluster(Z, t=k, criterion='maxclust')
    silhouette_scores.append(silhouette_score(df_standardized, clusters))

# Create a DataFrame to store the silhouette analysis data
silhouette_data = pd.DataFrame({'Number of Clusters (K)': range(2, len(data) + 1), 'Silhouette Score': silhouette_scores})
#"2 clusters (Silhouette of .194 and 6 clusters (silhouette score of .177)

# Plot the silhouette scores against the number of clusters (K)
# the highest silhouette score corresponds to the number of clusters producing the best results
plt.plot(range(2, len(data) + 1), silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Hierarchical Clustering')
plt.show()