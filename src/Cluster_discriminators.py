# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 16:34:10 2025

@author: edkwa
"""

#Task 1: Cluster Discriminators

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd  

# Load pre-clustered data
file_path = os.path.join("..", "sample data", "Hemodynamics_with_Kclusters.csv")
data = pd.read_csv(file_path)

# Task 1: Explain cluster membership
X = data.drop(columns=['k3_clusters'])
y = data['k3_clusters']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 2. Calculate SHAP values for ALL classes
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # Returns list of arrays [shap_cluster0, shap_cluster1, shap_cluster2]
shap_values_corrected = np.transpose(shap_values, (2, 0, 1))


# 3. Plot global feature importance (magnitude only)
shap.summary_plot(shap_values, X, plot_type="bar", class_names=['Cluster 0', 'Cluster 1', 'Cluster 2'])

print(f"SHAP values shape: {np.array(shap_values).shape}")  # Should be (n_clusters, n_samples, n_features)
print(f"X shape: {X.shape}")

# 5. Plot dot plots for each cluster
for i, cluster_name in enumerate(['Cluster 0', 'Cluster 1', 'Cluster 2']):
    plt.figure()
    shap.summary_plot(shap_values_corrected[i], X, plot_type="dot", max_display = 7, show=False)
    plt.title(f'SHAP Values for {cluster_name}\n(Red=high values push patients INTO this cluster)',
              fontsize=12, pad=20)
    plt.tight_layout()
    plt.show()
    
#Statistical validation
# 1. Select top features from SHAP analysis
top_features = ['dp/dt max', 'dp/dt min', 'AdjESP', 'HR', 'PVR', 'Eed','AdjEDP']

# 2. Run ANOVA for each feature
results = []
for feature in top_features:
    cluster_groups = [data[data['k3_clusters'] == i][feature] for i in [0, 1, 2]]
    f_stat, p_val = f_oneway(*cluster_groups)
    results.append({
        'Feature': feature,
        'F-statistic': f_stat,
        'p-value': p_val,
        'Significant (p < 0.05)': p_val < 0.05
    })

# 3. Display as a table
stats_df = pd.DataFrame(results)
print(stats_df)

for feature in top_features:  
    print(f"\n--- Tukey HSD for {feature} ---")  
    tukey = pairwise_tukeyhsd(  
        endog=data[feature],  
        groups=data['k3_clusters'],  
        alpha=0.05  
    )  
    print(tukey.summary())  