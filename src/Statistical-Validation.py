# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:14:28 2025

@author: edkwa
"""
import os
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Pre-clustered data
file_path = os.path.join("..", "sample data", "Hemodynamics_with_Kclusters.csv")
data = pd.read_csv(file_path)

# Select top features from SHAP analysis
top_features = ['dp/dt max', 'dp/dt min', 'AdjESP', 'HR', 'PVR', 'Eed','AdjEDP', 'EF', 'Ees']

# Run ANOVA omnibus for each feature
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

# Run Tukey posthoc
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