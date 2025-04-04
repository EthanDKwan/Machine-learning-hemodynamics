# Machine learning hemodynamics
 Repository for supervised and unsupervised ML phd project analyzing cardiac hemodynamics 


Content Notes:

1) "Hierarchical clustering script - ek.py"
Exploring Hemodynamic Groupings via Unsupervised Hierarchical Clustering Analysis
1. Objective
- Attempted clustering of multi-treatment hemodynamics data using PCA, hierarchical clustering, and DBSCAN.

2. Methods
- Performs hierarchical clustering using ward's linkage on the "PAH Project selected predictors spreadsheet - averaged.xlsx" dataset.
-(Hierarchical clustering doesn't require number of clusters to be set beforehand, but cuts dataset into clusters based on a threshold distance. # of clusters to choose should be revised via elbow method and silhouette score analysis to find best number of clusters)
-Steps: z-score standardization, PCA ("feature transformation")-> tells us how many PCs are important for explaining x% of total variance), Hierarchical clustering using linkage function
Preprocessing:
-Z-score standardized all 27 hemodynamic features (StandardScaler).
-Applied PCA (n_components=6, 70% variance retained) for noise reduction.

Clustering:
-Hierarchical clustering with k=3
-Validated stability across random seeds (random_state=42).



Outputs: PCA (eigenvalue cum sums), 3 clusters of samples
Plots: PCA: predictor bi-plots vs principal components
Elbow method + silhouette analysis
Clustering: Dendrogram (ward linkage)

3. Key Findings
Weak cluster structure: Silhouette scores consistently <0.2 across methods.

Possible reasons:
Data represents a continuum (e.g., gradual hemodynamic changes, not discrete groups).
High noise-to-signal ratio in measurements.
Visual evidence:
*** ATTACH PLOTS HERE
PCA e-cumsum
Silhouette & Elbow analyses
t-SNE visualizations
***
PCA/t-SNE showed overlapping distributions.
DBSCAN labeled most points as noise, with 0 clusters (eps and min_samples tuning failed to resolve this).

4. Interpretation
No biologically meaningful clusters emerged, suggesting:
-Treatments may affect hemodynamics uniformly.
-Alternative approaches (e.g., regression for dose-response) may be more appropriate.

5. Next Steps
Non-clustering analysis:
- PCA to identify dominant hemodynamic drivers.
- Regression to model treatment effects.

Data expansion: 
-Include more patients/timepoints to uncover subtle patterns.

6. Dependencies
-Python 3.8+, scikit-learn, scipy, matplotlib

7. Supplement
We ran PCA and found that it took ~6-7 PCs to explain 70% of the total variance -> PCA would lose significant variance -> since the features are biologically meaningful and we suspect subtle patterns beyond PCA, we decided to try hierarchical clustering on the z-score standardized data without PCA.
I tried DBScan and it found 0 clusters.

""Cluster analysis revealed weak structure (silhouette <0.2), suggesting hemodynamic responses vary continuously. Future work could model trends with regression.""

- Keep in mind this is unsupervised learning -> no goal

**************************************

2) "K-means clustering.py"
Exploring Hemodynamic Data via Unsupervised k-means Clustering Analysis
1. Objective
Identify natural groupings in hemodynamic responses across 137 samples using unsupervised clustering, independent of prior treatment labels.

2. Methods
Preprocessing:
-Z-score standardized all 27 hemodynamic features (StandardScaler).
-Applied PCA (n_components=6, 70% variance retained) for noise reduction.

Clustering:
-K-means with k=3 (silhouette = 0.51) and k=5 (silhouette = 0.42).
-Validated stability across random seeds (random_state=42).

3. Key Results
A. Three-Cluster Solution (k=3)

Cluster Profiles:
(Figure Cluster phenotyping k3)
C0: High Pressure, High Cardiac Output (e.g., "Adapted Function").
C1: Low Pressure, High Cardiac Output (e.g., "Healthier").
C2: High Pressure, Low Cardiac Output (e.g., "Sicker").

Biological Relevance:
C1 and C2 aligns with known hypertensive profiles, corresponding to a logical progression of the disease; C0 may indicate a uniquely robust adaptation phenotype.

B. Five-Cluster Solution (k=5)
(Figure cluster phenotyping k5)
Cluster Profiles:
Further splits K3C0 into nuanced subgroups K5C4/K5C1 (e.g., High Ejection Fraction K5C4 (60%) and Low Ejection Fraction K5C1 (42%). Identified a new emergent phenotype K5C0 created from K3C1 and K3C2 with the highest pressure and highest output. The K3C1 was largely consistent with the K5C2 while K3C2 was largely consistent with the K5C3.

Utility:
Captures subtle phenotypes but may overfit slightly (silhouette = 0.4).

4. Visualization
-Silhouette plot over range of k's
-Silhouette plot (k = 3) - shows more uniform, better separated equal clusters.
-Silhouette plot (k = 5) - 
-PCA Results: PC Loadings
-Cluster phenotyping k3
-Cluster phenotyping k5
- 

-PCA Plot (k=3): Clear separation between major clusters.
PCA Clusters

-t-SNE Plot (k = 3): Highlights sub-groupings.
-t-SNE Plot (k = 5): Emphasizes that K3C0 was further separated into K5C4 and K5C1. A new emergent phenotype K5C0 was created from parts of K3C1 and K3C2. K3C1 was largely preserved as K5C2 while K3C2 was largely consistent with K5C3.

5. Interpretation
Robustness:
- Nearly identical silhouette scores for raw vs. PCA reduced data confirm patterns are not noise-driven.

1) Consistently clustering phenotypes is a healthiest group (K3C1, K5C2) and a sickest group (K3C2, K5C3)
2) The adapted K3C0 phenotype can be further distinguished as later-stage sick and later-stage adapted, corresponding to length of disease and disease progression metrics.

Novelty:
Clusters do not perfectly align with treatment groups, suggesting:
-New hemodynamic subtypes exist that may be targets for more informed hemodynamic profile-specific therapies.
-Potential for personalized therapy based on hemodynamic responses.

6. Limitations
-Moderate silhouette scores (0.4–0.5): Clusters are separable but may overlap.
-Dependence on k: k=3 is conservative while k=5 is more exploratory and revealing.

7. Next Steps
-Clinical Correlation: Test if clusters predict outcomes (e.g., survival, drug response).
-Feature Importance: Use SHAP values or logistic regression to identify key drivers of each cluster.
-Validation Cohort: Replicate findings in an independent dataset.


- performs k-means clustering (6-clusters) on the "PAH Project selected predictors spreadsheet - averaged.xlsx" dataset.
3 and 5 clusters were chosen based on the silhouette analysis (largest k > 0.5 and largest k > 0.4). Interpretation of these clusters found on in ML powerpoint with labeled metrics.


