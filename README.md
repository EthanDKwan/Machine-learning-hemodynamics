# Machine Learning Hemodynamics  
**Repository for supervised and unsupervised ML PhD project analyzing cardiac hemodynamics**  

### Problem  
*"Hemodynamics are complex; can we find hidden patterns?"*  

### Solution  
*Fitting unsupervised (hierarchical, k-means clustering) and supervised (Random Forest Regression) machine learning models to a real-world pulmonary hypertension dataset (101 samples, 27 features), applying computational/numeric tools to validate our results (SHAP analysis, statistical validation) and applying our cardiac remodeling domain expertise to analyze and interpret the results.*

### Impact  
*"Identified 3 remodeling phenotypes in our pulmonary hypertension dataset with distinct treatment implications."*

---

## 1. Hierarchical Clustering 
**Exploring Hemodynamic Groupings via Unsupervised Hierarchical Clustering Analysis**  

### Objective  
- Cluster multi-treatment hemodynamics data using PCA, hierarchical clustering, and DBSCAN.  

### Methods  
**Preprocessing:**  
- Z-score standardized 27 hemodynamic features (`StandardScaler`).  
- Applied PCA (`n_components=6`, 70% variance retained) for noise reduction.  

**Clustering:**  
- Hierarchical clustering (Ward’s linkage, `k=3`).  
- Validated stability (`random_state=42`).  

**Outputs:**  
- PCA eigenvalue cumulative sums.  
- 3 sample clusters.  

**Plots:**  
- PCA bi-plots (predictors vs. principal components).  
- Elbow method + silhouette analysis.  
- Dendrogram (Ward linkage).  

### Key Findings  
- **Weak cluster structure**: Silhouette scores <0.2 across methods.  
- **Possible reasons**:  
  - Data continuum suggests gradual hemodynamic changes across samples.
  - High noise-to-signal ratio.  
- **Visual evidence**:  
  - PCA/t-SNE showed overlapping distributions.  
  - DBSCAN labeled most points as noise (`eps`/`min_samples` tuning failed).

### Interpretation  
No biologically meaningful clusters emerged, suggesting:  
- Treatments may affect hemodynamics uniformly.  
- Alternative approaches (e.g., regression for dose-response) may be better

### Next Steps  
- k-means to force clustering
- **Non-clustering analysis**:  
  - PCA to identify dominant hemodynamic drivers.  
  - Regression to model treatment effects.  
- **Data expansion**: Include more patients/timepoints.  

### Supplement  
- PCA required 6–7 PCs for 70% variance → opted for hierarchical clustering without PCA.  
- DBSCAN found 0 clusters.  
- Author’s note: Recall that clustering is fundamentally about discovery - no defined objective function exists.  

---

## 2. K-means Clustering 
**Exploring Hemodynamic Data via Unsupervised k-means Clustering Analysis**  

### Objective  
Identify natural groupings in hemodynamic responses across 101 samples.

### Methods  
**Preprocessing:**  
- Z-score standardized 27 features (`StandardScaler`).  
- PCA (`n_components=6`, 70% variance retained).  

**Clustering:**  
- K-means: `k=3` (silhouette = 0.51), `k=5` (silhouette = 0.42).  
- Validated stability (`random_state=42`).  

### Key Results  
#### A. Three-Cluster Solution (`k=3`)  
**Cluster Profiles:**  
- **C0**: (*"Baseline Adaptation"*)Low Pressure, few mechanical alterations (*"Least Severe"*).  
- **C1**: (*"Mechanical remodeling"*)High pressure, Most remodeling (increased ESP, Ees, dp/dt).  
- **C2**: (*"Functional Reserve"*)Medium Pressure, High EF, intermediate mechanical remodeling (*"Preserved mechanics"*).  

**Biological Relevance:**  
- Clustering indicates degree of mechanical adaptation aligns with hemodynamic profiles. C0 shows minimal changes; C1 suggests robust mechanical adaptation.

#### B. Five-Cluster Solution (`k=5`)  
**Cluster Profiles:**  
- Splits C0 into nuanced subgroups (e.g., High/Low Ejection Fraction).  
- Emergent phenotype (K5C0) from K3C1/K3C2 (highest pressure/output).  

**Utility:**  
- Captures subtle phenotypes but may overfit (silhouette = 0.4).  

### Visualization  
- Silhouette plots (`k=3`, `k=5`).  
- PCA/t-SNE plots showing cluster separation and sub-groupings.  

### Validation
- Silhouette scores: 0.51 (k = 3), stable across random seeds
- ANOVA confirmed significant differences in key hemodynamic features across clusters, including dp/dt, ESP, PVR (p<0.05).

### Interpretation  
**Robustness:**  
- Silhouette scores consistent across raw/PCA-reduced data.  
- Phenotypes:  
  - High ejection fraction (K3C1/K5C2) vs. High mechanical alterations (K3C2/K5C3).  

**Novelty:**  
- Clusters ≠ treatment groups → suggests new hemodynamic subtypes for targeted therapy.
- Mechanical adaptation linked to evolving hemodynamic profiles across multiple timepoints.



### Limitations  
- Moderate silhouette scores (0.4–0.5): clusters overlap.  
- Dependence on `k`: `k=3` (conservative) vs. `k=5` (exploratory).
- Timepoints are observational (not longitudinal).
- EF may not capture subtle dysfunction.

### Next Steps  
- **Clinical correlation**: Test cluster-outcome links (e.g., survival).  
- **Feature importance**: SHAP/logistic regression for key drivers.  
- **Validation cohort**: Replicate in independent dataset.  


---
## Supervised Analysis

### Motivation
A feature like EDP might be higher in cluster 3 than cluster 1, but PCA and analyzing cluster means do not explain what features drives the cluster assignment, nor their importance and interaction. SHAP analysis supplements this and can show whether it's pushing the model toward cluster 3, and how strongly — even in the presence of other features like contractility or vascular resistance (i.e. global and local importance, nonlinear effects, and feature directionality).

### Objective  
Investigate the hemodynamic features defining the boundaries between cluster phenotypes

### Methods 
- Random Forest Classifier model fitted to data
- SHAP analysis of Random Forest model
- Statistical validation

### Key Results
**SHAP Global Feature Importance**
- Primary discriminators:
  1. dp/dt max (contractility)
  2. dp/dt min (diastolic function) 
  3. Systolic pressure
  4. Heart rate (inotropic adaptation)
  5. PVR
- Collectively, these results indicate that pressure overload severity was the primary determinant of cluster assignment.

**Cluster-Specific Drivers**

| Cluster | Top Features                     | Phenotype Interpretation          |
|---------|----------------------------------|-----------------------------------|
| C0      | • Low systolic pressure          | Baseline adapted state            |
|         | • Low EDP/Eed                    | Minimal mechanical remodeling     |
|---------|----------------------------------|-----------------------------------|
| C1      | • High afterload (PVR, pressure) | Pressure-overloaded compensation  |
|         | • Diastolic dysfunction          | with inotropic adaptation        |
|---------|----------------------------------|-----------------------------------|
| C2      | • Reduced contractility          | Combined systolic/diastolic       |
|         | • High diastolic stiffness       | dysfunction                      |

**Key Findings**
• C0: Preserved mechanics in low-pressure state
• C1: Afterload-driven with compensatory changes
• C2: Advanced ventricular stiffness and impairment
• Heart rate emerged as unexpected important factor (C0/C1)

**Statistical Validation:
1-factor ANOVA showed significant effects of cluster on key features, including dp/dt max, dp/dt min, ESP, HR, PVR, Eed and Ees. Post-hoc analysis (Tukey HSD) indicated significant differences in all pair-wise cluster comparisons for dp/dt max, dp/dt min, and ESP (p<0.05). C1 showed a significantly increased HR compared to both C0 and C2 (p<0.05), as well as PVR, with C1 increased compared to both C0 and C2 (p<0.05). Both C2 and C1 showed significantly increased diastolic stiffness (Eed, p<0.05) compared to C0.

** Biological Relevance**

A balanced contractility and diastolic stiffness response in C2 imply an eccentric mechanical adaptation, in contrast to C1, which responded to the large pressure overload with greater diastolic dysfunction and recruited HR elevation to compensate.

--- 
## Conclusions
### Why this matters

Identified 3 clinically distinct phenotypes with:
- Mechanical adaptation linked to evolving hemodynamic profiles with divergent adaptive strategies, not just severity stages

- Tailored treatment implications:
	- C1 may benefit more from afterload reduction through traditional PAH vasodilator drugs.
	- C2 may require cardiac volume management (anti-congestives, diuretics).
	- Both C0 and C2 could benefit from traditional cardiac inotropic drugs.

---

### Key files
/src/

- Hierarchical-Clustering.py: Hierarchical clustering
- K-means-clustering.py: K-means clustering + Visualization
- Cluster-discriminators.py: Supervised classifier model + SHAP analysis
- Statistical-Validation.py: ANOVA + posthoc validation

-/Helpers/
	-plot_silhouette.py: for generating silhouette plots
	-plot_tsne_clusters.py: for generating tSNE plots

/Sample Data/
	- Hemodynamics_with_Kclusters.csv: Clustered data
	- PAH Project Selected Predictors - preprocessed.xlsx: Raw features
	- Parameter Legend: Relevant physiology, cardiac and mechanical terms

/Results/

-/Hierarchical clustering/

	- Hierarchical clustering dendrogram.png: clustering Dendrogram for samples with a distance threshold of 20

	- Hierarchical clustering elbow heuristic.png: Elbow analysis vs number of clusters

	- Silhouette analysis for hierarchical clustering.png: Total silhouette score vs number of clusters

	- tSNE visualization of Hierarchical clusters.png: tDistributed Stochastic Neighbor Embedding shows quality of hierarchical clustering

	- tSNE visualization of DBScan clusters.png: tDistributed Stochastic Neighbor Embedding shows quality of DBScan clustering (could not identify any clusters)

-/K Clustering/

	- Silhouette Analysis for k clustering.png: Total silhouette score vs number of clusters

	- silhouette_k3.png: Silhouette plot for each k3 cluster

	- silhouette_k5.png: Silhouette plot for each k5 cluster

	-tSNE visualization of k3 clusters.png

	-tSNE visualization of k5 clusters.png

	-KCluster Domain Phenotyping.xlsx: Counts of clustering phenotypes across key features (treatment duration, relative vs raw), as well as bar graphs showing means, stds, and significant differences between key features for k3 clustering results. Followup interpretation gided by domain expertise.

-/Principal Component Analysis

	- BiPlots for PC12.png: Visualization of sample distribution across PC1 and 2.

	- PCA eigenvalue cumsum.png: Relative % of cumulative variance explained by additional principal components.

-/Supervised Analysis/

	-/Cluster Discriminators/

		-SHAP global feature importance.png

		-SHAP Beeswarm C0.png

		-SHAP Beeswarm C1.png

		-SHAP Beeswarm C2.png


### How to use this Repository
Clone the repository:
Install requirements: pip install -r requirements.txt

Code: /src/ includes unsupervised, supervised, statistical analysis .py scripts

Figures: /Results/ includes data visualization and results figures

### License

This project is licensed under the CC by NC 4 License. See the [LICENSE](LICENSE) file for details.


Task 2: explains how function (EF) emerges in each phenotype.
Question: "Within each cluster, how do features influence EF?"
Method: Fit separate regressors to predict EF within each cluster, then SHAP to explain EF drivers.
Output: Features that drive EF in Cluster 0, 1, or 2.
Biological Insight: "How does function (EF) emerge in each phenotype?"
