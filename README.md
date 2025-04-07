# Machine Learning Hemodynamics  
**Repository for supervised and unsupervised ML PhD project analyzing cardiac hemodynamics**  

### Problem  
*"Hemodynamics are complex; can we find hidden patterns?"*  

### Solution  
ML clustering + rigorous validation.  

### Impact  
*"Identified 3 phenotypes with distinct treatment implications."*  

---

## 1. Hierarchical Clustering Script (`ek.py`)  
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
  - Data continuum (gradual hemodynamic changes).  
  - High noise-to-signal ratio.  
- **Visual evidence**:  
  - PCA/t-SNE showed overlapping distributions.  
  - DBSCAN labeled most points as noise (`eps`/`min_samples` tuning failed).  

### Interpretation  
No biologically meaningful clusters emerged, suggesting:  
- Treatments may affect hemodynamics uniformly.  
- Alternative approaches (e.g., regression for dose-response) may be better.  

### Next Steps  
- **Non-clustering analysis**:  
  - PCA to identify dominant hemodynamic drivers.  
  - Regression to model treatment effects.  
- **Data expansion**: Include more patients/timepoints.  

### Dependencies  
Python 3.8+, `scikit-learn`, `scipy`, `matplotlib`.  

### Supplement  
- PCA required 6–7 PCs for 70% variance → opted for hierarchical clustering without PCA.  
- DBSCAN found 0 clusters.  
- *Author’s note*: Clustering is unsupervised—no defined objective function exists.  

---

## 2. K-means Clustering (`K-means clustering.py`)  
**Exploring Hemodynamic Data via Unsupervised k-means Clustering Analysis**  

### Objective  
Identify natural groupings in hemodynamic responses across 137 samples (treatment labels ignored).  

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
- **C0**: High Pressure, High Cardiac Output (*"Adapted Function"*).  
- **C1**: Low Pressure, High Cardiac Output (*"Healthier"*).  
- **C2**: High Pressure, Low Cardiac Output (*"Sicker"*).  

**Biological Relevance:**  
- C1/C2 align with hypertensive profiles; C0 suggests robust adaptation.  

#### B. Five-Cluster Solution (`k=5`)  
**Cluster Profiles:**  
- Splits C0 into nuanced subgroups (e.g., High/Low Ejection Fraction).  
- Emergent phenotype (K5C0) from K3C1/K3C2 (highest pressure/output).  

**Utility:**  
- Captures subtle phenotypes but may overfit (silhouette = 0.4).  

### Visualization  
- Silhouette plots (`k=3`, `k=5`).  
- PCA/t-SNE plots showing cluster separation and sub-groupings.  

### Interpretation  
**Robustness:**  
- Silhouette scores consistent across raw/PCA-reduced data.  
- Phenotypes:  
  - Healthiest (K3C1/K5C2) vs. sickest (K3C2/K5C3).  
  - C0 further splits into disease progression stages.  

**Novelty:**  
- Clusters ≠ treatment groups → suggests new hemodynamic subtypes for targeted therapy.  

### Limitations  
- Moderate silhouette scores (0.4–0.5): clusters overlap.  
- Dependence on `k`: `k=3` (conservative) vs. `k=5` (exploratory).  

### Next Steps  
- **Clinical correlation**: Test cluster-outcome links (e.g., survival).  
- **Feature importance**: SHAP/logistic regression for key drivers.  
- **Validation cohort**: Replicate in independent dataset.  


---
## Supervised Analysis

Notes
Summary:
First, perform task 1: which is to identify global discriminators. We will do this by taking our cluster labels from the completed unsupervised k-means clustering (k = 3) and fit a supervised classifier model to this. We will then run SHAP analysis on this model to understand which features separate cluster 0, 1, and 2, and help to explain why these distinct phenotypic definitions exist. Then, we will perform task 2: to understand which features explain differential remodeling within clusters. We will feature three separate supervised regression models to each cluster to explain the continuous label of ejection fraction. We will then perform SHAP analysis to explain how features contribute towards the emergent cardiac function and how these mechanisms are distinct within each cluster phenotype.

(COMPLETE)
Task 1: explains why your clusters exist (phenotype definitions).
Explain Cluster Membership	
Question: "Which features define the boundaries between my clusters?"	
Method: Fit a classifier to predict cluster labels, then SHAP to explain feature importance.
Output: Features that discriminate Cluster 0 vs. 1 vs. 2.
Biological Insight: "Why do these phenotypes exist?"


Task 2: explains how function (EF) emerges in each phenotype.
Question: "Within each cluster, how do features influence EF?"
Method: Fit separate regressors to predict EF within each cluster, then SHAP to explain EF drivers.
Output: Features that drive EF in Cluster 0, 1, or 2.
Biological Insight: "How does function (EF) emerge in each phenotype?"

A feature like EDP might be higher in cluster 3 than cluster 1, but PCA and analyzing cluster means do not explain what features drives the cluster assignment, nor their importance and interaction. SHAP analysis supplements this and can show whether it's pushing the model toward cluster 3, and how strongly — even in the presence of other features like contractility or vascular resistance (i.e. global and local importance, nonlinear effects, and feature directionality).

Figures

Task 1:
Bar plot for all top features
Beeswarm plot for each cluster showing top 6 features
Results of ANOVA and tukey test in tabular form (or just written up)
Summary of significance