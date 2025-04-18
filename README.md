# Building Machine Learning for Hemodynamic Phenotyping 
*Identifying sex-associated cardiac remodeling patterns by mining clinical hemodynamic data*

*Used unsupervised learning and SHAP analysis to discover and validate three distinct adaptive remodeling patterns in clinical pulmonary hypertension data, validating prior clinical findings.*  

<<<<<<< HEAD
[Artificial intelligence won't replace your doctor's, but it could make them better](https://newsnetwork.mayoclinic.org/discussion/science-saturday-artificial-intelligence-wont-replace-your-doctors-but-it-could-make-them-better)
=======
*(Interactive dashboard deployed for trained model exploration and interactive phenotyping: [[Streamlit App Link]](https://cardiacphenotyping.streamlit.app/))* 
>>>>>>> 6366718f8cc3982721640a128d4793c01aaf8819


[▶️ Try the live demo](https://cardiacphenotyping.streamlit.app)

*For more information about the clinical demo dashboard, see [/deploy/](https://github.com/EthanDKwan/Machine-learning-hemodynamics/tree/main/deploy)
### Technologies
- **Frontend**: Streamlit (interactive dashboard)
- **Deployment**: Streamlit Cloud
- **Languages**: Python
- **ML**: scikit-learn
- **Data Handling**: Pandas, numpy, openpyxl
- **Visualization**: Matplotlib, plotly, seaborn

### Problem  
*"Cardiac hemodynamic remodeling is complex in health and disease; can we identify hidden patterns that improve our understanding of disease adaptation to personalize treatment?"*  

### Methods  
*Applied unsupervised (hierarchical, k-means clustering) and interpretable ML (Random Forest classifier, SHAP) models to a pulmonary hypertension clinical dataset (184 samples, 25 features), integrating statistical validation (ANOVA, post-hoc testing, SHAP analysis) and domain expertise to phenotype clusters.*

### Key Findings

#### Unsupervised Clustering (k=3)
Identified three **sex-associated adaptive phenotypes**:  
- **Cluster 0**: *Low-Pressure Adaptation* (Healthy function, ↓ PVR, ↑ EF)  
- **Cluster 1**: *Compensated Pressure Overload* (↑ PVR, preserved EF, ♀-predominant)  
- **Cluster 2**: *Diastolic-Driven Remodeling* (↑↑ EDP, ↑ elastance, ♂-predominant)  

#### Supervised Learning + Validation
- **Top Discriminators**:  
  - Vascular Resistance, systolic pressure, and hypertrophy drove cluster separation (SHAP).
  - ANOVA confirmed significant inter-cluster differences in systolic pressure, diastolic elastance, and ejection fraction (p<0.05)
- **Sex Bias**:  
  - Cluster 1: 87% female (linked to preserved EF under load).  
  - Cluster 2: 78% male (associated with diastolic stiffness).  

**Cluster Profiles**  
| Cluster | Top Features                     | Phenotype Interpretation          | Sex Bias |  
|---------|----------------------------------|-----------------------------------|----------|  
| **C0**  | • Low PVR, EDP                   | *Healthy adaptation*              | Neutral  |  
|         | • High EF, SV                    | Optimal biventricular function    |          |  
| **C1**  | • Moderate ↑ PVR, ESP            | *Compensated pressure overload*   | ♀ 78%    |  
|         | • Preserved EF                   | Female-predominant adaptation     |          |  
| **C2**  | • High EDP, elastance            | *Diastolic-driven remodeling*     | ♂ 82%    |  
|         | • Reduced SV, ↑ HR               | Male-predominant stiffness        |          |       |

## Impact
*"Identified sex-divergent adaptive strategies with implications for targeted therapy."*  

#### Clinical Implications  
| Phenotype                 | Sex Association       | Therapeutic Considerations        |  
|---------------------------|-----------------------|-----------------------------------|  
| Low-Pressure Adaptation   | Neutral               | Monitor; minimal intervention    |  
| Compensated Overload      | Female-predominant    | Vasodilators, volume management  |  
| Diastolic Remodeling      | Male-predominant      | Stiffness modulators, rate control |  

--- 
### Conclusions
#### Why this matters

1. **Sex-Specific Adaptations**:  
   - Females compensate via preserved EF under load (Cluster 1).  
   - Males develop diastolic-driven remodeling (Cluster 2), aligning with known sex differences in heart failure.  
2. **Non-Progressive Phenotypes**:  
   - Clusters represent distinct strategies (not severity stages), supporting tailored interventions.  
3. **Validation of Prior Work**:  
   - Confirms clinical findings published in [https://doi.org/10.1152/ajpheart.00098.2024] with computational rigor.

---

## Repository Structure
Machine-learning-hemodynamics/

├── deploy/ # Interactive dashboard deployment

├── results/ # Analysis outputs

├── src/ # Python source code

│ ├── hierarchical-clustering.py

│ ├── k-means-clustering.py

│ ├── cluster-discriminators.py

│ ├── statistical-validation.py

├── sample data/

│ ├── Parameter Legend.xlsx

│ ├── synthetic data.csv

└── README.md # Project overview

### How to use this Repository
1. Install requirements:  
   `pip install -r requirements.txt`
2. Run notebooks in order
3. See `results/` for final outputs

> **Note**: Clinical interpretations require domain expertise - consult cardiology literature for phenotype correlations.

### License

This project is licensed under the CC by NC 4 License. See the [LICENSE](LICENSE) file for details.

--- 

--- 
# Detailed Appendix

## 1. Hierarchical Clustering 
**Exploring Hemodynamic Groupings via Unsupervised Hierarchical Clustering Analysis**  

### Objective  
- Cluster multi-treatment hemodynamics data using PCA, hierarchical clustering, and DBSCAN.  

### Methods  
**Preprocessing:**  
- Z-score standardized 25 hemodynamic features (`StandardScaler`).  
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
Identify natural groupings in hemodynamic responses across 184 samples.

### Methods  
**Preprocessing:**  
- Z-score standardized 25 features (`StandardScaler`).  
- PCA (`n_components=6`, 70% variance retained).  

**Clustering:**  
- K-means: `k=3` (silhouette = 0.51), `k=5` (silhouette = 0.42).  
- Validated stability (`random_state=42`).  

### Key Results  
#### A. Three-Cluster Solution (`k=3`)  

**Cluster Profiles**  

| Cluster | Top Features                     | Phenotype Interpretation          | Sex Bias |  
|---------|----------------------------------|-----------------------------------|----------|  
| **C0**  | • Low PVR, EDP                   | *Healthy adaptation*              | Neutral  |  
|         | • High EF, SV                    | Optimal biventricular function    |          |  
| **C1**  | • Moderate ↑ PVR, ESP            | *Compensated pressure overload*   | ♀ 78%    |  
|         | • Preserved EF                   | Female-predominant adaptation     |          |  
| **C2**  | • High EDP, elastance            | *Diastolic-driven remodeling*     | ♂ 82%    |  
|         | • Reduced SV, ↑ HR               | Male-predominant stiffness        |          |       |


**Biological Relevance:**  
- Clustering indicates degree of mechanical adaptation aligns with hemodynamic profiles. C0 shows minimal changes; C1 suggests mechanical adaptation, C2 shows stiffening with reduced function.

#### B. Five-Cluster Solution (`k=5`)  
**Cluster Profiles:**  
- Splits C0 into nuanced subgroups (e.g., High/Low Ejection Fraction).  
- Emergent phenotype (K5C0) from K3C1/K3C2 (highest pressure/output).  

**Utility:**  
- Captures subtle phenotypes but may overfit (silhouette = 0.25).  

### Visualization  
- Silhouette plots (`k=3`, `k=5`).  
- PCA/t-SNE plots showing cluster separation and sub-groupings.  

### Validation
- Silhouette scores: 0.25 (k = 3), stable across random seeds
- ANOVA confirmed significant differences in key hemodynamic features across clusters, including dp/dt, ESP, PVR (p<0.05).

### Interpretation  
**Robustness:**  
- Silhouette scores consistent across raw/PCA-reduced data.  
- Phenotypes:  
  - High ejection fraction (K3C1/K5C2) vs. High mechanical alterations (K3C2/K5C3).  

**Novelty:**  
- Clusters ≠ treatment groups → suggests new hemodynamic subtypes for targeted therapy with sex-bias.
- Mechanical adaptation linked to evolving hemodynamic profiles across multiple timepoints.

### Limitations  
- Moderate silhouette scores (0.2–0.3): clusters overlap.  
- Dependence on `k`: `k=3` (conservative) vs. `k=5` (exploratory).
- Timepoints are observational (not longitudinal).
- EF alone may not capture subtle dysfunction.

### Next Steps  
- **Clinical correlation**: Test cluster-outcome links (e.g., survival, heart failure).  
- **Validation cohort**: Replicate in independent dataset.  


---
## Supervised Analysis

### Motivation
A feature like EDP might be higher in c2 than c0, but PCA and cluster means do not explain what features drives the cluster assignment, nor their importance and interaction. SHAP analysis supplements this by showing how features push the model toward clusters, and how strongly — even in the presence of other features like contractility or vascular resistance (i.e. global and local importance, nonlinear effects, and feature directionality).

### Objective  
Investigate the hemodynamic features defining the boundaries between cluster phenotypes

### Methods 
- Random Forest Classifier model fitted to data
- SHAP analysis of Random Forest model
- Statistical validation

### Key Results
**SHAP Global Feature Importance**
- Primary discriminators:
  1. PVR (afterload)
  2. Systolic pressure (pressure overload) 
  3. Fulton Index (Hypertrophy)
  4. RV Mass (Hypertrophy)

- Collectively, these results indicate that pressure overload severity was the primary determinant of cluster assignment.

**Cluster-Specific Drivers**

| Cluster | Top Features                     | Phenotype Interpretation          | Sex Bias |  
|---------|----------------------------------|-----------------------------------|----------|  
| **C0**  | • Low PVR, EDP                   | *Healthy adaptation*              | Neutral  |  
|         | • High EF, SV                    | Optimal biventricular function    |          |  
| **C1**  | • Moderate ↑ PVR, ESP            | *Compensated pressure overload*   | ♀ 78%    |  
|         | • Preserved EF                   | Female-predominant adaptation     |          |  
| **C2**  | • High EDP, elastance            | *Diastolic-driven remodeling*     | ♂ 82%    |  
|         | • Reduced SV, ↑ HR               | Male-predominant stiffness        |          |       |
**Key Findings**

• C0: Preserved mechanics in low-pressure state

• C1: Afterload-driven with compensatory changes

• C2: Advanced ventricular stiffness and impairment

**Statistical Validation**

1-factor ANOVA showed significant effects of cluster on key features, including dp/dt max, dp/dt min, ESP, HR, PVR, Eed and Ees. Post-hoc analysis (Tukey HSD) indicated significant differences in all pair-wise cluster comparisons for dp/dt max, dp/dt min, ESP, PVR, and Eed (p<0.05). C2 showed a significantly decreased HR compared to c0 (p<0.05). Both C2 and C1 showed significantly increased systolic and diastolic stiffness (Ees, Eed, p<0.05) compared to C0.

**Biological Relevance**

A significant contractility and diastolic stiffness response explained C1's mechanical adaptation, in contrast to C1, which responded to a larger pressure overload with greater diastolic pressure and dysfunction, corresponding to significantly decreased heart rate and ejection fraction.

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

- tSNE visualization of k3 clusters.png

- tSNE visualization of k5 clusters.png

- KCluster Domain Phenotyping.xlsx: Counts of clustering phenotypes across key features (treatment duration, relative vs raw), as well as bar graphs showing means, stds, and significant differences between key features for k3 clustering results. Followup interpretation gided by domain expertise.

-/Principal Component Analysis

- BiPlots for PC12.png: Visualization of sample distribution across PC1 and 2.

- PCA eigenvalue cumsum.png: Relative % of cumulative variance explained by additional principal components.

-/Supervised Analysis/

-/Cluster Discriminators/

- SHAP global feature importance.png

- SHAP Beeswarm C0.png

- SHAP Beeswarm C1.png

- SHAP Beeswarm C2.png

