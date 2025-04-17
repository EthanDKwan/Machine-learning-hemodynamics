# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:56:54 2025

@author: edkwa
"""

# pages/About.py
import streamlit as st

def show():
    st.title("Summary")     

       
    st.markdown("""
*Identifying sex-associated cardiac remodeling patterns by mining clinical hemodynamic data*

*Used unsupervised learning and SHAP analysis to discover and validate three distinct adaptive remodeling patterns in clinical pulmonary hypertension data, validating prior clinical findings.*  

*Interactive dashboard deployed for trained model exploration and interactive phenotyping on streamlit*

### Problem  
*"Cardiac hemodynamic remodeling is complex in health and disease; can we identify hidden patterns that improve our understanding of disease adaptation to personalize treatment?"*  

### Methods  
*Applied unsupervised (hierarchical, k-means clustering) and interpretable ML (Random Forest classifier, SHAP) models to a pulmonary hypertension clinical dataset (184 samples, 25 features), integrating statistical validation (ANOVA, post-hoc testing, SHAP analysis) and domain expertise to phenotype clusters.*

### Technologies
- **Frontend**: Streamlit (interactive dashboard)
- **Deployment**: Streamlit Cloud
- **Languages**: Python
- **ML**: scikit-learn
- **Data Handling**: Pandas, numpy, openpyxl
- **Visualization**: Matplotlib, plotly, seaborn

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

### License

This project is licensed under the CC by NC 4 License.

###Contact
EthanDKwan@gmail.com
""")

show()