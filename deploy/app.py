# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:34:38 2025

@author: edkwa
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from pathlib import Path

# --- Config ---
st.set_page_config(layout="wide", page_title="Hemodynamic Phenotyping")

# --- Load Assets ---
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "trainedmodels/trained_k3_model.joblib"
    model = joblib.load(model_path)
    model_path = Path(__file__).parent / "trainedmodels/pca_model.joblib"
    pca = joblib.load(model_path)
    model_path = Path(__file__).parent / "trainedmodels/scaler.joblib"
    scaler = joblib.load(model_path)
    return model, pca, scaler

@st.cache_resource
def load_explainer():
    model_path = Path(__file__).parent / "trainedmodels/SHAP_explainer.joblib"
    explainer = joblib.load(model_path)
    return explainer

model, pca, scaler = load_model()
explainer = load_explainer()

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def get_prediction(input_25_features):
    #Verify shape of input features
    X = np.array(input_25_features).reshape(1,-1)
    assert X.shape == (1,25), f"Expected (1,25), got{X.shape}"
    
    scaled_input = scaler.transform(X)
    pca_features = pca.transform(scaled_input)
    #Predict Cluster in PCA space
    cluster = model.predict(pca_features)[0]
    distances = model.transform(pca_features)
    
    #Confidence calculation #1 using softmax over distances
    cluster_probs = softmax(-distances) #Higher distance = lower probability
    #Confidence calculation #2 using inverse distances (scaled)    
    #cluster_probs = 1/(1+distances)
    #cluster_probs /= cluster_probs.sum(axis=1,keepdims=True) #normalize
    confidence = cluster_probs[0,cluster]
    
    return {
        'cluster': cluster,
        'pca_components': pca_features[0],  # Return reduced features, should be 1D array (6,)
        'confidence': confidence,
        'cluster_probs':cluster_probs
        }

# --- Clinical Descriptions ---
PHENOTYPE_DESCRIPTIONS = {
    0: {
        "name": "Hyperdynamic",
        "clinical": "Early compensated state with elevated contractility",
        "management": "Monitor for decompensation"
    },
    1: {
        "name": "Afterload-dominant",
        "clinical": "Hypertensive mechanical remodeling",
        "management": "Consider vasodilators"
    },
    2: {
        "name": "Functional Reserve Adaptation with Stiff Ventricle",
        "clinical": "Early HFpEF + Diastolic dysfunction pattern",
        "management": "Rate control strategies, inotropes"
    }
}

# --- Input Widgets ---
def create_inputs():
    st.sidebar.header("Features: Hemodynamic Parameters")
    
    # Organize into clinical categories
    with st.sidebar.expander("🩸 Pressure Overload"):
        AdjESP = st.slider("Systolic BP (mmHg)", 15, 120, 25, 1) #low, high, default
        AdjEDP = st.slider("Diastolic BP (mmHg)", 0.0, 10.0, 2.0,0.1)
        PVR = st.slider("Pulonary Vascular Resistance (dyn·s/cm³)", 0.0,3.0,0.35,0.01)
        Ea = st.slider("Arterial Elastance (dyn)",0.0,2.0,0.2,0.01)
        dpdt_max = st.slider("dP/dt Max (mmHg/s)",0,3500,1000)
        dpdt_min = st.slider("dP/dt_min (mmHg/s)",-3500,0,-1000)
        
    with st.sidebar.expander("💓 Cardiac Chamber Geometry + Mechanics"):
        HR = st.slider("Heart Rate (bpm)",0,400,250)
        EDV = st.slider("Diastolic Volume (uL)", 50, 360, 260)
        ESV = st.slider("Systolic Volume(uL)",30,250,100)
        SV = st.slider("Stroke Volume(uL)",20,330,EDV-ESV)
        EF = st.slider("Ejection Fraction (%)",0,100,round(SV/EDV*100))
        CO = st.slider("Cardiac Output (uL)",0.0,80.0,HR*SV/1000)
        EDPo = st.slider("RV Minimum Pressure (mmHg)", 0.0,7.0,1.5)
        Ees = st.slider("Systolic Elastance (mmHg/uL)",0.0,2.0,0.2,0.01)
        Eed = st.slider("Diastolic Elastance (mmHg/uL)",0.0,0.5,0.01,0.01)
        #VVC = st.slider("Cardio-pulmonary coupling (ratio)",0.0,4.0,1.0,0.1)
                        
    with st.sidebar.expander("🫀 Morphology"):
        RVmass = st.slider("RV Mass (g)", 0.1,1.5,0.4,0.05)
        RVthickness = st.slider("RV Thickness (mm)",0.6,2.5,1.1,0.05)
        FultonIndex = st.slider("Fulton Index (Ratio)", 0.1,1.0,0.3,0.01)
        RAmass = st.slider("Right Atrial Mass (g)", 0.01,0.1,0.03,0.01)
        LAmass = st.slider("Left Atrial Mass (g)", 0.01, 0.1, 0.03,0.01)
        Mass = st.slider("Total Normal Mass (g)", 240,440,200)
        SHI = st.slider("Septal Hypertrophy Index (Ratio)", 0.2, 1.05, 0.5,0.01)
        #Livermass = st.slider("Liver Mass (g)", 5, 30, 15)
    
    with st.sidebar.expander("🩺 Misc Metrics"):
         sex_label = st.select_slider("Sex",options=["Male", "Female"],value="Female")
         sex_numeric = 0 if sex_label == "Male" else 1  # Convert to 0/1
         age = st.slider("Age (years)", 0,100,30)
         Treatmentduration = st.slider("Treatment Duration(wks)", 0,12,0)
        
    # Return as dictionary matching model's expected features
    return {
        'Treatment Duration': Treatmentduration,
        'HR': HR,
        'SV': SV,
        'EDV': EDV,
        'ESV': ESV,
        'EF': EF,
        'CO': CO,
        'dp/dt max': dpdt_max,
        'dp/dt min': dpdt_min,
        'PVR': PVR,
        'Occlusion EDPo': EDPo,
        'AdjEDP': AdjEDP,        
        'AdjESP': AdjESP,
        'Ees': Ees,
        'Eed': Eed,
        'Ea': Ea,
        'Sex': sex_numeric,
        'Age': age/14+14,
        'RV Thickness': RVthickness,
        'RV Mass': RVmass,
        'Fulton Index': FultonIndex,
        'Total Mass': Mass,
        'Right Atrial Mass': RAmass,
        'Left Atrial Mass': LAmass,
        'SHI': SHI
    }

# --- Visualization Functions ---
def plot_radar(input_features, cluster_center, features, title):
    """Compare user inputs to cluster centroids"""
    assert len(input_features) == len(cluster_center), \
           f"Got {len(input_features)} features but cluster has {len(cluster_center)}"
    
    features = list(features)
    input_features = np.array(list(input_features.values())) if isinstance(input_features, dict) else np.array (input_features)
    cluster_center = np.array(cluster_center)
    #original_space_approx = pca.inverse_transform(model.cl)
    df = pd.DataFrame({
            'Feature': features,
            'Your Values': input_features,
            'Typical Values (Cluster Mean)': cluster_center
        })
    fig = px.line_polar(df,
        r='Your Values',
        theta='Feature',
        line_close=True,color_discrete_sequence=['red'],
        title=f"<b>{title}</b><br><sub>Red: Your Inputs | Blue: Cluster Average</sub>"
    )
    fig.add_trace(px.line_polar(df, r='Typical Values (Cluster Mean)', theta='Feature').data[0])

    fig.update_layout(
        legend=dict(
            title="Legend:",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.1
        ),
        polar=dict(
            radialaxis=dict(visible=True, gridcolor='lightgray'),
            angularaxis=dict(gridcolor='lightgray')
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_shap(shap_values, features, cluster_idx, feature_names):
    shap_values = np.transpose(shap_values, (2, 0, 1))
    explanation = shap.Explanation(
        values = shap_values[cluster_idx][0],
        base_values = explainer.expected_value[cluster_idx],
        data = features.iloc[0].values,
        feature_names = feature_names
        )
    
    st.subheader(f"Feature Contributions to {PHENOTYPE_DESCRIPTIONS[cluster_idx]['name']}") 
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, max_display=10)
    st.pyplot(fig)
    

# --- Main App Logic ---
def main():    
     
    st.title("Cardiac Phenotyping Dashboard")
    st.write("⇐ Define some hemodynamic inputs on the left")
    
    # Get inputs   
    inputs = create_inputs()
    
    #Original space
    model_path = Path(__file__).parent / "trainedmodels/cluster means.csv"
    cluster_means = pd.read_csv(model_path, index_col=0)
    #cluster_means = pd.read_csv("trainedmodels/cluster means.csv",index_col = 0)

    input_df = pd.DataFrame([inputs])
    
    if st.sidebar.button("Predict Phenotype"):
        # Predict
        result = get_prediction(input_df.values)
        #PCA Space
        predicted_cluster = result['cluster']
        pca_center = model.cluster_centers_[predicted_cluster]
        pca_components = result['pca_components']
        confidence = result['confidence']
        cluster_probs = result['cluster_probs']
        probabilities = cluster_probs[0]
        shap_values = explainer.shap_values(input_df)
        cluster_means.index = cluster_means.index.astype(int)
        original_center = cluster_means.loc[predicted_cluster].values #should be 25D array
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Predicted Phenotype")
            
            st.success(f"{PHENOTYPE_DESCRIPTIONS[predicted_cluster]['name']}\n"f"({confidence*100:.1f}% confidence)")
            
            # Clinical implications
            st.markdown("### Clinical Interpretation")
            st.info(PHENOTYPE_DESCRIPTIONS[predicted_cluster]["clinical"])
            st.write("### Potential Management")
            st.warning(PHENOTYPE_DESCRIPTIONS[predicted_cluster]["management"])
        with col2:
            # Probability distribution
            st.subheader("Cluster Probabilities")
            prob_df = pd.DataFrame({
                'Phenotype': [PHENOTYPE_DESCRIPTIONS[i]['name'] for i in range(3)],
                'Probability': probabilities
            })
            st.bar_chart(prob_df.set_index('Phenotype'))
        
        # Visualizations
        st.write("### Model Visualizations")
        plot_radar(pca_components, pca_center, [f"PC{i+1}" for i in range(6)], "PCA Space")
        plot_radar(inputs, original_center, inputs.keys(), "Original feature space")
        
        plot_shap(shap_values = shap_values, features = input_df, cluster_idx = predicted_cluster, feature_names = input_df.columns.tolist())
        
        # Download results
        csv_data = input_df.to_csv(index=False)
        csv_data += f"\nPrediction Phenotype,{PHENOTYPE_DESCRIPTIONS[predicted_cluster]['name']}"
        csv_data += f"\nPrediction Confidence,{confidence*100:.1f}%"
        csv_data += f"\nClinical Description,{PHENOTYPE_DESCRIPTIONS[predicted_cluster]['clinical']}"
        csv_data += f"\nRecommended Actions,{PHENOTYPE_DESCRIPTIONS[predicted_cluster]['management']}"
        st.download_button(
            label="Download Prediction Report (CSV)",
            data=csv_data,
            file_name=f"hemodynamic_prediction_report{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()