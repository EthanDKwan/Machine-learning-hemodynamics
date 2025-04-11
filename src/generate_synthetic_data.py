# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:47:42 2025

@author: edkwa
"""
import pandas as pd
import numpy as np
import os
"""
Generates synthetic data over physiologic feature ranges
"""

feature_ranges = {
    "GroundTruth": "binary", #Baseline/PAH
    "ID": (1,10),   #random ID generator
    "TreatmentDuration": (4,12),
    "HR": (240,360),
    "SV": (30,200),
    "EDV": (80,360),
    "ESV": (40,240),
    "EF": (40,70),          #Ejection fraction [%]
    "CO": (10,70),
    "dp/dt max": (1500,3500),
    "dp/dt min": (-3500,-1500),
    "PVR": (0.5,3.5),       #Pulmonary vascular resistance [dyn·s/cm³]
    "Occlusion EDPo": (0,7),
    "AdjEDP": (0.1,12),     #Diastolic pressure [mmHg]
    "AdjESP": (30,100),     #Systolic pressure [mmHg]
    "Ees": (0.3, 1.8),      #systolic elastance [mmHg/mL]
    "Eed": (0.001, 0.8),    #diastolic elastance [mmHg/mL]
    "Ea": (0.05, 1.2),
    "VVC": (0.3, 5),
    "Sex": "binary",
    "Age": (15, 22),
    "RV Thickness": (.9, 2.2),
    "RV Mass": (0.2, 0.9),
    "Fulton Index": (0.3, 0.6),
    "Mass": (240, 440),
    "Right Atrial Mass": (0.02, 0.1),
    "Left Atrial Mass": (0.02, 0.1),
    "SHI": (0.2, 1.05),
    "Liver Mass": (8, 25)
}

# Generate data
np.random.seed(42)
sample_data = pd.DataFrame({
    feature: (
        np.random.randint(0, 2, size=15)  # Binary 0/1
        if range_spec == "binary" 
        else np.random.uniform(low=range_spec[0], high=range_spec[1], size=15)
    )
    for feature, range_spec in feature_ranges.items()
})

save_dir = os.path.join("..", "Sample Data")
save_path = os.path.join(save_dir, "synthetic_data.csv")
# Save to CSV
sample_data.to_csv(save_path, index=False)