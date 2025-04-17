# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 16:05:25 2025

@author: edkwa
"""
import streamlit as st

def show():
    st.title("Glossary of Feature Names")     
    glossary = {
        "PVR": "Pulmonary Vascular Resistance: Resistance of the vascular bed the heart pumps blood into",
        "HR": "Heart rate",
        "CO": "Cardiac Output: Volume of blood pumped per time interval (typically per minute)",
        "SV": "Stroke Volume: Volume of blood pumped per single contraction",
        "EDV": "End-diastolic volume: Volume of blood that fills the ventricle following full relaxation",
        "ESV": "End-systolic volume: Volume of blood remaining in the ventricle following contraction",
        "EF": "Ejection Fraction: Fraction of blood filling the heart that is pumped out - a measure of cardiac efficiency",
        "dp/dt max": "maximum rate of increase in RV pressure experienced by the heart - an index of myocardial contractility",
        "dp/dt min": "maximum rate of decrease in RV pressure experienced by the heart - an index of relaxative function",
        "occlusion EDPo": "Ventricular minimum pressure: representing the residual pressure in a realistic unloaded state - typically clinically unfeasible",
        "AdjEDP": "End-diastolic pressure: blood pressure in the heart's ventricle chamber at the end of relaxation",
        "AdjESP": "End-systolic pressure: blood pressure in the heart's ventricle chamber at the end of contraction",
        "Ees" : "End-systolic elastance: Slope of the pressure-volume relationship at the end of contraction - a load independent index of contractile stiffness",
        "Eed": "End-diastolic elastance: Slope of the pressure-volume relationship at the end of relaxation - a load independent index of passive stiffness",
        "Ea": "Arterial elastance: a lumped parameter estimate of the total stiffness of the vascular bed the heart pumps against",
        "VVC": "Vascular-ventricular coupling: an index of the balance between ventricular function and arterial stiffness",
        "RV Thickness": "Thickness of the right ventricular myocardium free wall: an index of the hypertrophic muscle growth response to disease - clinically obtained from imaging",
        "RV Mass": "Mass of the right ventricular myocardial free wall: an index of the hypertrophic muscle growth response to disease - clinically estimated from imaging data",
        "Fulton Index": "The ratio of RV free wall mass to the mass of the remaining ventricular/septum walls - an index of the acuity of the RV hypertrophic response",
        "Atrial Mass": "The mass of the atrium wall - typically interpreted as an index of volume overload",
        "SHI": "Septum-hypertrophy index: the ratio of septum wall hypertrophy to LV mass hypertrophy - an index of the acuity of the RV hypertrophic response to pressures acute to the RV chamber"
    }
    for term, definition in glossary.items():
        with st.expander(f"**{term}**"):
            st.write(definition)   
            
show()