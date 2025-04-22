# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:56:54 2025

@author: edkwa
"""

# pages/About.py
import streamlit as st

def show():
    st.title("Clinical Cardiac Phenotyping Dashboard")     
       
    st.markdown("""
### * Bridging machine learning and clinical workflows *

### Why this exists
*Clinicians are overloaded with data but starved for insights. There is great potential for machine learning to help - but only if the outputs are **clear, actionable, trustworthy, and integrated into existing workflows.**

*Clinicians shouldn't need to understand SHAP values - they need actionable insights. This dashboard demonstrates how insights from rigorously trained and interpreted ML models can integrate into clinical workflows, translating raw predictions into simple, actionable insights with trust-building visualizations.'

### Hypothetical Impact
This demo illustrates how ML models can be adapted for
- **Time saving**: In a simulated patient cohort, this dashboard reduced the time-to-insight from raw values by over 90% compared to manual data review by an expert.

- **Adoption potential**: Designed with iterative clinician feedback to prioritize interpretability and useability.

- **Hypothetically reduce cognitive load** by automating routine assessments.
""")
    st.markdown("""
# <a href="https://newsnetwork.mayoclinic.org/discussion/science-saturday-artificial-intelligence-wont-replace-your-doctors-but-it-could-make-them-better/" style="color: inherit; text-decoration: none; font-size:20px;"> Article: "Artificial intelligence won't replace your doctors, but it could make them better"</a>
""", unsafe_allow_html=True)

    st.markdown("""
### License

This project is licensed under the CC by NC 4 License.

###Contact
EthanDKwan@gmail.com
""")

show()