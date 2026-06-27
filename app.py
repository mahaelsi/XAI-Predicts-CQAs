import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Biopharma AI CQA Predictor", layout="wide")

st.title("🧬 Biopharmaceutical Explainable AI Platform")
st.markdown("### MSC Manufacturing: Critical Quality Attribute (CQA) Predictor")
st.markdown("---")

# Interface setup
st.sidebar.header("Operator Input Panel")
st.sidebar.info("Enter current bioreactor parameters below to simulate a real-time batch prediction.")

# Sliders for user input
pH = st.sidebar.slider("pH", 6.8, 7.4, 7.1)
DO = st.sidebar.slider("Dissolved Oxygen (%)", 10.0, 100.0, 50.0)
glucose = st.sidebar.slider("Glucose (mM)", 0.0, 30.0, 15.0)
lactate = st.sidebar.slider("Lactate (mM)", 0.0, 40.0, 10.0)

# Simulate feature engineering
lactate_glucose_ratio = lactate / (glucose + 1e-5)
stress_index = lactate / (DO + 1e-5)

col1, col2, col3 = st.columns(3)

with col1:
    st.header("🔬 Process Status")
    st.write(f"**Lactate-Glucose Ratio:** {lactate_glucose_ratio:.2f}")
    st.write(f"**Culture Stress Index:** {stress_index:.2f}")
    
with col2:
    st.header("⚙️ Drift Detection")
    # Simulated drift logic for the prototype dashboard
    if stress_index > 0.8:
        st.error("🚨 HIGH: Process Drift Detected!")
    else:
        st.success("✅ NORMAL: Process within control limits.")

with col3:
    st.header("🎯 CQA Prediction")
    # Simulated prediction based on inputs (In a real app, you load the .json model here)
    simulated_yield = 2.5 - (stress_index * 0.5) + (pH * 0.1)
    st.metric(label="Predicted Cell Yield (10^6 cells/mL)", value=f"{simulated_yield:.2f}")

st.markdown("---")
st.header("🧠 Explainable AI (SHAP Interpretation)")
st.info(f"The model predicted a yield of {simulated_yield:.2f}. The primary driving factor for this prediction was the Culture Stress Index. Ensure Dissolved Oxygen levels are maintained to prevent hypoxic lactate accumulation.")
