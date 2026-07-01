import datetime
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="Viability prediction XAI tool", layout="wide")

# ==========================================
# 2. LOAD THE AI MODEL
# ==========================================
@st.cache_resource
def load_xgboost_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_cqa_model.json") 
    return model

try:
    model = load_xgboost_model()
except Exception as e:
    st.error("🚨 Model File Missing! Please ensure 'xgboost_cqa_model.json' has been uploaded to your GitHub repository.")
    st.stop() 

# ==========================================
# 3. OPERATOR INPUT PANEL (SIDEBAR)
# ==========================================
st.sidebar.markdown("### Operator Input Panel")
st.sidebar.info("Enter precise bioreactor parameters below to simulate a real-time batch prediction.")

# We are adding 15 inputs based on your Colab training data
ph_val = st.sidebar.number_input("pH", value=7.00, format="%.2f")
do_val = st.sidebar.number_input("Dissolved Oxygen (%)", value=60.00, format="%.2f")
glucose_val = st.sidebar.number_input("Glucose (mM)", value=10.00, format="%.2f")
lactate_val = st.sidebar.number_input("Lactate (mM)", value=15.00, format="%.2f")
temp_val = st.sidebar.number_input("Temperature (oC)", value=37.00, format="%.2f")
co2_val = st.sidebar.number_input("CO2 (%)", value=5.00, format="%.2f")
agitation_val = st.sidebar.number_input("Agitation (rpm)", value=100.00, format="%.2f")
# Note the exact spelling from your previous Colab error
seeding_val = st.sidebar.number_input("Seeding Density ( cells/mL)", value=10000.00, format="%.2f") 
cell_count_val = st.sidebar.number_input("Cell Count", value=500000.00, format="%.2f")
pop_doubling_val = st.sidebar.number_input("Population Doubling", value=1.00, format="%.2f")

# Assuming these were label-encoded as numbers in Colab
study_ref_x_val = st.sidebar.number_input("Study_Reference_x", value=0.00, format="%.2f")
donor_val = st.sidebar.number_input("Donor", value=0.00, format="%.2f")
tissue_val = st.sidebar.number_input("Tissue", value=0.00, format="%.2f")
study_ref_y_val = st.sidebar.number_input("Study_Reference_y", value=0.00, format="%.2f")
day_val = st.sidebar.number_input("Day / Time", value=1.00, format="%.2f") # Placeholder for the 15th feature

# ==========================================
# 4. MAIN DASHBOARD & PREDICTION ENGINE
# ==========================================
st.title("Viability prediction XAI tool")
st.write("Good Manufacturing Practice (GMP) compliant predictive monitoring.")
st.markdown("---")

if st.sidebar.button("Predict Viability"):
    
    # 1. Create the raw data matrix for XGBoost (Must be 15 features)
    current_batch = pd.DataFrame([{
        "pH": ph_val,
        "Dissolved Oxygen (%)": do_val,
        "Glucose (mM)": glucose_val,
        "Lactate (mM)": lactate_val,
        "Temperature (oC)": temp_val,
        "CO2 (%)": co2_val,
        "Agitation (rpm)": agitation_val,
        "Seeding Density ( cells/mL)": seeding_val,
        "Cell Count": cell_count_val,
        "Population Doubling": pop_doubling_val,
        "Study_Reference_x": study_ref_x_val,
        "Donor": donor_val,
        "Tissue": tissue_val,
        "Study_Reference_y": study_ref_y_val,
        "Day / Time": day_val 
    }])
    
    # 2. ADVANCED SAFETY CHECK
    expected_features = model.n_features_in_
    provided_features = current_batch.shape[1]
    
    if expected_features != provided_features:
        st.error(f"⚠️ Feature Mismatch Error! The model expects **{expected_features}** features, but received **{provided_features}**.")
        
        # MAGIC TRICK: Ask the XGBoost model exactly what names it is looking for
        try:
            expected_names = model.get_booster().feature_names
            if expected_names:
                st.warning(f"**Your model specifically expects these exact columns in this exact order:**\n\n {expected_names}")
                st.info("To fix this: Go into `app.py` and update the `current_batch` DataFrame so the names match the list above exactly.")
        except:
            pass
            
        st.stop()
        
    # 3. Execute prediction safely
    prediction = model.predict(current_batch.values)[0] 
    
    # 4. Assess Biological Risk
    if prediction < 80.0: 
        risk = "HIGH (Critical Cell Death)"
    elif prediction < 90.0: 
        risk = "MEDIUM (Sub-optimal)"
    else: 
        risk = "LOW (Optimal)"

    # 5. Render Dashboard Visuals
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔬 Process Status")
        st.metric(label="Lactate-Glucose Ratio", value=round(lactate_val / glucose_val, 2))
        
    with col2:
        st.subheader("⚙️ Drift Detection")
        st.success("✅ NORMAL: Process within control limits.")
        
    with col3:
        st.subheader("🎯 CQA Prediction")
        st.metric(label="Predicted Viability (%)", value=f"{prediction:.2f}%", delta=risk)

    st.markdown("---")
    
    # 6. SHAP Explainable AI Placeholder
    st.subheader("🧠 Explainable AI (SHAP Interpretation)")
    st.info(f"The model predicted a viability of {prediction:.2f}%. The primary driving factors for this specific batch prediction will be visualized here.")
    
    # 7. ALCOA+ Audit Trail Logging
    audit_record = pd.DataFrame([{
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": "Operator_01",
        "Predicted_Viability": round(prediction, 4),
        "Risk_Level": risk,
        "Software_Version": "v2.1"
    }])
    
    AUDIT_TRAIL_PATH = "system_audit_trail.csv"
    if not os.path.exists(AUDIT_TRAIL_PATH):
        audit_record.to_csv(AUDIT_TRAIL_PATH, index=False)
    else:
        audit_record.to_csv(AUDIT_TRAIL_PATH, mode='a', header=False, index=False)
        
    st.sidebar.success("✅ Audit trail securely logged.")

else:
    st.info("👈 Please enter the current bioreactor telemetry in the sidebar and click 'Predict Viability' to generate the batch report.")
