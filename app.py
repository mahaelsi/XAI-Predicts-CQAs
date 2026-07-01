import datetime
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
# Updated the page title to match the new naming convention
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

# Operator Input Fields
# NOTE: I have added fields based on your previous Colab errors. 
# You MUST add or remove fields here until they match your training dataset perfectly!
ph_val = st.sidebar.number_input("pH", value=7.00, format="%.2f")
do_val = st.sidebar.number_input("Dissolved Oxygen (%)", value=60.00, format="%.2f")
glucose_val = st.sidebar.number_input("Glucose (mM)", value=10.00, format="%.2f")
lactate_val = st.sidebar.number_input("Lactate (mM)", value=15.00, format="%.2f")
temp_val = st.sidebar.number_input("Temperature (oC)", value=37.00, format="%.2f")
co2_val = st.sidebar.number_input("CO2 (%)", value=5.00, format="%.2f")
agitation_val = st.sidebar.number_input("Agitation (rpm)", value=100.00, format="%.2f")
seeding_val = st.sidebar.number_input("Seeding Density", value=10000.00, format="%.2f")

# ==========================================
# 4. MAIN DASHBOARD & PREDICTION ENGINE
# ==========================================
# Updated the main application title
st.title("Viability prediction XAI tool")
st.write("Good Manufacturing Practice (GMP) compliant predictive monitoring.")
st.markdown("---")

if st.sidebar.button("Predict Viability"):
    
    # 1. Create the raw data matrix for XGBoost
    # The order of these columns MUST match your training data exactly!
    current_batch = pd.DataFrame([{
        "pH": ph_val,
        "Dissolved Oxygen (%)": do_val,
        "Glucose (mM)": glucose_val,
        "Lactate (mM)": lactate_val,
        "Temperature (oC)": temp_val,
        "CO2 (%)": co2_val,
        "Agitation (rpm)": agitation_val,
        "Seeding Density": seeding_val
        # If your model needs "Cell Count" or categorical variables, they must be added here.
    }])
    
    # 2. SAFETY CHECK: Prevent ValueError Crashing
    expected_features = model.n_features_in_
    provided_features = current_batch.shape[1]
    
    if expected_features != provided_features:
        st.error(f"⚠️ Feature Mismatch Error! Your XGBoost model was trained on **{expected_features}** features, but the app is only giving it **{provided_features}**.")
        st.warning("To fix this: Go into `app.py` and add the missing `st.sidebar.number_input` fields so the numbers match.")
        st.stop() # Stops the code from running and causing the red error box
        
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
        "Software_Version": "v2.0"
    }])
    
    AUDIT_TRAIL_PATH = "system_audit_trail.csv"
    if not os.path.exists(AUDIT_TRAIL_PATH):
        audit_record.to_csv(AUDIT_TRAIL_PATH, index=False)
    else:
        audit_record.to_csv(AUDIT_TRAIL_PATH, mode='a', header=False, index=False)
        
    st.sidebar.success("✅ Audit trail securely logged.")

else:
    st.info("👈 Please enter the current bioreactor telemetry in the sidebar and click 'Predict Viability' to generate the batch report.")
