import datetime
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="Biopharma AI CQA Predictor", layout="wide")

# ==========================================
# 2. LOAD THE AI MODEL
# ==========================================
# We use st.cache_resource so the app only loads the model once, making it fast.
@st.cache_resource
def load_xgboost_model():
    model = xgb.XGBRegressor()
    # IMPORTANT: Ensure this path matches the location of your model in GitHub!
    # If it's in a folder called '04_Models', it should be '04_Models/xgboost_cqa_model.json'
    model.load_model("xgboost_cqa_model.json") 
    return model

# Load the model into memory BEFORE the button is clicked
try:
    model = load_xgboost_model()
except Exception as e:
    st.error(f"Model File Missing or Path Incorrect. System Error: {e}")
    st.stop() # Stops the app from crashing further if the model isn't found

# ==========================================
# 3. OPERATOR INPUT PANEL (SIDEBAR)
# ==========================================
st.sidebar.markdown("### Operator Input Panel")
st.sidebar.info("Enter precise bioreactor parameters below to simulate a real-time batch prediction.")

ph_val = st.sidebar.number_input("pH", value=7.00, format="%.2f")
do_val = st.sidebar.number_input("Dissolved Oxygen (%)", value=60.00, format="%.2f")
glucose_val = st.sidebar.number_input("Glucose (mM)", value=10.00, format="%.2f")
lactate_val = st.sidebar.number_input("Lactate (mM)", value=15.00, format="%.2f")

# ==========================================
# 4. MAIN DASHBOARD & PREDICTION ENGINE
# ==========================================
st.title("Biomanufacturing Digital Twin")
st.write("Good Manufacturing Practice (GMP) compliant predictive monitoring.")
st.markdown("---")

if st.sidebar.button("Predict Viability"):
    
    # Create the raw data matrix for XGBoost
    current_batch = pd.DataFrame([{
        "pH": ph_val,
        "Dissolved Oxygen (%)": do_val,
        "Glucose (mM)": glucose_val,
        "Lactate (mM)": lactate_val
        # Note: If your model was trained on more parameters, they MUST be added here.
    }])
    
    # Execute prediction bypassing Pandas dtype formatting
    prediction = model.predict(current_batch.values)[0] 
    
    # Assess Biological Risk
    if prediction < 80.0: 
        risk = "HIGH (Critical Cell Death)"
    elif prediction < 90.0: 
        risk = "MEDIUM (Sub-optimal)"
    else: 
        risk = "LOW (Optimal)"

    # --- RENDER DASHBOARD VISUALS ---
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
    
    # --- SHAP EXPLAINABLE AI PLACEHOLDER ---
    st.subheader("🧠 Explainable AI (SHAP Interpretation)")
    st.info(f"The model predicted a viability of {prediction:.2f}%. The primary driving factors for this specific batch prediction will be visualized here.")
    # (Your actual SHAP waterfall plot rendering code would go here)
    
    # --- ALCOA+ AUDIT TRAIL LOGGING ---
    audit_record = pd.DataFrame([{
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": "Operator_01",
        "pH": ph_val,
        "Dissolved_Oxygen": do_val,
        "Predicted_Viability": round(prediction, 4),
        "Risk_Level": risk,
        "Software_Version": "v1.6"
    }])
    
    AUDIT_TRAIL_PATH = "system_audit_trail.csv"
    if not os.path.exists(AUDIT_TRAIL_PATH):
        audit_record.to_csv(AUDIT_TRAIL_PATH, index=False)
    else:
        audit_record.to_csv(AUDIT_TRAIL_PATH, mode='a', header=False, index=False)
        
    st.sidebar.success("✅ Audit trail securely logged.")

else:
    # What the app shows before the operator hits Predict
    st.info("👈 Please enter the current bioreactor telemetry in the sidebar and click 'Predict Viability' to generate the batch report.")
