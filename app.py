import streamlit as pd
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ==========================================
# 1. PAGE INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Viability Prediction XAI Tool", layout="wide")

st.title("🧪 Viability Prediction XAI Tool")
st.markdown("##### Good Manufacturing Practice (GMP) Compliant Predictive Monitoring Dashboard")
st.write("---")

# ==========================================
# 2. CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_xgboost_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_cqa_model.json")
    return model

try:
    model = load_xgboost_model()
except Exception as e:
    st.error(f"❌ Model File Missing! Please ensure 'xgboost_cqa_model.json' has been committed to your repository.")
    st.stop()

# ==========================================
# 3. LIVE GOOGLE SHEETS LOGGING FUNCTION
# ==========================================
def log_to_google_sheets(row_data):
    try:
        # Define API authorization access scopes
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        
        # Access credentials directly from Streamlit's secure secrets vault
        secret_info = st.secrets["gcp_service_account"]
        
        # Defensive fix for private key newline rendering issues
        if "private_key" in secret_info and "\\n" in secret_info["private_key"]:
            secret_info = dict(secret_info)
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")

        creds = Credentials.from_service_account_info(secret_info, scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open the master spreadsheet workbook
        spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1upEoaEmuhZeLseIXF-Ym7Ym5EAnvFqE69pE8nF29hI4/edit")
        
        # FIX: Select worksheet sheet1 before appending row data
        worksheet = spreadsheet.sheet1
        worksheet.append_row(row_data)
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Cloud Audit Logging Failed: {str(e)}")
        return False

# ==========================================
# 4. OPERATOR INPUT PANEL (SIDEBAR)
# ==========================================
st.sidebar.markdown("### 🎛️ Operator Input Panel")
st.sidebar.info("Enter precise bioreactor parameters below to simulate a real-time batch prediction.")

# Core Bioprocess Features
ph_val = st.sidebar.number_input("pH", value=7.00, format="%.2f")
do_val = st.sidebar.number_input("Dissolved Oxygen (%)", value=60.00, format="%.2f")
glucose_val = st.sidebar.number_input("Glucose (mM)", value=10.00, format="%.2f")
lactate_val = st.sidebar.number_input("Lactate (mM)", value=15.00, format="%.2f")
temp_val = st.sidebar.number_input("Temperature (oC)", value=37.00, format="%.2f")
co2_val = st.sidebar.number_input("CO2 (%)", value=5.00, format="%.2f")
agitation_val = st.sidebar.number_input("Agitation (rpm)", value=100.00, format="%.2f")
seeding_val = st.sidebar.number_input("Seeding Density (cells/mL)", value=10000.00, format="%.2f")
cell_count_val = st.sidebar.number_input("Cell Count", value=500000.00, format="%.2f")
pop_doubling_val = st.sidebar.number_input("Population Doubling", value=1.00, format="%.2f")
study_ref_x = st.sidebar.number_input("Study_Reference_x", value=0.00, format="%.2f")
donor_val = st.sidebar.number_input("Donor", value=0.00, format="%.2f")
tissue_val = st.sidebar.number_input("Tissue (0=BoneMarrow, 1=Adipose)", value=1.00, format="%.2f")
study_ref_y = st.sidebar.number_input("Study_Reference_y", value=0.00, format="%.2f")
day_time_val = st.sidebar.number_input("Day / Time", value=1.00, format="%.2f")

predict_button = st.sidebar.button("Predict Viability")

# ==========================================
# 5. DASHBOARD LAYOUT & EXECUTION FLOW
# ==========================================
if predict_button:
    # Package into a DataFrame matching model requirements exactly
    feature_dict = {
        "pH": [ph_val],
        "Dissolved Oxygen (%)": [do_val],
        "Glucose (mM)": [glucose_val],
        "Lactate (mM)": [lactate_val],
        "Temperature (oC)": [temp_val],
        "CO2 (%)": [co2_val],
        "Agitation (rpm)": [agitation_val],
        "Seeding Density (cells/mL)": [seeding_val],
        "Cell Count": [cell_count_val],
        "Population Doubling": [pop_doubling_val],
        "Study_Reference_x": [study_ref_x],
        "Donor": [donor_val],
        "Tissue (0=BoneMarrow, 1=Adipose)": [tissue_val],
        "Study_Reference_y": [study_ref_y],
        "Day / Time": [day_time_val]
    }
    current_batch = pd.DataFrame(feature_dict)

    # Compute live inference
    prediction = float(model.predict(current_batch)[0])
    
    # Process calculations
    lac_glu_ratio = round(lactate_val / glucose_val, 2) if glucose_val != 0 else 0.0
    drift_status = "NORMAL" if (7.0 <= ph_val <= 7.4 and 40.0 <= do_val <= 80.0) else "DRIFT DETECTED"
    risk = "HIGH (Critical)" if prediction < 80.0 else "LOW (Stable)"

    # Render top KPI metrics interface layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔬 Process Status")
        st.metric(label="Lactate-Glucose Ratio", value=f"{lac_glu_ratio}")
    with col2:
        st.markdown("### ⚙️ Drift Detection")
        if drift_status == "NORMAL":
            st.success("✅ NORMAL: Process within control limits.")
        else:
            st.sidebar.warning("⚠️ OUT OF BOUNDS: Dynamic drift verified.")
    with col3:
        st.markdown("### 🎯 CQA Prediction")
        st.metric(label="Predicted Viability (%)", value=f"{prediction:.2f}%")
        st.caption(f"Status alert: **{risk}**")

    # Construct complete structured transaction line for security ledger
    audit_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # A: Time_Stamp
        "System_Operator",                            # B: User
        float(round(prediction, 4)),                  # C: Predicted_Viability
        risk,                                         # D: Risk_Level
        drift_status,                                 # E: Drift_Status
        "v1.2.0-GMP",                                 # F: Software_Version
        float(temp_val),                              # G: Temperature
        float(agitation_val),                         # H: Agitation
        float(ph_val),                                # I: pH
        float(do_val),                                # J: DO
        float(seeding_val),                           # K: Seeding_Density
        "Adipose" if tissue_val == 1.0 else "BoneMarrow", # L: Tissue
        float(glucose_val),                           # M: Glucose
        float(lactate_val)                            # N: Lactate
    ]

    # Cloud submission tracking
    with st.spinner("Securing audit record in cloud ledger..."):
        success = log_to_google_sheets(audit_row)

    if success:
        st.sidebar.success("✅ Audit trail securely pushed to live Google Ledger.")
    else:
        st.sidebar.warning("⚠️ Warning: Output calculated but cloud log synchronization failed.")

    # 🧠 EXPLAINABLE AI SECTION (Runs synchronously inside button scope)
    st.write("---")
    st.subheader("🧠 Explainable AI (SHAP Interpretation)")
    
    with st.spinner("Calculating local feature attributions..."):
        explainer = shap.Explainer(model)
        shap_values = explainer(current_batch)

        # Build clean visual plot canvas
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)

else:
    # Prompt shown before the operator hits submit
    st.info("👉 Please enter the current bioreactor telemetry parameters in the sidebar and click 'Predict Viability' to view live predictions and parameter attributions.")
