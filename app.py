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
    model_obj = xgb.XGBRegressor()
    model_obj.load_model("xgboost_cqa_model.json")
    return model_obj

# Initialize global variable placeholder to avoid NameErrors
model = None

try:
    model = load_xgboost_model()
except Exception as e:
    st.error(f"❌ Model File Error: Please ensure 'xgboost_cqa_model.json' is present in your root directory. Detail: {str(e)}")

# ==========================================
# 3. LIVE GOOGLE SHEETS LOGGING FUNCTION
# ==========================================
def log_to_google_sheets(row_data):
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        secret_info = st.secrets["gcp_service_account"]
        
        if "private_key" in secret_info and "\\n" in secret_info["private_key"]:
            secret_info = dict(secret_info)
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")

        creds = Credentials.from_service_account_info(secret_info, scopes=scopes)
        client = gspread.authorize(creds)
        
        spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1upEoaEmuhZeLseIXF-Ym7Ym5EAnvFqE69pE8nF29hI4/edit")
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
donor_val = st.sidebar.number_input("Donor", value=0.00, format="%.2f")
tissue_val = st.sidebar.number_input("Tissue (0=BoneMarrow, 1=Adipose)", value=1.00, format="%.2f")

predict_button = st.sidebar.button("Predict Viability")

# ==========================================
# 5. DASHBOARD LAYOUT & EXECUTION FLOW
# ==========================================
if predict_button:
    if model is None:
        st.error("❌ Cannot calculate prediction because the XGBoost model failed to initialize. Please check your repository files.")
    else:
        # Match signature parameters exactly to the model topology structure
        feature_dict = {
            "pH": [ph_val],
            "Dissolved Oxygen (%)": [do_val],
            "Glucose (mM)": [glucose_val],
            "Lactate (mM)": [lactate_val],
            "Temperature (OC)": [temp_val], 
            "CO2 (%)": [co2_val],
            "Agitation (rpm)": [agitation_val],
            "Seeding Density (cells/mL)": [seeding_val],
            "Cell Count": [cell_count_val],
            "Population Doubling": [pop_doubling_val],
            "Study_Reference_x": [0.00],
            "Donor": [donor_val],
            "Tissue (0=BoneMarrow, 1=Adipose)": [tissue_val],
            "Study_Reference_y": [0.00],
            "Day / Time": [1.00]
        }
        
        current_batch = pd.DataFrame(feature_dict)

        # Compute live inference safely
        prediction = float(model.predict(current_batch)[0])
        
        # Process calculations
        lac_glu_ratio = round(lactate_val / glucose_val, 2) if glucose_val != 0 else 0.0
        drift_status = "NORMAL" if (7.0 <= ph_val <= 7.4 and 40.0 <= do_val <= 80.0) else "DRIFT DETECTED"
        risk = "HIGH (Critical)" if prediction < 80.0 else "LOW (Stable)"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔬 Process Status")
            st.metric(label="Lactate-Glucose Ratio", value=f"{lac_glu_ratio}")
        with col2:
            st.markdown("### ⚙️ Drift Detection")
            if drift_status == "NORMAL":
                st.success("✅ NORMAL: Within Limits")
            else:
                st.warning("⚠️ DRIFT DETECTED")
        with col3:
            st.markdown("### 🎯 CQA Prediction")
            st.metric(label="Predicted Viability", value=f"{prediction:.2f}%")
            st.caption(f"Risk Evaluation: **{risk}**")

        # Cloud Logging Transaction
        audit_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "System_Operator",
            float(round(prediction, 4)), risk, drift_status, "v1.8.0-GMP",
            float(temp_val), float(agitation_val), float(ph_val), float(do_val),
            float(seeding_val), "Adipose" if tissue_val == 1.0 else "BoneMarrow",
            float(glucose_val), float(lactate_val)
        ]

        with st.spinner("Securing audit record in cloud ledger..."):
            success = log_to_google_sheets(audit_row)

        if success:
            st.sidebar.success("✅ Audit trail pushed to Google Sheets.")
        else:
            st.sidebar.warning("⚠️ App functional, but cloud log sync failed.")

        # 🧠 EXPLAINABLE AI SECTION
        st.write("---")
        st.subheader("🧠 Explainable AI (SHAP Interpretation)")
        
        with st.spinner("Calculating local feature attributions..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(current_batch)

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_values[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as shap_error:
                st.error(f"Visualizer Notice: Prediction succeeded, but SHAP generation skipped: {str(shap_error)}")

else:
    st.info("👉 Please enter the current bioreactor telemetry parameters in the sidebar and click 'Predict Viability' to view live predictions and parameter attributions.")
