import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
import json

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

model = None

try:
    model = load_xgboost_model()
except Exception as e:
    st.error(f"❌ Model File Error: Please ensure 'xgboost_cqa_model.json' is present in your repository root. Detail: {str(e)}")

# ==========================================
# 3. APPEND-ONLY AUDIT LEDGER FUNCTION
# ==========================================
def log_to_audit_ledger(row_data, header_names):
    """
    Writes predictions to both an append-only local ledger CSV file
    and Google Sheets for double redundancy and audit compliance.
    """
    # 1. Append to local immutable CSV ledger
    ledger_file = "audit_ledger.csv"
    try:
        file_exists = os.path.exists(ledger_file)
        ledger_df = pd.DataFrame([row_data], columns=header_names)
        ledger_df.to_csv(ledger_file, mode='a', header=not file_exists, index=False)
        local_success = True
    except Exception:
        local_success = False

    # 2. Sync to Google Sheets cloud ledger
    cloud_success = False
    cloud_msg = ""
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Load secret credentials
        if "gcp_json" in st.secrets:
            secret_dict = json.loads(st.secrets["gcp_json"])
        elif "gcp_service_account" in st.secrets:
            secret_dict = dict(st.secrets["gcp_service_account"])
        else:
            raise ValueError("No GCP credentials found in Streamlit Secrets.")

        # Unescape escaped newlines if present
        if "private_key" in secret_dict:
            secret_dict["private_key"] = secret_dict["private_key"].replace("\\n", "\n")

        creds = Credentials.from_service_account_info(secret_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open Google Sheet by Spreadsheet Key
        sheet_key = "1upEoaEmuhZeLseIXF-Ym7Ym5EAnvFqE69pE8nF29hI4"
        spreadsheet = client.open_by_key(sheet_key)
        worksheet = spreadsheet.sheet1
        worksheet.append_row(row_data)
        cloud_success = True
    except Exception as e:
        cloud_msg = str(e)

    return local_success, cloud_success, cloud_msg

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
tissue_val = st.sidebar.number_input("Tissue (0=BoneMarrow, 1=Adipose)", value=1.00, format="%.2f")

predict_button = st.sidebar.button("Predict Viability")

# ==========================================
# 5. DASHBOARD LAYOUT & EXECUTION FLOW
# ==========================================
if predict_button:
    if model is None:
        st.error("❌ Model failed to initialize. Please verify your repository configuration.")
    else:
        # Map raw operator inputs to possible feature keys
        raw_inputs = {
            "pH": ph_val,
            "Dissolved Oxygen (%)": do_val,
            "Glucose (mM)": glucose_val,
            "Lactate (mM)": lactate_val,
            "Temperature (oC)": temp_val,
            "Temperature (OC)": temp_val,
            "CO2 (%)": co2_val,
            "Agitation (rpm)": agitation_val,
            "Seeding Density (cells/mL)": seeding_val,
            "Cell Count": cell_count_val,
            "Population Doubling": pop_doubling_val,
            "Tissue (0=BoneMarrow, 1=Adipose)": tissue_val,
            "Donor": 0.0,
            "Study_Reference_x": 0.0,
            "Study_Reference_y": 0.0,
            "Day / Time": 1.0
        }

        # Extract exact feature list from booster to guarantee alignment
        booster_features = model.get_booster().feature_names

        if booster_features:
            row_data = {col: [float(raw_inputs.get(col, 0.0))] for col in booster_features}
            current_batch = pd.DataFrame(row_data, columns=booster_features)
        else:
            current_batch = pd.DataFrame([raw_inputs])

        current_batch = current_batch.astype(np.float64)

        # Robust inference using native DMatrix to bypass C-API Columnar issues
        try:
            dmatrix_input = xgb.DMatrix(current_batch)
            prediction = float(model.get_booster().predict(dmatrix_input)[0])
        except Exception:
            prediction = float(model.predict(current_batch)[0])

        # Process metrics
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

        # Prepare audit record entries
        ledger_headers = [
            "Timestamp", "Operator", "Predicted_Viability", "Risk_Evaluation",
            "Drift_Status", "App_Version", "Temperature", "Agitation",
            "pH", "Dissolved_Oxygen", "Seeding_Density", "Tissue_Type",
            "Glucose", "Lactate"
        ]
        
        audit_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "System_Operator",
            float(round(prediction, 4)), risk, drift_status, "v1.9.0-GMP",
            float(temp_val), float(agitation_val), float(ph_val), float(do_val),
            float(seeding_val), "Adipose" if tissue_val == 1.0 else "BoneMarrow",
            float(glucose_val), float(lactate_val)
        ]

        with st.spinner("Recording entry in append-only audit ledger..."):
            local_ok, cloud_ok, cloud_err = log_to_audit_ledger(audit_row, ledger_headers)

        if cloud_ok:
            st.sidebar.success("✅ Cloud & Local audit ledger updated.")
        elif local_ok:
            st.sidebar.warning(f"⚠️ Appended to local audit_ledger.csv (Cloud sync pending: {cloud_err})")
        else:
            st.sidebar.error("❌ Failed to update audit ledger.")

        # 🧠 EXPLAINABLE AI SECTION
        st.write("---")
        st.subheader("🧠 Explainable AI (SHAP Interpretation)")
        
        with st.spinner("Calculating local feature attributions..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(current_batch)

                num_features = len(current_batch.columns)
                
                # Dynamic height ensures all feature rows fit comfortably without crowding
                fig_height = max(6, int(num_features * 0.45))
                fig, ax = plt.subplots(figsize=(10, fig_height))

                # max_display ensures EVERY feature is listed individually
                shap.plots.waterfall(
                    shap_values[0], 
                    max_display=num_features, 
                    show=False
                )
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as shap_error:
                st.error(f"Visualizer Notice: Prediction succeeded, but SHAP generation skipped: {str(shap_error)}")

else:
    st.info("👉 Please enter the current bioreactor telemetry parameters in the sidebar and click 'Predict Viability' to view live predictions and parameter attributions.")
