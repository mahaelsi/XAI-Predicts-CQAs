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
import base64
import re

# ==========================================
# 1. PAGE INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Viability Prediction XAI Tool", layout="wide")

st.title("🧪 Viability Prediction XAI Tool")
st.markdown("##### Good Manufacturing Practice (GMP) Compliant Predictive Monitoring Dashboard")
st.write("---")

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1upEoaEmuhZeLseIXfSz-9wBeUtVJTORXHh_lf8B2AFQ/edit"

def extract_spreadsheet_id(url_or_id: str) -> str:
    """Safely extracts the 44-character Google Sheet ID from any URL or raw ID string."""
    url_or_id = str(url_or_id).strip()
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id

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
# 3. CRASH-PROOF AUDIT LEDGER FUNCTION
# ==========================================
def log_to_audit_ledger(row_data, header_names):
    """
    Writes predictions to local CSV and syncs to Google Sheets.
    Includes defensive JSON slicing to prevent JSONDecodeError crashes.
    """
    # 1. Append to local CSV ledger
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
    sheet_id = "UNKNOWN_ID"
    sa_email = "UNKNOWN_EMAIL"

    try:
        target_input = st.secrets.get("sheet_url", DEFAULT_SHEET_URL)
        sheet_id = extract_spreadsheet_id(target_input)

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        secret_dict = None

        if "gcp_service_account" in st.secrets:
            secret_dict = dict(st.secrets["gcp_service_account"])
            if "private_key" in secret_dict:
                secret_dict["private_key"] = str(secret_dict["private_key"]).replace("\\n", "\n")
        elif "gcp_service_account_b64" in st.secrets:
            raw_b64 = str(st.secrets["gcp_service_account_b64"]).strip().strip('"').strip("'")
            json_bytes = base64.b64decode(raw_b64)
            json_str = json_bytes.decode("utf-8", errors="ignore").strip()
            # Safely truncate extra characters past the closing brace
            if "}" in json_str:
                json_str = json_str[:json_str.rfind("}") + 1]
            secret_dict = json.loads(json_str)
        else:
            raise ValueError("No GCP credentials found in Streamlit Secrets.")

        sa_email = secret_dict.get("client_email", "Unknown Email")
        creds = Credentials.from_service_account_info(secret_dict, scopes=scopes)
        client = gspread.authorize(creds)

        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.sheet1
        worksheet.append_row(row_data)
        cloud_success = True

    except Exception as e:
        cloud_msg = f"{type(e).__name__}: {str(e)} | Target Sheet ID: [{sheet_id}] | Shared with: [{sa_email}]"

    return local_success, cloud_success, cloud_msg

# ==========================================
# 4. OPERATOR INPUT PANEL (SIDEBAR)
# ==========================================
st.sidebar.markdown("### 🎛️ Operator Input Panel")
st.sidebar.info("Enter precise bioreactor parameters below to simulate a real-time batch prediction.")

ph_val = st.sidebar.number_input("pH", value=7.20, format="%.2f")
do_val = st.sidebar.number_input("Dissolved Oxygen (%)", value=50.00, format="%.2f")
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
        # Calculate derived metabolic metrics
        lac_glu_ratio = float(round(lactate_val / glucose_val, 4)) if glucose_val > 0 else 0.0
        metabolic_load = float(round((lactate_val * cell_count_val) / 1e6, 4))
        stress_index = float(round(abs(ph_val - 7.2) + abs(temp_val - 37.0) + abs(co2_val - 5.0), 4))

        # Comprehensive lookup map for potential feature name variants
        feature_val_map = {
            "Donor": 0.0,
            "Tissue": float(tissue_val),
            "pH": float(ph_val),
            "CO2 (%)": float(co2_val),
            "DO": float(do_val),
            "Glucose": float(glucose_val),
            "Lactate": float(lactate_val),
            "Temperature (oC)": float(temp_val),
            "Agitation (rpm)": float(agitation_val),
            "Seeding Density ( cells/mL)": float(seeding_val),  # Matches exact trained model key
            "Seeding Density (cells/mL)": float(seeding_val),
            "Cell Count": float(cell_count_val),
            "Population Doubling": float(pop_doubling_val),
            "Lactate_Glucose_Ratio": lac_glu_ratio,
            "Metabolic_Load": metabolic_load,
            "Culture_Stress_Index": stress_index,
            "Day / Time": 1.0,
            "Study_Reference_x": 0.0,
            "Study_Reference_y": 0.0,
        }

        # Retrieve exact feature names required by trained XGBoost model
        booster = model.get_booster()
        expected_features = booster.feature_names

        if expected_features:
            batch_dict = {f_name: [feature_val_map.get(f_name, 0.0)] for f_name in expected_features}
            current_batch = pd.DataFrame(batch_dict)
        else:
            fallback_cols = [
                "Donor", "Tissue", "pH", "CO2 (%)", "DO", "Glucose", "Lactate",
                "Temperature (oC)", "Agitation (rpm)", "Seeding Density ( cells/mL)",
                "Cell Count", "Population Doubling", "Lactate_Glucose_Ratio",
                "Metabolic_Load", "Culture_Stress_Index"
            ]
            batch_dict = {f_name: [feature_val_map.get(f_name, 0.0)] for f_name in fallback_cols}
            current_batch = pd.DataFrame(batch_dict)

        current_batch = current_batch.astype(np.float64)

        # Execute prediction
        raw_prediction = float(model.predict(current_batch)[0])

        # Model outputs percentage directly (e.g. 95.5 = 95.5%)
        predicted_viability_pct = max(0.0, min(100.0, raw_prediction))

        # Process Drift Detection
        drift_reasons = []
        if not (7.00 <= ph_val <= 7.40):
            drift_reasons.append(f"pH ({ph_val})")
        if not (40.0 <= do_val <= 80.0):
            drift_reasons.append(f"DO ({do_val}%)")
        if not (36.5 <= temp_val <= 37.5):
            drift_reasons.append(f"Temp ({temp_val}°C)")
        if not (4.5 <= co2_val <= 5.5):
            drift_reasons.append(f"CO2 ({co2_val}%)")
        if not (80.0 <= agitation_val <= 120.0):
            drift_reasons.append(f"Agitation ({agitation_val} rpm)")
        if glucose_val < 3.0:
            drift_reasons.append(f"Glucose Low ({glucose_val} mM)")
        if lactate_val > 25.0:
            drift_reasons.append(f"Lactate High ({lactate_val} mM)")

        if drift_reasons:
            drift_status = "DRIFT DETECTED"
            drift_display_text = f"⚠️ DRIFT: {', '.join(drift_reasons)}"
        else:
            drift_status = "NORMAL"
            drift_display_text = "✅ NORMAL: Within Limits"

        risk = "HIGH (Critical)" if predicted_viability_pct < 80.0 else "LOW (Stable)"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔬 Process Status")
            st.metric(label="Lactate-Glucose Ratio", value=f"{lac_glu_ratio:.2f}")
        with col2:
            st.markdown("### ⚙️ Drift Detection")
            if drift_status == "NORMAL":
                st.success(drift_display_text)
            else:
                st.warning(drift_display_text)
        with col3:
            st.markdown("### 🎯 CQA Prediction")
            st.metric(label="Predicted Viability", value=f"{predicted_viability_pct:.2f}%")
            st.caption(f"Risk Evaluation: **{risk}**")

        # Audit ledger logging (Google Sheets & Local CSV)
        ledger_headers = [
            "Timestamp", "Operator", "Predicted_Viability", "Risk_Evaluation",
            "Drift_Status", "App_Version", "Temperature", "Agitation",
            "pH", "Dissolved_Oxygen", "Seeding_Density", "Tissue_Type",
            "Glucose", "Lactate"
        ]
        
        audit_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "System_Operator",
            float(round(predicted_viability_pct, 2)), risk, drift_status, "v2.4.0-GMP",
            float(temp_val), float(agitation_val), float(ph_val), float(do_val),
            float(seeding_val), "Adipose" if tissue_val == 1.0 else "BoneMarrow",
            float(glucose_val), float(lactate_val)
        ]

        with st.spinner("Recording entry in append-only audit ledger..."):
            local_ok, cloud_ok, cloud_err = log_to_audit_ledger(audit_row, ledger_headers)

        if cloud_ok:
            st.sidebar.success("✅ Cloud & Local audit ledger updated.")
        elif local_ok:
            st.sidebar.warning(f"Appended to local audit_ledger.csv\n\n(Cloud sync status: {cloud_err})")
        else:
            st.sidebar.error("❌ Failed to update audit ledger.")

        # 🧠 EXPLAINABLE AI SECTION (DYNAMIC SHAP WATERFALL GENERATOR)
        st.write("---")
        st.subheader("🧠 Explainable AI (SHAP Interpretation)")
        
        with st.spinner("Calculating local feature attributions..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(current_batch)

                # Format clean feature names for waterfall visualization
                clean_display_names = [f.replace("( cells/mL)", "(cells/mL)") for f in current_batch.columns]
                shap_values.feature_names = clean_display_names

                num_features = len(clean_display_names)
                fig_height = max(6, int(num_features * 0.45))
                fig, ax = plt.subplots(figsize=(10, fig_height))

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
    st.info("👉 Please enter current bioreactor telemetry parameters in sidebar and click 'Predict Viability'.")
