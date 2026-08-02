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
# 3. SANITIZED GOOGLE CREDENTIALS LOADING
# ==========================================
try:
    target_input = st.secrets.get("sheet_url", DEFAULT_SHEET_URL)
    sheet_id = extract_spreadsheet_id(target_input)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # 1. Check if secrets exist
    if "gcp_service_account" not in st.secrets:
        raise ValueError("Missing 'gcp_service_account' block in Streamlit Secrets.")

    # 2. Make a mutable dictionary copy (Streamlit secrets are read-only)
    secret_dict = dict(st.secrets["gcp_service_account"])

    # 3. Clean and sanitize the private key string
    if "private_key" in secret_dict:
        pk = str(secret_dict["private_key"]).strip()
        
        # Replace escaped literal backslashes (\n) with real system newlines
        pk = pk.replace("\\n", "\n")
        
        # Remove any surrounding wrapping quotes that may have been saved
        pk = pk.strip('"').strip("'")
        
        secret_dict["private_key"] = pk

    # 4. Authenticate service account
    sa_email = secret_dict.get("client_email", "Unknown Email")
    creds = Credentials.from_service_account_info(secret_dict, scopes=scopes)
    client = gspread.authorize(creds)

    # 5. Write to target Google Sheet
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.sheet1
    worksheet.append_row(row_data)
    cloud_success = True

except Exception as e:
    cloud_msg = f"{type(e).__name__}: {str(e)} | Target Sheet ID: [{sheet_id}] | Shared with: [{sa_email}]"

# ==========================================
# 4. OPERATOR INPUT PANEL (SIDEBAR)
# ==========================================
st.sidebar.markdown("### 🎛️ Operator Input Panel")
st.sidebar.info("Enter precise bioreactor parameters below to simulate a real-time batch prediction.")

ph_val = st.sidebar.number_input("pH", value=7.20, format="%.2f")
do_val = st.sidebar.number_input("Dissolved Oxygen (%)", value=50.00, format="%.2f")
glucose_val = st.sidebar.number_input("Glucose (mM)", value=10.00, format="%.2f")
lactate_val = st.sidebar.number_input("Lactate (mM)", value=15.00, format="%.2f")
temp_val = st.sidebar.number_input("Temperature (°C)", value=37.00, format="%.2f")
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

        clean_feature_names = [
            "Donor", "Tissue", "pH", "CO2 (%)", "DO", "Glucose", "Lactate",
            "Temperature (°C)", "Agitation (rpm)", "Seeding Density (cells/mL)",
            "Cell Count", "Population Doubling", "Lactate/Glucose Ratio",
            "Metabolic Load", "Culture Stress Index"
        ]

        # Construct exact 15-element numeric input vector
        X_np = np.array([[
            0.0,                        # 0: Donor
            float(tissue_val),          # 1: Tissue
            float(ph_val),              # 2: pH
            float(co2_val),             # 3: CO2 (%)
            float(do_val),              # 4: DO
            float(glucose_val),         # 5: Glucose
            float(lactate_val),         # 6: Lactate
            float(temp_val),            # 7: Temperature
            float(agitation_val),       # 8: Agitation
            float(seeding_val),         # 9: Seeding Density
            float(cell_count_val),      # 10: Cell Count
            float(pop_doubling_val),    # 11: Population Doubling
            lac_glu_ratio,              # 12: Lactate_Glucose_Ratio
            metabolic_load,             # 13: Metabolic_Load
            stress_index                # 14: Culture_Stress_Index
        ]], dtype=np.float32)

        # Predict viability using Booster to bypass feature-name mismatches
        try:
            dmat = xgb.DMatrix(X_np)
            raw_prediction = float(model.get_booster().predict(dmat)[0])
        except Exception:
            raw_prediction = float(model.predict(X_np)[0])

        # Convert decimal predictions (0.0 to 1.0) to percentage scale (0% to 100%)
        is_decimal_scale = (0.0 <= raw_prediction <= 1.0)
        predicted_viability_pct = raw_prediction * 100.0 if is_decimal_scale else raw_prediction
        predicted_viability_pct = max(0.0, min(100.0, predicted_viability_pct))

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
                plt.close('all')

                booster = model.get_booster()
                explainer = shap.TreeExplainer(booster)
                
                exp = explainer(X_np)[0]
                exp.feature_names = clean_feature_names

                if is_decimal_scale:
                    exp.values = exp.values * 100.0
                    if hasattr(exp, "base_values"):
                        exp.base_values = exp.base_values * 100.0

                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Standard compatible call across all SHAP versions
                shap.plots.waterfall(
                    exp, 
                    max_display=len(clean_feature_names), 
                    show=False
                )
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as shap_error:
                st.error(f"Visualizer Notice: Prediction succeeded, but SHAP generation skipped: {str(shap_error)}")

else:
    st.info("👉 Please enter current bioreactor telemetry parameters in sidebar and click 'Predict Viability'.")
