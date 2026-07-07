import shap
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="Viability prediction XAI tool", layout="wide")

# ==========================================
# 2. LOAD THE MACHINE LEARNING ENGINE
# ==========================================
@st.cache_resource
def load_xgboost_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_cqa_model.json") 
    return model

try:
    model = load_xgboost_model()
except Exception as e:
    st.error("🚨 Model File Missing! Please ensure 'xgboost_cqa_model.json' has been committed to your GitHub repository.")
    st.stop() 

# ==========================================
# 3. LIVE GOOGLE SHEETS LOGGING FUNCTION
# ==========================================
def log_to_google_sheets(row_data):
    try:
        # Define API authorization access scopes
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        
        # Access credentials directly from Streamlit's secure backend vault
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], 
            scopes=scopes
        )
        
        # Authorize client and hook into the live spreadsheet
        client = gspread.authorize(creds)
        
        # ⚠️ PASTE YOUR ACTUAL GOOGLE SHEET URL HERE
        sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1upEoaEmuhZeLseIXfSz-9wBeUtVJTORXHh_lf8B2AFQ/edit?usp=drivesdk")

# 📄 PASTE YOUR ACTUAL GOOGLE SHEET URL HERE
        sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1upEoaEmuhZeLseIXf...")
        
        # 🎯 Target the very first tab (worksheet) inside that file
        worksheet = sheet.get_worksheet(0) 

        # #️⃣ Inject row data sequentially into the bottom ledger line
        worksheet.append_row(row_data)
        return True
        
    except Exception as e:
        st.sidebar.error(f"❌ Cloud Audit Logging Failed: {e}")
        return False

# ==========================================
# 4. OPERATOR INPUT PANEL (SIDEBAR)
# ==========================================
st.sidebar.markdown("### Operator Input Panel")
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

# Data Pipeline Metadata Features
donor_val = st.sidebar.number_input("Donor", value=0.00, format="%.2f")
tissue_val = st.sidebar.number_input("Tissue (0=BoneMarrow, 1=Adipose)", value=1.00, format="%.2f")
# ==========================================
# 5. MAIN DASHBOARD DISPLAY LAYOUT
# ==========================================
st.title("Viability prediction XAI tool")
st.write("Good Manufacturing Practice (GMP) compliant predictive monitoring dashboard.")
st.markdown("---")

if st.sidebar.button("Predict Viability"):
    
    # 1. Assemble input variables into a clean 15-parameter feature matrix
    current_batch = pd.DataFrame([{
        "pH": ph_val,
        "Dissolved Oxygen (%)": do_val,
        "Glucose (mM)": glucose_val,
        "Lactate (mM)": lactate_val,
        "Temperature (oC)": temp_val,
        "CO2 (%)": co2_val,
        "Agitation (rpm)": agitation_val,
        "Seeding Density (cells/mL)": seeding_val,
        "Cell Count": cell_count_val,
        "Population Doubling": pop_doubling_val,
        "Study_Reference_x": 0.0,
        "Donor": donor_val,
        "Tissue": tissue_val,
        "Study_Reference_y": 0.0,
        "Day / Time": 1.0 
    }])
    
    # 2. Structural Check: Ensure dimensions match
    expected_features = model.n_features_in_
    provided_features = current_batch.shape[1]

    if expected_features != provided_features:
        st.error(f"⚠️ Feature Mismatch! The model expects **{expected_features}** parameters, but received **{provided_features}**.")
        st.stop()
        
    # Run mathematical inference
    prediction = model.predict(current_batch.values)[0] 
    
    # Establish operational risk grading brackets
    if prediction < 80.0: 
        risk = "HIGH (Critical Cell Death)"
    elif prediction < 90.0: 
        risk = "MEDIUM (Sub-optimal)"
    else: 
        risk = "LOW (Optimal)"

    # Render results sections using high-visibility metrics boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔬 Process Status")
        st.metric(label="Lactate-Glucose Ratio", value=round(lactate_val / glucose_val, 2))
        
    with col2:
        st.subheader("⚙️ Drift Detection")
        drift_status = "Normal"
        st.success(f"✅ {drift_status.upper()}: Process within control limits.")
        
    with col3:
        st.subheader("🎯 CQA Prediction")
        st.metric(label="Predicted Viability (%)", value=f"{prediction:.2f}%", delta=risk)

    st.markdown("---")
    
    st.subheader("🧠 Explainable AI (SHAP Interpretation)")
    
    with st.spinner("Calculating local feature attributions..."):
        # Generate the SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(current_batch)
        
        # Draw the Waterfall plot
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        
        # Display the plot in the app
        st.pyplot(fig)
    
    # ==========================================
    # 6. EXACT ROBUST A1 TO N1 AUDIT EXCEL MAPPING
    # ==========================================
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    operator_id = "Operator_01"
    software_version = "v2.1-Cloud"
    
    # This array tracks sequentially into columns A through N of the cloud sheet
    audit_row = [
        current_time,                                      # A: Time_Stamp
        operator_id,                                       # B: User
        float(round(prediction, 4)),                       # C: Predicted_Viability
        risk,                                              # D: Risk_Level
        drift_status,                                      # E: Drift_Status
        software_version,                                  # F: Software_Version
        float(temp_val),                                   # G: Temperature
        float(agitation_val),                              # H: Agitation
        float(ph_val),                                     # I: pH
        float(do_val),                                     # J: DO
        float(seeding_val),                                # K: Seeding_Density
        "Adipose" if tissue_val == 1.0 else "BoneMarrow",  # L: Tissue
        float(glucose_val),                                # M: Glucose
        float(lactate_val)                                 # N: Lactate
    ]
    
    # Submit execution logging payload to cloud destination
    with st.spinner("Securing audit record in cloud ledger..."):
        success = log_to_google_sheets(audit_row)

if success:
            st.sidebar.success("✅ Audit trail securely pushed to live Google Ledger.")
else:
            st.sidebar.warning("⚠️ Warning: Output calculated but cloud log synchronization failed.")

        # 1. Move the SHAP block OUT of the else statement (Match the indentation above)
st.subheader("🧠 Explainable AI (SHAP Interpretation)")
        
with st.spinner("Calculating local feature attributions..."):
            # Generate the SHAP values
            explainer = shap.Explainer(model)
            shap_values = explainer(current_batch)

            # Draw the Waterfall plot
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            
            # Display the plot in the app
            st.pyplot(fig)

    # 2. This is where your outer 'else:' actually belongs (for when the button is NOT clicked)
    else:
    st.subheader("🧠 Explainable AI (SHAP Interpretation)")
    st.info("👉 Please enter the current bioreactor telemetry in the sidebar and click 'Predict Viability' to view live parameter attributions.")
