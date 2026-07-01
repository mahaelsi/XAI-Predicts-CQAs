import datetime
import os
import pandas as pd
import streamlit as st

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



# 1. CREATE THE BUTTON IN THE SIDEBAR OR MAIN PAGE
if st.sidebar.button("Execute Batch Prediction"):
    
    # 2. RUN PREDICTION (Using the variables from your sliders)
    # Ensure current_batch is correctly formatted as we did in Colab
    prediction = model.predict(current_batch.values)[0] 
    
    # Calculate Risk Level based on Viability
    if prediction < 80.0: 
        risk = "HIGH (Critical Cell Death)"
    elif prediction < 90.0: 
        risk = "MEDIUM (Sub-optimal)"
    else: 
        risk = "LOW (Optimal)"

    # 3. DISPLAY THE VIABILITY RESULT
    st.subheader("🎯 CQA Prediction")
    st.metric(label="Predicted Viability", value=f"{prediction:.2f}%", delta=risk)
    
    # 4. LOG TO AUDIT TRAIL
    # Create the log entry
    audit_record = pd.DataFrame([{
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": "Operator_01", # In a real app, this would be the logged-in user
        "Predicted_Viability": round(prediction, 4),
        "Risk_Level": risk,
        "Software_Version": "v1.4"
    }])
    
    # Define path for audit trail
    AUDIT_TRAIL_PATH = "system_audit_trail.csv"
    
    # Append to CSV (Streamlit will save this in its local container)
    if not os.path.exists(AUDIT_TRAIL_PATH):
        audit_record.to_csv(AUDIT_TRAIL_PATH, index=False)
    else:
        audit_record.to_csv(AUDIT_TRAIL_PATH, mode='a', header=False, index=False)
        
    st.success("✅ Prediction successful and saved to secure audit trail.")
    
    # ... [Your existing SHAP explanation code can go here, inside the button block] ...

else:
    # This shows when the app first loads, before the button is clicked
    st.info("👈 Enter parameters in the sidebar and click 'Execute Batch Prediction' to begin.")
