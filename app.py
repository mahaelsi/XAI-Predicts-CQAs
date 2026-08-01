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

        # Positional feature array matching model's expected 18 inputs
        ordered_input_values = [
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
            stress_index,               # 14: Culture_Stress_Index
            1.0,                        # 15: Day / Time
            0.0,                        # 16: Study_Reference_x
            0.0                         # 17: Study_Reference_y
        ]

        standard_feature_names = [
            "Donor", "Tissue", "pH", "CO2 (%)", "DO", "Glucose", "Lactate",
            "Temperature (oC)", "Agitation (rpm)", "Seeding Density (cells/mL)",
            "Cell Count", "Population Doubling", "Lactate_Glucose_Ratio",
            "Metabolic_Load", "Culture_Stress_Index", "Day / Time",
            "Study_Reference_x", "Study_Reference_y"
        ]

        # 1. Convert to 2D numpy array for position-based inference (bypasses feature_names validation)
        X_input = np.array([ordered_input_values], dtype=np.float64)

        # 2. Predict using model native predict call on raw numeric array
        try:
            raw_pred_arr = model.predict(X_input)
            raw_prediction = float(raw_pred_arr[0])
        except Exception:
            # Fallback to booster DMatrix without string feature names
            dmat = xgb.DMatrix(X_input)
            raw_prediction = float(model.get_booster().predict(dmat)[0])

        # Scale prediction between 0% and 100%
        predicted_viability_pct = raw_prediction * 100.0 if raw_prediction <= 1.0 else raw_prediction
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

        # Audit ledger logging (Preserves Google Sheet & Local CSV sync)
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
                # Create DataFrame for SHAP so feature names display nicely on plot
                df_shap = pd.DataFrame(X_input, columns=standard_feature_names)
                
                # Initialize TreeExplainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(df_shap)

                num_features = len(standard_feature_names)
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
