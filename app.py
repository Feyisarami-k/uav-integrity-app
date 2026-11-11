import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page title and layout
st.set_page_config(page_title="UAV GNSS Integrity Predictor", layout="centered")

# App title and description
st.title("🛸 UAV GNSS Integrity Alarm Predictor")
st.write("""
This interactive web app predicts the probability of a **GNSS Integrity Alarm (rx_integrity_alarm)** 
using an optimized Random Forest Classifier trained on flight sensor and receiver features.
""")

# Load the trained model
@st.cache_resource
def load_model():
    with open('uav_randomforest.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading the model: {e}. Please ensure 'uav_randomforest.pkl' is in the same directory.")

st.subheader("🔧 Input Flight & Signal Parameters")

# Layout inputs in columns
col1, col2 = st.columns(2)

with col1:
    altitude = st.number_input("Altitude (meters)", value=1000.0, step=10.0)
    cn0_mean = st.number_input("C/N0 mean (dB-Hz)", value=42.0, step=0.1)
    cn0_std = st.number_input("C/N0 Standard Deviation (dB)", value=1.5, step=0.1)
    sats = st.slider("Number of Satellites (Sats)", min_value=0, max_value=30, value=10)
    hdop = st.number_input("HDOP", value=1.0, step=0.05)

with col2:
    velocity_inconsistency = st.number_input("Velocity Inconsistency (ms)", value=0.5, step=0.01)
    clock_drift = st.number_input("Clock Drift (ppb)", value=-10.0, step=0.1)
    power = st.number_input("Power (dBm)", value=-130.0, step=0.5)
    imu_gnss_div = st.number_input("IMU-GNSS Divergence", value=2.0, step=0.1)
    
    # Categorical drop-down for UAV Platform
    uav_platform = st.selectbox(
        "UAV Platform",
        options=["Tactical", "Quad", "VTOL", "Fixed-Wing", "Unknown"]
    )

# One-Hot Encoding preparation (matching training features)
# Features expected by model: 
# ['altitude_m', 'C/N0 mean', 'cn0_std_db', 'Sats', 'hdop', 'velocity_inconsistency_ms', 
#  'clock_drift_ppb', 'Power', 'IMU-GNSS Div', 'uav_platform_fixed-wing', 
#  'uav_platform_quad', 'uav_platform_tactical', 'uav_platform_vtol']

# Process platform encoding
platform_fixed_wing = 1 if uav_platform == "Fixed-Wing" else 0
platform_quad = 1 if uav_platform == "Quad" else 0
platform_tactical = 1 if uav_platform == "Tactical" else 0
platform_vtol = 1 if uav_platform == "VTOL" else 0

# Prepare final data payload matching model format
input_data = pd.DataFrame({
    'altitude_m': [altitude],
    'C/N0 mean': [cn0_mean],
    'cn0_std_db': [cn0_std],
    'Sats': [sats],
    'hdop': [hdop],
    'velocity_inconsistency_ms': [velocity_inconsistency],
    'clock_drift_ppb': [clock_drift],
    'Power': [power],
    'IMU-GNSS Div': [imu_gnss_div],
    'uav_platform_fixed-wing': [platform_fixed_wing],
    'uav_platform_quad': [platform_quad],
    'uav_platform_tactical': [platform_tactical],
    'uav_platform_vtol': [platform_vtol]
})

st.markdown("---")

# Predict button
if st.button("🔍 Run GNSS Integrity Diagnostic", type="primary"):
    try:
        # Make predictions
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        st.subheader("📊 Diagnostic Results")
        
        if prediction == 1:
            st.error(f"🚨 **ALERT: Receiver Integrity Alarm Triggered!** (Probability: {prediction_proba:.2%})")
            st.warning("Recommendation: Check GNSS environment for spoofing, jamming, or heavy multipath signals.")
        else:
            st.success(f"✅ **GNSS Status: NORMAL** (Alarm Probability: {prediction_proba:.2%})")
            st.info("System integrity is behaving within nominal parameters.")
            
    except Exception as e:
        st.error(f"Error performing prediction: {e}")
