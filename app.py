import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Label mapping based on training encoder
risk_levels = {0: 'High', 1: 'Low', 2: 'Moderate', 3: 'Very High'}
risk_emojis = {
    "Low": "🟢",
    "Moderate": "🟡",
    "High": "🟠",
    "Very High": "🔴"
}



# Streamlit page config
st.set_page_config(page_title="Landslide Risk Predictor", page_icon="🏔️", layout="centered")

# --- Title and Description ---
st.title("🏔️ Landslide Risk Prediction")
st.markdown("Get a quick assessment of landslide risk based on environmental conditions.")

st.info(" Enter environmental measurements below and click **Predict** to see the landslide risk level.")

# --- Input Form ---
with st.form("input_form"):
    st.subheader("📝 Input Environmental Data")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=-50.0, max_value=60.0, value=0.0, step=0.1, help="Enter temperature")
        precipitation = st.number_input("🌧️ Precipitation (mm)", min_value=0.0, max_value=500.0, value=0.0, step=1.0, help="Enter precipitation")
        elevation = st.number_input("⛰️ Elevation (m)", min_value=0.0, max_value=9000.0, value=0.0, step=1.0, help="Enter elevation")

    with col2:
        humidity = st.slider("💧 Humidity (%)", min_value=0, max_value=100, value=0, help="Select humidity")
        soil_moisture = st.slider("🌱 Soil Moisture (%)", min_value=0, max_value=100, value=0, help="Select soil moisture")


    
    submit_btn = st.form_submit_button("🚨 Predict Landslide Risk")

# --- Prediction Result ---
if submit_btn:
    if any(v == 0 for v in [humidity, soil_moisture]):
        st.warning("⚠️ Please provide valid values for both **Humidity** and **Soil Moisture** — they cannot be zero.")
    else:
        input_data = np.array([[temperature, humidity, precipitation, soil_moisture, elevation]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        risk_label = risk_levels.get(prediction, "Unknown")
        emoji = risk_emojis.get(risk_label, "❓")

        # Custom color mapping for background
        bg_colors = {
            "Low": "#d4edda",
            "Moderate": "#fff3cd",
            "High": "#ffe5b4",
            "Very High": "#f8d7da"
        }

        text_colors = {
            "Low": "#155724",
            "Moderate": "#856404",
            "High": "#7f3d00",
            "Very High": "#721c24"
        }

        st.markdown("### 🔍 Prediction Result")

        
        st.markdown(f"""
        <div style='
            background-color: {bg_colors.get(risk_label, "#f0f0f0")};
            color: {text_colors.get(risk_label, "#000")};
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
        '>
            {emoji} Landslide Risk Level: {risk_label}
        </div>
        """, unsafe_allow_html=True)

        #  Probabilities
        probs = model.predict_proba(input_scaled)[0]
        st.markdown("### 📊 Class Probabilities")
        for idx, prob in enumerate(probs):
            label = risk_levels.get(idx, f"Class {idx}")
            st.write(f"{risk_emojis.get(label, '')} **{label}**: {prob:.2%}")


# --- Sidebar ---
st.sidebar.markdown("💻 **About the Model**")
st.sidebar.info("""
This landslide risk prediction model is trained using:
- XGBoost Classifier
- Features: Temperature, Humidity, Precipitation, Soil Moisture, Elevation
- Labels: Low, Moderate, High, Very High
""")

