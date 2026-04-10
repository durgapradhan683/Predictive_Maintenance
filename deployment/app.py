import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download, HfApi

# Set Hugging Face token from environment variables or Streamlit secrets
hf_token = os.getenv("HF_TOKEN")

# Initialize HfApi
api = HfApi(token=hf_token)

# Repository details
model_repo_id = "dmpradhan/PredictiveMaintenance"
model_filename = "best_PredictiveMaintenance_model_v1.joblib"

# Download and load the model
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=model_repo_id,
            filename=model_filename,
            repo_type="model",
            token=hf_token
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.set_page_config(page_title="Predictive Maintenance App", layout="wide")

st.title("Engine Predictive Maintenance Dashboard")
st.markdown("This application predicts whether an engine requires maintenance based on its sensor readings.")

if model is not None:
    st.header("Enter Engine Sensor Readings:")

    # Input fields for sensor data
    col1, col2 = st.columns(2)
    with col1:
        engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=750)
        lub_oil_pressure = st.number_input("Lub Oil Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=3.0, format="%.2f")
        fuel_pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.0, max_value=25.0, value=6.5, format="%.2f")
    with col2:
        coolant_pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=2.2, format="%.2f")
        lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=0.0, max_value=100.0, value=77.0, format="%.2f")
        coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=78.0, format="%.2f")

    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'Engine rpm': [engine_rpm],
        'Lub oil pressure': [lub_oil_pressure],
        'Fuel pressure': [fuel_pressure],
        'Coolant pressure': [coolant_pressure],
        'lub oil temp': [lub_oil_temp],
        'Coolant temp': [coolant_temp]
    })

    if st.button("Predict Engine Condition"):
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[:, 1]

            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error(f"The engine is likely **Faulty** (Probability: {prediction_proba[0]:.2f})")
                st.markdown("---\n"+
                            "**Recommendations:**\n"+
                            "- Check for low lubricant oil pressure.\n"+
                            "- Investigate potential issues with fuel pressure.\n"+
                            "- Examine the cooling system for high coolant pressure and temperature.")
            else:
                st.success(f"The engine is likely **Normal** (Probability: {prediction_proba[0]:.2f})")
                st.markdown("---\n"+
                            "**Status:** Engine operating within normal parameters.")

            st.subheader("Input Data:")
            st.write(input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Model could not be loaded. Please check the Hugging Face token and repository access.")
