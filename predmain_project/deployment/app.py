import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import warnings
import os
warnings.filterwarnings("ignore")

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="navzen2000/predmain-model", filename="best_predmain_model_prod.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for predicting Engine Maintenance
st.title("Predictive Maintenance App for Vehicle Breakdowns")
st.write("The Predictive Maintenance App can analyze historical and real-time engine sensor data to identify whether an engine requires maintenance or is operating normally")
st.write("Kindly enter the engine sensor data to predict if mainteance is required")

# Collect sensor input
Engine_RPM = st.number_input("The number of revolutions per minute (RPM) of the engine, indicating engine speed", min_value=0, max_value=6000, value=750)
Lub_Oil_Pressure = st.number_input("The pressure of the lubricating oil in the engine, essential for reducing friction and wear in bar or kilopascals (kPa)", min_value=0, max_value=30, value=5)
Fuel_Pressure = st.number_input("The pressure at which fuel is supplied to the engine, critical for proper combustion in kilopascals (kPa)", min_value=0, max_value=30, value=7)
Coolant_Pressure = st.number_input("The pressure of the engine coolant, affecting engine temperature regulation in kilopascals (kPa)", min_value=0, max_value=30, value=2)
Lub_Oil_Temperature = st.number_input("The temperature of the lubricating oil, which impacts viscosity and engine performance in degrees Celsius (°C)", min_value=25, max_value=200, value=78)
Coolant_Temperature = st.number_input("The temperature of the engine coolant, crucial for preventing overheating in degrees Celsius (°C)", min_value=25, max_value=200, value=78)

# Store in Data Frame
input_data = pd.DataFrame([{
    'Engine rpm': Engine_RPM,
    'Lub oil pressure': Lub_Oil_Pressure,
    'Fuel pressure': Fuel_Pressure,
    'Coolant pressure': Coolant_Pressure,
    'lub oil temp': Lub_Oil_Temperature,
    'Coolant temp': Coolant_Temperature
}])


# Set the classification threshold
workspace = os.environ.get('GITHUB_WORKSPACE', os.getcwd())
abs_path = os.path.join(workspace, 'threshold.txt')

# Open in "r" (read) mode
with open(abs_path, "r") as f: 
    # Read the content and convert it to a float
    classification_threshold = float(f.read().strip())

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "requires maintenance" if prediction == 1 else "does not require maintenance"
    st.write(f"Based on the information provided engine sensor data, the engine {result}.")
