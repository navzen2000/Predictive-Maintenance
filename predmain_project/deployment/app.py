import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import warnings
warnings.filterwarnings("ignore")

# Define the single source of truth for the model hub repository
REPO_ID = "navzen2000/predmain-model"

# Streamlit UI for predicting Engine Maintenance
st.title("Predictive Maintenance App for Vehicle Breakdowns")
st.write("The Predictive Maintenance App can analyze historical and real-time engine sensor data to identify whether an engine requires maintenance or is operating normally")
st.write("Kindly enter the engine sensor data to predict if mainteance is required")

@st.cache_resource(show_spinner="Initializing production model artifacts...")
def load_production_assets():
    """
    Downloads and caches the model pipeline and classification threshold.
    Executes once on startup, preventing repeated network/disk IO calls.
    """
    m_path = hf_hub_download(repo_id=REPO_ID, filename="best_predmain_model_prod.joblib")
    t_path = hf_hub_download(repo_id=REPO_ID, filename="threshold.txt")
    
    fitted_model = joblib.load(m_path)
    with open(t_path, "r") as f: 
        optimal_threshold = float(f.read().strip())
        
    return fitted_model, optimal_threshold

# Protected asset initialization block
try:
    model, classification_threshold = load_production_assets()
except Exception as e:
    st.error(
        "⚠️ **System Initialization Error:** The application was unable to retrieve the latest "
        "model production configuration or decision threshold metrics from the registry. "
        "Please verify your pipeline deployment statuses."
    )
    with st.expander("View technical exception details"):
        st.code(str(e))
    st.stop()

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

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)
    result = "requires maintenance" if prediction == 1 else "does not require maintenance"
    st.write(f"Based on the information provided engine sensor data, the engine {result}.")
