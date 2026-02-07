
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="⚡ Smart Grid Energy Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .metric-card {
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .highlight {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Path Setup - Use script's directory
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# MODEL LOADER
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "XGBoost (Champion 🏆)": "xgboost_energy_model.pkl",
        "Random Forest": "rf_model.pkl",
        "Support Vector Regressor": "svr_model.pkl",
        "Linear Regression": "linear_model.pkl"
    }

    st.sidebar.write(f"📁 Models dir: {MODELS_DIR}")
    
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        exists = os.path.exists(path)
        st.sidebar.write(f"{'✅' if exists else '❌'} {filename}")
        
        if exists:
            try:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                st.sidebar.error(f"Error loading {filename}: {e}")
                models[name] = None
        else:
            models[name] = None
    return models

models_data = load_models()

# ==========================================
# SIDEBAR - INPUTS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3109/3109837.png", width=100)
st.sidebar.title("Configuration")
st.sidebar.markdown("Define conditions for forecasting:")

# 1. Date & Time
st.sidebar.subheader("📅 Date & Time")
selected_date = st.sidebar.date_input("Date", datetime.today())
selected_hour = st.sidebar.slider("Hour (0-23)", 0, 23, 18)

# 2. Weather Conditions
st.sidebar.subheader("🌤️ Weather")
temp_k = st.sidebar.number_input("Temperature (Kelvin)", 250.0, 320.0, 298.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 65)
pressure = st.sidebar.number_input("Pressure (hPa)", 950, 1050, 1012)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 3.5)
clouds = st.sidebar.slider("Cloud Cover (%)", 0, 100, 20)
rain_1h = st.sidebar.number_input("Rain (1h mm)", 0.0, 50.0, 0.0)

# 3. Advanced Features (Lags) - Required for XGBoost
st.sidebar.subheader("⚙️ Historical Lags (Simulated)")
st.sidebar.caption("Since previous load is needed for lags, estimate past values:")
lag_1h = st.sidebar.number_input("Load 1 Hour Ago (MW)", 10000, 40000, 28000)
lag_24h = st.sidebar.number_input("Load 24 Hours Ago (MW)", 10000, 40000, 27500)
rolling_mean = st.sidebar.number_input("Avg Load Last 24h", 10000, 40000, 28000)

weather_deg = 180 # Default
temp_min = temp_k - 2
temp_max = temp_k + 2

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("⚡ Energy Consumption Forecaster")
st.markdown("Predicting Power Grid Demand using Multi-Model Ensemble Analysis")

# Create Input DataFrame (Raw)
input_dict = {
    'temp': temp_k,
    'temp_min': temp_min,
    'temp_max': temp_max,
    'pressure': pressure,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'wind_deg': weather_deg,
    'rain_1h': rain_1h,
    'clouds_all': clouds,
    'hour': selected_hour,
    'month': selected_date.month,
    'day_of_week': selected_date.weekday(),
    'is_weekend': 1 if selected_date.weekday() >= 5 else 0,
    
    # Cyclical
    'hour_sin': np.sin(2 * np.pi * selected_hour/24),
    'hour_cos': np.cos(2 * np.pi * selected_hour/24),
    
    # Lags
    'lag_1h': lag_1h,
    'lag_24h': lag_24h,
    'lag_168h': lag_24h, # Approximate 1 week lag as 1 day for demo
    'rolling_mean_24': rolling_mean,
    'rolling_std_24': 1000 # fixed assumption
}

df_input = pd.DataFrame([input_dict])

if st.button("🔮 Generate Forecast"):
    
    st.markdown("### 🔍 Model Predictions")
    
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    predictions = {}
    
    i = 0
    for name, data in models_data.items():
        if data is None:
            continue
            
        model = data['model']
        required_features = data['features']
        
        # Prepare input specifically for this model
        # (Filter columns to match exactly what the model was trained on)
        valid_input = df_input.copy()
        
        # Add missing columns if any (fill 0)
        for col in required_features:
            if col not in valid_input.columns:
                valid_input[col] = 0
        
        # Reorder
        X_in = valid_input[required_features]
        
        # Scaling if needed (SVR/Linear might have scalers inside data dict?)
        # Based on your previous code, Linear/SVR saved 'scaler' or 'scaler_X' in pickle.
        # We need to handle that.
        
        try:
            pred = 0
            # SVR
            if "svr" in name.lower() or "support vector" in name.lower():
                scaler_X = data.get('scaler_X')
                scaler_y = data.get('scaler_y')
                if scaler_X and scaler_y:
                    X_scaled = scaler_X.transform(X_in)
                    y_scaled = model.predict(X_scaled)
                    pred = scaler_y.inverse_transform(y_scaled.reshape(-1,1)).flatten()[0]
                else:
                    pred = model.predict(X_in)[0]

            # Linear Check for scaler
            elif "linear" in name.lower():
                scaler = data.get('scaler')
                if scaler:
                    X_scaled = scaler.transform(X_in)
                    pred = model.predict(X_scaled)[0]
                else:
                    pred = model.predict(X_in)[0]
            
            # Trees (RF, XGB)
            else:
                pred = model.predict(X_in)[0]
            
            predictions[name] = pred
            
            # Display Card
            with cols[i % 4]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <h2 style="color: #007bff;">{pred:,.0f} MW</h2>
                </div>
                """, unsafe_allow_html=True)
            i += 1
            
        except Exception as e:
            st.error(f"Error predicting with {name}: {str(e)}")

    # ==========================================
    # COMPARATIVE ANALYSIS
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Comparative Analysis")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Bar Chart
        if predictions:
            chart_data = pd.DataFrame({
                "Model": list(predictions.keys()),
                "Predicted Load (MW)": list(predictions.values())
            })
            st.bar_chart(chart_data.set_index("Model"))
    
    with c2:
        # Insight
        xg_pred = predictions.get("XGBoost (Champion 🏆)", 0)
        st.info(f"**Champion Model Insight**: The XGBoost model predicts **{xg_pred:,.0f} MW**. This model has the highest historical accuracy (R²=0.98) effectively capturing the complex non-linear relationships and time-lag dependencies.")
        
        if len(predictions) > 1:
            avg_pred = np.mean(list(predictions.values()))
            diff = xg_pred - avg_pred
            st.write(f"Deviation from Ensemble Mean: **{diff:+.0f} MW**")

else:
    st.info("Adjust parameters in the sidebar and click **Generate Forecast**.")
