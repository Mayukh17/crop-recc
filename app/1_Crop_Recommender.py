# app/pages/1_Crop_Recommender.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crop_recommender import CropRecommender
from utils.advanced_features import (
    get_weather_data,
    get_market_prices,
    apply_regional_factors,
    REGIONAL_FACTORS,
    CROP_ROTATION
)

st.title("🌱 Crop Recommender")

# Load Model
@st.cache_resource
def load_model():
    recommender = CropRecommender()
    recommender.load_model()
    return recommender

recommender = load_model()

# Input Sections
col1, col2, col3 = st.columns([2, 2, 1])

# Parameter Ranges
param_ranges = {
    'N': (0.0, 140.0),
    'P': (5.0, 145.0),
    'K': (5.0, 205.0),
    'temperature': (8.0, 44.0),
    'humidity': (14.0, 100.0),
    'ph': (3.5, 10.0),
    'rainfall': (20.0, 300.0)
}

# Soil Parameters
with col1:
    st.header("🧪 Soil Parameters")
    nitrogen = st.number_input("Nitrogen (N)", *param_ranges['N'], value=50.0)
    phosphorus = st.number_input("Phosphorus (P)", *param_ranges['P'], value=50.0)
    potassium = st.number_input("Potassium (K)", *param_ranges['K'], value=50.0)
    ph = st.number_input("pH Value", *param_ranges['ph'], value=6.5)

# Environmental Parameters
with col2:
    st.header("🌦️ Environmental Parameters")
    with st.expander("📍 Fetch Current Weather by Location"):
        lat = st.number_input("Latitude", format="%.4f")
        lon = st.number_input("Longitude", format="%.4f")
        if st.button("Fetch Weather"):
            with st.spinner("Fetching weather..."):
                weather = get_weather_data(lat, lon)
                if weather:
                    st.session_state.temperature = weather['current_weather']['temperature']
                    st.session_state.humidity = weather.get('hourly', {}).get('relative_humidity_2m', [70])[0]
                    st.session_state.rainfall = sum(weather.get('hourly', {}).get('precipitation', [0]))
                    st.success("Weather Data Fetched!")

    temperature = st.number_input("Temperature (°C)", *param_ranges['temperature'], value=float(st.session_state.get('temperature', 25.0)))
    humidity = st.number_input("Humidity (%)", *param_ranges['humidity'], value=float(st.session_state.get('humidity', 70.0)))
    rainfall = st.number_input("Rainfall (mm)", *param_ranges['rainfall'], value=float(st.session_state.get('rainfall', 100.0)))

# Regional Factors
with col3:
    st.header("📍 Regional Factors")
    region = st.selectbox("Select Region", options=list(REGIONAL_FACTORS.keys()))

# Predict Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Predict Crop", use_container_width=True)

if predict_button:
    features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]

    try:
        predictions = recommender.predict_top_crops(features, region=region)

        st.success("Here are the Top Recommended Crops 🌾")
        for idx, (crop, confidence) in enumerate(predictions, start=1):
            st.subheader(f"{idx}. {crop} ({confidence*100:.1f}% confidence)")

            # Market Info
            market = get_market_prices(crop)
            st.info(f"💰 Price: ${market['price']}/kg | 📈 Trend: {market['trend'].capitalize()}")

            # Crop Rotation
            if crop in CROP_ROTATION:
                st.warning(f"🔄 Good Next Crops: {', '.join(CROP_ROTATION[crop])}")

        # Calculate adjusted features for display
        if region and region in REGIONAL_FACTORS:
            factors = REGIONAL_FACTORS[region]
            adjusted = features.copy()
            adjusted[3] *= factors['temperature_multiplier']
            adjusted[4] *= factors['humidity_multiplier']
            adjusted[6] *= factors['rainfall_multiplier']
        else:
            adjusted = features

        # Summary Table
        st.subheader("📊 Summary")
        summary = pd.DataFrame({
            'Parameter': ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
            'Original Value': features,
            'Adjusted Value': adjusted
        })
        st.dataframe(summary, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
