import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crop_recommender import CropRecommender
from utils.advanced_features import (
    get_weather_data,
    get_market_prices,
    apply_regional_factors,
    REGIONAL_FACTORS,
    CROP_ROTATION
)

# Set up Streamlit page
st.set_page_config(page_title="Crop Prediction", page_icon="🌱", layout="wide")

# Custom CSS styles
st.markdown("""
    <style>
    .prediction-title {
        font-size: 36px;
        font-weight: bold;
        color: #1565C0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        color: #1565C0;
    }
    .parameter-section {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .market-info {
        padding: 1rem;
        background-color: #F1F8E9;
        border-radius: 10px;
        margin: 1rem 0;
        color: black;
    }
    .rotation-info {
        padding: 1rem;
        background-color: #FFF3E0;
        border-radius: 10px;
        margin: 1rem 0;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    recommender = CropRecommender()
    recommender.load_model()
    return recommender

recommender = load_model()

# Title
st.markdown('<p class="prediction-title">🌱 Crop Prediction</p>', unsafe_allow_html=True)

# Input Columns
col1, col2, col3 = st.columns([2, 2, 1])

# Soil Parameters
with col1:
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.subheader("Soil Parameters")
    nitrogen = st.number_input("Nitrogen content (N)", min_value=0.0, max_value=140.0, value=50.0)
    phosphorus = st.number_input("Phosphorus content (P)", min_value=5.0, max_value=145.0, value=50.0)
    potassium = st.number_input("Potassium content (K)", min_value=5.0, max_value=205.0, value=50.0)
    ph = st.number_input("pH value", min_value=3.5, max_value=10.0, value=6.5)
    st.markdown('</div>', unsafe_allow_html=True)

# Environmental Parameters
with col2:
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.subheader("Environmental Parameters")

    with st.expander("📍 Use Current Location Weather"):
        st.write("Enter your coordinates to fetch current weather data:")
        lat = st.number_input("Latitude", value=0.0, format="%.4f")
        lon = st.number_input("Longitude", value=0.0, format="%.4f")
        if st.button("Fetch Weather"):
            with st.spinner("Fetching weather data..."):
                weather_data = get_weather_data(lat, lon)
                if weather_data:
                    st.session_state.temperature = float(weather_data['current_weather']['temperature'])
                    st.session_state.humidity = float(weather_data.get('hourly', {}).get('relative_humidity_2m', [70.0])[0])
                    st.session_state.rainfall = float(sum(weather_data.get('hourly', {}).get('precipitation', [0.0])))
                    st.success("Weather data fetched successfully!")

    temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0, value=float(st.session_state.get('temperature', 25.0)))
    humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, value=float(st.session_state.get('humidity', 70.0)))
    rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=float(st.session_state.get('rainfall', 100.0)))
    st.markdown('</div>', unsafe_allow_html=True)

# Regional Factors
with col3:
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.subheader("Regional Factors")
    region = st.selectbox("Select Region", options=["skip"]+list(REGIONAL_FACTORS.keys()))
    if region=="skip":
        region=None
    st.markdown('</div>', unsafe_allow_html=True)

# Predict Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Predict Crop", use_container_width=True)

# Run prediction
if predict_button:
    features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]

    try:
        predictions = recommender.predict_top_crops(features, region=region)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Top Crop Recommendations")

        for i, (crop, confidence) in enumerate(predictions, 1):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{i}. {crop}**")
            with c2:
                st.progress(confidence)
                st.text(f"{confidence*100:.1f}%")

            market_info = get_market_prices(crop)
            st.markdown(f"""
            <div class="market-info">
                <p><strong>Market Price:</strong> ${market_info['price']}/kg
                <br><strong>Market Trend:</strong> {market_info['trend'].title()}</p>
            </div>
            """, unsafe_allow_html=True)

            if crop in CROP_ROTATION:
                st.markdown(f"""
                <div class="rotation-info">
                    <p><strong>Recommended Next Crops:</strong> {', '.join(CROP_ROTATION[crop])}</p>
                </div>
                """, unsafe_allow_html=True)


        # Parameter Summary Table
        st.subheader("Parameter Summary")
        summary_data = {
            'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Temperature (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)'],
            'Original Value': features,
            'Adjusted Value': features
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while making the prediction: {str(e)}")

# Parameter Info
with st.expander("ℹ️ Parameter Information"):
    st.markdown("""
    ### Understanding the Parameters

    #### Soil Parameters
    - **Nitrogen (N)**: Essential for leaf growth and chlorophyll production
    - **Phosphorus (P)**: Important for root development and flower/fruit production
    - **Potassium (K)**: Enhances crop quality and disease resistance
    - **pH**: Affects nutrient availability and soil microorganism activity

    #### Environmental Parameters
    - **Temperature**: Influences plant growth rate and metabolism
    - **Humidity**: Affects plant transpiration and disease susceptibility
    - **Rainfall**: Determines water availability for plant growth

    #### Regional Factors
    Different regions have different multipliers that affect:
    - Rainfall expectations
    - Temperature tolerance
    - Humidity requirements

    ### Tips for Accurate Predictions
    1. Use recent soil test results for N, P, K, and pH values
    2. Consider using the weather integration for accurate environmental data
    3. Select your region carefully for more accurate predictions
    4. Consider crop rotation suggestions for sustainable farming
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <p>Make sure to validate the recommended crops with local agricultural experts</p>
</div>
""", unsafe_allow_html=True)
