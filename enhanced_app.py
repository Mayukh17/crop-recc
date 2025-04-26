#!/usr/bin/env python3
"""
Enhanced Crop Recommendation System - Web Interface
================================================

This script provides a comprehensive web interface that combines:
- Advanced soil analysis
- Crop lifecycle management
- Real-time weather integration
- Machine learning predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Import our custom modules
from advanced_crop_analysis import AdvancedSoilAnalyzer, CropLifecycleManager
from weather_integration import WeatherService
from model_improvements import EnhancedCropRecommender

# Page configuration
st.set_page_config(
    page_title="Enhanced Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .result-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .feature-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .weather-card {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FFA000;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'weather_service' not in st.session_state:
        api_key = st.secrets.get("openweathermap_api_key", "your_api_key_here")
        st.session_state.weather_service = WeatherService(api_key)
    
    if 'soil_analyzer' not in st.session_state:
        st.session_state.soil_analyzer = AdvancedSoilAnalyzer()
    
    if 'lifecycle_manager' not in st.session_state:
        st.session_state.lifecycle_manager = CropLifecycleManager()
    
    if 'crop_recommender' not in st.session_state:
        st.session_state.crop_recommender = EnhancedCropRecommender()

def display_weather_section():
    """Display weather information and forecasts."""
    st.markdown('<h2 class="sub-header">Weather Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input("Latitude", value=40.7128, format="%.4f")
        longitude = st.number_input("Longitude", value=-74.0060, format="%.4f")
    
    with col2:
        location_name = st.text_input("Location Name", value="NYC")
    
    if st.button("Get Weather Data"):
        try:
            current = st.session_state.weather_service.get_current_weather(latitude, longitude)
            forecast = st.session_state.weather_service.get_forecast(latitude, longitude, days=5)
            
            # Display current weather
            st.markdown('<div class="weather-card">', unsafe_allow_html=True)
            st.subheader("Current Weather")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Temperature", f"{current.temperature:.1f}°C")
            with col2:
                st.metric("Humidity", f"{current.humidity}%")
            with col3:
                st.metric("Rainfall", f"{current.rainfall}mm")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display forecast
            st.subheader("5-Day Forecast")
            
            # Create forecast chart
            forecast_data = pd.DataFrame([{
                'Date': f.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Temperature': f.temperature,
                'Rainfall': f.rainfall,
                'Humidity': f.humidity
            } for f in forecast])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Temperature'],
                name='Temperature (°C)',
                line=dict(color='#FF9800')
            ))
            
            fig.add_trace(go.Bar(
                x=forecast_data['Date'],
                y=forecast_data['Rainfall'],
                name='Rainfall (mm)',
                marker_color='#2196F3'
            ))
            
            fig.update_layout(
                title='Weather Forecast',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cache the weather data
            st.session_state.weather_service.cache_weather_data(location_name, current)
            
        except Exception as e:
            st.error(f"Error fetching weather data: {str(e)}")

def display_soil_analysis_section():
    """Display soil analysis interface."""
    st.markdown('<h2 class="sub-header">Soil Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        n = st.number_input("Nitrogen content (N)", min_value=0, max_value=140, value=90)
        p = st.number_input("Phosphorous content (P)", min_value=0, max_value=140, value=42)
        k = st.number_input("Potassium content (K)", min_value=0, max_value=200, value=43)
        ph = st.number_input("pH value", min_value=0.0, max_value=14.0, value=6.5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        sand = st.slider("Sand %", 0, 100, 40)
        silt = st.slider("Silt %", 0, 100, 40)
        clay = st.slider("Clay %", 0, 100, 20)
        organic_matter = st.slider("Organic Matter %", 0, 30, 5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Analyze Soil"):
        soil_analysis = st.session_state.soil_analyzer.analyze_soil(
            n=n, p=p, k=k, ph=ph,
            sand_pct=sand, silt_pct=silt, clay_pct=clay,
            organic_matter=organic_matter
        )
        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Soil Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Soil Type:** {soil_analysis.soil_type.value}")
            st.write(f"**pH Level:** {soil_analysis.ph_level:.1f}")
            st.write(f"**Organic Matter:** {soil_analysis.organic_matter:.1f}%")
        
        with col2:
            st.write(f"**Water Retention:** {soil_analysis.water_retention:.2f}")
            st.write(f"**Drainage Rate:** {soil_analysis.drainage_rate:.2f}")
        
        st.subheader("Nutrient Levels")
        for nutrient, level in soil_analysis.nutrient_levels.items():
            st.write(f"**{nutrient}:** {level.value}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_crop_recommendation_section():
    """Display crop recommendation interface."""
    st.markdown('<h2 class="sub-header">Crop Recommendation</h2>', unsafe_allow_html=True)
    
    # Load data
    data = pd.read_csv('Crop_Recommendation.csv')
    
    # Get user input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.8)
        humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=202.9)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Get Recommendation"):
        try:
            # Get soil analysis
            soil_analysis = st.session_state.soil_analyzer.analyze_soil(
                n=n, p=p, k=k, ph=ph,
                sand_pct=sand, silt_pct=silt, clay_pct=clay,
                organic_matter=organic_matter
            )
            
            # Make prediction
            features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            prediction = st.session_state.crop_recommender.predict(features)[0]
            
            # Get crop requirements
            requirements = st.session_state.lifecycle_manager.get_crop_requirements(prediction, data)
            
            # Display results
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("Recommendation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Recommended Crop:** {prediction}")
                st.write(f"**Soil Type:** {soil_analysis.soil_type.value}")
                st.write(f"**Growing Season:** {st.session_state.lifecycle_manager.detect_season(temperature, rainfall)}")
            
            with col2:
                st.write("**Optimal Growing Conditions:**")
                for param, values in requirements.items():
                    st.write(f"- {param}: {values['min']:.1f} - {values['max']:.1f}")
            
            # Plot feature importance
            feature_importance = st.session_state.crop_recommender.get_feature_importance()
            fig = px.bar(
                x=list(feature_importance.keys()),
                y=list(feature_importance.values()),
                title="Feature Importance in Recommendation",
                labels={'x': 'Feature', 'y': 'Importance'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making recommendation: {str(e)}")

def main():
    """Main function for the Streamlit app."""
    st.markdown('<h1 class="main-header">🌾 Enhanced Crop Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">Navigation</h2>', unsafe_allow_html=True)
    page = st.sidebar.radio("Go to", 
                          ["Weather Information", "Soil Analysis", "Crop Recommendation"])
    
    # Display selected page
    if page == "Weather Information":
        display_weather_section()
    elif page == "Soil Analysis":
        display_soil_analysis_section()
    else:
        display_crop_recommendation_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="info-text" style="text-align: center;">
    Enhanced Crop Recommendation System v2.0<br>
    Combines machine learning, soil analysis, and real-time weather data
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 