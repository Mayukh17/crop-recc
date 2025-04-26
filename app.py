#!/usr/bin/env python3
"""
Crop Recommendation System - Streamlit App
=========================================

This script provides a web-based interface for the crop recommendation system
using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from crop_recommender import CropRecommender
import requests
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import seaborn as sns
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="��",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .st-emotion-cache-1wivap2 {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        
    def get_weather_data(self, lat, lon):
        if self.api_key == 'your_api_key_here':
            return None
            
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

class CropRecommendationApp:
    def __init__(self):
        self.recommender = CropRecommender()
        self.weather_service = WeatherService()
        self.geolocator = Nominatim(user_agent="crop_recommendation_app")
        
        # Load and prepare data
        self.df = pd.read_csv('Crop_Recommendation.csv')
        self.crop_info = {
            'rice': {'description': 'A staple food crop that grows in paddy fields.', 'ideal_conditions': 'Requires warm temperatures and high rainfall.'},
            'maize': {'description': 'Also known as corn, a versatile crop used for food and feed.', 'ideal_conditions': 'Needs moderate temperatures and well-drained soil.'},
            'chickpea': {'description': 'A protein-rich legume crop.', 'ideal_conditions': 'Thrives in cool, dry conditions.'},
            'kidneybeans': {'description': 'A variety of common bean, rich in protein and fiber.', 'ideal_conditions': 'Prefers warm temperatures and well-drained soil.'},
            'pigeonpeas': {'description': 'A drought-resistant legume crop.', 'ideal_conditions': 'Adapts well to semi-arid conditions.'},
            'mothbeans': {'description': 'A drought-resistant legume.', 'ideal_conditions': 'Grows well in hot, dry conditions.'},
            'mungbean': {'description': 'A small, green legume rich in nutrients.', 'ideal_conditions': 'Requires warm temperatures and moderate rainfall.'},
            'blackgram': {'description': 'A pulse crop rich in protein.', 'ideal_conditions': 'Grows best in warm, humid conditions.'},
            'lentil': {'description': 'A small, lens-shaped legume.', 'ideal_conditions': 'Prefers cool temperatures and well-drained soil.'},
            'pomegranate': {'description': 'A fruit-bearing shrub with red, juicy seeds.', 'ideal_conditions': 'Adapts to various climates but prefers semi-arid conditions.'},
            'banana': {'description': 'A tropical fruit crop.', 'ideal_conditions': 'Needs warm temperatures and high rainfall.'},
            'mango': {'description': 'A tropical fruit tree.', 'ideal_conditions': 'Requires warm temperatures and seasonal rainfall.'},
            'grapes': {'description': 'A climbing vine producing fruit clusters.', 'ideal_conditions': 'Grows best in Mediterranean-like climates.'},
            'watermelon': {'description': 'A vine crop producing large, juicy fruits.', 'ideal_conditions': 'Needs warm temperatures and moderate water.'},
            'muskmelon': {'description': 'A sweet, aromatic melon variety.', 'ideal_conditions': 'Requires warm temperatures and well-drained soil.'},
            'apple': {'description': 'A temperate fruit tree.', 'ideal_conditions': 'Needs cold winters and moderate summers.'},
            'orange': {'description': 'A citrus fruit tree.', 'ideal_conditions': 'Grows best in subtropical climates.'},
            'papaya': {'description': 'A tropical fruit tree with large fruits.', 'ideal_conditions': 'Requires warm temperatures and regular rainfall.'},
            'coconut': {'description': 'A tropical palm tree.', 'ideal_conditions': 'Needs tropical climate with high humidity.'},
            'cotton': {'description': 'A fiber crop.', 'ideal_conditions': 'Grows well in warm climates with moderate rainfall.'},
            'jute': {'description': 'A fiber crop used for making rope and fabric.', 'ideal_conditions': 'Requires warm, humid conditions.'},
            'coffee': {'description': 'A beverage crop grown in plantations.', 'ideal_conditions': 'Prefers shade and moderate temperatures.'}
        }

    def get_location_weather(self, location_name):
        try:
            location = self.geolocator.geocode(location_name)
            if location:
                weather_data = self.weather_service.get_weather_data(location.latitude, location.longitude)
                return weather_data
            return None
        except GeocoderTimedOut:
            return None

    def create_feature_importance_plot(self):
        importance_df = pd.DataFrame({
            'Feature': ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
            'Importance': self.recommender.get_feature_importance()
        })
        fig = px.bar(importance_df, x='Feature', y='Importance',
                    title='Feature Importance in Crop Prediction',
                    color='Importance',
                    color_continuous_scale='Viridis')
        return fig

    def create_crop_distribution_plot(self):
        crop_counts = self.df['label'].value_counts()
        fig = px.pie(values=crop_counts.values, names=crop_counts.index,
                    title='Distribution of Crops in Training Data')
        return fig

    def run(self):
        st.title("🌾 Smart Crop Recommendation System")
        
        # Sidebar
        st.sidebar.title("🌾 Navigation")
        page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "Data Insights"])
        
        if page == "Home":
            self.show_home_page()
        elif page == "Make Prediction":
            self.show_prediction_page()
        else:
            self.show_data_insights()

    def show_home_page(self):
        st.markdown("""
        Welcome to the Smart Crop Recommendation System! This application helps farmers make 
        informed decisions about which crops to plant based on various environmental and soil parameters.
        
        ### Features:
        - 🎯 Get personalized crop recommendations
        - 📊 Explore data insights and visualizations
        - 🔍 Analyze soil and environmental parameters
        
        ### How to use:
        1. Navigate to the "Make Prediction" page
        2. Enter your soil and environmental parameters
        3. Click "Get Recommendation" to receive your personalized crop suggestion
        
        ### Dataset Overview:
        The system is trained on a comprehensive dataset containing information about various crops 
        and their optimal growing conditions.
        """)

    def show_prediction_page(self):
        st.title("🎯 Crop Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Soil Parameters")
            n = st.number_input("Nitrogen (N) content in soil", 0, 140, 50)
            p = st.number_input("Phosphorous (P) content in soil", 0, 145, 50)
            k = st.number_input("Potassium (K) content in soil", 0, 205, 50)
            temp = st.number_input("Temperature (°C)", -20.0, 50.0, 25.0)
            
        with col2:
            st.subheader("Environmental Parameters")
            humidity = st.number_input("Relative Humidity (%)", 0.0, 100.0, 50.0)
            ph = st.number_input("pH value of soil", 0.0, 14.0, 7.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

        if st.button("Get Recommendation"):
            try:
                # Prepare input data
                input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
                
                # Load the model and make prediction
                prediction = self.recommender.predict(input_data)
                
                # Display result with styling
                st.success(f"### Recommended Crop: {prediction[0]}")
                
                # Display input parameters summary
                st.subheader("Input Parameters Summary")
                data = {
                    'Parameter': ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                    'Value': [n, p, k, temperature, humidity, ph, rainfall]
                }
                df_summary = pd.DataFrame(data)
                st.table(df_summary)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    def show_data_insights(self):
        st.title("📊 Data Insights")
        
        # Load the dataset
        df = pd.read_csv('Crop_Recommendation.csv')
        
        st.subheader("Dataset Overview")
        st.write(f"Total number of samples: {len(df)}")
        st.write(f"Number of unique crops: {df['Crop'].nunique()}")
        
        # Crop distribution
        st.subheader("Crop Distribution")
        fig = px.bar(df['Crop'].value_counts().reset_index(), 
                    x='Crop', y='count',
                    title='Distribution of Crops in Dataset')
        st.plotly_chart(fig)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        for feature in feature_cols:
            fig = px.box(df, y=feature, x='Crop',
                        title=f'{feature} Distribution by Crop')
            st.plotly_chart(fig)

if __name__ == "__main__":
    app = CropRecommendationApp()
    app.run() 