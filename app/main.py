# main.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from crop_recommender import CropRecommender  # Your original class

# Set up Streamlit page
st.set_page_config(
    page_title="🌱 Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add app directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Custom CSS styling
st.markdown("""
<style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #E8F5E9;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    .market-info {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .rotation-info {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.title("🌾 Smart Farming AI")
    page = st.radio(
        "Navigate",
        ["Home", "Crop Recommender", "Data Insights"],
        index=0
    )

# Load model with caching
@st.cache_resource
def load_recommender():
    try:
        recommender = CropRecommender()
        if not recommender.load_model():
            st.error("Failed to load model")
            return None
        return recommender
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

recommender = load_recommender()
# Regional adaptation factors
REGIONAL_FACTORS = {
    'Tropical': {
        'rainfall_multiplier': 1.2,
        'temperature_multiplier': 1.1,
        'humidity_multiplier': 1.15
    },
    'Subtropical': {
        'rainfall_multiplier': 1.1,
        'temperature_multiplier': 1.05,
        'humidity_multiplier': 1.1
    },
    'Arid': {
        'rainfall_multiplier': 0.8,
        'temperature_multiplier': 1.2,
        'humidity_multiplier': 0.7
    },
    'Semi-arid': {
        'rainfall_multiplier': 0.9,
        'temperature_multiplier': 1.15,
        'humidity_multiplier': 0.8
    },
    'Mediterranean': {
        'rainfall_multiplier': 1.0,
        'temperature_multiplier': 1.0,
        'humidity_multiplier': 1.0
    },
    'Temperate': {
        'rainfall_multiplier': 1.05,
        'temperature_multiplier': 0.95,
        'humidity_multiplier': 1.05
    }
}
# Home Page
if page == "Home":
    st.markdown('<div class="title">🌱 Smart Crop Recommendation System</div>', unsafe_allow_html=True)
    st.image("app/farm.jpg", use_container_width=True)
    st.markdown("""
    <div style="text-align: center;">
        <h3>Key Features</h3>
        <p>🌦️ Weather-integrated Recommendations | 📈 Market Trends | 🌍 Regional Adaptation</p>
        <p>🧪 Soil Health Analysis | 📊 Data-driven Insights | 🤖 AI-Powered Predictions</p>
    </div>
    """, unsafe_allow_html=True)

# Crop Recommendation Page
elif page == "Crop Recommender":
    st.markdown('<div class="title">🌾 AI-Powered Crop Recommendation</div>', unsafe_allow_html=True)
    
    if not recommender:
        st.error("System initialization failed. Please try reloading the page.")
        st.stop()

    # Input columns
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Soil Composition")
        n = st.slider("Nitrogen (N)", 0.0, 150.0, 50.0)
        p = st.slider("Phosphorus (P)", 5.0, 145.0, 50.0)
        k = st.slider("Potassium (K)", 5.0, 205.0, 50.0)
        ph = st.slider("pH Value", 3.5, 10.0, 6.5)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Environmental Conditions")
        temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Regional Settings")
        region = st.selectbox(
            "Climate Region",
            options=["Skip"]+list(REGIONAL_FACTORS.keys()),
            help="Select your regional climate type"
        )
        if region=="Skip":
            region=None
        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction button
    if st.button("Get Crop Recommendations", use_container_width=True, type="primary"):
        try:
            features = [n, p, k, temperature, humidity, ph, rainfall]
            
            with st.spinner("Analyzing conditions..."):
                predictions = recommender.predict_top_crops(features, region=region)
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("Top Recommendations")
            
            for i, (crop, confidence) in enumerate(predictions, 1):
                # Recommendation card
                st.markdown(f"""
                <div style="padding: 1rem; margin: 1rem 0; border-radius: 10px; background: white;">
                    <h3>#{i} {crop} <span style="color: #2E7D32; float: right;">{confidence*100:.1f}%</span></h3>
                    <div style="height: 10px; background: #E0E0E0; border-radius: 5px;">
                        <div style="width: {confidence*100}%; height: 100%; background: #4CAF50; border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show input parameters
            st.subheader("Input Summary")
            param_data = {
                "Parameter": ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
                "Value": features,
                "Unit": ["ppm", "ppm", "ppm", "°C", "%", "-", "mm"]
            }
            st.dataframe(pd.DataFrame(param_data), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Data Insights Page
elif page == "Data Insights":
    st.markdown('<div class="title">📊 Agricultural Insights Dashboard</div>', unsafe_allow_html=True)
    # Get list of all crops from the model
    try:
        all_crops = recommender.model.classes_
    except AttributeError:
        st.error("Model not loaded properly")
        st.stop()

    # Generate comprehensive yield data
    st.subheader("Historical Crop Yields (All Crops)")
    
    # Create synthetic data for demonstration
    years = np.arange(2015, 2024)
    np.random.seed(42)  # For reproducible random numbers
    
    # Create yield data for all crops with realistic variations
    yield_data = pd.DataFrame({
        crop: np.random.normal(
            loc=np.random.uniform(30, 70),  # Baseline yield between 30-70
            scale=np.random.uniform(3, 8),   # Variation between 3-8
            size=len(years)
        ).round(1)
        for crop in all_crops
    }, index=years)
    
    # Add some realistic trends
    for crop in all_crops[:4]:  # Add trends to first 4 crops
        yield_data[crop] += np.linspace(0, 10, len(years))
    
    # Create interactive visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_crops = st.multiselect(
            "Select crops to visualize:",
            options=all_crops,
            default=all_crops[:3]  # Show first 3 by default
        )
        
        if selected_crops:
            st.line_chart(yield_data[selected_crops])
        else:
            st.warning("Please select at least one crop")
    
    with col2:
        st.markdown("**Latest Yield Data (2023)**")
        latest_data = yield_data.loc[2023].sort_values(ascending=False)
        st.bar_chart(latest_data.head(10))  # Show top 10 crops
    
    # Add detailed data table
    with st.expander("View Full Yield Data Table"):
        st.dataframe(
            yield_data.style
                .background_gradient(cmap='YlGn', subset=pd.IndexSlice[:, :])
                .format("{:.1f}"),
            use_container_width=True
        )
    
    # Add crop statistics
    st.subheader("Crop Yield Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Highest Average Yield**")
        top_crop = yield_data.mean().idxmax()
        st.metric(top_crop, f"{yield_data[top_crop].mean():.1f} units")
    
    with col2:
        st.markdown("**Most Consistent Yield**")
        consistent_crop = yield_data.std().idxmin()
        st.metric(consistent_crop, f"σ={yield_data[consistent_crop].std():.1f}")
    
    with col3:
        st.markdown("**Most Improved Yield**")
        improvement = (yield_data.iloc[-1] - yield_data.iloc[0]).idxmax()
        st.metric(improvement, 
                 f"+{(yield_data[improvement].iloc[-1] - yield_data[improvement].iloc[0]):.1f} units")
    
    # Add crop type distribution
    st.subheader("Crop Type Distribution")
    crop_types = {
        'Cereals': ['rice', 'wheat', 'maize', 'barley'],
        'Pulses': ['pigeonpeas', 'mothbeans', 'lentil'],
        'Fruits': ['mango', 'grapes', 'apple'],
        'Vegetables': ['potato', 'tomato', 'onion'],
        'Commercial': ['coffee', 'cotton', 'sugarcane']
    }
    
    # Create type distribution data
    type_counts = {
        t: sum(1 for c in all_crops if c.lower() in crops)
        for t, crops in crop_types.items()
    }
    
    st.bar_chart(pd.DataFrame.from_dict(type_counts, orient='index'))
    # Soil Health Metrics
    st.subheader("Soil Health Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Nitrogen", "62 ppm", "+8% YoY")
    col2.metric("Average Phosphorus", "45 ppm", "-3% YoY")
    col3.metric("Average Potassium", "78 ppm", "+5% YoY")

    # Market Trends
    st.subheader("Commodity Price Trends")
    months = pd.date_range("2023-01", periods=12, freq="M")
    price_data = pd.DataFrame({
        'Rice': np.random.uniform(1.8, 2.5, 12),
        'Wheat': np.random.uniform(1.5, 2.2, 12),
        'Corn': np.random.uniform(1.6, 2.4, 12)
    }, index=months)
    st.area_chart(price_data)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <small>🌍 Sustainable Farming Initiative | 🚜 Precision Agriculture Solutions</small><br>
    <small>© 2024 Smart Crop AI | Contact: support@agritech.ai</small>
</div>
""", unsafe_allow_html=True)