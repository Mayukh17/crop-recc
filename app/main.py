import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))
from crop_recommender import CropRecommender

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = CropRecommender()
    try:
        if not st.session_state.recommender.load_model():
            st.info("Training new model...")
            try:
                X, y = st.session_state.recommender.load_data('Crop_Recommendation.csv')
                st.session_state.recommender.train_model(X, y)
                st.session_state.recommender.save_model()
                st.success("Model trained and saved successfully!")
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.stop()
        else:
            st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error initializing the model: {str(e)}")
        st.stop()

# Custom CSS with background image
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin: 0.5em 0;
    }
    .stApp {
        background-image: url("app/assets/farm.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .content-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

def home_page():
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.title("Smart Crop Recommendation System 🌾")
    st.markdown("""
    Welcome to the Smart Crop Recommendation System! This tool helps farmers and agricultural 
    professionals choose the best crop to grow based on soil and environmental conditions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display a welcome message with information about the system
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("""
    ## About This System
    
    This application uses machine learning to recommend the most suitable crops based on:
    
    - **Soil Parameters**: Nitrogen, Phosphorus, Potassium, and pH levels
    - **Environmental Conditions**: Temperature, Humidity, and Rainfall
    
    ### How to Use
    
    1. Navigate to the **Crop Recommender** page using the sidebar
    2. Enter your soil and environmental parameters
    3. Click "Recommend Crop" to get personalized crop recommendations
    4. View additional insights in the **Data Insights** page
    
    ### Features
    
    - **Multi-crop Recommendations**: Get multiple crop options with confidence scores
    - **Weather Integration**: Use real-time weather data for more accurate predictions
    - **Regional Adaptation**: Adjust recommendations based on your geographical region
    - **Market Insights**: View current market prices and trends for recommended crops
    - **Crop Rotation Advice**: Get suggestions for sustainable crop rotation
    - **Soil Health Monitoring**: Track and analyze your soil health over time
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a call-to-action button
    if st.button("Get Started", use_container_width=True):
        st.session_state.page = "Crop Recommender"

def crop_recommender_page():
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.title("Crop Recommendation 🌱")
    st.markdown("Enter soil and environmental parameters to get crop recommendations:")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    with st.form("crop_form"):
        n = st.number_input("Nitrogen content (N)", min_value=0.0, format="%.2f")
        p = st.number_input("Phosphorus content (P)", min_value=0.0, format="%.2f")
        k = st.number_input("Potassium content (K)", min_value=0.0, format="%.2f")
        temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, format="%.2f")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, format="%.2f")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, format="%.2f")
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, format="%.2f")
        
        submitted = st.form_submit_button("Recommend Crop")
        if submitted:
            try:
                features = [n, p, k, temp, humidity, ph, rainfall]
                prediction = st.session_state.recommender.predict_crop(features)
                st.success(f"Recommended Crop: **{prediction}**")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

def data_insights_page():
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.title("Data Insights 📊")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    try:
        df = pd.read_csv('Crop_Recommendation.csv')
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        
        st.subheader("Feature Distributions")
        selected_col = st.selectbox("Select feature to visualize", df.columns[:-1])
        
        # Create histogram using plotly
        fig = px.histogram(df, x=selected_col, nbins=30)
        fig.update_layout(
            title=f"Distribution of {selected_col}",
            xaxis_title=selected_col,
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Dataset file not found! Please make sure 'Crop_Recommendation.csv' is in the correct location.")
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Crop Recommender", "Data Insights"])

# Page routing
if page == "Home":
    home_page()
elif page == "Crop Recommender":
    crop_recommender_page()
elif page == "Data Insights":
    data_insights_page()