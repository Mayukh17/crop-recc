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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
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
</style>
""", unsafe_allow_html=True)

class CropRecommender:
    """
    A class to handle crop recommendation using machine learning.
    """
    
    def __init__(self, model_path='crop_recommendation_model.pkl'):
        """
        Initialize the CropRecommender.
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        
    def load_data(self, file_path):
        """
        Load and prepare the dataset.
        
        Args:
            file_path (str): Path to the CSV file containing crop data
            
        Returns:
            tuple: Features (X) and target variable (y)
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data format is incorrect
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Validate data structure
            required_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall', 'Crop']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in the dataset")
            
            # Separate features and target
            X = df.drop('Crop', axis=1)
            y = df['Crop']
            
            logger.info(f"Loaded {len(df)} samples with {len(X.columns)} features")
            return X, y
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def train_model(self, X, y):
        """
        Train the Random Forest model.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target variable
            
        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info("Training Random Forest model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        X_test_scaled = self.scaler.transform(X_test)
        accuracy = self.model.score(X_test_scaled, y_test)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        
        return self.model, accuracy

    def save_model(self):
        """
        Save the trained model and scaler.
        
        Raises:
            ValueError: If model or scaler is not trained
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not trained yet")
            
        logger.info(f"Saving model to {self.model_path}")
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
            
        logger.info("Model saved successfully")
        
    def load_model(self):
        """
        Load a previously trained model and scaler.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            return False
            
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict_crop(self, features):
        """
        Make crop predictions for new data.
        
        Args:
            features (array): Input features to predict
                Shape should be (n_samples, n_features)
                Features order: [N, P, K, temperature, humidity, ph, rainfall]
            
        Returns:
            str: Predicted crop
            
        Raises:
            ValueError: If model or scaler is not trained
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not trained yet")
            
        # Validate input shape
        expected_features = 7
        if len(features) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(features)}")
            
        # Reshape for sklearn
        features_array = np.array(features).reshape(1, -1)
            
        # Scale the features
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        
        # Get feature importances
        feature_importances = self.model.feature_importances_
        feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importance_dict = dict(zip(feature_names, feature_importances))
        
        return prediction[0], importance_dict

def main():
    """
    Main function for the Streamlit app.
    """

    st.write("hello there somnath gandu you dont know anything about coding only you know fuzzy rule")
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">About</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-text">
        This application uses machine learning to recommend the most suitable crop based on soil and environmental parameters.
        
        The model is trained on a dataset of soil and environmental parameters and their corresponding crop recommendations.
        
        Simply enter the parameters in the form and click 'Get Recommendation' to get a crop recommendation.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">Parameters</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-text">
        - **Nitrogen (N)**: Essential for leaf growth
        - **Phosphorus (P)**: Promotes root and flower development
        - **Potassium (K)**: Helps with disease resistance
        - **Temperature**: Average temperature in degree Celsius
        - **Humidity**: Relative humidity in percentage
        - **pH**: pH value of soil (0-14)
        - **Rainfall**: Rainfall in mm
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">Model Info</h2>', unsafe_allow_html=True)
        
        # Initialize the recommender
        recommender = CropRecommender()
        
        # Check if model exists, if not train a new one
        if not recommender.load_model():
            with st.spinner("Training new model..."):
                X, y = recommender.load_data('Crop_Recommendation.csv')
                
                # Train model
                model, accuracy = recommender.train_model(X, y)
                
                # Save model
                recommender.save_model()
                
                st.success(f"Model trained successfully with accuracy: {accuracy:.2f}")
        else:
            st.success("Model loaded successfully")
    
    # Main content
    st.markdown('<h2 class="sub-header">Enter Soil and Environmental Parameters</h2>', unsafe_allow_html=True)
    
    # Create two columns for the form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="">', unsafe_allow_html=True)
        n = st.number_input("Nitrogen content in soil (N)", min_value=0, max_value=140, value=90, step=1)
        p = st.number_input("Phosphorous content in soil (P)", min_value=0, max_value=140, value=42, step=1)
        k = st.number_input("Potassium content in soil (K)", min_value=0, max_value=200, value=43, step=1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="">', unsafe_allow_html=True)
        temperature = st.number_input("Temperature in degree Celsius", min_value=0.0, max_value=50.0, value=20.8, step=0.1)
        humidity = st.number_input("Relative humidity in %", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        ph = st.number_input("pH value of soil", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.number_input("Rainfall in mm", min_value=0.0, max_value=300.0, value=202.9, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Submit button
    if st.button("Get Recommendation", key="recommend_button"):
        with st.spinner("Analyzing parameters..."):
            # Get features
            features = [n, p, k, temperature, humidity, ph, rainfall]
            
            # Make prediction
            predicted_crop, importance_dict = recommender.predict_crop(features)
            
            # Display result
            st.markdown('<div class="">', unsafe_allow_html=True)
            st.markdown(f"<h3>Recommended Crop: <span style='color: #1E88E5;'>{predicted_crop}</span></h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display feature importance
            st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
            
            # Sort features by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Create a bar chart
            import plotly.express as px
            
            df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
            fig = px.bar(df, x='Feature', y='Importance', 
                         title='Feature Importance in Crop Recommendation',
                         color='Importance',
                         color_continuous_scale='Blues')
            
            fig.update_layout(
                xaxis_title="Feature",
                yaxis_title="Importance",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display parameter ranges for the recommended crop
            st.markdown('<h3 class="sub-header">Parameter Ranges for Recommended Crop</h3>', unsafe_allow_html=True)
            
            # Load the dataset
            df = pd.read_csv('Crop_Recommendation.csv')
            
            # Filter for the recommended crop
            crop_data = df[df['Crop'] == predicted_crop]
            
            # Calculate min and max for each parameter
            param_ranges = {
                'Nitrogen': (crop_data['Nitrogen'].min(), crop_data['Nitrogen'].max()),
                'Phosphorus': (crop_data['Phosphorus'].min(), crop_data['Phosphorus'].max()),
                'Potassium': (crop_data['Potassium'].min(), crop_data['Potassium'].max()),
                'Temperature': (crop_data['Temperature'].min(), crop_data['Temperature'].max()),
                'Humidity': (crop_data['Humidity'].min(), crop_data['Humidity'].max()),
                'pH': (crop_data['pH_Value'].min(), crop_data['pH_Value'].max()),
                'Rainfall': (crop_data['Rainfall'].min(), crop_data['Rainfall'].max())
            }
            
            # Display parameter ranges
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            for param, (min_val, max_val) in param_ranges.items():
                st.markdown(f"**{param}**: {min_val:.1f} - {max_val:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 