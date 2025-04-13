#!/usr/bin/env python3
"""
Crop Recommendation System
=========================

This script implements a machine learning model to recommend crops based on soil and environmental parameters.
The system uses a Random Forest Classifier to predict the most suitable crop based on various soil and
environmental features.

Features used for prediction:
- N: Nitrogen content in soil
- P: Phosphorous content in soil
- K: Potassium content in soil
- temperature: Temperature in degree Celsius
- humidity: Relative humidity in %
- ph: pH value of soil
- rainfall: Rainfall in mm

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class CropRecommender:
    """
    A class to handle crop recommendation using machine learning.
    
    This class encapsulates all the functionality needed to load data,
    preprocess it, train a model, and make predictions.
    """
    
    def __init__(self, model_path: str = 'crop_recommendation_model.pkl'):
        """
        Initialize the CropRecommender.
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare the dataset.
        
        Args:
            file_path (str): Path to the CSV file containing crop data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target variable (y)
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data format is incorrect
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Validate data structure
            required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'Crop']
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

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """
        Preprocess the data by splitting into train/test sets and scaling features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Tuple containing:
            - Scaled training features
            - Scaled testing features
            - Training labels
            - Testing labels
            - Fitted scaler
        """
        logger.info("Preprocessing data...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train the Random Forest model.
        
        Args:
            X_train (np.ndarray): Scaled training features
            y_train (np.ndarray): Training labels
            
        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info("Training Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        return self.model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model Accuracy: {accuracy:.2f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }

    def save_model(self) -> None:
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

    def predict_crop(self, features: np.ndarray) -> str:
        """
        Make crop predictions for new data.
        
        Args:
            features (np.ndarray): Input features to predict
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
        if features.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {features.shape[1]}")
            
        # Scale the features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        return prediction[0]

def main():
    """
    Main function to demonstrate the crop recommendation system.
    """
    try:
        # Initialize the recommender
        recommender = CropRecommender()
        
        # Load and prepare data
        X, y = recommender.load_data('Crop_Recommendation.csv')
        
        # Preprocess data
        X_train_scaled, X_test_scaled, y_train, y_test, _ = recommender.preprocess_data(X, y)
        
        # Train model
        recommender.train_model(X_train_scaled, y_train)
        
        # Evaluate model
        recommender.evaluate_model(X_test_scaled, y_test)
        
        # Save model
        recommender.save_model()
        
        # Example prediction
        logger.info("\nMaking example prediction...")
        sample_features = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])
        predicted_crop = recommender.predict_crop(sample_features)
        logger.info(f"Predicted crop for sample data: {predicted_crop}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 