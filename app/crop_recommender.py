#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class CropRecommender:
    """
    A class to handle crop recommendation using machine learning.
    
    This class encapsulates all the functionality needed to load data,
    preprocess it, train a model, and make predictions.
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
        
    def load_data(self, file_path='Crop_Recommendation.csv'):
        """
        Load and prepare the dataset.
        
        Args:
            file_path (str): Path to the CSV file containing crop data
            
        Returns:
            tuple: Features (X) and target variable (y)
        """
        try:
            # Get the absolute path to the CSV file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_path = os.path.join(base_dir, file_path)
            
            # If file not found in current directory, try parent directory
            if not os.path.exists(absolute_path):
                absolute_path = os.path.join(os.path.dirname(base_dir), file_path)
            
            logger.info(f"Loading data from {absolute_path}")
            df = pd.read_csv(absolute_path)
            
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
            logger.error(f"Data file not found: {absolute_path}")
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
        
        return self.model

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

    def predict_top_crops(self, features, region=None, top_n=3):
        """
        Predict top N suitable crops with confidence scores.
        
        Args:
            features (list): List of 7 features in order [N, P, K, temperature, humidity, ph, rainfall]
            region (str, optional): Region name for adjusting features. If None, no adjustment is made.
            top_n (int, optional): Number of top crops to return. Defaults to 3.
            
        Returns:
            list: List of tuples (crop_name, confidence_score)
            
        Raises:
            ValueError: If model or scaler is not trained, or if features are invalid
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not trained yet")
            
        # Validate input shape
        if len(features) != 7:
            raise ValueError(f"Expected 7 features, got {len(features)}")
            
        # Create a copy of features to avoid modifying the original
        adjusted_features = features.copy()
            
        # Apply regional factors if specified
        if region and region in REGIONAL_FACTORS:
            try:
                factors = REGIONAL_FACTORS[region]
                adjusted_features[3] *= factors['temperature_multiplier']  # temperature
                adjusted_features[4] *= factors['humidity_multiplier']     # humidity
                adjusted_features[6] *= factors['rainfall_multiplier']     # rainfall
                logger.info(f"Applied regional factors for {region}")
            except Exception as e:
                logger.warning(f"Error applying regional factors: {str(e)}")
                # If regional factors fail, use original features
                adjusted_features = features
            
        # Reshape and scale features
        features_array = np.array(adjusted_features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Get probabilities for all crops
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top N predictions
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        return [(self.model.classes_[i], probabilities[i]) for i in top_indices]

    def predict_crop(self, features, region=None):
        """
        Predict the best crop (returns single prediction).
        
        Args:
            features (list): List of 7 features in order [N, P, K, temperature, humidity, ph, rainfall]
            region (str, optional): Region name for adjusting features
            
        Returns:
            str: Name of the best predicted crop
        """
        predictions = self.predict_top_crops(features, region, top_n=1)
        return predictions[0][0]

def get_user_input():
    """
    Get soil and environmental parameters from user input.
    
    Returns:
        list: List of features in the order [N, P, K, temperature, humidity, ph, rainfall]
    """
    print("\n=== Crop Recommendation System ===")
    print("Please enter the following soil and environmental parameters:")
    
    try:
        n = float(input("Nitrogen content in soil (N): "))
        p = float(input("Phosphorous content in soil (P): "))
        k = float(input("Potassium content in soil (K): "))
        temperature = float(input("Temperature in degree Celsius: "))
        humidity = float(input("Relative humidity in %: "))
        ph = float(input("pH value of soil: "))
        rainfall = float(input("Rainfall in mm: "))
        
        # Get region input
        print("\nAvailable regions:", ", ".join(REGIONAL_FACTORS.keys()))
        region = input("Enter region (press Enter to skip): ").strip()
        if not region or region not in REGIONAL_FACTORS:
            region = None
        
        return [n, p, k, temperature, humidity, ph, rainfall], region
    except ValueError:
        print("Error: Please enter valid numerical values.")
        sys.exit(1)

def main():
    """
    Main function to demonstrate the crop recommendation system.
    """
    try:
        # Initialize the recommender
        recommender = CropRecommender()
        
        # Check if model exists, if not train a new one
        if not recommender.load_model():
            logger.info("Training new model...")
            # Load and prepare data
            X, y = recommender.load_data()
            
            # Train model
            recommender.train_model(X, y)
            
            # Save model
            recommender.save_model()
        
        # Get user input
        features, region = get_user_input()
        
        # Make prediction
        predictions = recommender.predict_top_crops(features, region=region)
        
        # Display results
        print("\n=== Recommendations ===")
        for i, (crop, confidence) in enumerate(predictions, 1):
            print(f"{i}. {crop}: {confidence:.1%} confidence")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 