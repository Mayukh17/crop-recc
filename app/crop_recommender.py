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

class CropRecommender:
    """
    A class to handle crop recommendation using machine learning.
    
    This class encapsulates all the functionality needed to load data,
    preprocess it, train a model, and make predictions.
    """
    file_path = 'Crop_Recommendation.csv'
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

    def predict_top_crops(self, features, top_n=3):
        """
        Predict top N suitable crops with confidence scores.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not trained yet")
            
        # Validate input shape
        if len(features) != 7:
            raise ValueError(f"Expected 7 features, got {len(features)}")
            
        # Reshape and scale features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Get probabilities for all crops
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top N predictions
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        return [(self.model.classes_[i], probabilities[i]) for i in top_indices]

    def predict_crop(self, features):
        """
        Predict the best crop (returns single prediction).
        """
        predictions = self.predict_top_crops(features, top_n=1)
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
        
        return [n, p, k, temperature, humidity, ph, rainfall]
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
            X, y = recommender.load_data('Crop_Recommendation.csv')
            
            # Train model
            recommender.train_model(X, y)
            
            # Save model
            recommender.save_model()
        
        # Get user input
        features = get_user_input()
        
        # Make prediction
        predicted_crop = recommender.predict_crop(features)
        
        # Display result
        print("\n=== Recommendation ===")
        print(f"Based on the provided parameters, the recommended crop is: {predicted_crop}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 