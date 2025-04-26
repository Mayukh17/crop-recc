#!/usr/bin/env python3
"""
Enhanced Crop Recommendation System
=================================

This script extends the basic crop recommendation system with advanced features:
- Cross-validation
- Feature importance analysis
- Model comparison
- Hyperparameter tuning
- Advanced visualizations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedCropRecommender:
    """
    Enhanced version of the crop recommender with additional features.
    """
    
    def __init__(self, model_path: str = 'enhanced_crop_model.pkl'):
        """
        Initialize the enhanced recommender.
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = Path(model_path)
        self.models = {}
        self.scaler = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
    def analyze_soil_type(self, n: float, p: float, k: float) -> str:
        """
        Classify soil type based on NPK values.
        
        Args:
            n (float): Nitrogen content
            p (float): Phosphorus content
            k (float): Potassium content
            
        Returns:
            str: Soil type classification
        """
        # Simple soil classification based on NPK values
        npk_sum = n + p + k
        if npk_sum < 50:
            return "Poor soil"
        elif npk_sum < 100:
            return "Average soil"
        else:
            return "Rich soil"
    
    def detect_season(self, temperature: float, rainfall: float) -> str:
        """
        Detect season based on temperature and rainfall.
        
        Args:
            temperature (float): Temperature in Celsius
            rainfall (float): Rainfall in mm
            
        Returns:
            str: Detected season
        """
        if temperature > 25 and rainfall > 200:
            return "Monsoon"
        elif temperature > 30:
            return "Summer"
        elif temperature < 20:
            return "Winter"
        else:
            return "Spring"

    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                               cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation for all models.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            cv (int): Number of folds
            
        Returns:
            Dict containing cross-validation scores for each model
        """
        cv_scores = {}
        
        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf_scores = cross_val_score(rf, X, y, cv=cv)
        cv_scores['random_forest'] = rf_scores
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_scores = cross_val_score(xgb_model, X, y, cv=cv)
        cv_scores['xgboost'] = xgb_scores
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(random_state=42)
        lgb_scores = cross_val_score(lgb_model, X, y, cv=cv)
        cv_scores['lightgbm'] = lgb_scores
        
        return cv_scores

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for Random Forest.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            
        Returns:
            Dict containing best parameters and score
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

    def plot_feature_importance(self, model: Any, feature_names: List[str]) -> None:
        """
        Plot feature importance using SHAP values.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        plt.figure(figsize=(10, 6))
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title('Feature Importances')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str]) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    def analyze_correlations(self, data: pd.DataFrame) -> None:
        """
        Analyze and plot feature correlations.
        
        Args:
            data: DataFrame containing features
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig('correlations.png')
        plt.close()

    def get_crop_requirements(self, crop: str, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Get optimal growing conditions for a specific crop.
        
        Args:
            crop (str): Crop name
            data (pd.DataFrame): Dataset containing crop information
            
        Returns:
            Dict containing optimal ranges for each parameter
        """
        crop_data = data[data['Crop'] == crop]
        requirements = {}
        
        for feature in self.feature_names:
            requirements[feature] = {
                'min': crop_data[feature].min(),
                'max': crop_data[feature].max(),
                'mean': crop_data[feature].mean(),
                'std': crop_data[feature].std()
            }
        
        return requirements

def main():
    """
    Main function to demonstrate enhanced crop recommendation system.
    """
    try:
        # Initialize enhanced recommender
        recommender = EnhancedCropRecommender()
        
        # Load data
        data = pd.read_csv('Crop_Recommendation.csv')
        X = data.drop('Crop', axis=1)
        y = data['Crop']
        
        # Perform cross-validation
        cv_scores = recommender.perform_cross_validation(X, y)
        logger.info("\nCross-validation scores:")
        for model, scores in cv_scores.items():
            logger.info(f"{model}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Tune hyperparameters
        tuning_results = recommender.tune_hyperparameters(X, y)
        logger.info("\nBest hyperparameters:")
        logger.info(tuning_results['best_params'])
        
        # Train final model with best parameters
        best_rf = RandomForestClassifier(**tuning_results['best_params'], random_state=42)
        best_rf.fit(X, y)
        
        # Plot feature importance
        recommender.plot_feature_importance(best_rf, X.columns)
        
        # Analyze correlations
        recommender.analyze_correlations(X)
        
        # Example prediction with additional information
        sample_features = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])
        prediction = best_rf.predict(sample_features)[0]
        
        # Get soil type and season
        soil_type = recommender.analyze_soil_type(90, 42, 43)
        season = recommender.detect_season(20.8, 202.9)
        
        # Get crop requirements
        requirements = recommender.get_crop_requirements(prediction, data)
        
        logger.info("\nPrediction Results:")
        logger.info(f"Predicted crop: {prediction}")
        logger.info(f"Soil type: {soil_type}")
        logger.info(f"Season: {season}")
        logger.info("\nOptimal growing conditions:")
        for param, values in requirements.items():
            logger.info(f"{param}: {values['min']:.1f} - {values['max']:.1f} (mean: {values['mean']:.1f})")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 