#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the app directory to the Python path
app_dir = Path(__file__).parent / 'app'
sys.path.append(str(app_dir))

from crop_recommender import CropRecommender

def test_model_loading():
    """
    Test if the model can be loaded correctly.
    """
    try:
        # Initialize the recommender
        recommender = CropRecommender()
        
        # Try to load the model
        if recommender.load_model():
            logger.info("Model loaded successfully")
            
            # Test prediction with sample data
            sample_features = [90, 42, 43, 20.8, 82.0, 6.5, 202.9]
            predictions = recommender.predict_top_crops(sample_features)
            
            logger.info("Sample prediction results:")
            for crop, confidence in predictions:
                logger.info(f"{crop}: {confidence:.1%} confidence")
                
            return True
        else:
            logger.error("Failed to load model")
            return False
            
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1) 