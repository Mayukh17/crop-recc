import requests
import numpy as np
from typing import Dict, List, Tuple
import streamlit as st

# Crop rotation knowledge base
CROP_ROTATION = {
    'Rice': ['Legumes', 'Oilseeds', 'Vegetables'],
    'Wheat': ['Pulses', 'Mustard', 'Legumes'],
    'Maize': ['Soybeans', 'Peanuts', 'Beans'],
    'Cotton': ['Wheat', 'Sorghum', 'Chickpeas'],
    'Sugarcane': ['Legumes', 'Rice', 'Vegetables'],
    'Coffee': ['Beans', 'Vegetables', 'Spices'],
    'Chickpea': ['Wheat', 'Sorghum', 'Maize'],
    'Kidney Beans': ['Corn', 'Potatoes', 'Carrots'],
    'Pigeonpeas': ['Cotton', 'Sorghum', 'Millet'],
    'Mothbeans': ['Wheat', 'Vegetables', 'Spices'],
    'Mungbean': ['Rice', 'Maize', 'Vegetables'],
    'Blackgram': ['Rice', 'Maize', 'Vegetables'],
    'Lentil': ['Wheat', 'Rice', 'Vegetables'],
    'Pomegranate': ['Vegetables', 'Herbs', 'Flowers'],
    'Banana': ['Legumes', 'Vegetables', 'Cover Crops'],
    'Mango': ['Cover Crops', 'Vegetables', 'Herbs'],
    'Grapes': ['Cover Crops', 'Herbs', 'Flowers'],
    'Watermelon': ['Corn', 'Beans', 'Peas'],
    'Muskmelon': ['Corn', 'Beans', 'Peas'],
    'Apple': ['Cover Crops', 'Herbs', 'Flowers'],
    'Orange': ['Cover Crops', 'Herbs', 'Vegetables'],
    'Papaya': ['Legumes', 'Vegetables', 'Cover Crops'],
    'Coconut': ['Cover Crops', 'Spices', 'Vegetables']
}

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

def get_weather_data(latitude: float, longitude: float) -> Dict:
    """
    Fetch real-time weather data from Open-Meteo API.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&hourly=temperature_2m,relative_humidity_2m,precipitation"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def get_market_prices(crop: str) -> Dict:
    """
    Get current market prices for crops (mock data for demonstration).
    """
    # Mock market data - replace with actual API integration
    MARKET_DATA = {
        'Rice': {'price': 4.5, 'trend': 'rising'},
        'Wheat': {'price': 3.8, 'trend': 'stable'},
        'Maize': {'price': 2.9, 'trend': 'falling'},
        'Cotton': {'price': 6.2, 'trend': 'stable'},
        'Sugarcane': {'price': 1.8, 'trend': 'rising'},
        'Coffee': {'price': 8.5, 'trend': 'rising'},
        'Chickpea': {'price': 4.2, 'trend': 'stable'},
        'Kidney Beans': {'price': 5.1, 'trend': 'rising'},
        'Pigeonpeas': {'price': 3.9, 'trend': 'stable'},
        'Mothbeans': {'price': 4.0, 'trend': 'stable'},
        'Mungbean': {'price': 4.3, 'trend': 'rising'},
        'Blackgram': {'price': 4.1, 'trend': 'stable'},
        'Lentil': {'price': 4.8, 'trend': 'rising'},
        'Pomegranate': {'price': 7.2, 'trend': 'stable'},
        'Banana': {'price': 2.5, 'trend': 'stable'},
        'Mango': {'price': 5.5, 'trend': 'rising'},
        'Grapes': {'price': 6.8, 'trend': 'rising'},
        'Watermelon': {'price': 1.9, 'trend': 'falling'},
        'Muskmelon': {'price': 2.2, 'trend': 'stable'},
        'Apple': {'price': 4.9, 'trend': 'rising'},
        'Orange': {'price': 3.5, 'trend': 'stable'},
        'Papaya': {'price': 2.8, 'trend': 'stable'},
        'Coconut': {'price': 3.2, 'trend': 'rising'}
    }
    return MARKET_DATA.get(crop, {'price': 0.0, 'trend': 'unknown'})

def apply_regional_factors(features: List[float], region: str) -> List[float]:
    """
    Adjust features based on regional factors.
    """
    if region not in REGIONAL_FACTORS:
        return features
    
    factors = REGIONAL_FACTORS[region]
    adjusted_features = features.copy()
    
    # Adjust temperature (index 3)
    adjusted_features[3] *= factors['temperature_multiplier']
    # Adjust humidity (index 4)
    adjusted_features[4] *= factors['humidity_multiplier']
    # Adjust rainfall (index 6)
    adjusted_features[6] *= factors['rainfall_multiplier']
    
    return adjusted_features

def get_soil_health_status(n: float, p: float, k: float, ph: float) -> Dict:
    """
    Analyze soil health based on NPK values and pH.
    """
    health_status = {
        'overall_health': 'Good',
        'recommendations': [],
        'warnings': []
    }
    
    # Check Nitrogen levels
    if n < 30:
        health_status['overall_health'] = 'Poor'
        health_status['recommendations'].append("Add nitrogen-rich fertilizers")
    elif n > 100:
        health_status['warnings'].append("High nitrogen levels - reduce fertilization")
    
    # Check Phosphorus levels
    if p < 10:
        health_status['overall_health'] = 'Poor'
        health_status['recommendations'].append("Add phosphorus-rich fertilizers")
    elif p > 100:
        health_status['warnings'].append("High phosphorus levels - reduce fertilization")
    
    # Check Potassium levels
    if k < 20:
        health_status['overall_health'] = 'Poor'
        health_status['recommendations'].append("Add potassium-rich fertilizers")
    elif k > 150:
        health_status['warnings'].append("High potassium levels - reduce fertilization")
    
    # Check pH levels
    if ph < 5.5:
        health_status['recommendations'].append("Soil is too acidic - consider adding lime")
    elif ph > 8.5:
        health_status['recommendations'].append("Soil is too alkaline - consider adding sulfur")
    
    return health_status

def get_crop_calendar(crop: str) -> Dict:
    """
    Get crop calendar information (mock data for demonstration).
    """
    # Mock calendar data - replace with actual database
    CROP_CALENDAR = {
        'Rice': {
            'planting_season': ['June-July', 'November-December'],
            'growth_duration': '3-4 months',
            'harvesting_time': ['October-November', 'March-April']
        },
        # Add more crops...
    }
    
    default_calendar = {
        'planting_season': ['Spring', 'Fall'],
        'growth_duration': '3-6 months',
        'harvesting_time': ['Summer', 'Winter']
    }
    
    return CROP_CALENDAR.get(crop, default_calendar) 