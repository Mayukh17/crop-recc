#!/usr/bin/env python3
"""
Weather Integration Module
========================

This module provides real-time weather data integration using OpenWeatherMap API
and historical weather data analysis.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    cloud_cover: float
    timestamp: datetime

class WeatherService:
    """
    Provides weather data from OpenWeatherMap API and historical analysis.
    """
    
    def __init__(self, api_key: str, cache_dir: str = "weather_cache"):
        """
        Initialize weather service.
        
        Args:
            api_key: OpenWeatherMap API key
            cache_dir: Directory to store cached weather data
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_current_weather(self, lat: float, lon: float) -> WeatherData:
        """
        Get current weather data for location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            WeatherData object with current conditions
        """
        endpoint = f"{self.base_url}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            return WeatherData(
                temperature=data["main"]["temp"],
                humidity=data["main"]["humidity"],
                rainfall=data["rain"]["1h"] if "rain" in data else 0,
                wind_speed=data["wind"]["speed"],
                cloud_cover=data["clouds"]["all"],
                timestamp=datetime.fromtimestamp(data["dt"])
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching current weather: {str(e)}")
            raise
    
    def get_forecast(self, lat: float, lon: float, days: int = 7) -> List[WeatherData]:
        """
        Get weather forecast for specified number of days.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days to forecast
            
        Returns:
            List of WeatherData objects for each day
        """
        endpoint = f"{self.base_url}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data["list"][:days * 8]:  # 8 readings per day
                forecasts.append(WeatherData(
                    temperature=item["main"]["temp"],
                    humidity=item["main"]["humidity"],
                    rainfall=item["rain"]["3h"] if "rain" in item else 0,
                    wind_speed=item["wind"]["speed"],
                    cloud_cover=item["clouds"]["all"],
                    timestamp=datetime.fromtimestamp(item["dt"])
                ))
            
            return forecasts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching forecast: {str(e)}")
            raise
    
    def cache_weather_data(self, location: str, weather_data: WeatherData):
        """
        Cache weather data for historical analysis.
        """
        cache_file = self.cache_dir / f"{location}_weather.json"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Add new data
            data.append({
                "temperature": weather_data.temperature,
                "humidity": weather_data.humidity,
                "rainfall": weather_data.rainfall,
                "wind_speed": weather_data.wind_speed,
                "cloud_cover": weather_data.cloud_cover,
                "timestamp": weather_data.timestamp.isoformat()
            })
            
            # Save updated data
            with open(cache_file, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.error(f"Error caching weather data: {str(e)}")
    
    def get_historical_data(self, location: str, 
                          start_date: datetime, 
                          end_date: datetime) -> List[WeatherData]:
        """
        Get historical weather data from cache.
        """
        cache_file = self.cache_dir / f"{location}_weather.json"
        
        try:
            if not cache_file.exists():
                return []
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Filter and convert data
            historical_data = []
            for item in data:
                timestamp = datetime.fromisoformat(item["timestamp"])
                if start_date <= timestamp <= end_date:
                    historical_data.append(WeatherData(
                        temperature=item["temperature"],
                        humidity=item["humidity"],
                        rainfall=item["rainfall"],
                        wind_speed=item["wind_speed"],
                        cloud_cover=item["cloud_cover"],
                        timestamp=timestamp
                    ))
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error reading historical data: {str(e)}")
            return []
    
    def analyze_growing_season(self, location: str, 
                             min_temp: float, 
                             max_temp: float) -> Tuple[datetime, datetime]:
        """
        Analyze historical data to determine optimal growing season.
        
        Args:
            location: Location name
            min_temp: Minimum required temperature
            max_temp: Maximum acceptable temperature
            
        Returns:
            Tuple of start and end dates for optimal growing season
        """
        historical_data = self.get_historical_data(
            location,
            datetime.now() - timedelta(days=365),
            datetime.now()
        )
        
        if not historical_data:
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            "temperature": d.temperature,
            "timestamp": d.timestamp,
            "month": d.timestamp.month
        } for d in historical_data])
        
        # Calculate monthly averages
        monthly_avg = df.groupby("month")["temperature"].mean()
        
        # Find consecutive months within temperature range
        suitable_months = monthly_avg[(monthly_avg >= min_temp) & 
                                   (monthly_avg <= max_temp)].index.tolist()
        
        if not suitable_months:
            return None
        
        # Find longest sequence of suitable months
        sequences = []
        current_seq = [suitable_months[0]]
        
        for i in range(1, len(suitable_months)):
            if suitable_months[i] == suitable_months[i-1] + 1:
                current_seq.append(suitable_months[i])
            else:
                sequences.append(current_seq)
                current_seq = [suitable_months[i]]
        sequences.append(current_seq)
        
        best_sequence = max(sequences, key=len)
        
        # Convert to dates
        current_year = datetime.now().year
        start_date = datetime(current_year, best_sequence[0], 1)
        end_date = datetime(current_year, best_sequence[-1], 
                          28)  # Using 28 to be safe for all months
        
        return start_date, end_date

def main():
    """
    Example usage of weather integration.
    """
    # Initialize weather service with API key
    api_key = "your_api_key_here"  # Replace with actual API key
    weather_service = WeatherService(api_key)
    
    try:
        # Example coordinates (New York City)
        lat, lon = 40.7128, -74.0060
        
        # Get current weather
        current = weather_service.get_current_weather(lat, lon)
        print("\nCurrent Weather:")
        print(f"Temperature: {current.temperature}°C")
        print(f"Humidity: {current.humidity}%")
        print(f"Rainfall: {current.rainfall}mm")
        
        # Get forecast
        forecast = weather_service.get_forecast(lat, lon, days=5)
        print("\nForecast:")
        for day in forecast:
            print(f"{day.timestamp.date()}: {day.temperature}°C, "
                  f"Rain: {day.rainfall}mm")
        
        # Cache the data
        weather_service.cache_weather_data("NYC", current)
        
        # Analyze growing season
        season = weather_service.analyze_growing_season("NYC", 15, 30)
        if season:
            start, end = season
            print(f"\nOptimal Growing Season:")
            print(f"Start: {start.strftime('%B %d')}")
            print(f"End: {end.strftime('%B %d')}")
        
    except Exception as e:
        logger.error(f"Error in weather analysis: {str(e)}")

if __name__ == "__main__":
    main() 