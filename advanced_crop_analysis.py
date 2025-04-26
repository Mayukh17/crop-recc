#!/usr/bin/env python3
"""
Advanced Crop Analysis Module
===========================

This module provides sophisticated soil analysis and crop-specific features:
- Detailed soil composition analysis
- Crop growth stage prediction
- Nutrient deficiency detection
- Crop yield estimation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class SoilType(Enum):
    CLAY = "Clay"
    SANDY = "Sandy"
    LOAMY = "Loamy"
    SILTY = "Silty"
    PEATY = "Peaty"
    CHALKY = "Chalky"
    
class NutrientLevel(Enum):
    DEFICIENT = "Deficient"
    LOW = "Low"
    OPTIMAL = "Optimal"
    HIGH = "High"
    EXCESSIVE = "Excessive"

@dataclass
class SoilAnalysis:
    soil_type: SoilType
    ph_level: float
    organic_matter: float
    nutrient_levels: Dict[str, NutrientLevel]
    water_retention: float
    drainage_rate: float

@dataclass
class CropGrowthStage:
    stage: str
    days_from_planting: int
    expected_duration: int
    required_conditions: Dict[str, Tuple[float, float]]

class AdvancedSoilAnalyzer:
    """
    Provides detailed soil analysis and recommendations.
    """
    
    def __init__(self):
        self.soil_type_ranges = {
            SoilType.CLAY: {"sand": (0, 45), "silt": (0, 45), "clay": (40, 100)},
            SoilType.SANDY: {"sand": (80, 100), "silt": (0, 20), "clay": (0, 20)},
            SoilType.LOAMY: {"sand": (25, 50), "silt": (25, 50), "clay": (10, 25)},
            SoilType.SILTY: {"sand": (0, 20), "silt": (80, 100), "clay": (0, 20)},
            SoilType.PEATY: {"organic_matter": (30, 100)},
            SoilType.CHALKY: {"calcium_carbonate": (30, 100)}
        }
        
    def determine_soil_type(self, sand: float, silt: float, clay: float, 
                          organic_matter: float = 0, calcium_carbonate: float = 0) -> SoilType:
        """
        Determine soil type based on composition percentages.
        """
        if organic_matter > 30:
            return SoilType.PEATY
        if calcium_carbonate > 30:
            return SoilType.CHALKY
            
        compositions = {
            SoilType.CLAY: clay,
            SoilType.SANDY: sand,
            SoilType.LOAMY: min(sand, silt),
            SoilType.SILTY: silt
        }
        return max(compositions.items(), key=lambda x: x[1])[0]
    
    def analyze_nutrient_levels(self, n: float, p: float, k: float) -> Dict[str, NutrientLevel]:
        """
        Analyze NPK levels and determine deficiencies or excesses.
        """
        def get_level(value: float, optimal_range: Tuple[float, float]) -> NutrientLevel:
            if value < optimal_range[0] * 0.5:
                return NutrientLevel.DEFICIENT
            elif value < optimal_range[0]:
                return NutrientLevel.LOW
            elif value <= optimal_range[1]:
                return NutrientLevel.OPTIMAL
            elif value <= optimal_range[1] * 1.5:
                return NutrientLevel.HIGH
            else:
                return NutrientLevel.EXCESSIVE
        
        return {
            "N": get_level(n, (40, 80)),
            "P": get_level(p, (30, 60)),
            "K": get_level(k, (20, 50))
        }
    
    def calculate_water_retention(self, soil_type: SoilType, organic_matter: float) -> float:
        """
        Calculate soil's water retention capacity.
        """
        base_retention = {
            SoilType.CLAY: 0.8,
            SoilType.SANDY: 0.3,
            SoilType.LOAMY: 0.6,
            SoilType.SILTY: 0.7,
            SoilType.PEATY: 0.9,
            SoilType.CHALKY: 0.4
        }
        
        # Organic matter improves water retention
        organic_matter_factor = 1 + (organic_matter / 100)
        return min(1.0, base_retention[soil_type] * organic_matter_factor)
    
    def analyze_soil(self, n: float, p: float, k: float, ph: float, 
                    sand_pct: float, silt_pct: float, clay_pct: float,
                    organic_matter: float = 5.0, calcium_carbonate: float = 0.0) -> SoilAnalysis:
        """
        Perform comprehensive soil analysis.
        """
        soil_type = self.determine_soil_type(sand_pct, silt_pct, clay_pct, 
                                           organic_matter, calcium_carbonate)
        nutrient_levels = self.analyze_nutrient_levels(n, p, k)
        water_retention = self.calculate_water_retention(soil_type, organic_matter)
        drainage_rate = 1 - water_retention  # Simplified inverse relationship
        
        return SoilAnalysis(
            soil_type=soil_type,
            ph_level=ph,
            organic_matter=organic_matter,
            nutrient_levels=nutrient_levels,
            water_retention=water_retention,
            drainage_rate=drainage_rate
        )

class CropLifecycleManager:
    """
    Manages crop-specific growth stages and requirements.
    """
    
    def __init__(self):
        self.growth_stages = {
            "rice": [
                CropGrowthStage("Germination", 0, 5, {
                    "temperature": (20, 35),
                    "humidity": (80, 90)
                }),
                CropGrowthStage("Seedling", 5, 15, {
                    "temperature": (20, 30),
                    "humidity": (70, 85)
                }),
                CropGrowthStage("Tillering", 15, 40, {
                    "temperature": (25, 31),
                    "humidity": (60, 80)
                }),
                CropGrowthStage("Flowering", 40, 70, {
                    "temperature": (22, 29),
                    "humidity": (70, 80)
                }),
                CropGrowthStage("Maturity", 70, 110, {
                    "temperature": (20, 25),
                    "humidity": (60, 75)
                })
            ]
            # Add more crops here
        }
        
    def get_growth_stage(self, crop: str, days_from_planting: int) -> CropGrowthStage:
        """
        Determine current growth stage based on days from planting.
        """
        if crop not in self.growth_stages:
            raise ValueError(f"No growth stage data available for {crop}")
            
        cumulative_days = 0
        for stage in self.growth_stages[crop]:
            cumulative_days += stage.expected_duration
            if days_from_planting <= cumulative_days:
                return stage
                
        return self.growth_stages[crop][-1]  # Return final stage if beyond expected duration
    
    def estimate_yield(self, crop: str, soil_analysis: SoilAnalysis, 
                      temperature: float, rainfall: float) -> float:
        """
        Estimate crop yield based on conditions.
        """
        # Base yield potential (tons per hectare)
        base_yields = {
            "rice": 4.5,
            "wheat": 3.0,
            "maize": 5.5,
            "potato": 25.0,
            "cotton": 2.5
        }
        
        if crop not in base_yields:
            raise ValueError(f"No yield data available for {crop}")
        
        # Calculate modifiers based on conditions
        soil_modifier = self._calculate_soil_modifier(soil_analysis)
        weather_modifier = self._calculate_weather_modifier(crop, temperature, rainfall)
        nutrient_modifier = self._calculate_nutrient_modifier(soil_analysis.nutrient_levels)
        
        return base_yields[crop] * soil_modifier * weather_modifier * nutrient_modifier
    
    def _calculate_soil_modifier(self, soil_analysis: SoilAnalysis) -> float:
        """Calculate yield modifier based on soil conditions."""
        ph_optimal = 0.5 + min(1.0, 1 - abs(soil_analysis.ph_level - 6.5) / 2)
        water_optimal = 0.5 + min(1.0, soil_analysis.water_retention)
        return (ph_optimal + water_optimal) / 2
    
    def _calculate_weather_modifier(self, crop: str, temperature: float, rainfall: float) -> float:
        """Calculate yield modifier based on weather conditions."""
        optimal_conditions = {
            "rice": {"temp": (25, 30), "rain": (200, 300)},
            "wheat": {"temp": (15, 20), "rain": (100, 150)},
            "maize": {"temp": (20, 25), "rain": (150, 200)}
        }
        
        if crop not in optimal_conditions:
            return 1.0
            
        temp_range = optimal_conditions[crop]["temp"]
        rain_range = optimal_conditions[crop]["rain"]
        
        temp_modifier = 0.5 + min(1.0, 1 - abs(temperature - sum(temp_range)/2) / (temp_range[1] - temp_range[0]))
        rain_modifier = 0.5 + min(1.0, 1 - abs(rainfall - sum(rain_range)/2) / (rain_range[1] - rain_range[0]))
        
        return (temp_modifier + rain_modifier) / 2
    
    def _calculate_nutrient_modifier(self, nutrient_levels: Dict[str, NutrientLevel]) -> float:
        """Calculate yield modifier based on nutrient levels."""
        modifiers = {
            NutrientLevel.DEFICIENT: 0.4,
            NutrientLevel.LOW: 0.7,
            NutrientLevel.OPTIMAL: 1.0,
            NutrientLevel.HIGH: 0.8,
            NutrientLevel.EXCESSIVE: 0.6
        }
        
        return sum(modifiers[level] for level in nutrient_levels.values()) / len(nutrient_levels)

def main():
    """
    Example usage of advanced crop analysis.
    """
    # Initialize analyzers
    soil_analyzer = AdvancedSoilAnalyzer()
    lifecycle_manager = CropLifecycleManager()
    
    # Example soil analysis
    soil_analysis = soil_analyzer.analyze_soil(
        n=60, p=45, k=30,
        ph=6.5,
        sand_pct=40,
        silt_pct=40,
        clay_pct=20,
        organic_matter=5.0
    )
    
    print("\nSoil Analysis Results:")
    print(f"Soil Type: {soil_analysis.soil_type.value}")
    print(f"pH Level: {soil_analysis.ph_level}")
    print("Nutrient Levels:")
    for nutrient, level in soil_analysis.nutrient_levels.items():
        print(f"  {nutrient}: {level.value}")
    print(f"Water Retention: {soil_analysis.water_retention:.2f}")
    print(f"Drainage Rate: {soil_analysis.drainage_rate:.2f}")
    
    # Example crop lifecycle analysis
    crop = "rice"
    days = 45
    stage = lifecycle_manager.get_growth_stage(crop, days)
    
    print(f"\nCrop Growth Analysis for {crop} at {days} days:")
    print(f"Current Stage: {stage.stage}")
    print("Required Conditions:")
    for condition, (min_val, max_val) in stage.required_conditions.items():
        print(f"  {condition}: {min_val}-{max_val}")
    
    # Example yield estimation
    estimated_yield = lifecycle_manager.estimate_yield(
        crop=crop,
        soil_analysis=soil_analysis,
        temperature=28,
        rainfall=250
    )
    
    print(f"\nEstimated Yield: {estimated_yield:.2f} tons/hectare")

if __name__ == "__main__":
    main() 