"""
Domain expertise module for demining applications.
Contains specialized knowledge about mine types, clearance methods, and operational constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MineType(Enum):
    """Types of landmines and explosive hazards."""
    ANTI_PERSONNEL = "AP"
    ANTI_TANK = "AT"
    UNEXPLODED_ORDNANCE = "UXO"
    IMPROVISED_EXPLOSIVE_DEVICE = "IED"

class ClearanceMethod(Enum):
    """Methods for mine clearance."""
    MANUAL_DETECTION = "manual"
    METAL_DETECTOR = "metal_detector"
    DEMINING_DOG = "dog"
    MECHANICAL_CLEARANCE = "mechanical"
    EXPLOSIVE_DETONATION = "explosive"

class DeminingExpertise:
    """
    Domain expertise for demining operations.
    Contains specialized knowledge about mine characteristics, clearance methods, and operational constraints.
    """
    
    def __init__(self):
        self.mine_characteristics = self._initialize_mine_characteristics()
        self.clearance_constraints = self._initialize_clearance_constraints()
        self.operational_factors = self._initialize_operational_factors()
    
    def _initialize_mine_characteristics(self) -> Dict:
        """Initialize mine characteristics database."""
        return {
            MineType.ANTI_PERSONNEL: {
                "detection_difficulty": 0.7,  # 0-1 scale
                "metal_content": 0.3,  # Low metal content
                "size_range": (0.05, 0.15),  # Diameter in meters
                "depth_range": (0.02, 0.15),  # Burial depth in meters
                "trigger_pressure": 5,  # kg
                "blast_radius": 5,  # meters
                "common_materials": ["plastic", "wood", "minimal_metal"],
                "detection_methods": [ClearanceMethod.MANUAL_DETECTION, ClearanceMethod.DEMINING_DOG]
            },
            MineType.ANTI_TANK: {
                "detection_difficulty": 0.4,  # Easier to detect
                "metal_content": 0.8,  # High metal content
                "size_range": (0.2, 0.4),  # Larger diameter
                "depth_range": (0.1, 0.3),  # Deeper burial
                "trigger_pressure": 100,  # kg
                "blast_radius": 20,  # meters
                "common_materials": ["metal", "steel"],
                "detection_methods": [ClearanceMethod.METAL_DETECTOR, ClearanceMethod.MECHANICAL_CLEARANCE]
            },
            MineType.UNEXPLODED_ORDNANCE: {
                "detection_difficulty": 0.6,
                "metal_content": 0.9,  # Very high metal content
                "size_range": (0.1, 1.0),  # Variable size
                "depth_range": (0.05, 2.0),  # Variable depth
                "trigger_pressure": 50,  # kg
                "blast_radius": 50,  # meters
                "common_materials": ["metal", "steel", "explosive"],
                "detection_methods": [ClearanceMethod.METAL_DETECTOR, ClearanceMethod.MECHANICAL_CLEARANCE]
            }
        }
    
    def _initialize_clearance_constraints(self) -> Dict:
        """Initialize clearance operational constraints."""
        return {
            "weather_constraints": {
                "rain_threshold": 10,  # mm/hour
                "wind_threshold": 25,  # km/h
                "temperature_range": (-10, 45),  # Celsius
                "visibility_threshold": 100  # meters
            },
            "terrain_constraints": {
                "max_slope": 30,  # degrees
                "vegetation_density_threshold": 0.8,
                "rocky_terrain_penalty": 0.3,
                "wet_soil_penalty": 0.4
            },
            "safety_constraints": {
                "min_team_size": 3,
                "max_team_size": 12,
                "experience_requirement": 1,  # years
                "safety_distance": 50,  # meters between team members
                "equipment_maintenance_interval": 7  # days
            }
        }
    
    def _initialize_operational_factors(self) -> Dict:
        """Initialize operational factors affecting clearance."""
        return {
            "seasonal_factors": {
                "winter": {"efficiency": 0.7, "safety_risk": 0.3},
                "spring": {"efficiency": 0.9, "safety_risk": 0.1},
                "summer": {"efficiency": 0.8, "safety_risk": 0.2},
                "autumn": {"efficiency": 0.85, "safety_risk": 0.15}
            },
            "time_factors": {
                "morning_efficiency": 0.9,
                "afternoon_efficiency": 0.8,
                "evening_efficiency": 0.6,
                "night_operations": False  # Generally not allowed
            },
            "equipment_factors": {
                "metal_detector_accuracy": 0.85,
                "dog_detection_accuracy": 0.92,
                "manual_detection_accuracy": 0.95,
                "mechanical_clearance_efficiency": 0.8
            }
        }
    
    def calculate_detection_probability(self, mine_type: MineType, terrain_features: Dict, 
                                     clearance_method: ClearanceMethod) -> float:
        """
        Calculate probability of detecting a mine based on domain expertise.
        
        Args:
            mine_type: Type of mine
            terrain_features: Terrain characteristics
            clearance_method: Method used for clearance
            
        Returns:
            Detection probability (0-1)
        """
        mine_char = self.mine_characteristics[mine_type]
        
        # Base detection probability
        base_prob = 1 - mine_char["detection_difficulty"]
        
        # Terrain factors
        slope_factor = max(0.5, 1 - terrain_features.get("slope", 0) / 45)
        vegetation_factor = max(0.3, 1 - terrain_features.get("vegetation_density", 0))
        soil_factor = 1 - terrain_features.get("soil_moisture", 0) * 0.3
        
        # Method-specific factors
        method_accuracy = self.operational_factors["equipment_factors"].get(
            f"{clearance_method.value}_accuracy", 0.8
        )
        
        # Weather factors
        weather_factor = 1 - terrain_features.get("precipitation", 0) * 0.2
        
        # Calculate final probability
        detection_prob = (base_prob * slope_factor * vegetation_factor * 
                         soil_factor * method_accuracy * weather_factor)
        
        return min(1.0, max(0.1, detection_prob))
    
    def calculate_clearance_efficiency(self, terrain_features: Dict, team_features: Dict,
                                     environmental_features: Dict) -> float:
        """
        Calculate clearance efficiency based on operational factors.
        
        Args:
            terrain_features: Terrain characteristics
            team_features: Team characteristics
            environmental_features: Environmental conditions
            
        Returns:
            Clearance efficiency (0-1)
        """
        # Base efficiency
        base_efficiency = 0.8
        
        # Team factors
        team_size = team_features.get("team_size", 6)
        team_experience = team_features.get("experience_years", 2)
        team_training = team_features.get("training_level", "intermediate")
        
        # Optimal team size is 6-8 people
        size_factor = 1 - abs(team_size - 7) * 0.05
        experience_factor = min(1.0, team_experience / 5)
        training_factor = {"basic": 0.7, "intermediate": 0.85, "advanced": 1.0}.get(team_training, 0.85)
        
        # Terrain factors
        slope = terrain_features.get("slope", 0)
        vegetation = terrain_features.get("vegetation_density", 0.5)
        accessibility = terrain_features.get("accessibility_score", 0.5)
        
        slope_factor = max(0.3, 1 - slope / 30)
        vegetation_factor = max(0.4, 1 - vegetation * 0.3)
        accessibility_factor = accessibility
        
        # Environmental factors
        weather = environmental_features.get("weather_condition", "good")
        season = environmental_features.get("season", "spring")
        
        weather_factor = {"excellent": 1.0, "good": 0.9, "fair": 0.7, "poor": 0.4}.get(weather, 0.7)
        seasonal_factor = self.operational_factors["seasonal_factors"][season]["efficiency"]
        
        # Calculate final efficiency
        efficiency = (base_efficiency * size_factor * experience_factor * training_factor *
                     slope_factor * vegetation_factor * accessibility_factor *
                     weather_factor * seasonal_factor)
        
        return min(1.0, max(0.1, efficiency))
    
    def calculate_safety_risk(self, terrain_features: Dict, team_features: Dict,
                            environmental_features: Dict) -> float:
        """
        Calculate safety risk for clearance operations.
        
        Args:
            terrain_features: Terrain characteristics
            team_features: Team characteristics
            environmental_features: Environmental conditions
            
        Returns:
            Safety risk (0-1, higher is more dangerous)
        """
        # Base risk
        base_risk = 0.1
        
        # Terrain risk factors
        slope = terrain_features.get("slope", 0)
        vegetation = terrain_features.get("vegetation_density", 0.5)
        rocky_terrain = terrain_features.get("rocky_terrain", 0.3)
        
        terrain_risk = (slope / 45) * 0.3 + vegetation * 0.2 + rocky_terrain * 0.3
        
        # Team risk factors
        team_experience = team_features.get("experience_years", 2)
        team_training = team_features.get("training_level", "intermediate")
        safety_record = team_features.get("safety_score", 0.9)
        
        experience_risk = max(0, 0.3 - team_experience * 0.05)
        training_risk = {"basic": 0.2, "intermediate": 0.1, "advanced": 0.05}.get(team_training, 0.1)
        safety_risk = 1 - safety_record
        
        # Environmental risk factors
        weather = environmental_features.get("weather_condition", "good")
        season = environmental_features.get("season", "spring")
        visibility = environmental_features.get("visibility", 1000)
        
        weather_risk = {"excellent": 0.0, "good": 0.05, "fair": 0.15, "poor": 0.3}.get(weather, 0.1)
        seasonal_risk = self.operational_factors["seasonal_factors"][season]["safety_risk"]
        visibility_risk = max(0, 0.2 - visibility / 5000)
        
        # Calculate final risk
        total_risk = (base_risk + terrain_risk + experience_risk + training_risk + 
                     safety_risk + weather_risk + seasonal_risk + visibility_risk)
        
        return min(1.0, max(0.0, total_risk))
    
    def recommend_clearance_method(self, terrain_features: Dict, mine_type: MineType,
                                 team_capabilities: Dict) -> ClearanceMethod:
        """
        Recommend optimal clearance method based on domain expertise.
        
        Args:
            terrain_features: Terrain characteristics
            mine_type: Type of mine expected
            team_capabilities: Team's available methods and experience
            
        Returns:
            Recommended clearance method
        """
        mine_char = self.mine_characteristics[mine_type]
        
        # Method suitability scores
        method_scores = {}
        
        for method in ClearanceMethod:
            score = 0
            
            # Method-specific factors
            if method == ClearanceMethod.METAL_DETECTOR:
                score += mine_char["metal_content"] * 0.8
                score += (1 - terrain_features.get("soil_moisture", 0.5)) * 0.3
                score += (1 - terrain_features.get("vegetation_density", 0.5)) * 0.2
                
            elif method == ClearanceMethod.DEMINING_DOG:
                score += mine_char["detection_difficulty"] * 0.6  # Dogs good for difficult cases
                score += (1 - terrain_features.get("temperature", 20) / 40) * 0.2
                score += (1 - terrain_features.get("wind_speed", 10) / 30) * 0.2
                
            elif method == ClearanceMethod.MANUAL_DETECTION:
                score += mine_char["size_range"][0] * 5  # Larger mines easier to find manually
                score += (1 - terrain_features.get("vegetation_density", 0.5)) * 0.4
                score += terrain_features.get("accessibility_score", 0.5) * 0.3
                
            elif method == ClearanceMethod.MECHANICAL_CLEARANCE:
                score += (1 - mine_char["detection_difficulty"]) * 0.5
                score += (1 - terrain_features.get("slope", 0) / 30) * 0.3
                score += terrain_features.get("accessibility_score", 0.5) * 0.4
            
            # Team capability factors
            if method.value in team_capabilities.get("available_methods", []):
                score += 0.3
            
            if team_capabilities.get("experience_level", "intermediate") == "advanced":
                score += 0.2
            
            method_scores[method] = score
        
        # Return method with highest score
        return max(method_scores, key=method_scores.get)
    
    def estimate_clearance_duration(self, area_size: float, terrain_features: Dict,
                                    team_features: Dict, environmental_features: Dict) -> float:
        """
        Estimate clearance duration based on domain expertise.
        
        Args:
            area_size: Area to clear in hectares
            terrain_features: Terrain characteristics
            team_features: Team characteristics
            environmental_features: Environmental conditions
            
        Returns:
            Estimated duration in days
        """
        # Base clearance rate (hectares per day)
        base_rate = 0.5  # Conservative estimate
        
        # Team efficiency
        team_size = team_features.get("team_size", 6)
        team_experience = team_features.get("experience_years", 2)
        team_training = team_features.get("training_level", "intermediate")
        
        size_factor = min(1.5, team_size / 6)  # Optimal size is 6
        experience_factor = min(1.3, 1 + team_experience * 0.1)
        training_factor = {"basic": 0.7, "intermediate": 1.0, "advanced": 1.2}.get(team_training, 1.0)
        
        # Terrain factors
        slope = terrain_features.get("slope", 0)
        vegetation = terrain_features.get("vegetation_density", 0.5)
        accessibility = terrain_features.get("accessibility_score", 0.5)
        
        slope_factor = max(0.3, 1 - slope / 30)
        vegetation_factor = max(0.4, 1 - vegetation * 0.4)
        accessibility_factor = accessibility
        
        # Environmental factors
        weather = environmental_features.get("weather_condition", "good")
        season = environmental_features.get("season", "spring")
        
        weather_factor = {"excellent": 1.2, "good": 1.0, "fair": 0.8, "poor": 0.5}.get(weather, 1.0)
        seasonal_factor = self.operational_factors["seasonal_factors"][season]["efficiency"]
        
        # Calculate final rate
        clearance_rate = (base_rate * size_factor * experience_factor * training_factor *
                         slope_factor * vegetation_factor * accessibility_factor *
                         weather_factor * seasonal_factor)
        
        # Calculate duration
        duration = area_size / clearance_rate
        
        # Add safety buffer (20% extra time)
        duration *= 1.2
        
        return max(1, duration)  # Minimum 1 day
    
    def calculate_priority_score(self, hazard_features: Dict, population_features: Dict,
                               economic_features: Dict) -> float:
        """
        Calculate priority score for clearance operations.
        
        Args:
            hazard_features: Hazard characteristics
            population_features: Population impact
            economic_features: Economic factors
            
        Returns:
            Priority score (0-1, higher is more urgent)
        """
        # Hazard severity
        hazard_type = hazard_features.get("hazard_type", "AP")
        contamination_level = hazard_features.get("contamination_level", "Medium")
        survey_confidence = hazard_features.get("confidence_level", 0.8)
        
        # Hazard type weights
        type_weights = {"AP": 0.8, "AT": 0.6, "UXO": 0.9, "IED": 1.0}
        type_score = type_weights.get(hazard_type, 0.7)
        
        # Contamination level weights
        contamination_weights = {"Low": 0.3, "Medium": 0.6, "High": 1.0}
        contamination_score = contamination_weights.get(contamination_level, 0.6)
        
        # Population impact
        population_affected = population_features.get("population_affected", 100)
        population_density = population_features.get("population_density", 50)
        
        population_score = min(1.0, (population_affected / 1000) * 0.5 + (population_density / 1000) * 0.5)
        
        # Economic impact
        economic_impact = economic_features.get("economic_impact_usd", 10000)
        agricultural_land = economic_features.get("agricultural_land", 0.3)
        
        economic_score = min(1.0, (economic_impact / 100000) * 0.3 + agricultural_land * 0.7)
        
        # Calculate final priority score
        priority_score = (type_score * 0.3 + contamination_score * 0.2 + 
                         survey_confidence * 0.1 + population_score * 0.2 + 
                         economic_score * 0.2)
        
        return min(1.0, max(0.0, priority_score))
    
    def get_operational_recommendations(self, terrain_features: Dict, team_features: Dict,
                                      environmental_features: Dict) -> Dict:
        """
        Get operational recommendations based on domain expertise.
        
        Args:
            terrain_features: Terrain characteristics
            team_features: Team characteristics
            environmental_features: Environmental conditions
            
        Returns:
            Dictionary of operational recommendations
        """
        recommendations = {
            "safety_warnings": [],
            "efficiency_tips": [],
            "equipment_recommendations": [],
            "timing_recommendations": []
        }
        
        # Safety warnings
        if terrain_features.get("slope", 0) > 20:
            recommendations["safety_warnings"].append("High slope area - use additional safety measures")
        
        if terrain_features.get("vegetation_density", 0) > 0.7:
            recommendations["safety_warnings"].append("Dense vegetation - limited visibility, proceed with caution")
        
        if environmental_features.get("weather_condition", "good") == "poor":
            recommendations["safety_warnings"].append("Poor weather conditions - consider postponing operations")
        
        # Efficiency tips
        if team_features.get("team_size", 6) < 6:
            recommendations["efficiency_tips"].append("Consider increasing team size for better efficiency")
        
        if terrain_features.get("accessibility_score", 0.5) < 0.3:
            recommendations["efficiency_tips"].append("Low accessibility - plan for additional time and resources")
        
        # Equipment recommendations
        if terrain_features.get("soil_moisture", 0.5) > 0.7:
            recommendations["equipment_recommendations"].append("Wet soil conditions - use waterproof equipment")
        
        if terrain_features.get("rocky_terrain", 0.3) > 0.5:
            recommendations["equipment_recommendations"].append("Rocky terrain - use heavy-duty equipment")
        
        # Timing recommendations
        season = environmental_features.get("season", "spring")
        if season == "winter":
            recommendations["timing_recommendations"].append("Winter operations - plan for shorter working days")
        elif season == "summer":
            recommendations["timing_recommendations"].append("Summer operations - avoid midday heat, work early morning/evening")
        
        return recommendations
