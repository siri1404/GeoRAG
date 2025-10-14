from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum

class HazardLabel(str, Enum):
    SAFE = "safe"
    HAZARDOUS = "hazardous"
    UNKNOWN = "unknown"
    SUSPECTED = "suspected"

class MissionStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class FeedbackType(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    CORRECTION = "correction"

class QueryStrategy(str, Enum):
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    EXPECTED_CHANGE = "expected_change"
    HYBRID = "hybrid"
    IMPACT = "impact"

class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: List[Any]

class HazardAreaCreate(BaseModel):
    geometry: GeoJSONGeometry
    risk_score: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    model_version: str
    features: Optional[Dict[str, Any]] = {}
    priority_rank: Optional[int] = None

class HazardAreaResponse(BaseModel):
    id: str
    geometry: Dict[str, Any]
    risk_score: float
    uncertainty: float
    prediction_timestamp: datetime
    model_version: str
    features: Dict[str, Any]
    priority_rank: Optional[int]
    created_at: datetime

class GroundTruthLabelCreate(BaseModel):
    geometry: GeoJSONGeometry
    label: HazardLabel
    confidence: float = Field(default=1.0, ge=0, le=1)
    labeled_by: str
    verification_method: Optional[str] = None
    notes: Optional[str] = None
    mission_id: Optional[str] = None

class GroundTruthLabelResponse(BaseModel):
    id: str
    geometry: Dict[str, Any]
    label: HazardLabel
    confidence: float
    labeled_by: str
    labeled_at: datetime
    verification_method: Optional[str]
    notes: Optional[str]

class MissionCreate(BaseModel):
    mission_name: str
    mission_date: date
    team_id: str
    area_geometry: Optional[GeoJSONGeometry] = None
    priority_score: Optional[float] = Field(default=None, ge=0, le=1)
    resources_allocated: Optional[Dict[str, Any]] = {}
    expected_duration_hours: Optional[float] = None
    notes: Optional[str] = None

class MissionUpdate(BaseModel):
    status: Optional[MissionStatus] = None
    actual_duration_hours: Optional[float] = None
    areas_cleared: Optional[int] = None
    hazards_found: Optional[int] = None
    notes: Optional[str] = None

class MissionResponse(BaseModel):
    id: str
    mission_name: str
    mission_date: date
    team_id: str
    status: MissionStatus
    priority_score: Optional[float]
    resources_allocated: Dict[str, Any]
    areas_cleared: int
    hazards_found: int
    created_at: datetime

class UserFeedbackCreate(BaseModel):
    hazard_area_id: str
    feedback_type: FeedbackType
    corrected_label: Optional[HazardLabel] = None
    corrected_risk_score: Optional[float] = Field(default=None, ge=0, le=1)
    user_id: str
    comments: Optional[str] = None
    geometry: Optional[GeoJSONGeometry] = None

class ActiveLearningQuery(BaseModel):
    geometry: GeoJSONGeometry
    query_strategy: QueryStrategy
    selection_score: float
    expected_info_gain: Optional[float] = None
    uncertainty_score: Optional[float] = None
    diversity_score: Optional[float] = None
    impact_score: Optional[float] = None

class ModelExplanation(BaseModel):
    hazard_area_id: str
    feature_importance: Dict[str, float]
    top_features: List[Dict[str, Any]]
    explanation_text: str
    similar_areas: List[str]
    confidence_factors: Dict[str, Any]

class PredictionRequest(BaseModel):
    geometry: GeoJSONGeometry
    features: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    risk_score: float
    uncertainty: float
    confidence_interval: List[float]
    model_version: str

class PrioritizationRequest(BaseModel):
    area_geometries: List[GeoJSONGeometry]
    constraints: Optional[Dict[str, Any]] = {}
    weights: Optional[Dict[str, float]] = {
        "risk": 0.3,
        "uncertainty": 0.2,
        "population_impact": 0.3,
        "accessibility": 0.1,
        "cost": 0.1
    }

class PrioritizationResponse(BaseModel):
    ranked_areas: List[Dict[str, Any]]
    optimal_sequence: List[int]
    expected_impact: Dict[str, float]
