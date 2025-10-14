from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from app.models.schemas import (
    HazardAreaCreate, HazardAreaResponse, PredictionRequest,
    PredictionResponse, UserFeedbackCreate
)
from app.database import get_supabase
from app.config import get_settings
from supabase import Client
import json

router = APIRouter(prefix="/api/v1/hazards", tags=["hazards"])

@router.get("/", response_model=List[HazardAreaResponse])
async def get_hazard_areas(
    min_risk: Optional[float] = 0.0,
    max_risk: Optional[float] = 1.0,
    limit: int = 100,
    supabase: Client = Depends(get_supabase)
):
    try:
        query = supabase.table("hazard_areas").select("*")

        if min_risk > 0:
            query = query.gte("risk_score", min_risk)
        if max_risk < 1:
            query = query.lte("risk_score", max_risk)

        result = query.order("risk_score", desc=True).limit(limit).execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{hazard_id}", response_model=HazardAreaResponse)
async def get_hazard_area(
    hazard_id: str,
    supabase: Client = Depends(get_supabase)
):
    try:
        result = supabase.table("hazard_areas").select("*").eq("id", hazard_id).maybeSingle().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Hazard area not found")

        return result.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictionResponse)
async def predict_hazard(
    request: PredictionRequest,
    settings = Depends(get_settings)
):
    try:
        import torch
        import numpy as np
        from app.ml.model import TerrainHazardModel, extract_features
        from app.ml.model_persistence import get_model_manager

        model_manager = get_model_manager()
        model = model_manager.get_current_model()
        if model is None:
            model = TerrainHazardModel()

        features_dict = request.features or {}
        features = extract_features(features_dict)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        result = model.predict_with_uncertainty(features_tensor, n_samples=50)

        return PredictionResponse(
            risk_score=float(result['risk_score'][0]),
            uncertainty=float(result['uncertainty'][0]),
            confidence_interval=[
                float(result['confidence_interval'][0][0]),
                float(result['confidence_interval'][1][0])
            ],
            model_version=model_manager.get_model_version()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/uncertainty-map")
async def get_uncertainty_map(
    min_uncertainty: float = 0.5,
    limit: int = 100,
    supabase: Client = Depends(get_supabase)
):
    try:
        result = supabase.table("hazard_areas")\
            .select("id, geometry, uncertainty, risk_score")\
            .gte("uncertainty", min_uncertainty)\
            .order("uncertainty", desc=True)\
            .limit(limit)\
            .execute()

        return {"areas": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(
    feedback: UserFeedbackCreate,
    supabase: Client = Depends(get_supabase)
):
    try:
        feedback_dict = feedback.model_dump()

        if feedback.geometry:
            feedback_dict['geometry'] = json.dumps(feedback.geometry.model_dump())

        result = supabase.table("user_feedback").insert(feedback_dict).execute()

        return {"message": "Feedback submitted successfully", "id": result.data[0]['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/high-priority")
async def get_high_priority_areas(
    threshold: float = 0.7,
    limit: int = 50,
    supabase: Client = Depends(get_supabase)
):
    try:
        result = supabase.table("hazard_areas")\
            .select("*")\
            .gte("risk_score", threshold)\
            .order("priority_rank")\
            .limit(limit)\
            .execute()

        return {"areas": result.data, "count": len(result.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
