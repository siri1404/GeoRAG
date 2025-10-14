from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.schemas import (
    MissionCreate, MissionUpdate, MissionResponse,
    PrioritizationRequest, PrioritizationResponse
)
from app.database import get_supabase
from supabase import Client
import json
import numpy as np

router = APIRouter(prefix="/api/v1/missions", tags=["missions"])

@router.post("/", response_model=MissionResponse)
async def create_mission(
    mission: MissionCreate,
    supabase: Client = Depends(get_supabase)
):
    try:
        mission_dict = mission.model_dump()

        if mission.area_geometry:
            mission_dict['area_geometry'] = json.dumps(mission.area_geometry.model_dump())

        result = supabase.table("missions").insert(mission_dict).execute()

        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[MissionResponse])
async def get_missions(
    status: str = None,
    limit: int = 50,
    supabase: Client = Depends(get_supabase)
):
    try:
        query = supabase.table("missions").select("*")

        if status:
            query = query.eq("status", status)

        result = query.order("mission_date", desc=True).limit(limit).execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{mission_id}", response_model=MissionResponse)
async def get_mission(
    mission_id: str,
    supabase: Client = Depends(get_supabase)
):
    try:
        result = supabase.table("missions").select("*").eq("id", mission_id).maybeSingle().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Mission not found")

        return result.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{mission_id}", response_model=MissionResponse)
async def update_mission(
    mission_id: str,
    update: MissionUpdate,
    supabase: Client = Depends(get_supabase)
):
    try:
        update_dict = {k: v for k, v in update.model_dump().items() if v is not None}

        result = supabase.table("missions")\
            .update(update_dict)\
            .eq("id", mission_id)\
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Mission not found")

        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plan")
async def plan_mission(
    team_id: str,
    max_areas: int = 5,
    supabase: Client = Depends(get_supabase)
):
    try:
        high_priority_areas = supabase.table("hazard_areas")\
            .select("*")\
            .gte("risk_score", 0.6)\
            .order("priority_rank")\
            .limit(max_areas * 2)\
            .execute()

        if not high_priority_areas.data:
            return {"message": "No high-priority areas found", "areas": []}

        selected_areas = high_priority_areas.data[:max_areas]

        total_risk = sum(area['risk_score'] for area in selected_areas)
        avg_uncertainty = sum(area['uncertainty'] for area in selected_areas) / len(selected_areas)

        return {
            "recommended_areas": selected_areas,
            "count": len(selected_areas),
            "total_risk_score": total_risk,
            "average_uncertainty": avg_uncertainty,
            "team_id": team_id,
            "estimated_duration_hours": len(selected_areas) * 4
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prioritize", response_model=PrioritizationResponse)
async def prioritize_areas(
    request: PrioritizationRequest,
    supabase: Client = Depends(get_supabase)
):
    try:
        from app.services.prioritization import PrioritizationService

        prioritization_service = PrioritizationService()

        result = prioritization_service.prioritize_areas(
            area_geometries=request.area_geometries,
            constraints=request.constraints,
            weights=request.weights
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_mission_statistics(
    supabase: Client = Depends(get_supabase)
):
    try:
        all_missions = supabase.table("missions").select("*").execute()

        total_missions = len(all_missions.data)
        completed = len([m for m in all_missions.data if m['status'] == 'completed'])
        in_progress = len([m for m in all_missions.data if m['status'] == 'in_progress'])
        planned = len([m for m in all_missions.data if m['status'] == 'planned'])

        total_areas_cleared = sum(m.get('areas_cleared', 0) for m in all_missions.data)
        total_hazards_found = sum(m.get('hazards_found', 0) for m in all_missions.data)

        avg_duration = np.mean([
            m.get('actual_duration_hours', 0)
            for m in all_missions.data
            if m.get('actual_duration_hours') and m.get('actual_duration_hours') > 0
        ]) if all_missions.data else 0

        return {
            "total_missions": total_missions,
            "completed": completed,
            "in_progress": in_progress,
            "planned": planned,
            "total_areas_cleared": total_areas_cleared,
            "total_hazards_found": total_hazards_found,
            "average_duration_hours": float(avg_duration),
            "detection_rate": (total_hazards_found / max(total_areas_cleared, 1)) * 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
