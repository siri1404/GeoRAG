from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.schemas import (
    GroundTruthLabelCreate, GroundTruthLabelResponse,
    ActiveLearningQuery, QueryStrategy
)
from app.database import get_supabase
from supabase import Client
import json
import numpy as np

router = APIRouter(prefix="/api/v1/active-learning", tags=["active-learning"])

@router.get("/next-areas")
async def get_next_areas_to_label(
    strategy: QueryStrategy = QueryStrategy.HYBRID,
    n_queries: int = 5,
    supabase: Client = Depends(get_supabase)
):
    try:
        unlabeled_areas = supabase.table("active_learning_queries")\
            .select("*")\
            .eq("labeled", False)\
            .execute()

        if len(unlabeled_areas.data) == 0:
            hazard_areas = supabase.table("hazard_areas")\
                .select("*")\
                .order("uncertainty", desc=True)\
                .limit(n_queries * 2)\
                .execute()

            from app.ml.active_learning import ActiveLearningStrategy

            al_strategy = ActiveLearningStrategy(strategy=strategy.value)

            candidates = hazard_areas.data
            predictions = np.array([area['risk_score'] for area in candidates])
            uncertainties = np.array([area['uncertainty'] for area in candidates])

            features_list = []
            for area in candidates:
                features = area.get('features', {})
                feature_vector = [
                    features.get('elevation', 0),
                    features.get('slope', 0),
                    features.get('proximity_to_conflict', 0),
                    features.get('population_density', 0)
                ]
                features_list.append(feature_vector)
            features = np.array(features_list)

            selected_indices = al_strategy.select_queries(
                candidates, predictions, uncertainties, features, n_queries
            )

            recommended_areas = [candidates[i] for i in selected_indices]

            for idx, area in zip(selected_indices, recommended_areas):
                query_data = {
                    'geometry': json.dumps(area['geometry']),
                    'query_strategy': strategy.value,
                    'selection_score': float(uncertainties[idx]),
                    'expected_info_gain': float(al_strategy.compute_expected_info_gain(uncertainties[idx])),
                    'uncertainty_score': float(uncertainties[idx]),
                    'impact_score': float(predictions[idx]),
                    'model_version': area.get('model_version', 'v1.0.0'),
                    'labeled': False
                }

                supabase.table("active_learning_queries").insert(query_data).execute()

            return {
                "recommended_areas": recommended_areas,
                "strategy": strategy.value,
                "count": len(recommended_areas)
            }
        else:
            top_queries = sorted(
                unlabeled_areas.data,
                key=lambda x: x.get('selection_score', 0),
                reverse=True
            )[:n_queries]

            return {
                "recommended_areas": top_queries,
                "strategy": strategy.value,
                "count": len(top_queries)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/label", response_model=GroundTruthLabelResponse)
async def submit_label(
    label: GroundTruthLabelCreate,
    supabase: Client = Depends(get_supabase)
):
    try:
        label_dict = label.model_dump()
        label_dict['geometry'] = json.dumps(label.geometry.model_dump())

        result = supabase.table("ground_truth_labels").insert(label_dict).execute()

        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/labels", response_model=List[GroundTruthLabelResponse])
async def get_labels(
    limit: int = 100,
    supabase: Client = Depends(get_supabase)
):
    try:
        result = supabase.table("ground_truth_labels")\
            .select("*")\
            .order("labeled_at", desc=True)\
            .limit(limit)\
            .execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_active_learning_statistics(
    supabase: Client = Depends(get_supabase)
):
    try:
        total_labels = supabase.table("ground_truth_labels").select("id", count="exact").execute()

        labeled_queries = supabase.table("active_learning_queries")\
            .select("id", count="exact")\
            .eq("labeled", True)\
            .execute()

        unlabeled_queries = supabase.table("active_learning_queries")\
            .select("id", count="exact")\
            .eq("labeled", False)\
            .execute()

        recent_labels = supabase.table("ground_truth_labels")\
            .select("label")\
            .order("labeled_at", desc=True)\
            .limit(100)\
            .execute()

        label_distribution = {}
        for label_data in recent_labels.data:
            label_type = label_data['label']
            label_distribution[label_type] = label_distribution.get(label_type, 0) + 1

        return {
            "total_labels": total_labels.count,
            "labeled_queries": labeled_queries.count,
            "unlabeled_queries": unlabeled_queries.count,
            "label_distribution": label_distribution,
            "labeling_efficiency": (labeled_queries.count / max(total_labels.count, 1)) * 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
