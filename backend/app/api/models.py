from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict
from app.database import get_supabase
from app.models.schemas import ModelExplanation
from supabase import Client
import json

router = APIRouter(prefix="/api/v1/models", tags=["models"])

@router.post("/update")
async def trigger_model_update(
    background_tasks: BackgroundTasks,
    min_new_samples: int = 10,
    supabase: Client = Depends(get_supabase)
):
    try:
        recent_labels = supabase.table("ground_truth_labels")\
            .select("*")\
            .order("labeled_at", desc=True)\
            .limit(min_new_samples)\
            .execute()

        if len(recent_labels.data) < min_new_samples:
            return {
                "message": f"Not enough new samples. Need {min_new_samples}, got {len(recent_labels.data)}",
                "update_triggered": False
            }

        background_tasks.add_task(perform_incremental_update, recent_labels.data)

        return {
            "message": "Model update triggered",
            "update_triggered": True,
            "samples_count": len(recent_labels.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def perform_incremental_update(labels_data: list):
    try:
        import torch
        import numpy as np
        from app.ml.model import TerrainHazardModel, extract_features
        from app.ml.incremental_learning import IncrementalLearner
        from app.ml.model_persistence import get_model_manager
        from app.config import get_settings
        from datetime import datetime

        settings = get_settings()
        model_manager = get_model_manager()

        model = model_manager.get_current_model()
        if model is None:
            model = TerrainHazardModel()

        learner = IncrementalLearner(model)

        features_list = []
        labels_list = []

        for label_data in labels_data:
            features_dict = label_data.get('notes', {})
            if isinstance(features_dict, str):
                try:
                    features_dict = json.loads(features_dict)
                except:
                    features_dict = {}

            features = extract_features(features_dict)
            features_list.append(features)

            label_value = 1.0 if label_data['label'] in ['hazardous', 'suspected'] else 0.0
            labels_list.append(label_value)

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        training_result = learner.incremental_update(
            features_array,
            labels_array,
            epochs=10,
            batch_size=16
        )

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        new_version = f"v1.0.{timestamp}"

        metadata = {
            'training_samples': len(labels_data),
            'performance_metrics': training_result,
            'updated_at': datetime.now().isoformat()
        }

        model_manager.save_model(model, new_version, metadata)

        supabase = get_supabase()
        supabase.table("model_updates").update({"active": False}).neq("version", new_version).execute()
        supabase.table("model_updates").insert({
            'version': new_version,
            'training_samples': len(labels_data),
            'performance_metrics': training_result,
            'active': True,
            'model_config': {
                'input_dim': 15,
                'hidden_dims': [128, 64, 32],
                'dropout': 0.3
            },
            'notes': 'Incremental learning update'
        }).execute()

        print(f"Model updated successfully: {new_version}")

    except Exception as e:
        print(f"Error during incremental update: {str(e)}")

@router.get("/performance")
async def get_model_performance(
    version: str = None,
    supabase: Client = Depends(get_supabase)
):
    try:
        if version:
            result = supabase.table("model_updates")\
                .select("*")\
                .eq("version", version)\
                .maybeSingle()\
                .execute()
        else:
            result = supabase.table("model_updates")\
                .select("*")\
                .eq("active", True)\
                .maybeSingle()\
                .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Model version not found")

        return result.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/explain/{hazard_area_id}", response_model=ModelExplanation)
async def explain_prediction(
    hazard_area_id: str,
    supabase: Client = Depends(get_supabase)
):
    try:
        hazard_area = supabase.table("hazard_areas")\
            .select("*")\
            .eq("id", hazard_area_id)\
            .maybeSingle()\
            .execute()

        if not hazard_area.data:
            raise HTTPException(status_code=404, detail="Hazard area not found")

        existing_explanation = supabase.table("model_explanations")\
            .select("*")\
            .eq("hazard_area_id", hazard_area_id)\
            .maybeSingle()\
            .execute()

        if existing_explanation.data:
            return existing_explanation.data

        from app.ml.explainability import ModelExplainer
        from app.ml.model import TerrainHazardModel, extract_features
        from app.ml.model_persistence import get_model_manager
        import torch

        model_manager = get_model_manager()
        model = model_manager.get_current_model()
        if model is None:
            model = TerrainHazardModel()

        feature_names = [
            'elevation', 'slope', 'aspect', 'curvature',
            'proximity_to_conflict', 'proximity_to_roads',
            'proximity_to_buildings', 'proximity_to_water',
            'population_density', 'accessibility_score',
            'ndvi', 'land_cover_forest', 'land_cover_urban',
            'land_cover_agriculture', 'land_cover_bare'
        ]

        explainer = ModelExplainer(model, feature_names)

        features_dict = hazard_area.data.get('features', {})
        features = extract_features(features_dict)

        explanation = explainer.explain_prediction(
            features,
            hazard_area.data['risk_score'],
            hazard_area.data['uncertainty']
        )

        explanation_data = {
            'hazard_area_id': hazard_area_id,
            'model_version': hazard_area.data.get('model_version', 'v1.0.0'),
            'feature_importance': explanation['feature_importance'],
            'top_features': explanation['top_features'],
            'explanation_text': explanation['explanation_text'],
            'similar_areas': [],
            'confidence_factors': explanation['confidence_factors']
        }

        result = supabase.table("model_explanations").insert(explanation_data).execute()

        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions")
async def get_model_versions(
    limit: int = 10,
    supabase: Client = Depends(get_supabase)
):
    try:
        result = supabase.table("model_updates")\
            .select("*")\
            .order("trained_at", desc=True)\
            .limit(limit)\
            .execute()

        return {"versions": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
