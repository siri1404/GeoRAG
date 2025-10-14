export interface GeoJSONGeometry {
  type: string;
  coordinates: number[][] | number[][][];
}

export interface HazardArea {
  id: string;
  geometry: any;
  risk_score: number;
  uncertainty: number;
  prediction_timestamp: string;
  model_version: string;
  features: Record<string, any>;
  priority_rank?: number;
}

export interface Mission {
  id: string;
  mission_name: string;
  mission_date: string;
  team_id: string;
  status: 'planned' | 'in_progress' | 'completed' | 'cancelled';
  priority_score?: number;
  areas_cleared: number;
  hazards_found: number;
  created_at: string;
}

export interface GroundTruthLabel {
  id: string;
  geometry: any;
  label: 'safe' | 'hazardous' | 'unknown' | 'suspected';
  confidence: number;
  labeled_by: string;
  labeled_at: string;
  notes?: string;
}

export interface ActiveLearningQuery {
  id?: string;
  geometry: any;
  query_strategy: string;
  selection_score: number;
  expected_info_gain?: number;
  uncertainty_score?: number;
  impact_score?: number;
}

export interface ModelExplanation {
  feature_importance: Record<string, number>;
  top_features: Array<{ name: string; importance: number; rank: number }>;
  explanation_text: string;
  confidence_factors: {
    confidence_score: number;
    model_certainty: string;
    needs_verification: boolean;
  };
}

export interface PredictionResponse {
  risk_score: number;
  uncertainty: number;
  confidence_interval: [number, number];
  model_version: string;
}
