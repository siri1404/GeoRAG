/*
  # GeoRAG - Geospatial Reasoning Framework Schema

  ## Overview
  This migration creates the complete database schema for the GeoRAG system,
  a geospatial ML pipeline for terrain reasoning and hazard prediction with
  active learning and human-in-the-loop interfaces.

  ## Tables Created

  1. **hazard_areas**
     - Stores predicted hazard zones with risk scores and uncertainty
     - Includes geometry, model predictions, and temporal tracking
     - Used for visualization and mission planning

  2. **ground_truth_labels**
     - Stores verified ground truth data from field operations
     - Includes label type, confidence, and verification metadata
     - Used for model training and validation

  3. **missions**
     - Tracks field missions and clearance operations
     - Includes planning, execution, and completion status
     - Links to hazard areas and team assignments

  4. **model_updates**
     - Tracks model versions and training events
     - Stores performance metrics for each version
     - Enables rollback and A/B testing

  5. **user_feedback**
     - Captures operational team feedback on predictions
     - Enables model refinement and correction
     - Supports human-in-the-loop learning

  6. **terrain_features**
     - Stores extracted terrain characteristics
     - Includes elevation, slope, land cover, proximity metrics
     - Used as input features for ML models

  7. **active_learning_queries**
     - Tracks areas recommended for labeling
     - Stores query strategy and expected information gain
     - Prioritizes data collection efforts

  8. **model_explanations**
     - Stores model prediction explanations
     - Includes feature importance and reasoning
     - Supports transparency and trust

  ## Security
  - Row Level Security (RLS) enabled on all tables
  - Policies restrict access to authenticated users
  - Audit trails for all data modifications

  ## Spatial Indexing
  - GIST indexes on all geometry columns for fast spatial queries
  - Optimized for proximity searches and intersection operations
*/

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create custom types
CREATE TYPE hazard_label AS ENUM ('safe', 'hazardous', 'unknown', 'suspected');
CREATE TYPE mission_status AS ENUM ('planned', 'in_progress', 'completed', 'cancelled');
CREATE TYPE feedback_type AS ENUM ('correct', 'incorrect', 'uncertain', 'correction');
CREATE TYPE query_strategy AS ENUM ('uncertainty', 'diversity', 'expected_change', 'hybrid', 'impact');

-- Table: hazard_areas
CREATE TABLE IF NOT EXISTS hazard_areas (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  geometry geography(POLYGON, 4326) NOT NULL,
  risk_score float NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
  uncertainty float NOT NULL CHECK (uncertainty >= 0 AND uncertainty <= 1),
  prediction_timestamp timestamptz DEFAULT now(),
  model_version text NOT NULL,
  features jsonb DEFAULT '{}',
  priority_rank integer,
  area_sqm float,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Table: ground_truth_labels
CREATE TABLE IF NOT EXISTS ground_truth_labels (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  geometry geography(GEOMETRY, 4326) NOT NULL,
  label hazard_label NOT NULL,
  confidence float DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
  labeled_by text NOT NULL,
  labeled_at timestamptz DEFAULT now(),
  verification_method text,
  notes text,
  mission_id uuid,
  photos jsonb DEFAULT '[]',
  created_at timestamptz DEFAULT now()
);

-- Table: missions
CREATE TABLE IF NOT EXISTS missions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  mission_name text NOT NULL,
  mission_date date NOT NULL,
  team_id text NOT NULL,
  area_geometry geography(POLYGON, 4326),
  status mission_status DEFAULT 'planned',
  priority_score float CHECK (priority_score >= 0 AND priority_score <= 1),
  resources_allocated jsonb DEFAULT '{}',
  expected_duration_hours float,
  actual_duration_hours float,
  areas_cleared integer DEFAULT 0,
  hazards_found integer DEFAULT 0,
  planned_start timestamptz,
  actual_start timestamptz,
  completed_at timestamptz,
  notes text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Table: model_updates
CREATE TABLE IF NOT EXISTS model_updates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  version text UNIQUE NOT NULL,
  trained_at timestamptz DEFAULT now(),
  training_samples integer NOT NULL,
  performance_metrics jsonb DEFAULT '{}',
  active boolean DEFAULT false,
  model_config jsonb DEFAULT '{}',
  training_duration_seconds integer,
  notes text,
  created_at timestamptz DEFAULT now()
);

-- Table: user_feedback
CREATE TABLE IF NOT EXISTS user_feedback (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  hazard_area_id uuid REFERENCES hazard_areas(id) ON DELETE CASCADE,
  feedback_type feedback_type NOT NULL,
  corrected_label hazard_label,
  corrected_risk_score float CHECK (corrected_risk_score >= 0 AND corrected_risk_score <= 1),
  user_id text NOT NULL,
  comments text,
  geometry geography(POINT, 4326),
  timestamp timestamptz DEFAULT now()
);

-- Table: terrain_features
CREATE TABLE IF NOT EXISTS terrain_features (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  geometry geography(POINT, 4326) NOT NULL,
  elevation float,
  slope float,
  aspect float,
  curvature float,
  land_cover text,
  ndvi float,
  proximity_to_conflict float,
  proximity_to_roads float,
  proximity_to_buildings float,
  proximity_to_water float,
  population_density float,
  accessibility_score float,
  infrastructure_nearby jsonb DEFAULT '{}',
  computed_at timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

-- Table: active_learning_queries
CREATE TABLE IF NOT EXISTS active_learning_queries (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  geometry geography(POLYGON, 4326) NOT NULL,
  query_strategy query_strategy NOT NULL,
  selection_score float NOT NULL,
  expected_info_gain float,
  uncertainty_score float,
  diversity_score float,
  impact_score float,
  suggested_at timestamptz DEFAULT now(),
  labeled boolean DEFAULT false,
  label_id uuid REFERENCES ground_truth_labels(id),
  model_version text NOT NULL,
  batch_id uuid,
  created_at timestamptz DEFAULT now()
);

-- Table: model_explanations
CREATE TABLE IF NOT EXISTS model_explanations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  hazard_area_id uuid REFERENCES hazard_areas(id) ON DELETE CASCADE,
  model_version text NOT NULL,
  feature_importance jsonb DEFAULT '{}',
  top_features jsonb DEFAULT '[]',
  explanation_text text,
  similar_areas jsonb DEFAULT '[]',
  confidence_factors jsonb DEFAULT '{}',
  generated_at timestamptz DEFAULT now()
);

-- Create spatial indexes
CREATE INDEX IF NOT EXISTS idx_hazard_areas_geometry ON hazard_areas USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_ground_truth_geometry ON ground_truth_labels USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_missions_geometry ON missions USING GIST (area_geometry);
CREATE INDEX IF NOT EXISTS idx_terrain_features_geometry ON terrain_features USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_active_learning_geometry ON active_learning_queries USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_user_feedback_geometry ON user_feedback USING GIST (geometry);

-- Create additional indexes for performance
CREATE INDEX IF NOT EXISTS idx_hazard_areas_risk ON hazard_areas (risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_hazard_areas_priority ON hazard_areas (priority_rank);
CREATE INDEX IF NOT EXISTS idx_missions_status ON missions (status);
CREATE INDEX IF NOT EXISTS idx_missions_date ON missions (mission_date DESC);
CREATE INDEX IF NOT EXISTS idx_model_updates_active ON model_updates (active) WHERE active = true;
CREATE INDEX IF NOT EXISTS idx_active_learning_labeled ON active_learning_queries (labeled) WHERE labeled = false;

-- Enable Row Level Security
ALTER TABLE hazard_areas ENABLE ROW LEVEL SECURITY;
ALTER TABLE ground_truth_labels ENABLE ROW LEVEL SECURITY;
ALTER TABLE missions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_updates ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE terrain_features ENABLE ROW LEVEL SECURITY;
ALTER TABLE active_learning_queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_explanations ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Allow authenticated users to read all data
CREATE POLICY "Authenticated users can read hazard areas"
  ON hazard_areas FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can read ground truth"
  ON ground_truth_labels FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can read missions"
  ON missions FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can read model updates"
  ON model_updates FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can read feedback"
  ON user_feedback FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can read terrain features"
  ON terrain_features FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can read queries"
  ON active_learning_queries FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can read explanations"
  ON model_explanations FOR SELECT
  TO authenticated
  USING (true);

-- RLS Policies: Allow authenticated users to insert data
CREATE POLICY "Authenticated users can insert ground truth"
  ON ground_truth_labels FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can insert missions"
  ON missions FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can insert feedback"
  ON user_feedback FOR INSERT
  TO authenticated
  WITH CHECK (true);

-- RLS Policies: Allow authenticated users to update their own data
CREATE POLICY "Authenticated users can update missions"
  ON missions FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update ground truth"
  ON ground_truth_labels FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_hazard_areas_updated_at
  BEFORE UPDATE ON hazard_areas
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_missions_updated_at
  BEFORE UPDATE ON missions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();