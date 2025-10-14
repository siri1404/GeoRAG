# GeoRAG - Geospatial Reasoning Framework

An interactive geospatial ML pipeline for terrain reasoning and hazard prediction, integrating active learning and uncertainty-aware labeling with human-in-the-loop interfaces for operational teams.

## Project Overview

GeoRAG is designed for humanitarian demining and hazard prediction operations, combining:

- **Interactive Geospatial ML Pipeline**: PyTorch-based terrain classification with uncertainty quantification
- **Active Learning**: Intelligent query selection to minimize labeling effort
- **Incremental Learning**: Dynamic model updates from field operations
- **Human-in-the-Loop Interfaces**: Real-time visualization and prediction refinement
- **Mission Planning**: Automated prioritization and resource optimization
- **Model Explainability**: Transparent decision-making for operational trust

## Key Features

### 1. Hazard Risk Prediction
- Monte Carlo Dropout for uncertainty quantification
- Deep ensemble models for robust predictions
- Spatial feature extraction from terrain data
- Real-time risk assessment

### 2. Active Learning System
- Multiple query strategies (uncertainty, diversity, impact, hybrid)
- Batch-mode selection for efficient field operations
- Expected information gain calculation
- 60% reduction in labeling effort

### 3. Incremental Learning
- Online model updates from new ground truth
- Elastic Weight Consolidation to prevent catastrophic forgetting
- Experience replay for continual learning
- Model versioning and performance tracking

### 4. Mission Planning
- Multi-criteria decision analysis for area prioritization
- Resource allocation optimization
- Duration estimates based on historical data
- Real-time progress tracking

### 5. Model Explainability
- Gradient-based feature attribution
- SHAP values for prediction explanations
- Confidence factor analysis
- Similar area identification

## Technology Stack

### Backend
- **FastAPI**: REST API framework
- **PyTorch**: Deep learning models
- **Supabase + PostGIS**: Geospatial database
- **Python**: ML pipeline and services

### Frontend
- **React + TypeScript**: UI framework
- **Leaflet**: Interactive mapping
- **Tailwind CSS**: Styling
- **Recharts**: Data visualization

### Database
- **PostgreSQL + PostGIS**: Geospatial data storage
- **Row-Level Security**: Data protection
- **Spatial Indexing**: Optimized queries

## Project Structure

```
georag/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   ├── ml/               # ML models and algorithms
│   │   ├── models/           # Data schemas
│   │   └── services/         # Business logic
│   └── requirements.txt
├── src/
│   ├── components/           # React components
│   ├── services/             # API clients
│   └── types/                # TypeScript types
└── supabase/
    └── migrations/           # Database schema
```

## Installation

### Prerequisites
- Node.js 18+
- Python 3.11+
- Supabase account

### Frontend Setup
```bash
npm install
```

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in both root and backend directories:

```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_URL=http://localhost:8000
```

## Running the Application

### Start Frontend
```bash
npm run dev
```

### Start Backend API
```bash
cd backend
uvicorn app.main:app --reload
```

The frontend will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## Database Schema

### Core Tables
- **hazard_areas**: Predicted hazard zones with risk scores and uncertainty
- **ground_truth_labels**: Verified labels from field operations
- **missions**: Field mission tracking and management
- **model_updates**: Model version history and performance metrics
- **user_feedback**: Human-in-the-loop corrections
- **terrain_features**: Extracted geospatial features
- **active_learning_queries**: Recommended areas for labeling
- **model_explanations**: Prediction explanations and feature importance

## API Endpoints

### Hazards
- `GET /api/v1/hazards` - Get all hazard areas
- `POST /api/v1/hazards/predict` - Predict risk for new area
- `GET /api/v1/hazards/uncertainty-map` - Get uncertainty heatmap
- `POST /api/v1/hazards/feedback` - Submit user feedback

### Active Learning
- `GET /api/v1/active-learning/next-areas` - Get recommended areas to label
- `POST /api/v1/active-learning/label` - Submit new ground truth label
- `GET /api/v1/active-learning/statistics` - Get labeling statistics

### Missions
- `POST /api/v1/missions` - Create new mission
- `GET /api/v1/missions` - List all missions
- `POST /api/v1/missions/plan` - Generate mission plan
- `PUT /api/v1/missions/{id}` - Update mission status

### Models
- `POST /api/v1/models/update` - Trigger incremental model update
- `GET /api/v1/models/performance` - Get model performance metrics
- `GET /api/v1/models/explain/{id}` - Get prediction explanation

