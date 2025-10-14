# GeoRAG Quick Start Guide

This guide will help you get the GeoRAG system up and running quickly.

## Prerequisites

- Node.js 18+
- Python 3.11+
- npm installed
- Supabase account (already configured)

## Installation Steps

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 3. Environment Configuration

The environment files are already configured:
- **Frontend**: `.env` (root directory)
- **Backend**: `backend/.env`

Both are connected to the Supabase database.

## Database Setup

### Apply Database Migrations

The database schema is already created via the Supabase migration file. If you need to verify:

1. Log into your Supabase dashboard
2. Check the SQL Editor or Table Editor to confirm tables exist
3. Tables should include: hazard_areas, ground_truth_labels, missions, model_updates, etc.

### Seed Sample Data

To populate the database with sample data for testing:

```bash
cd backend
python -m app.seed_data
cd ..
```

This will create:
- 50 sample hazard areas across 5 regions
- Terrain feature points globally
- 30 ground truth labels
- 15 sample missions
- 5 model update records

## Running the Application

### Start the Backend API

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Start the Frontend (in a new terminal)

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Training the ML Model

### Initial Model Training

To train the initial ML model with sample data:

```bash
cd backend
python -m app.ml.train
```

This will:
- Load training data from the database
- Train a TerrainHazardModel with uncertainty quantification
- Evaluate performance metrics
- Save the trained model to `backend/models/`
- Create a model update record in the database

Training takes approximately 2-5 minutes depending on your hardware.

## Using the Application

### 1. Dashboard View
- View overall statistics
- Track missions and labeling progress
- Monitor active learning efficiency

### 2. Risk Map
- Interactive map showing hazard areas
- Color-coded by risk score (red = high risk)
- Click areas for detailed information
- Opacity indicates prediction uncertainty

### 3. Active Learning Panel
- View recommended areas for labeling
- Submit new ground truth labels
- System prioritizes high-value areas to label
- Reduces labeling effort by up to 60%

### 4. Mission Planning
- Create new field missions
- Automated area prioritization
- Resource allocation planning
- Track mission progress

### 5. Model Explanations
- Understand model predictions
- Feature importance visualization
- Confidence factor analysis
- Transparent decision-making

## API Endpoints

### Hazards
- `GET /api/v1/hazards` - List hazard areas
- `POST /api/v1/hazards/predict` - Predict risk for new area
- `GET /api/v1/hazards/uncertainty-map` - Get uncertainty heatmap

### Active Learning
- `GET /api/v1/active-learning/next-areas` - Get recommended areas to label
- `POST /api/v1/active-learning/label` - Submit ground truth label
- `GET /api/v1/active-learning/statistics` - Get labeling statistics

### Missions
- `POST /api/v1/missions` - Create new mission
- `GET /api/v1/missions` - List missions
- `PUT /api/v1/missions/{id}` - Update mission status

### Models
- `POST /api/v1/models/update` - Trigger incremental model update
- `GET /api/v1/models/performance` - Get model performance metrics
- `GET /api/v1/models/explain/{id}` - Get prediction explanation
- `GET /api/v1/models/versions` - List model versions

## Model Management

### Loading a Trained Model

Models are automatically loaded from the `backend/models/` directory. The system uses the most recent version by default.

### Incremental Learning

To update the model with new ground truth labels:

```bash
curl -X POST http://localhost:8000/api/v1/models/update?min_new_samples=10
```

This triggers incremental learning that:
- Uses new ground truth labels
- Updates model weights without forgetting
- Creates a new model version
- Preserves knowledge from previous training

### Model Persistence

Models are saved in PyTorch format:
- Location: `backend/models/`
- Format: `model_v{version}.pt`
- Metadata: `model_v{version}_metadata.json`

## Development Workflow

### Adding New Features

1. Make code changes
2. Run type checking: `npm run typecheck`
3. Run linting: `npm run lint`
4. Build project: `npm run build`
5. Test functionality

### Testing the API

Use the built-in API documentation:
- Navigate to `http://localhost:8000/docs`
- Interactive Swagger UI with all endpoints
- Try requests directly from the browser

## Troubleshooting

### Backend Won't Start

**Issue**: Module import errors
**Solution**: Ensure you're in the backend directory and have installed all requirements

```bash
cd backend
pip install -r requirements.txt
```

### Frontend Shows No Data

**Issue**: Database is empty
**Solution**: Run the seed data script

```bash
cd backend
python -m app.seed_data
```

### Model Predictions Seem Random

**Issue**: Model hasn't been trained
**Solution**: Train the initial model

```bash
cd backend
python -m app.ml.train
```

### CORS Errors

**Issue**: Backend not running or wrong port
**Solution**: Ensure backend is running on port 8000

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

## Next Steps

### For Development
- Explore the codebase in `src/` (frontend) and `backend/app/` (backend)
- Review the architecture documentation in `ARCHITECTURE.md`
- Check API documentation at `http://localhost:8000/docs`

### For Production
- Configure CORS for specific domains
- Set up proper authentication
- Use environment-specific configuration
- Deploy backend and frontend separately
- Use managed PostgreSQL for database
- Set up monitoring and logging

### For Research
- Experiment with different active learning strategies
- Adjust model architecture in `backend/app/ml/model.py`
- Implement custom feature extraction
- Add new terrain data sources
- Extend explainability methods

## Additional Resources

- **README.md**: Project overview and features
- **ARCHITECTURE.md**: Detailed system architecture
- **API Docs**: http://localhost:8000/docs (when backend is running)
- **Supabase Dashboard**: Database management and queries

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the architecture documentation
3. Inspect browser console for frontend errors
4. Check backend logs for API errors
5. Verify database connectivity in Supabase dashboard

## Summary

You now have a fully functional GeoRAG system with:
- Interactive web interface
- ML-powered hazard prediction with uncertainty
- Active learning for efficient data collection
- Mission planning and tracking
- Model explainability and transparency
- Incremental learning capabilities
- Database persistence with spatial support

Start exploring and building geospatial ML applications!
