# Database Successfully Seeded

## Summary

The GeoRAG Supabase database has been populated with sample data for testing and development.

## Data Inserted

### Hazard Areas: 10 records
- **Average Risk Score**: 0.65 (65%)
- **Average Uncertainty**: 0.19 (19%)
- **Max Risk Score**: 0.88 (88%)
- **Geographic Coverage**: Multiple regions including Middle East, Africa, and Southeast Asia
- **Features**: All areas include terrain features (elevation, slope, proximity metrics)

**Sample Locations**:
- Region A (Middle East): 4 high-risk areas
- Region B (Eastern Europe): 2 moderate-risk areas
- Region C (Africa): 2 low-risk areas
- Region D (West Africa): 2 moderate-risk areas

### Ground Truth Labels: 10 records
- **Hazardous**: 4 areas
- **Safe**: 3 areas
- **Suspected**: 2 areas
- **Unknown**: 1 area

**Verification Methods**:
- Field Survey: 6 labels
- Remote Sensing: 2 labels
- Expert Review: 2 labels

**Operators**: operator_1, operator_2, operator_3

### Missions: 6 records
- **Completed**: 2 missions (12 areas cleared, 6 hazards found)
- **In Progress**: 2 missions (6 areas cleared, 3 hazards found)
- **Planned**: 2 missions (upcoming operations)

**Teams**: team_1, team_2, team_3

**Mission Details**:
- Mission Alpha: Completed, high-priority zone cleared
- Mission Bravo: In progress, ongoing operations
- Mission Charlie: Planned, high-risk area with full team deployment
- Mission Delta: Completed, low-risk area
- Mission Echo: In progress, moderate risk
- Mission Foxtrot: Planned for next week

### Model Updates: 3 records
- **v1.0.0** (Active): 85% accuracy, 1000 training samples
- **v0.9.0** (Inactive): 80% accuracy, 750 training samples
- **v0.8.0** (Inactive): 76% accuracy, 500 training samples

**Performance Metrics** (Active Model):
- Accuracy: 0.85
- Precision: 0.82
- Recall: 0.79
- F1 Score: 0.80
- AUC-ROC: 0.88

### Terrain Features: 5 sample points
- Global coverage with diverse land cover types
- Elevation range: 95m - 380m
- Slope range: 6.8° - 22.1°
- Land cover: urban, bare, agriculture, forest
- Infrastructure data included

## Using the Sample Data

### View Hazard Areas in Frontend
1. Start the frontend: `npm run dev`
2. Navigate to "Risk Map" view
3. You should see 10 hazard areas displayed on the map
4. Click any area for detailed information

### Query via API
```bash
# Get all hazard areas
curl http://localhost:8000/api/v1/hazards

# Get high-risk areas
curl http://localhost:8000/api/v1/hazards/high-priority?threshold=0.7

# Get missions
curl http://localhost:8000/api/v1/missions

# Get active learning statistics
curl http://localhost:8000/api/v1/active-learning/statistics
```

### Database Queries
You can query the data directly from Supabase dashboard or using SQL:

```sql
-- View all hazard areas with risk levels
SELECT
  id,
  risk_score,
  uncertainty,
  model_version,
  priority_rank
FROM hazard_areas
ORDER BY risk_score DESC;

-- View mission progress
SELECT
  mission_name,
  status,
  areas_cleared,
  hazards_found,
  priority_score
FROM missions
ORDER BY mission_date DESC;

-- View ground truth label distribution
SELECT
  label,
  COUNT(*) as count,
  ROUND(AVG(confidence)::numeric, 2) as avg_confidence
FROM ground_truth_labels
GROUP BY label;
```

## Data Characteristics

### Realistic Scenarios
- Mix of high, medium, and low-risk areas
- Various verification methods and confidence levels
- Active and completed missions with progress tracking
- Geographic diversity for global testing

### ML Training Ready
- 10 ground truth labels for initial training
- 10 hazard areas with predicted risk scores
- 5 terrain feature points for feature extraction
- Model performance history tracked

### Active Learning Ready
- Areas with varying uncertainty levels
- Multiple query strategies can be tested
- Unlabeled areas available for recommendations

## Next Steps

### 1. View the Data
```bash
npm run dev
# Open http://localhost:5173
# Navigate to Dashboard and Risk Map views
```

### 2. Test Active Learning
- Go to "Active Learning" panel
- Click "Get Recommendations"
- System will suggest high-value areas to label

### 3. Create a Mission
- Go to "Mission Planning"
- Fill in mission details
- System will prioritize based on sample data

### 4. Train Model
```bash
cd backend
# Install dependencies if needed
pip install -r requirements.txt
# Train model with the sample data
python -m app.ml.train
```

## Database Status

✅ All tables populated with sample data
✅ Spatial data (PostGIS) working correctly
✅ Row-Level Security policies active
✅ Indexes created for performance
✅ Ready for frontend/API testing

The GeoRAG system is now fully operational with realistic sample data!
