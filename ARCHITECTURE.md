# GeoRAG System Architecture

## Overview

GeoRAG is a full-stack geospatial ML application built for humanitarian operations, featuring real-time hazard prediction, active learning, and human-in-the-loop interfaces.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Frontend (React)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │  Dashboard  │  │  Risk Map   │  │ Active Learn │       │
│  └─────────────┘  └─────────────┘  └──────────────┘       │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │   Missions  │  │ Explanations│                          │
│  └─────────────┘  └─────────────┘                          │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API
┌────────────────────────┴────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Hazards   │  │ Active Learn │  │   Missions   │      │
│  │     API     │  │     API      │  │     API      │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│  ┌─────────────┐  ┌──────────────┐                         │
│  │   Models    │  │ Prioritize   │                         │
│  │     API     │  │   Service    │                         │
│  └─────────────┘  └──────────────┘                         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    ML Pipeline (PyTorch)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Uncertainty │  │    Active    │  │ Incremental  │      │
│  │    Model    │  │   Learning   │  │   Learning   │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│  ┌─────────────┐                                            │
│  │Explainability│                                           │
│  └─────────────┘                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│              Database (Supabase + PostGIS)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Hazard Areas │  │Ground Truth  │  │   Missions   │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Models    │  │   Feedback   │  │   Features   │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend Layer

**Technology**: React + TypeScript + Tailwind CSS

**Components**:
- `MapView`: Interactive Leaflet map with risk visualization
- `Dashboard`: Statistics and metrics overview
- `ActiveLearningPanel`: Label recommendation interface
- `MissionPlanner`: Mission creation and planning
- `ExplanationPanel`: Model prediction explanations

**Key Features**:
- Responsive design (mobile-first)
- Real-time data updates
- Interactive geospatial visualization
- Low cognitive load UI design

### Backend Layer

**Technology**: FastAPI + Python

**API Modules**:

1. **Hazards API** (`/api/v1/hazards`)
   - Risk prediction endpoint
   - Uncertainty mapping
   - User feedback collection
   - High-priority area queries

2. **Active Learning API** (`/api/v1/active-learning`)
   - Query strategy selection
   - Label submission
   - Statistics tracking
   - Next areas recommendation

3. **Missions API** (`/api/v1/missions`)
   - Mission CRUD operations
   - Planning and prioritization
   - Status tracking
   - Resource optimization

4. **Models API** (`/api/v1/models`)
   - Incremental updates
   - Performance metrics
   - Explanation generation
   - Version management

### ML Pipeline Layer

**Technology**: PyTorch + NumPy + Scikit-learn

**Core Modules**:

1. **Terrain Hazard Model** (`ml/model.py`)
   ```python
   class TerrainHazardModel(nn.Module):
       - Feature extractor (conv + batch norm + dropout)
       - Risk prediction head (sigmoid output)
       - Uncertainty estimation head (softplus output)
       - Monte Carlo Dropout for epistemic uncertainty
   ```

2. **Active Learning** (`ml/active_learning.py`)
   ```python
   class ActiveLearningStrategy:
       - Uncertainty sampling
       - Diversity sampling
       - Expected model change
       - Hybrid strategy (recommended)
       - Impact-based selection
   ```

3. **Incremental Learning** (`ml/incremental_learning.py`)
   ```python
   class IncrementalLearner:
       - Experience replay buffer
       - Elastic Weight Consolidation (EWC)
       - Fine-tuning pipeline
       - Performance monitoring
   ```

4. **Explainability** (`ml/explainability.py`)
   ```python
   class ModelExplainer:
       - Gradient-based feature attribution
       - Natural language explanations
       - Feature importance ranking
       - Confidence factor analysis
   ```

### Database Layer

**Technology**: PostgreSQL + PostGIS

**Schema Design**:

1. **Spatial Tables**:
   - All geometry columns use `geography(GEOMETRY, 4326)` type
   - GIST spatial indexes for fast queries
   - Support for points, polygons, and complex geometries

2. **Security**:
   - Row-Level Security (RLS) enabled on all tables
   - Authenticated user policies
   - Audit trails via timestamps

3. **Performance**:
   - Spatial indexing on all geometry columns
   - B-tree indexes on frequently queried columns
   - Optimized for proximity searches

## Data Flow

### 1. Prediction Request Flow
```
User → Frontend → API → ML Model → Database
                          ↓
                    Uncertainty
                    Quantification
                          ↓
                      Response ← ← ← ←
```

### 2. Active Learning Flow
```
System → Query Strategy → Candidate Selection
           ↓
      Rank by Score
           ↓
      Present to User → Feedback → Update Model
```

### 3. Incremental Learning Flow
```
New Labels → Memory Buffer → Fine-tune Model
                                    ↓
                             EWC Regularization
                                    ↓
                            Update Database
```

### 4. Mission Planning Flow
```
User Input → Prioritization Service → Risk Analysis
                                            ↓
                                    Multi-criteria
                                    Optimization
                                            ↓
                                    Mission Plan
```

## Key Design Decisions

### 1. Uncertainty Quantification
**Decision**: Use Monte Carlo Dropout + Deep Ensembles
**Rationale**:
- Provides both epistemic and aleatoric uncertainty
- Computationally efficient during inference
- Well-calibrated probability estimates

### 2. Active Learning Strategy
**Decision**: Hybrid approach (uncertainty + diversity + impact)
**Rationale**:
- Balances exploration and exploitation
- Considers operational constraints
- Maximizes information gain per label

### 3. Incremental Learning
**Decision**: EWC + Experience Replay
**Rationale**:
- Prevents catastrophic forgetting
- Maintains performance on old data
- Fast adaptation to new patterns

### 4. Database Choice
**Decision**: Supabase (PostgreSQL + PostGIS)
**Rationale**:
- Native geospatial support
- ACID compliance for critical data
- Built-in auth and RLS
- Scalable for production

### 5. API Design
**Decision**: RESTful with FastAPI
**Rationale**:
- Standard HTTP conventions
- Auto-generated OpenAPI docs
- Type safety with Pydantic
- High performance async support

## Scalability Considerations

### Horizontal Scaling
- **API**: Stateless design enables load balancing
- **Database**: Read replicas for query performance
- **ML Models**: Model serving with batching

### Vertical Scaling
- **Computation**: GPU support for model inference
- **Storage**: PostGIS spatial indexes for large datasets
- **Memory**: Efficient data structures and caching

### Performance Optimization
- **Database**: Materialized views for aggregate queries
- **API**: Response caching for frequent requests
- **Frontend**: Code splitting and lazy loading
- **ML**: Model quantization for faster inference

## Security Architecture

### Authentication & Authorization
- JWT-based authentication
- Row-Level Security in database
- API key management
- Audit logging

### Data Protection
- Encrypted connections (HTTPS/TLS)
- Sensitive data masking
- Input validation and sanitization
- CORS configuration

### Operational Security
- Environment variable management
- Secret rotation policies
- Backup and recovery procedures
- Monitoring and alerting

## Deployment Architecture

### Development
```
Local Machine
├── Frontend (Vite dev server :5173)
├── Backend (Uvicorn :8000)
└── Database (Supabase cloud)
```

### Production
```
Cloud Infrastructure
├── Frontend (CDN + Static hosting)
├── Backend (Container orchestration)
└── Database (Managed Supabase)
```

## Monitoring & Observability

### Metrics to Track
- API response times
- Model inference latency
- Database query performance
- Active learning efficiency
- Model accuracy over time
- User engagement metrics

### Logging Strategy
- Structured JSON logs
- Request/response logging
- Error tracking with stack traces
- Audit trails for data modifications

### Alerting Rules
- High error rates
- Slow query detection
- Model drift detection
- System resource exhaustion

## Future Architecture Enhancements

### Short-term (3-6 months)
- Redis caching layer
- Background job queue (Celery)
- WebSocket for real-time updates
- Mobile app backend

### Medium-term (6-12 months)
- Microservices architecture
- Event-driven communication
- GraphQL API option
- Multi-region deployment

### Long-term (12+ months)
- Kubernetes orchestration
- Service mesh for microservices
- Machine learning pipeline automation
- Edge computing for field operations
