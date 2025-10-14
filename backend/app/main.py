from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import hazards, active_learning, missions, models

app = FastAPI(
    title="GeoRAG - Geospatial Reasoning Framework",
    description="ML-powered geospatial hazard prediction with active learning and human-in-the-loop interfaces",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hazards.router)
app.include_router(active_learning.router)
app.include_router(missions.router)
app.include_router(models.router)

@app.get("/")
async def root():
    return {
        "message": "GeoRAG API",
        "version": "1.0.0",
        "endpoints": {
            "hazards": "/api/v1/hazards",
            "active_learning": "/api/v1/active-learning",
            "missions": "/api/v1/missions",
            "models": "/api/v1/models"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
