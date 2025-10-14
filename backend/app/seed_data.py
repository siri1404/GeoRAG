import asyncio
import json
import random
from datetime import datetime, timedelta
from supabase import create_client
from app.config import get_settings

settings = get_settings()
supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

def generate_sample_polygon(center_lat: float, center_lon: float, size_km: float = 1.0):
    offset = size_km / 111.0

    coords = [
        [center_lon - offset, center_lat - offset],
        [center_lon + offset, center_lat - offset],
        [center_lon + offset, center_lat + offset],
        [center_lon - offset, center_lat + offset],
        [center_lon - offset, center_lat - offset]
    ]

    return {
        "type": "Polygon",
        "coordinates": [coords]
    }

def generate_sample_point(center_lat: float, center_lon: float):
    return {
        "type": "Point",
        "coordinates": [center_lon, center_lat]
    }

def generate_terrain_features():
    return {
        "elevation": random.uniform(0, 500),
        "slope": random.uniform(0, 45),
        "aspect": random.uniform(0, 360),
        "curvature": random.uniform(-0.5, 0.5),
        "proximity_to_conflict": random.uniform(0, 50000),
        "proximity_to_roads": random.uniform(100, 20000),
        "proximity_to_buildings": random.uniform(200, 15000),
        "proximity_to_water": random.uniform(500, 25000),
        "population_density": random.uniform(0, 1000),
        "accessibility_score": random.uniform(0.1, 1.0),
        "ndvi": random.uniform(-0.2, 0.9),
        "land_cover_forest": random.choice([0, 1]),
        "land_cover_urban": random.choice([0, 1]),
        "land_cover_agriculture": random.choice([0, 1]),
        "land_cover_bare": random.choice([0, 1])
    }

async def seed_hazard_areas():
    print("Seeding hazard areas...")

    test_regions = [
        {"name": "Region A", "lat": 15.5, "lon": 32.5, "base_risk": 0.7},
        {"name": "Region B", "lat": 33.3, "lon": 44.4, "base_risk": 0.5},
        {"name": "Region C", "lat": -10.2, "lon": 25.8, "base_risk": 0.3},
        {"name": "Region D", "lat": 41.0, "lon": 29.0, "base_risk": 0.8},
        {"name": "Region E", "lat": 8.5, "lon": -11.5, "base_risk": 0.6}
    ]

    hazard_areas = []

    for region in test_regions:
        for i in range(10):
            lat_offset = random.uniform(-2, 2)
            lon_offset = random.uniform(-2, 2)

            geometry = generate_sample_polygon(
                region["lat"] + lat_offset,
                region["lon"] + lon_offset,
                random.uniform(0.5, 2.0)
            )

            risk_variation = random.uniform(-0.2, 0.2)
            risk_score = max(0.0, min(1.0, region["base_risk"] + risk_variation))
            uncertainty = random.uniform(0.05, 0.3)

            features = generate_terrain_features()

            area = {
                "geometry": json.dumps(geometry),
                "risk_score": risk_score,
                "uncertainty": uncertainty,
                "model_version": "v1.0.0",
                "features": features,
                "priority_rank": None,
                "area_sqm": random.uniform(500000, 2000000)
            }

            hazard_areas.append(area)

    for i in range(0, len(hazard_areas), 10):
        batch = hazard_areas[i:i+10]
        result = supabase.table("hazard_areas").insert(batch).execute()
        print(f"Inserted {len(batch)} hazard areas")

    print(f"✓ Seeded {len(hazard_areas)} hazard areas")

async def seed_terrain_features():
    print("Seeding terrain features...")

    terrain_points = []

    for lat in range(-40, 60, 5):
        for lon in range(-180, 180, 10):
            point_geometry = generate_sample_point(lat, lon)

            terrain_data = generate_terrain_features()

            feature_point = {
                "geometry": json.dumps(point_geometry),
                "elevation": terrain_data["elevation"],
                "slope": terrain_data["slope"],
                "aspect": terrain_data["aspect"],
                "curvature": terrain_data["curvature"],
                "land_cover": random.choice(["forest", "urban", "agriculture", "bare", "water"]),
                "ndvi": terrain_data["ndvi"],
                "proximity_to_conflict": terrain_data["proximity_to_conflict"],
                "proximity_to_roads": terrain_data["proximity_to_roads"],
                "proximity_to_buildings": terrain_data["proximity_to_buildings"],
                "proximity_to_water": terrain_data["proximity_to_water"],
                "population_density": terrain_data["population_density"],
                "accessibility_score": terrain_data["accessibility_score"],
                "infrastructure_nearby": {
                    "roads": random.randint(0, 10),
                    "buildings": random.randint(0, 50),
                    "health_facilities": random.randint(0, 3)
                }
            }

            terrain_points.append(feature_point)

    for i in range(0, len(terrain_points), 20):
        batch = terrain_points[i:i+20]
        result = supabase.table("terrain_features").insert(batch).execute()
        print(f"Inserted {len(batch)} terrain features")

    print(f"✓ Seeded {len(terrain_points)} terrain feature points")

async def seed_ground_truth_labels():
    print("Seeding ground truth labels...")

    labels = []
    label_types = ["safe", "hazardous", "unknown", "suspected"]

    for i in range(30):
        lat = random.uniform(-40, 60)
        lon = random.uniform(-180, 180)

        geometry = generate_sample_polygon(lat, lon, 0.3)

        label = {
            "geometry": json.dumps(geometry),
            "label": random.choice(label_types),
            "confidence": random.uniform(0.7, 1.0),
            "labeled_by": f"operator_{random.randint(1, 5)}",
            "verification_method": random.choice(["field_survey", "remote_sensing", "expert_review"]),
            "notes": f"Sample ground truth label {i+1}",
            "photos": []
        }

        labels.append(label)

    result = supabase.table("ground_truth_labels").insert(labels).execute()
    print(f"✓ Seeded {len(labels)} ground truth labels")

async def seed_missions():
    print("Seeding missions...")

    missions = []
    statuses = ["planned", "in_progress", "completed"]

    for i in range(15):
        mission_date = datetime.now() - timedelta(days=random.randint(0, 90))

        lat = random.uniform(-40, 60)
        lon = random.uniform(-180, 180)
        area_geometry = generate_sample_polygon(lat, lon, 5.0)

        status = random.choice(statuses)

        mission = {
            "mission_name": f"Mission {chr(65 + i)}",
            "mission_date": mission_date.date().isoformat(),
            "team_id": f"team_{random.randint(1, 5)}",
            "area_geometry": json.dumps(area_geometry),
            "status": status,
            "priority_score": random.uniform(0.3, 1.0),
            "resources_allocated": {
                "personnel": random.randint(3, 12),
                "vehicles": random.randint(1, 4),
                "equipment": ["metal_detectors", "gps", "protective_gear"]
            },
            "expected_duration_hours": random.uniform(8, 72),
            "areas_cleared": random.randint(0, 50) if status == "completed" else 0,
            "hazards_found": random.randint(0, 20) if status == "completed" else 0,
            "notes": f"Sample mission {i+1}"
        }

        if status == "in_progress":
            mission["actual_start"] = (mission_date - timedelta(hours=random.randint(1, 24))).isoformat()
        elif status == "completed":
            mission["actual_start"] = (mission_date - timedelta(hours=random.randint(24, 72))).isoformat()
            mission["completed_at"] = mission_date.isoformat()
            mission["actual_duration_hours"] = random.uniform(6, 80)

        missions.append(mission)

    result = supabase.table("missions").insert(missions).execute()
    print(f"✓ Seeded {len(missions)} missions")

async def seed_model_updates():
    print("Seeding model updates...")

    updates = []

    for i in range(5):
        version_date = datetime.now() - timedelta(days=(5-i)*10)

        update = {
            "version": f"v1.{i}.0",
            "trained_at": version_date.isoformat(),
            "training_samples": random.randint(500, 2000),
            "performance_metrics": {
                "accuracy": random.uniform(0.75, 0.92),
                "precision": random.uniform(0.70, 0.90),
                "recall": random.uniform(0.68, 0.88),
                "f1_score": random.uniform(0.72, 0.89),
                "auc_roc": random.uniform(0.80, 0.95)
            },
            "active": i == 4,
            "model_config": {
                "input_dim": 15,
                "hidden_dims": [128, 64, 32],
                "dropout": 0.3,
                "learning_rate": 0.001
            },
            "training_duration_seconds": random.randint(300, 1800),
            "notes": f"Model update {i+1} - Training iteration"
        }

        updates.append(update)

    result = supabase.table("model_updates").insert(updates).execute()
    print(f"✓ Seeded {len(updates)} model updates")

async def main():
    print("=" * 60)
    print("GeoRAG Database Seeding")
    print("=" * 60)

    await seed_hazard_areas()
    await seed_terrain_features()
    await seed_ground_truth_labels()
    await seed_missions()
    await seed_model_updates()

    print("=" * 60)
    print("✓ Database seeding completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
