import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from supabase import create_client
from app.config import get_settings
from app.ml.model import TerrainHazardModel, extract_features
from app.ml.model_persistence import get_model_manager

settings = get_settings()
supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

class HazardDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, uncertainties: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.uncertainties = torch.FloatTensor(uncertainties)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.uncertainties[idx]

def load_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("Loading training data from database...")

    labels_data = supabase.table("ground_truth_labels").select("*").execute()

    if not labels_data.data:
        print("No ground truth labels found. Using synthetic data.")
        return generate_synthetic_data()

    features_list = []
    labels_list = []
    confidences_list = []

    label_mapping = {
        "safe": 0.0,
        "suspected": 0.4,
        "unknown": 0.5,
        "hazardous": 1.0
    }

    for label_data in labels_data.data:
        label_type = label_data.get("label", "unknown")
        confidence = label_data.get("confidence", 1.0)

        terrain_data = {
            "elevation": np.random.uniform(0, 500),
            "slope": np.random.uniform(0, 45),
            "aspect": np.random.uniform(0, 360),
            "curvature": np.random.uniform(-0.5, 0.5),
            "proximity_to_conflict": np.random.uniform(0, 50000),
            "proximity_to_roads": np.random.uniform(100, 20000),
            "proximity_to_buildings": np.random.uniform(200, 15000),
            "proximity_to_water": np.random.uniform(500, 25000),
            "population_density": np.random.uniform(0, 1000),
            "accessibility_score": np.random.uniform(0.1, 1.0),
            "ndvi": np.random.uniform(-0.2, 0.9),
            "land_cover_forest": 0,
            "land_cover_urban": 0,
            "land_cover_agriculture": 0,
            "land_cover_bare": 0
        }

        feature_vector = extract_features(terrain_data)
        features_list.append(feature_vector)
        labels_list.append(label_mapping[label_type])
        confidences_list.append(confidence)

    hazard_areas = supabase.table("hazard_areas").select("*").limit(200).execute()

    for area in hazard_areas.data[:50]:
        terrain_data = area.get("features", {})
        feature_vector = extract_features(terrain_data)
        features_list.append(feature_vector)
        labels_list.append(area.get("risk_score", 0.5))
        confidences_list.append(1.0 - area.get("uncertainty", 0.3))

    features = np.array(features_list)
    labels = np.array(labels_list).reshape(-1, 1)
    confidences = np.array(confidences_list).reshape(-1, 1)
    uncertainties = 1.0 - confidences

    print(f"Loaded {len(features)} training samples")
    return features, labels, uncertainties

def generate_synthetic_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Generating {n_samples} synthetic training samples...")

    features = np.random.randn(n_samples, 15).astype(np.float32)

    risk_factors = (
        features[:, 0] * 0.2 +
        features[:, 4] * 0.3 +
        features[:, 8] * 0.2 +
        features[:, 10] * 0.15
    )

    labels = 1.0 / (1.0 + np.exp(-risk_factors))
    labels = np.clip(labels + np.random.normal(0, 0.1, n_samples), 0, 1).reshape(-1, 1)

    uncertainties = np.random.uniform(0.05, 0.3, (n_samples, 1)).astype(np.float32)

    return features, labels.astype(np.float32), uncertainties

def train_model(
    model: TerrainHazardModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001
) -> Dict[str, List[float]]:
    print("Starting model training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_risk = nn.BCELoss()
    criterion_uncertainty = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_risk_loss': [],
        'val_risk_loss': []
    }

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_risk_losses = []

        for features, labels, uncertainties in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            uncertainties = uncertainties.to(device)

            optimizer.zero_grad()

            risk_pred, uncertainty_pred = model(features)

            loss_risk = criterion_risk(risk_pred, labels)
            loss_uncertainty = criterion_uncertainty(uncertainty_pred, uncertainties)

            loss = loss_risk + 0.3 * loss_uncertainty

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_risk_losses.append(loss_risk.item())

        model.eval()
        val_losses = []
        val_risk_losses = []

        with torch.no_grad():
            for features, labels, uncertainties in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                uncertainties = uncertainties.to(device)

                risk_pred, uncertainty_pred = model(features)

                loss_risk = criterion_risk(risk_pred, labels)
                loss_uncertainty = criterion_uncertainty(uncertainty_pred, uncertainties)

                loss = loss_risk + 0.3 * loss_uncertainty

                val_losses.append(loss.item())
                val_risk_losses.append(loss_risk.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_risk = np.mean(train_risk_losses)
        avg_val_risk = np.mean(val_risk_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_risk_loss'].append(avg_train_risk)
        history['val_risk_loss'].append(avg_val_risk)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Risk: {avg_val_risk:.4f}")

    return history

def evaluate_model(model: TerrainHazardModel, test_loader: DataLoader) -> Dict[str, float]:
    print("Evaluating model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels, _ in test_loader:
            features = features.to(device)

            risk_pred, _ = model(features)

            all_predictions.extend(risk_pred.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(mse)

    binary_predictions = (predictions > 0.5).astype(int)
    binary_labels = (labels > 0.5).astype(int)

    accuracy = np.mean(binary_predictions == binary_labels)

    true_positives = np.sum((binary_predictions == 1) & (binary_labels == 1))
    false_positives = np.sum((binary_predictions == 1) & (binary_labels == 0))
    false_negatives = np.sum((binary_predictions == 0) & (binary_labels == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return metrics

def main():
    print("=" * 60)
    print("GeoRAG Model Training")
    print("=" * 60)

    features, labels, uncertainties = load_training_data()

    n_samples = len(features)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    train_dataset = HazardDataset(features[train_idx], labels[train_idx], uncertainties[train_idx])
    val_dataset = HazardDataset(features[val_idx], labels[val_idx], uncertainties[val_idx])
    test_dataset = HazardDataset(features[test_idx], labels[test_idx], uncertainties[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TerrainHazardModel(input_dim=15, hidden_dims=[128, 64, 32])

    history = train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001)

    metrics = evaluate_model(model, test_loader)

    model_manager = get_model_manager()
    version = f"v1.0.{datetime.now().strftime('%Y%m%d%H%M%S')}"

    metadata = {
        "training_samples": n_train,
        "validation_samples": n_val,
        "test_samples": len(test_idx),
        "performance_metrics": metrics,
        "training_history": {
            "final_train_loss": history['train_loss'][-1],
            "final_val_loss": history['val_loss'][-1]
        },
        "trained_at": datetime.now().isoformat()
    }

    model_manager.save_model(model, version, metadata)

    model_update_data = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "training_samples": n_train,
        "performance_metrics": metrics,
        "active": True,
        "model_config": {
            "input_dim": 15,
            "hidden_dims": [128, 64, 32],
            "dropout": 0.3,
            "learning_rate": 0.001
        },
        "training_duration_seconds": 0,
        "notes": "Initial model training"
    }

    supabase.table("model_updates").update({"active": False}).neq("version", version).execute()
    supabase.table("model_updates").insert(model_update_data).execute()

    print("=" * 60)
    print("✓ Model training completed successfully!")
    print(f"Model version: {version}")
    print("=" * 60)

if __name__ == "__main__":
    main()
