import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np

class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, training=True)

class TerrainHazardModel(nn.Module):
    def __init__(self, input_dim: int = 15, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                MCDropout(0.3)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.risk_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            MCDropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        risk = self.risk_head(features)
        uncertainty = self.uncertainty_head(features)
        return risk, uncertainty

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> Dict[str, np.ndarray]:
        self.train()

        predictions = []
        uncertainties = []

        with torch.no_grad():
            for _ in range(n_samples):
                risk, uncertainty = self.forward(x)
                predictions.append(risk.cpu().numpy())
                uncertainties.append(uncertainty.cpu().numpy())

        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)

        mean_prediction = predictions.mean(axis=0)
        epistemic_uncertainty = predictions.std(axis=0)
        aleatoric_uncertainty = uncertainties.mean(axis=0)
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)

        confidence_lower = np.percentile(predictions, 5, axis=0)
        confidence_upper = np.percentile(predictions, 95, axis=0)

        return {
            'risk_score': mean_prediction.flatten(),
            'uncertainty': total_uncertainty.flatten(),
            'epistemic_uncertainty': epistemic_uncertainty.flatten(),
            'aleatoric_uncertainty': aleatoric_uncertainty.flatten(),
            'confidence_interval': [confidence_lower.flatten(), confidence_upper.flatten()]
        }

class EnsembleModel:
    def __init__(self, n_models: int = 5, input_dim: int = 15):
        self.models = [TerrainHazardModel(input_dim=input_dim) for _ in range(n_models)]
        self.n_models = n_models

    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        all_predictions = []

        for model in self.models:
            result = model.predict_with_uncertainty(x, n_samples=20)
            all_predictions.append(result['risk_score'])

        all_predictions = np.array(all_predictions)

        mean_prediction = all_predictions.mean(axis=0)
        ensemble_uncertainty = all_predictions.std(axis=0)

        confidence_lower = np.percentile(all_predictions, 5, axis=0)
        confidence_upper = np.percentile(all_predictions, 95, axis=0)

        return {
            'risk_score': mean_prediction,
            'uncertainty': ensemble_uncertainty,
            'confidence_interval': [confidence_lower, confidence_upper]
        }

    def save(self, path: str):
        torch.save({
            f'model_{i}': model.state_dict()
            for i, model in enumerate(self.models)
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        for i, model in enumerate(self.models):
            model.load_state_dict(checkpoint[f'model_{i}'])

def extract_features(terrain_data: Dict[str, float]) -> np.ndarray:
    feature_names = [
        'elevation', 'slope', 'aspect', 'curvature',
        'proximity_to_conflict', 'proximity_to_roads',
        'proximity_to_buildings', 'proximity_to_water',
        'population_density', 'accessibility_score',
        'ndvi', 'land_cover_forest', 'land_cover_urban',
        'land_cover_agriculture', 'land_cover_bare'
    ]

    features = []
    for fname in feature_names:
        features.append(terrain_data.get(fname, 0.0))

    return np.array(features, dtype=np.float32)
