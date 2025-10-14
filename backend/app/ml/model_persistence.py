import torch
import os
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from app.ml.model import TerrainHazardModel, EnsembleModel

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

class ModelManager:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.current_model = None
        self.model_version = None

    def save_model(
        self,
        model: TerrainHazardModel,
        version: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        model_path = self.model_dir / f"model_{version}.pt"
        metadata_path = self.model_dir / f"model_{version}_metadata.json"

        torch.save({
            'model_state_dict': model.state_dict(),
            'version': version,
            'saved_at': datetime.now().isoformat(),
        }, model_path)

        if metadata is None:
            metadata = {}

        metadata.update({
            'version': version,
            'saved_at': datetime.now().isoformat(),
            'model_path': str(model_path),
        })

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved: {model_path}")
        return model_path

    def load_model(
        self,
        version: Optional[str] = None,
        input_dim: int = 15,
        hidden_dims: list = [128, 64, 32]
    ) -> TerrainHazardModel:
        if version is None:
            version = self._get_latest_version()

        model_path = self.model_dir / f"model_{version}.pt"

        if not model_path.exists():
            print(f"Model {version} not found, creating new model")
            model = TerrainHazardModel(input_dim=input_dim, hidden_dims=hidden_dims)
            self.current_model = model
            self.model_version = "v1.0.0-untrained"
            return model

        model = TerrainHazardModel(input_dim=input_dim, hidden_dims=hidden_dims)

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])

        self.current_model = model
        self.model_version = version

        print(f"Model loaded: {model_path}")
        return model

    def load_metadata(self, version: str) -> Dict:
        metadata_path = self.model_dir / f"model_{version}_metadata.json"

        if not metadata_path.exists():
            return {}

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _get_latest_version(self) -> str:
        model_files = list(self.model_dir.glob("model_v*.pt"))

        if not model_files:
            return "v1.0.0"

        versions = []
        for f in model_files:
            name = f.stem
            version = name.replace("model_", "")
            versions.append(version)

        versions.sort(reverse=True)
        return versions[0]

    def list_models(self) -> list:
        model_files = list(self.model_dir.glob("model_v*.pt"))

        models = []
        for f in model_files:
            name = f.stem
            version = name.replace("model_", "")

            metadata = self.load_metadata(version)

            models.append({
                'version': version,
                'path': str(f),
                'size_mb': f.stat().st_size / (1024 * 1024),
                'metadata': metadata
            })

        return sorted(models, key=lambda x: x['version'], reverse=True)

    def save_ensemble(
        self,
        ensemble: EnsembleModel,
        version: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        ensemble_path = self.model_dir / f"ensemble_{version}.pt"

        ensemble.save(str(ensemble_path))

        if metadata:
            metadata_path = self.model_dir / f"ensemble_{version}_metadata.json"
            metadata.update({
                'version': version,
                'saved_at': datetime.now().isoformat(),
                'model_path': str(ensemble_path),
                'n_models': ensemble.n_models
            })

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"Ensemble saved: {ensemble_path}")
        return ensemble_path

    def load_ensemble(
        self,
        version: Optional[str] = None,
        n_models: int = 5,
        input_dim: int = 15
    ) -> EnsembleModel:
        if version is None:
            version = self._get_latest_ensemble_version()

        ensemble_path = self.model_dir / f"ensemble_{version}.pt"

        if not ensemble_path.exists():
            print(f"Ensemble {version} not found, creating new ensemble")
            return EnsembleModel(n_models=n_models, input_dim=input_dim)

        ensemble = EnsembleModel(n_models=n_models, input_dim=input_dim)
        ensemble.load(str(ensemble_path))

        print(f"Ensemble loaded: {ensemble_path}")
        return ensemble

    def _get_latest_ensemble_version(self) -> str:
        ensemble_files = list(self.model_dir.glob("ensemble_v*.pt"))

        if not ensemble_files:
            return "v1.0.0"

        versions = []
        for f in ensemble_files:
            name = f.stem
            version = name.replace("ensemble_", "")
            versions.append(version)

        versions.sort(reverse=True)
        return versions[0]

    def get_current_model(self) -> Optional[TerrainHazardModel]:
        if self.current_model is None:
            self.current_model = self.load_model()

        return self.current_model

    def get_model_version(self) -> str:
        if self.model_version is None:
            self.load_model()

        return self.model_version or "v1.0.0"

_model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    return _model_manager
