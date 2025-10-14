import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine

class ModelExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def explain_prediction(
        self,
        features: np.ndarray,
        prediction: float,
        uncertainty: float
    ) -> Dict:
        feature_importance = self._compute_feature_importance(features)

        explanation_text = self._generate_explanation_text(
            features, prediction, uncertainty, feature_importance
        )

        top_features = self._get_top_features(feature_importance, top_k=5)

        confidence_factors = self._analyze_confidence_factors(
            prediction, uncertainty, feature_importance
        )

        return {
            'feature_importance': feature_importance,
            'top_features': top_features,
            'explanation_text': explanation_text,
            'confidence_factors': confidence_factors
        }

    def _compute_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        features_tensor.requires_grad = True

        risk, _ = self.model(features_tensor)
        risk.backward()

        gradients = features_tensor.grad.detach().numpy()[0]

        importance = np.abs(gradients * features)

        importance_normalized = importance / (importance.sum() + 1e-8)

        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance_normalized)
        }

    def _generate_explanation_text(
        self,
        features: np.ndarray,
        prediction: float,
        uncertainty: float,
        feature_importance: Dict[str, float]
    ) -> str:
        risk_level = "HIGH" if prediction > 0.7 else "MODERATE" if prediction > 0.4 else "LOW"
        confidence_level = "LOW" if uncertainty > 0.5 else "MODERATE" if uncertainty > 0.25 else "HIGH"

        top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]

        explanation = f"Risk Assessment: {risk_level} ({prediction:.2%})\n"
        explanation += f"Confidence: {confidence_level} (uncertainty: {uncertainty:.2%})\n\n"
        explanation += "Key Contributing Factors:\n"

        for i, (feature_name, importance) in enumerate(top_factors, 1):
            feature_idx = self.feature_names.index(feature_name)
            feature_value = features[feature_idx]
            readable_name = feature_name.replace('_', ' ').title()
            explanation += f"{i}. {readable_name}: {feature_value:.2f} (importance: {importance:.2%})\n"

        if prediction > 0.6 and uncertainty < 0.3:
            explanation += "\n⚠️ Strong indicators suggest hazardous conditions. Recommend immediate clearance prioritization."
        elif prediction > 0.6 and uncertainty > 0.5:
            explanation += "\n⚠️ High risk predicted but with significant uncertainty. Recommend additional ground verification."
        elif prediction < 0.3 and uncertainty < 0.3:
            explanation += "\n✓ Area shows low hazard probability with high confidence. May be suitable for lower priority."
        else:
            explanation += "\n→ Moderate risk assessment. Recommend standard clearance protocols."

        return explanation

    def _get_top_features(self, feature_importance: Dict[str, float], top_k: int = 5) -> List[Dict]:
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {
                'name': name.replace('_', ' ').title(),
                'importance': importance,
                'rank': i + 1
            }
            for i, (name, importance) in enumerate(sorted_features)
        ]

    def _analyze_confidence_factors(
        self,
        prediction: float,
        uncertainty: float,
        feature_importance: Dict[str, float]
    ) -> Dict:
        confidence_score = 1.0 - uncertainty

        model_certainty = "high" if uncertainty < 0.25 else "moderate" if uncertainty < 0.5 else "low"

        max_importance = max(feature_importance.values())
        feature_concentration = "concentrated" if max_importance > 0.4 else "distributed"

        prediction_strength = "strong" if abs(prediction - 0.5) > 0.3 else "weak"

        return {
            'confidence_score': float(confidence_score),
            'model_certainty': model_certainty,
            'feature_concentration': feature_concentration,
            'prediction_strength': prediction_strength,
            'uncertainty': float(uncertainty),
            'needs_verification': uncertainty > 0.5 or (prediction > 0.6 and uncertainty > 0.3)
        }

    def find_similar_areas(
        self,
        target_features: np.ndarray,
        database_features: List[np.ndarray],
        database_ids: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        similarities = []

        for db_features, db_id in zip(database_features, database_ids):
            similarity = 1 - cosine(target_features, db_features)
            similarities.append((db_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return [
            {'area_id': area_id, 'similarity': float(sim)}
            for area_id, sim in similarities[:top_k]
        ]

    def counterfactual_explanation(
        self,
        features: np.ndarray,
        target_prediction: float = 0.3
    ) -> Dict:
        current_features = features.copy()
        features_tensor = torch.FloatTensor(current_features).unsqueeze(0)

        with torch.no_grad():
            current_risk, _ = self.model(features_tensor)
            current_prediction = current_risk.item()

        modifications = {}

        for i, feature_name in enumerate(self.feature_names):
            if 'proximity' in feature_name.lower():
                test_features = current_features.copy()
                test_features[i] *= 1.5

                test_tensor = torch.FloatTensor(test_features).unsqueeze(0)
                with torch.no_grad():
                    test_risk, _ = self.model(test_tensor)
                    new_prediction = test_risk.item()

                impact = current_prediction - new_prediction
                if abs(impact) > 0.05:
                    modifications[feature_name] = {
                        'current_value': float(current_features[i]),
                        'suggested_value': float(test_features[i]),
                        'impact': float(impact),
                        'change_type': 'increase'
                    }

        return {
            'current_prediction': float(current_prediction),
            'target_prediction': float(target_prediction),
            'suggested_modifications': modifications
        }
