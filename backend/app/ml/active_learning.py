import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class ActiveLearningStrategy:
    def __init__(self, strategy: str = 'hybrid'):
        self.strategy = strategy

    def select_queries(
        self,
        candidates: List[Dict],
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        features: np.ndarray,
        n_queries: int = 5,
        constraints: Dict = None
    ) -> List[int]:

        if self.strategy == 'uncertainty':
            return self._uncertainty_sampling(uncertainties, n_queries)
        elif self.strategy == 'diversity':
            return self._diversity_sampling(features, n_queries)
        elif self.strategy == 'expected_change':
            return self._expected_model_change(predictions, uncertainties, n_queries)
        elif self.strategy == 'impact':
            return self._impact_sampling(candidates, predictions, uncertainties, n_queries)
        elif self.strategy == 'hybrid':
            return self._hybrid_sampling(
                candidates, predictions, uncertainties, features, n_queries, constraints
            )
        else:
            return self._uncertainty_sampling(uncertainties, n_queries)

    def _uncertainty_sampling(self, uncertainties: np.ndarray, n_queries: int) -> List[int]:
        indices = np.argsort(uncertainties)[::-1][:n_queries]
        return indices.tolist()

    def _diversity_sampling(self, features: np.ndarray, n_queries: int) -> List[int]:
        if len(features) <= n_queries:
            return list(range(len(features)))

        kmeans = KMeans(n_clusters=n_queries, random_state=42)
        kmeans.fit(features)

        selected_indices = []
        for cluster_center in kmeans.cluster_centers_:
            distances = cdist([cluster_center], features, metric='euclidean')[0]
            nearest_idx = np.argmin(distances)
            selected_indices.append(nearest_idx)

        return selected_indices

    def _expected_model_change(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        n_queries: int
    ) -> List[int]:
        expected_change = predictions * (1 - predictions) * uncertainties

        indices = np.argsort(expected_change)[::-1][:n_queries]
        return indices.tolist()

    def _impact_sampling(
        self,
        candidates: List[Dict],
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        n_queries: int
    ) -> List[int]:
        impact_scores = []

        for i, candidate in enumerate(candidates):
            population = candidate.get('population_density', 0)
            accessibility = candidate.get('accessibility_score', 0.5)
            strategic_value = candidate.get('strategic_value', 0.5)

            impact = (
                predictions[i] * 0.4 +
                uncertainties[i] * 0.2 +
                population * 0.2 +
                accessibility * 0.1 +
                strategic_value * 0.1
            )
            impact_scores.append(impact)

        impact_scores = np.array(impact_scores)
        indices = np.argsort(impact_scores)[::-1][:n_queries]
        return indices.tolist()

    def _hybrid_sampling(
        self,
        candidates: List[Dict],
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        features: np.ndarray,
        n_queries: int,
        constraints: Dict = None
    ) -> List[int]:
        if constraints is None:
            constraints = {}

        uncertainty_weight = constraints.get('uncertainty_weight', 0.35)
        diversity_weight = constraints.get('diversity_weight', 0.25)
        impact_weight = constraints.get('impact_weight', 0.40)

        uncertainty_scores = uncertainties / (uncertainties.max() + 1e-8)

        if len(features) > 1:
            distances = cdist(features, features, metric='euclidean')
            diversity_scores = distances.mean(axis=1)
            diversity_scores = diversity_scores / (diversity_scores.max() + 1e-8)
        else:
            diversity_scores = np.ones(len(features))

        impact_scores = []
        for candidate in candidates:
            population = candidate.get('population_density', 0)
            accessibility = candidate.get('accessibility_score', 0.5)
            impact = population * 0.6 + accessibility * 0.4
            impact_scores.append(impact)
        impact_scores = np.array(impact_scores)
        impact_scores = impact_scores / (impact_scores.max() + 1e-8)

        combined_scores = (
            uncertainty_weight * uncertainty_scores +
            diversity_weight * diversity_scores +
            impact_weight * impact_scores
        )

        indices = np.argsort(combined_scores)[::-1][:n_queries]
        return indices.tolist()

    def compute_expected_info_gain(
        self,
        uncertainty: float,
        current_model_entropy: float = 1.0
    ) -> float:
        expected_gain = uncertainty * np.log2(2) * current_model_entropy
        return float(expected_gain)

def batch_mode_selection(
    candidates: List[Dict],
    uncertainties: np.ndarray,
    features: np.ndarray,
    n_queries: int,
    max_distance: float = None
) -> List[int]:
    selected = []
    remaining = list(range(len(candidates)))

    first_idx = np.argmax(uncertainties)
    selected.append(first_idx)
    remaining.remove(first_idx)

    for _ in range(n_queries - 1):
        if not remaining:
            break

        best_score = -float('inf')
        best_idx = None

        for idx in remaining:
            uncertainty_score = uncertainties[idx]

            diversity_score = min([
                np.linalg.norm(features[idx] - features[s])
                for s in selected
            ])

            if max_distance is not None:
                spatial_distances = [
                    candidates[idx].get('distance_to', {}).get(str(s), float('inf'))
                    for s in selected
                ]
                min_spatial_dist = min(spatial_distances)
                if min_spatial_dist < max_distance:
                    continue

            combined_score = 0.6 * uncertainty_score + 0.4 * diversity_score

            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected
