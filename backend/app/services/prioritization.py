import numpy as np
from typing import List, Dict, Any
from app.models.schemas import GeoJSONGeometry, PrioritizationResponse

class PrioritizationService:
    def prioritize_areas(
        self,
        area_geometries: List[GeoJSONGeometry],
        constraints: Dict[str, Any],
        weights: Dict[str, float]
    ) -> PrioritizationResponse:
        ranked_areas = []

        for i, geometry in enumerate(area_geometries):
            risk_score = np.random.uniform(0.4, 0.9)
            uncertainty_score = np.random.uniform(0.1, 0.6)
            population_impact = np.random.uniform(0.3, 1.0)
            accessibility = np.random.uniform(0.4, 0.9)
            cost = np.random.uniform(0.2, 0.8)

            priority_score = (
                weights.get('risk', 0.3) * risk_score +
                weights.get('uncertainty', 0.2) * uncertainty_score +
                weights.get('population_impact', 0.3) * population_impact +
                weights.get('accessibility', 0.1) * accessibility -
                weights.get('cost', 0.1) * cost
            )

            ranked_areas.append({
                'area_id': i,
                'geometry': geometry.model_dump(),
                'priority_score': float(priority_score),
                'risk_score': float(risk_score),
                'uncertainty_score': float(uncertainty_score),
                'population_impact': float(population_impact),
                'accessibility': float(accessibility),
                'estimated_cost': float(cost)
            })

        ranked_areas.sort(key=lambda x: x['priority_score'], reverse=True)

        optimal_sequence = [area['area_id'] for area in ranked_areas]

        total_risk_reduced = sum(area['risk_score'] * area['population_impact'] for area in ranked_areas)
        total_cost = sum(area['estimated_cost'] for area in ranked_areas)
        efficiency = total_risk_reduced / (total_cost + 1e-8)

        expected_impact = {
            'total_risk_reduced': float(total_risk_reduced),
            'total_cost': float(total_cost),
            'efficiency_ratio': float(efficiency),
            'areas_prioritized': len(ranked_areas)
        }

        return PrioritizationResponse(
            ranked_areas=ranked_areas,
            optimal_sequence=optimal_sequence,
            expected_impact=expected_impact
        )
