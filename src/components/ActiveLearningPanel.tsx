import { useState } from 'react';
import { Target, CheckCircle, AlertCircle, HelpCircle } from 'lucide-react';
import { api } from '../services/api';
import { ActiveLearningQuery } from '../types';

interface ActiveLearningPanelProps {
  onLabelSubmitted?: () => void;
}

export function ActiveLearningPanel({ onLabelSubmitted }: ActiveLearningPanelProps) {
  const [recommendedAreas, setRecommendedAreas] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState('hybrid');

  const loadRecommendations = async () => {
    setLoading(true);
    try {
      const response = await api.activeLearning.getNextAreas(selectedStrategy, 5);
      setRecommendedAreas(response.recommended_areas || []);
    } catch (error) {
      console.error('Error loading recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLabelSubmit = async (areaId: string, label: 'safe' | 'hazardous' | 'unknown') => {
    try {
      await api.activeLearning.submitLabel({
        geometry: { type: 'Point', coordinates: [0, 0] },
        label,
        confidence: 1.0,
        labeled_by: 'user',
        verification_method: 'field_verification',
        notes: `Labeled from active learning recommendation`
      });

      setRecommendedAreas(areas => areas.filter(a => a.id !== areaId));
      onLabelSubmitted?.();
    } catch (error) {
      console.error('Error submitting label:', error);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Target className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-bold text-gray-900">Active Learning</h2>
        </div>
        <button
          onClick={loadRecommendations}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
      </div>

      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Query Strategy
        </label>
        <select
          value={selectedStrategy}
          onChange={(e) => setSelectedStrategy(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="uncertainty">Uncertainty Sampling</option>
          <option value="diversity">Diversity Sampling</option>
          <option value="expected_change">Expected Model Change</option>
          <option value="hybrid">Hybrid (Recommended)</option>
          <option value="impact">Impact-Based</option>
        </select>
      </div>

      {recommendedAreas.length > 0 ? (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
            Recommended Areas to Label ({recommendedAreas.length})
          </h3>
          {recommendedAreas.map((area, index) => (
            <div
              key={area.id || index}
              className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h4 className="font-semibold text-gray-900">Area #{index + 1}</h4>
                  <p className="text-sm text-gray-600 mt-1">
                    Risk: {((area.risk_score || 0) * 100).toFixed(1)}% |
                    Uncertainty: {((area.uncertainty || 0) * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => handleLabelSubmit(area.id, 'safe')}
                  className="flex-1 px-3 py-2 bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors flex items-center justify-center gap-2"
                >
                  <CheckCircle className="w-4 h-4" />
                  Safe
                </button>
                <button
                  onClick={() => handleLabelSubmit(area.id, 'hazardous')}
                  className="flex-1 px-3 py-2 bg-red-50 text-red-700 rounded-lg hover:bg-red-100 transition-colors flex items-center justify-center gap-2"
                >
                  <AlertCircle className="w-4 h-4" />
                  Hazardous
                </button>
                <button
                  onClick={() => handleLabelSubmit(area.id, 'unknown')}
                  className="flex-1 px-3 py-2 bg-gray-50 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors flex items-center justify-center gap-2"
                >
                  <HelpCircle className="w-4 h-4" />
                  Unknown
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-gray-500">
          <Target className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>Click "Get Recommendations" to find areas that need labeling</p>
          <p className="text-sm mt-2">Using {selectedStrategy} strategy</p>
        </div>
      )}
    </div>
  );
}
