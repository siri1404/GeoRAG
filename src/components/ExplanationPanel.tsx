import { useState } from 'react';
import { Brain, Info, TrendingUp } from 'lucide-react';
import { api } from '../services/api';
import { ModelExplanation } from '../types';

interface ExplanationPanelProps {
  hazardAreaId?: string;
}

export function ExplanationPanel({ hazardAreaId }: ExplanationPanelProps) {
  const [explanation, setExplanation] = useState<ModelExplanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [areaIdInput, setAreaIdInput] = useState('');

  const loadExplanation = async (areaId: string) => {
    setLoading(true);
    try {
      const response = await api.models.explain(areaId);
      setExplanation(response);
    } catch (error) {
      console.error('Error loading explanation:', error);
      alert('Could not load explanation. Make sure the hazard area ID is valid.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => {
    const idToUse = hazardAreaId || areaIdInput;
    if (idToUse) {
      loadExplanation(idToUse);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-6 h-6 text-purple-600" />
        <h2 className="text-xl font-bold text-gray-900">Model Explanation</h2>
      </div>

      {!hazardAreaId && (
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Hazard Area ID
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={areaIdInput}
              onChange={(e) => setAreaIdInput(e.target.value)}
              placeholder="Enter hazard area ID"
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
            <button
              onClick={handleSubmit}
              disabled={loading || !areaIdInput}
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors"
            >
              Explain
            </button>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-purple-600"></div>
        </div>
      )}

      {!loading && explanation && (
        <div className="space-y-6">
          <div className="bg-purple-50 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
              <Info className="w-5 h-5 text-purple-600" />
              Explanation
            </h3>
            <p className="text-sm text-gray-700 whitespace-pre-line">
              {explanation.explanation_text}
            </p>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-purple-600" />
              Top Contributing Features
            </h3>
            <div className="space-y-2">
              {explanation.top_features.map((feature) => (
                <div key={feature.name} className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm font-medium text-gray-700">
                        {feature.rank}. {feature.name}
                      </span>
                      <span className="text-sm text-gray-600">
                        {(feature.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-purple-600 rounded-full transition-all"
                        style={{ width: `${feature.importance * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="border-t border-gray-200 pt-4">
            <h3 className="font-semibold text-gray-900 mb-3">Confidence Factors</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-xs text-gray-600 mb-1">Model Certainty</p>
                <p className="font-semibold text-gray-900 capitalize">
                  {explanation.confidence_factors.model_certainty}
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-xs text-gray-600 mb-1">Confidence Score</p>
                <p className="font-semibold text-gray-900">
                  {(explanation.confidence_factors.confidence_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {explanation.confidence_factors.needs_verification && (
              <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                <p className="text-sm text-yellow-800">
                  ⚠️ This prediction has high uncertainty. Ground verification is recommended.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {!loading && !explanation && (
        <div className="text-center py-12 text-gray-500">
          <Brain className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>Enter a hazard area ID to see model explanation</p>
        </div>
      )}
    </div>
  );
}
