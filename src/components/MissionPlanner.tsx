import { useState } from 'react';
import { Calendar, Users, MapPin, Clock } from 'lucide-react';
import { api } from '../services/api';
import { Mission } from '../types';

interface MissionPlannerProps {
  onMissionCreated?: (mission: Mission) => void;
}

export function MissionPlanner({ onMissionCreated }: MissionPlannerProps) {
  const [missionName, setMissionName] = useState('');
  const [teamId, setTeamId] = useState('');
  const [missionDate, setMissionDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [recommendation, setRecommendation] = useState<any>(null);

  const handleGetRecommendation = async () => {
    if (!teamId) {
      alert('Please enter a team ID first');
      return;
    }

    setLoading(true);
    try {
      const response = await api.missions.plan(teamId, 5);
      setRecommendation(response);
    } catch (error) {
      console.error('Error getting recommendation:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateMission = async () => {
    if (!missionName || !teamId || !missionDate) {
      alert('Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      const mission = await api.missions.create({
        mission_name: missionName,
        mission_date: missionDate,
        team_id: teamId,
        expected_duration_hours: recommendation?.estimated_duration_hours || 8,
        priority_score: recommendation ? recommendation.total_risk_score / recommendation.count : 0.5
      });

      onMissionCreated?.(mission);
      setMissionName('');
      setRecommendation(null);
    } catch (error) {
      console.error('Error creating mission:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center gap-3 mb-6">
        <MapPin className="w-6 h-6 text-green-600" />
        <h2 className="text-xl font-bold text-gray-900">Mission Planner</h2>
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Mission Name
          </label>
          <input
            type="text"
            value={missionName}
            onChange={(e) => setMissionName(e.target.value)}
            placeholder="e.g., Northern Sector Clearance"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Users className="inline w-4 h-4 mr-1" />
              Team ID
            </label>
            <input
              type="text"
              value={teamId}
              onChange={(e) => setTeamId(e.target.value)}
              placeholder="TEAM-001"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="inline w-4 h-4 mr-1" />
              Date
            </label>
            <input
              type="date"
              value={missionDate}
              onChange={(e) => setMissionDate(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
            />
          </div>
        </div>

        <button
          onClick={handleGetRecommendation}
          disabled={loading || !teamId}
          className="w-full px-4 py-2 bg-green-50 text-green-700 rounded-lg hover:bg-green-100 disabled:opacity-50 transition-colors font-medium"
        >
          {loading ? 'Analyzing...' : 'Get Area Recommendations'}
        </button>
      </div>

      {recommendation && (
        <div className="border-t border-gray-200 pt-6 space-y-4">
          <h3 className="font-semibold text-gray-900">Recommended Mission Plan</h3>

          <div className="bg-green-50 rounded-lg p-4 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Areas to Clear</span>
              <span className="font-semibold text-gray-900">{recommendation.count}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Total Risk Score</span>
              <span className="font-semibold text-gray-900">
                {recommendation.total_risk_score.toFixed(2)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Avg Uncertainty</span>
              <span className="font-semibold text-gray-900">
                {(recommendation.average_uncertainty * 100).toFixed(1)}%
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 flex items-center gap-1">
                <Clock className="w-4 h-4" />
                Estimated Duration
              </span>
              <span className="font-semibold text-gray-900">
                {recommendation.estimated_duration_hours}h
              </span>
            </div>
          </div>

          <button
            onClick={handleCreateMission}
            disabled={loading || !missionName || !missionDate}
            className="w-full px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors font-semibold"
          >
            Create Mission
          </button>
        </div>
      )}
    </div>
  );
}
