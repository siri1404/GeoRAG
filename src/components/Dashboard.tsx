import { useEffect, useState } from 'react';
import { Activity, AlertTriangle, MapPin, TrendingUp } from 'lucide-react';
import { api } from '../services/api';

interface Statistics {
  missions: {
    total_missions: number;
    completed: number;
    in_progress: number;
    planned: number;
    total_areas_cleared: number;
    total_hazards_found: number;
  };
  activeLearning: {
    total_labels: number;
    labeled_queries: number;
    unlabeled_queries: number;
    labeling_efficiency: number;
  };
}

export function Dashboard() {
  const [stats, setStats] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    try {
      const [missionStats, alStats] = await Promise.all([
        api.missions.getStatistics(),
        api.activeLearning.getStatistics()
      ]);

      setStats({
        missions: missionStats,
        activeLearning: alStats
      });
    } catch (error) {
      console.error('Error loading statistics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard
        title="Total Missions"
        value={stats?.missions.total_missions || 0}
        icon={<MapPin className="w-6 h-6" />}
        color="blue"
        subtitle={`${stats?.missions.in_progress || 0} in progress`}
      />

      <StatCard
        title="Areas Cleared"
        value={stats?.missions.total_areas_cleared || 0}
        icon={<Activity className="w-6 h-6" />}
        color="green"
        subtitle={`${stats?.missions.total_hazards_found || 0} hazards found`}
      />

      <StatCard
        title="Labels Collected"
        value={stats?.activeLearning.total_labels || 0}
        icon={<TrendingUp className="w-6 h-6" />}
        color="purple"
        subtitle={`${stats?.activeLearning.unlabeled_queries || 0} pending`}
      />

      <StatCard
        title="Labeling Efficiency"
        value={`${(stats?.activeLearning.labeling_efficiency || 0).toFixed(1)}%`}
        icon={<AlertTriangle className="w-6 h-6" />}
        color="orange"
        subtitle="Active learning impact"
      />
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'purple' | 'orange';
  subtitle?: string;
}

function StatCard({ title, value, icon, color, subtitle }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    purple: 'bg-purple-50 text-purple-600',
    orange: 'bg-orange-50 text-orange-600',
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          {icon}
        </div>
      </div>
      <h3 className="text-gray-600 text-sm font-medium mb-1">{title}</h3>
      <p className="text-3xl font-bold text-gray-900">{value}</p>
      {subtitle && (
        <p className="text-sm text-gray-500 mt-2">{subtitle}</p>
      )}
    </div>
  );
}
