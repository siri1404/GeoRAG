import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseKey);

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  hazards: {
    getAll: async (params?: { min_risk?: number; max_risk?: number; limit?: number }) => {
      const queryParams = new URLSearchParams();
      if (params?.min_risk) queryParams.append('min_risk', params.min_risk.toString());
      if (params?.max_risk) queryParams.append('max_risk', params.max_risk.toString());
      if (params?.limit) queryParams.append('limit', params.limit.toString());

      const response = await fetch(`${API_BASE_URL}/api/v1/hazards?${queryParams}`);
      return response.json();
    },

    getById: async (id: string) => {
      const response = await fetch(`${API_BASE_URL}/api/v1/hazards/${id}`);
      return response.json();
    },

    predict: async (geometry: any, features?: any) => {
      const response = await fetch(`${API_BASE_URL}/api/v1/hazards/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ geometry, features })
      });
      return response.json();
    },

    getUncertaintyMap: async (min_uncertainty = 0.5) => {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/hazards/uncertainty-map?min_uncertainty=${min_uncertainty}`
      );
      return response.json();
    },

    submitFeedback: async (feedback: any) => {
      const response = await fetch(`${API_BASE_URL}/api/v1/hazards/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedback)
      });
      return response.json();
    }
  },

  activeLearning: {
    getNextAreas: async (strategy = 'hybrid', n_queries = 5) => {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/active-learning/next-areas?strategy=${strategy}&n_queries=${n_queries}`
      );
      return response.json();
    },

    submitLabel: async (label: any) => {
      const response = await fetch(`${API_BASE_URL}/api/v1/active-learning/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(label)
      });
      return response.json();
    },

    getStatistics: async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/active-learning/statistics`);
      return response.json();
    }
  },

  missions: {
    create: async (mission: any) => {
      const response = await fetch(`${API_BASE_URL}/api/v1/missions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mission)
      });
      return response.json();
    },

    getAll: async (status?: string) => {
      const url = status
        ? `${API_BASE_URL}/api/v1/missions?status=${status}`
        : `${API_BASE_URL}/api/v1/missions`;
      const response = await fetch(url);
      return response.json();
    },

    update: async (id: string, update: any) => {
      const response = await fetch(`${API_BASE_URL}/api/v1/missions/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(update)
      });
      return response.json();
    },

    plan: async (teamId: string, maxAreas = 5) => {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/missions/plan?team_id=${teamId}&max_areas=${maxAreas}`,
        { method: 'POST' }
      );
      return response.json();
    },

    getStatistics: async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/missions/statistics`);
      return response.json();
    }
  },

  models: {
    triggerUpdate: async (minSamples = 10) => {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/models/update?min_new_samples=${minSamples}`,
        { method: 'POST' }
      );
      return response.json();
    },

    getPerformance: async (version?: string) => {
      const url = version
        ? `${API_BASE_URL}/api/v1/models/performance?version=${version}`
        : `${API_BASE_URL}/api/v1/models/performance`;
      const response = await fetch(url);
      return response.json();
    },

    explain: async (hazardAreaId: string) => {
      const response = await fetch(`${API_BASE_URL}/api/v1/models/explain/${hazardAreaId}`);
      return response.json();
    },

    getVersions: async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/models/versions`);
      return response.json();
    }
  }
};
