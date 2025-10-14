import { useState, useEffect } from 'react';
import { MapPin, Activity, Target, Brain, Menu, X } from 'lucide-react';
import { MapView } from './components/MapView';
import { Dashboard } from './components/Dashboard';
import { ActiveLearningPanel } from './components/ActiveLearningPanel';
import { MissionPlanner } from './components/MissionPlanner';
import { ExplanationPanel } from './components/ExplanationPanel';
import { HazardArea } from './types';
import { api } from './services/api';

type View = 'dashboard' | 'map' | 'active-learning' | 'missions' | 'explanations';

function App() {
  const [currentView, setCurrentView] = useState<View>('dashboard');
  const [hazardAreas, setHazardAreas] = useState<HazardArea[]>([]);
  const [selectedArea, setSelectedArea] = useState<HazardArea | null>(null);
  const [loading, setLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    loadHazardAreas();
  }, []);

  const loadHazardAreas = async () => {
    setLoading(true);
    try {
      const data = await api.hazards.getAll({ limit: 100 });
      setHazardAreas(data || []);
    } catch (error) {
      console.error('Error loading hazard areas:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAreaClick = (area: HazardArea) => {
    setSelectedArea(area);
    setCurrentView('explanations');
  };

  const navigation = [
    { id: 'dashboard', name: 'Dashboard', icon: Activity },
    { id: 'map', name: 'Risk Map', icon: MapPin },
    { id: 'active-learning', name: 'Active Learning', icon: Target },
    { id: 'missions', name: 'Mission Planning', icon: MapPin },
    { id: 'explanations', name: 'Explanations', icon: Brain },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <MapPin className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">GeoRAG</h1>
                <p className="text-xs text-gray-500">Geospatial Reasoning Framework</p>
              </div>
            </div>

            <div className="hidden md:flex items-center gap-2">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => setCurrentView(item.id as View)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      currentView === item.id
                        ? 'bg-blue-50 text-blue-700 font-semibold'
                        : 'text-gray-600 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {item.name}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-gray-600 hover:bg-gray-50 rounded-lg"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {mobileMenuOpen && (
          <div className="md:hidden border-t border-gray-200 bg-white">
            <div className="px-4 py-2 space-y-1">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setCurrentView(item.id as View);
                      setMobileMenuOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                      currentView === item.id
                        ? 'bg-blue-50 text-blue-700 font-semibold'
                        : 'text-gray-600 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    {item.name}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading && currentView === 'map' ? (
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading hazard data...</p>
            </div>
          </div>
        ) : (
          <>
            {currentView === 'dashboard' && <Dashboard />}

            {currentView === 'map' && (
              <div>
                <div className="mb-6 bg-white rounded-lg shadow-sm p-4">
                  <h2 className="text-lg font-semibold text-gray-900 mb-2">
                    Hazard Risk Map
                  </h2>
                  <p className="text-sm text-gray-600">
                    Showing {hazardAreas.length} hazard areas. Click on any area for details.
                  </p>
                </div>
                <div className="h-[600px]">
                  <MapView
                    hazardAreas={hazardAreas}
                    onAreaClick={handleAreaClick}
                    center={[20, 0]}
                    zoom={3}
                  />
                </div>
              </div>
            )}

            {currentView === 'active-learning' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ActiveLearningPanel onLabelSubmitted={loadHazardAreas} />
                <div className="space-y-6">
                  <div className="bg-white rounded-lg shadow-md p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      Active Learning Benefits
                    </h3>
                    <ul className="space-y-3 text-sm text-gray-700">
                      <li className="flex items-start gap-2">
                        <span className="text-green-600 font-bold">✓</span>
                        <span>Reduces labeling effort by up to 60%</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-600 font-bold">✓</span>
                        <span>Prioritizes high-uncertainty areas for verification</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-600 font-bold">✓</span>
                        <span>Improves model accuracy with fewer samples</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-600 font-bold">✓</span>
                        <span>Adapts to new patterns as operations progress</span>
                      </li>
                    </ul>
                  </div>
                  <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
                    <h4 className="font-semibold text-blue-900 mb-2">How It Works</h4>
                    <p className="text-sm text-blue-800">
                      The system analyzes all unlabeled areas and recommends which ones would
                      most improve the model if labeled. This intelligent selection ensures
                      maximum learning efficiency during field operations.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {currentView === 'missions' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <MissionPlanner onMissionCreated={() => {}} />
                <div className="space-y-6">
                  <div className="bg-white rounded-lg shadow-md p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      Mission Planning Features
                    </h3>
                    <ul className="space-y-3 text-sm text-gray-700">
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 font-bold">→</span>
                        <span>Automated area prioritization based on risk and impact</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 font-bold">→</span>
                        <span>Resource optimization for team allocation</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 font-bold">→</span>
                        <span>Duration estimates based on historical data</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 font-bold">→</span>
                        <span>Real-time progress tracking and updates</span>
                      </li>
                    </ul>
                  </div>
                  <div className="bg-green-50 rounded-lg p-6 border border-green-200">
                    <h4 className="font-semibold text-green-900 mb-2">Prioritization Algorithm</h4>
                    <p className="text-sm text-green-800">
                      Missions are optimized using multi-criteria decision analysis,
                      balancing risk scores, population impact, accessibility, and
                      operational costs to maximize civilian safety benefits.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {currentView === 'explanations' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ExplanationPanel hazardAreaId={selectedArea?.id} />
                <div className="space-y-6">
                  <div className="bg-white rounded-lg shadow-md p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      Model Transparency
                    </h3>
                    <ul className="space-y-3 text-sm text-gray-700">
                      <li className="flex items-start gap-2">
                        <span className="text-purple-600 font-bold">●</span>
                        <span>Understand why the model made specific predictions</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-purple-600 font-bold">●</span>
                        <span>Identify key terrain features driving risk assessment</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-purple-600 font-bold">●</span>
                        <span>Build trust through transparent decision-making</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-purple-600 font-bold">●</span>
                        <span>Validate model reasoning with domain expertise</span>
                      </li>
                    </ul>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-6 border border-purple-200">
                    <h4 className="font-semibold text-purple-900 mb-2">Explainability Methods</h4>
                    <p className="text-sm text-purple-800">
                      The system uses gradient-based feature attribution to quantify the
                      contribution of each terrain characteristic to the final risk prediction,
                      providing actionable insights for operational teams.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-600">
            GeoRAG - Geospatial Reasoning Framework for Hazard Prediction and Humanitarian Operations
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
