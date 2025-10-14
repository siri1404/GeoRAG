import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Circle, Popup, useMap } from 'react-leaflet';
import { HazardArea } from '../types';
import 'leaflet/dist/leaflet.css';

interface MapViewProps {
  hazardAreas: HazardArea[];
  onAreaClick?: (area: HazardArea) => void;
  center?: [number, number];
  zoom?: number;
}

function getRiskColor(riskScore: number): string {
  if (riskScore >= 0.7) return '#DC2626';
  if (riskScore >= 0.5) return '#F59E0B';
  if (riskScore >= 0.3) return '#FCD34D';
  return '#10B981';
}

function getUncertaintyOpacity(uncertainty: number): number {
  return Math.max(0.3, 1 - uncertainty);
}

export function MapView({ hazardAreas, onAreaClick, center = [0, 0], zoom = 6 }: MapViewProps) {
  const [mapCenter, setMapCenter] = useState<[number, number]>(center);

  useEffect(() => {
    if (hazardAreas.length > 0 && hazardAreas[0].geometry?.coordinates) {
      const firstArea = hazardAreas[0];
      if (firstArea.geometry.type === 'Point') {
        setMapCenter([firstArea.geometry.coordinates[1], firstArea.geometry.coordinates[0]]);
      } else if (firstArea.geometry.type === 'Polygon' && firstArea.geometry.coordinates[0]) {
        const coords = firstArea.geometry.coordinates[0][0];
        setMapCenter([coords[1], coords[0]]);
      }
    }
  }, [hazardAreas]);

  return (
    <div className="h-full w-full rounded-lg overflow-hidden shadow-lg">
      <MapContainer
        center={mapCenter}
        zoom={zoom}
        className="h-full w-full"
        style={{ minHeight: '500px' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {hazardAreas.map((area) => {
          let position: [number, number] = [0, 0];
          let radius = 500;

          if (area.geometry?.coordinates) {
            if (area.geometry.type === 'Point') {
              position = [area.geometry.coordinates[1], area.geometry.coordinates[0]];
            } else if (area.geometry.type === 'Polygon' && area.geometry.coordinates[0]) {
              const coords = area.geometry.coordinates[0][0];
              position = [coords[1], coords[0]];
              radius = 800;
            }
          }

          return (
            <Circle
              key={area.id}
              center={position}
              radius={radius}
              pathOptions={{
                fillColor: getRiskColor(area.risk_score),
                fillOpacity: getUncertaintyOpacity(area.uncertainty),
                color: getRiskColor(area.risk_score),
                weight: 2,
              }}
              eventHandlers={{
                click: () => onAreaClick?.(area),
              }}
            >
              <Popup>
                <div className="p-2">
                  <h3 className="font-bold text-lg mb-2">Hazard Area</h3>
                  <div className="space-y-1">
                    <p className="text-sm">
                      <span className="font-semibold">Risk Score:</span>{' '}
                      <span className={`font-bold ${area.risk_score >= 0.7 ? 'text-red-600' : area.risk_score >= 0.5 ? 'text-orange-600' : 'text-yellow-600'}`}>
                        {(area.risk_score * 100).toFixed(1)}%
                      </span>
                    </p>
                    <p className="text-sm">
                      <span className="font-semibold">Uncertainty:</span> {(area.uncertainty * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm">
                      <span className="font-semibold">Model:</span> {area.model_version}
                    </p>
                    {area.priority_rank && (
                      <p className="text-sm">
                        <span className="font-semibold">Priority:</span> #{area.priority_rank}
                      </p>
                    )}
                  </div>
                </div>
              </Popup>
            </Circle>
          );
        })}
      </MapContainer>
    </div>
  );
}
