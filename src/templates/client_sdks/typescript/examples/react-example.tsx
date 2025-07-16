/**
 * React Example: Pynomaly SDK Integration
 * Demonstrates how to use the Pynomaly TypeScript SDK in React applications
 */

import React, { useState, useEffect, useCallback } from 'react';
import { PynomaliClient } from '../src/index';
import type {
  DetectionRequest,
  DetectionResponse,
  StreamDetectionResult,
  StreamAlert,
  HealthStatus,
} from '../src/types';

// Custom hook for Pynomaly client
export const usePynomaliClient = (config?: any) => {
  const [client] = useState(() => new PynomaliClient({
    baseUrl: process.env.REACT_APP_PYNOMALY_API_URL || 'https://api.pynomaly.com',
    apiKey: process.env.REACT_APP_PYNOMALY_API_KEY,
    debug: process.env.NODE_ENV === 'development',
    websocket: {
      enabled: true,
      autoReconnect: true,
    },
    ...config,
  }));

  return client;
};

// Authentication component
export const AuthComponent: React.FC = () => {
  const client = usePynomaliClient();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
    mfaCode: '',
  });

  useEffect(() => {
    // Check if already authenticated
    const clientInfo = client.getClientInfo();
    setIsAuthenticated(clientInfo.isAuthenticated);
    setUser(clientInfo.sessionInfo);
  }, [client]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await client.auth.login(credentials);
      setIsAuthenticated(true);
      setUser(response.user);
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    setLoading(true);
    try {
      await client.auth.logout();
      setIsAuthenticated(false);
      setUser(null);
    } catch (err: any) {
      setError(err.message || 'Logout failed');
    } finally {
      setLoading(false);
    }
  };

  if (isAuthenticated) {
    return (
      <div className="auth-container">
        <h2>Welcome, {user?.username}!</h2>
        <button onClick={handleLogout} disabled={loading}>
          {loading ? 'Logging out...' : 'Logout'}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
    );
  }

  return (
    <div className="auth-container">
      <form onSubmit={handleLogin}>
        <h2>Login to Pynomaly</h2>
        
        <div>
          <label>Username:</label>
          <input
            type="text"
            value={credentials.username}
            onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
            required
          />
        </div>

        <div>
          <label>Password:</label>
          <input
            type="password"
            value={credentials.password}
            onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
            required
          />
        </div>

        <div>
          <label>MFA Code (optional):</label>
          <input
            type="text"
            value={credentials.mfaCode}
            onChange={(e) => setCredentials({ ...credentials, mfaCode: e.target.value })}
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Logging in...' : 'Login'}
        </button>

        {error && <div className="error">{error}</div>}
      </form>
    </div>
  );
};

// Anomaly detection component
export const AnomalyDetectionComponent: React.FC = () => {
  const client = usePynomaliClient();
  const [inputData, setInputData] = useState('1,2,3,4,5,100,6,7,8,9');
  const [algorithm, setAlgorithm] = useState('isolation_forest');
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDetection = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = inputData.split(',').map(Number);
      
      const request: DetectionRequest = {
        data,
        algorithm: algorithm as any,
        parameters: {
          contamination: 0.1,
          n_estimators: 100,
        },
        includeExplanations: true,
      };

      const response = await client.detection.detect(request);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Detection failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="detection-container">
      <h2>Anomaly Detection</h2>
      
      <div>
        <label>Input Data (comma-separated):</label>
        <input
          type="text"
          value={inputData}
          onChange={(e) => setInputData(e.target.value)}
          placeholder="1,2,3,4,5,100,6,7,8,9"
        />
      </div>

      <div>
        <label>Algorithm:</label>
        <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
          <option value="isolation_forest">Isolation Forest</option>
          <option value="one_class_svm">One-Class SVM</option>
          <option value="local_outlier_factor">Local Outlier Factor</option>
          <option value="elliptic_envelope">Elliptic Envelope</option>
        </select>
      </div>

      <button onClick={handleDetection} disabled={loading}>
        {loading ? 'Detecting...' : 'Detect Anomalies'}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <h3>Results</h3>
          <div>
            <strong>Anomaly Rate:</strong> {(result.anomalyRate * 100).toFixed(2)}%
          </div>
          <div>
            <strong>Processing Time:</strong> {result.processingTime}ms
          </div>
          <div>
            <strong>Algorithm:</strong> {result.algorithm}
          </div>
          
          <h4>Scores and Predictions:</h4>
          <table>
            <thead>
              <tr>
                <th>Index</th>
                <th>Score</th>
                <th>Prediction</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {result.scores.map((score, index) => (
                <tr key={index} className={result.predictions[index] ? 'anomaly' : ''}>
                  <td>{index}</td>
                  <td>{score.toFixed(4)}</td>
                  <td>{result.predictions[index] ? 'Anomaly' : 'Normal'}</td>
                  <td>{result.confidence[index]?.toFixed(4) || 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

// Real-time streaming component
export const StreamingComponent: React.FC = () => {
  const client = usePynomaliClient();
  const [isConnected, setIsConnected] = useState(false);
  const [streamData, setStreamData] = useState<StreamDetectionResult[]>([]);
  const [alerts, setAlerts] = useState<StreamAlert[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConnect = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      await client.connectWebSocket({
        onConnect: () => {
          setIsConnected(true);
          console.log('WebSocket connected');
        },
        onDisconnect: () => {
          setIsConnected(false);
          console.log('WebSocket disconnected');
        },
        onData: (data: StreamDetectionResult) => {
          setStreamData(prev => [...prev.slice(-19), data]); // Keep last 20 results
        },
        onAlert: (alert: StreamAlert) => {
          setAlerts(prev => [...prev.slice(-9), alert]); // Keep last 10 alerts
        },
        onError: (error: Error) => {
          setError(error.message);
        },
      });
    } catch (err: any) {
      setError(err.message || 'Connection failed');
    } finally {
      setLoading(false);
    }
  }, [client]);

  const handleDisconnect = useCallback(() => {
    client.disconnectWebSocket();
    setIsConnected(false);
  }, [client]);

  const sendTestData = useCallback(async () => {
    if (!isConnected) return;

    try {
      const streamingManager = client.getStreamingManager();
      if (!streamingManager) return;

      // Generate random test data
      const data = Array.from({ length: 10 }, () => Math.random() * 100);
      
      await streamingManager.sendStreamData('test-stream', data);
    } catch (err: any) {
      setError(err.message || 'Failed to send data');
    }
  }, [client, isConnected]);

  return (
    <div className="streaming-container">
      <h2>Real-time Streaming</h2>
      
      <div className="controls">
        {!isConnected ? (
          <button onClick={handleConnect} disabled={loading}>
            {loading ? 'Connecting...' : 'Connect WebSocket'}
          </button>
        ) : (
          <>
            <button onClick={handleDisconnect}>Disconnect</button>
            <button onClick={sendTestData}>Send Test Data</button>
          </>
        )}
      </div>

      <div className="status">
        Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
      </div>

      {error && <div className="error">{error}</div>}

      {alerts.length > 0 && (
        <div className="alerts">
          <h3>Recent Alerts</h3>
          {alerts.map((alert, index) => (
            <div key={index} className={`alert alert-${alert.level}`}>
              <strong>{alert.level.toUpperCase()}:</strong> {alert.message}
              <span className="score">Score: {alert.score.toFixed(4)}</span>
            </div>
          ))}
        </div>
      )}

      {streamData.length > 0 && (
        <div className="stream-data">
          <h3>Real-time Results</h3>
          <div className="data-grid">
            {streamData.map((data, index) => (
              <div key={index} className="data-item">
                <div>Stream: {data.streamId}</div>
                <div>Anomalies: {data.result.anomalyRate.toFixed(2)}%</div>
                <div>Time: {new Date(data.timestamp).toLocaleTimeString()}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Health monitoring component
export const HealthComponent: React.FC = () => {
  const client = usePynomaliClient();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const healthStatus = await client.health.getHealth();
      setHealth(healthStatus);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch health status');
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchHealth, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, fetchHealth]);

  return (
    <div className="health-container">
      <h2>System Health</h2>
      
      <div className="controls">
        <button onClick={fetchHealth} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
        <label>
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          Auto-refresh (5s)
        </label>
      </div>

      {error && <div className="error">{error}</div>}

      {health && (
        <div className="health-status">
          <div className={`overall-status status-${health.status}`}>
            <h3>Overall Status: {health.status.toUpperCase()}</h3>
            <div>Version: {health.version}</div>
            <div>Uptime: {Math.floor(health.uptime / 3600)}h {Math.floor((health.uptime % 3600) / 60)}m</div>
          </div>

          <div className="services">
            <h4>Services</h4>
            {health.services.map((service, index) => (
              <div key={index} className={`service status-${service.status}`}>
                <strong>{service.name}</strong>
                <span className="status">{service.status}</span>
                {service.responseTime && (
                  <span className="response-time">{service.responseTime}ms</span>
                )}
                {service.error && (
                  <div className="service-error">{service.error}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Main application component
export const PynomaliApp: React.FC = () => {
  const [activeTab, setActiveTab] = useState('auth');

  const tabs = [
    { id: 'auth', label: 'Authentication', component: AuthComponent },
    { id: 'detection', label: 'Anomaly Detection', component: AnomalyDetectionComponent },
    { id: 'streaming', label: 'Real-time Streaming', component: StreamingComponent },
    { id: 'health', label: 'System Health', component: HealthComponent },
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || AuthComponent;

  return (
    <div className="pynomaly-app">
      <header>
        <h1>Pynomaly React Example</h1>
        <nav>
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={activeTab === tab.id ? 'active' : ''}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main>
        <ActiveComponent />
      </main>

      <style jsx>{`
        .pynomaly-app {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
          font-family: Arial, sans-serif;
        }

        header {
          border-bottom: 2px solid #eee;
          margin-bottom: 20px;
          padding-bottom: 20px;
        }

        nav button {
          margin-right: 10px;
          padding: 10px 20px;
          border: 1px solid #ddd;
          background: white;
          cursor: pointer;
        }

        nav button.active {
          background: #007bff;
          color: white;
        }

        .error {
          color: #dc3545;
          background: #f8d7da;
          padding: 10px;
          border-radius: 4px;
          margin: 10px 0;
        }

        .status-healthy { color: #28a745; }
        .status-degraded { color: #ffc107; }
        .status-unhealthy { color: #dc3545; }

        .alert-info { background: #d1ecf1; }
        .alert-warning { background: #fff3cd; }
        .alert-critical { background: #f8d7da; }

        table {
          width: 100%;
          border-collapse: collapse;
          margin: 10px 0;
        }

        th, td {
          border: 1px solid #ddd;
          padding: 8px;
          text-align: left;
        }

        tr.anomaly {
          background-color: #ffe6e6;
        }

        .data-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 10px;
          margin: 10px 0;
        }

        .data-item {
          border: 1px solid #ddd;
          padding: 10px;
          border-radius: 4px;
        }
      `}</style>
    </div>
  );
};

export default PynomaliApp;