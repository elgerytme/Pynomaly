/**
 * React Framework Integration Examples
 * 
 * This file demonstrates how to integrate the Pynomaly JavaScript SDK
 * with React applications using hooks, context, and components.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  PynomalyProvider,
  usePynomalyClient,
  usePynomalyAuth,
  useAnomalyDetection,
  useWebSocket
} from '@pynomaly/js-sdk/react';

// Example 1: Basic App Setup with Provider
export function PynomalyApp() {
  const config = {
    apiKey: process.env.REACT_APP_PYNOMALY_API_KEY,
    baseUrl: process.env.REACT_APP_PYNOMALY_BASE_URL || 'https://api.pynomaly.com',
    debug: process.env.NODE_ENV === 'development'
  };

  return (
    <PynomalyProvider
      config={config}
      autoConnect={true}
      onReady={(client) => console.log('Pynomaly client ready:', client)}
      onError={(error) => console.error('Pynomaly error:', error)}
    >
      <div className="app">
        <Header />
        <MainContent />
        <Footer />
      </div>
    </PynomalyProvider>
  );
}

// Example 2: Authentication Component
export function AuthComponent() {
  const { client, isReady } = usePynomalyClient();
  const {
    authState,
    isLoading,
    error,
    login,
    loginWithApiKey,
    logout
  } = usePynomalyAuth({ client });

  const [credentials, setCredentials] = useState({
    email: '',
    password: ''
  });

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      await login(credentials);
    } catch (err) {
      console.error('Login failed:', err);
    }
  };

  const handleApiKeyLogin = async () => {
    const apiKey = prompt('Enter your API key:');
    if (apiKey) {
      try {
        await loginWithApiKey(apiKey);
      } catch (err) {
        console.error('API key login failed:', err);
      }
    }
  };

  if (!isReady) {
    return <div className="loading">Loading Pynomaly SDK...</div>;
  }

  if (authState.isAuthenticated) {
    return (
      <div className="auth-container">
        <div className="user-info">
          <h3>Welcome, {authState.user?.name || authState.user?.email}!</h3>
          <p>Logged in at: {authState.lastActivity.toLocaleString()}</p>
          <button onClick={logout} className="btn btn-secondary">
            Logout
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-container">
      <h2>Login to Pynomaly</h2>
      
      {error && (
        <div className="error-message">
          Error: {error.message}
        </div>
      )}

      <form onSubmit={handleLogin} className="login-form">
        <div className="form-group">
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            value={credentials.email}
            onChange={(e) => setCredentials(prev => ({ ...prev, email: e.target.value }))}
            required
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            value={credentials.password}
            onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
            required
            disabled={isLoading}
          />
        </div>

        <div className="form-actions">
          <button type="submit" disabled={isLoading} className="btn btn-primary">
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
          
          <button 
            type="button" 
            onClick={handleApiKeyLogin} 
            disabled={isLoading}
            className="btn btn-secondary"
          >
            Login with API Key
          </button>
        </div>
      </form>
    </div>
  );
}

// Example 3: Anomaly Detection Component
export function AnomalyDetectionComponent() {
  const { client, isReady } = usePynomalyClient();
  const {
    result,
    isLoading,
    error,
    progress,
    detectAnomalies,
    detectAnomaliesAsync,
    clear
  } = useAnomalyDetection({ client });

  const [dataset, setDataset] = useState([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [100, 200, 300] // This should be detected as an anomaly
  ]);

  const [algorithm, setAlgorithm] = useState('isolation_forest');
  const [useAsync, setUseAsync] = useState(false);

  const handleDetection = async () => {
    try {
      const params = {
        data: dataset,
        algorithm,
        parameters: {
          contamination: 0.25,
          n_estimators: 100
        }
      };

      if (useAsync) {
        await detectAnomaliesAsync(params);
      } else {
        await detectAnomalies(params);
      }
    } catch (err) {
      console.error('Detection failed:', err);
    }
  };

  const addDataPoint = () => {
    const newPoint = [
      Math.random() * 10,
      Math.random() * 10,
      Math.random() * 10
    ];
    setDataset(prev => [...prev, newPoint]);
  };

  const addAnomalyPoint = () => {
    const anomalyPoint = [
      Math.random() * 100 + 50,
      Math.random() * 100 + 50,
      Math.random() * 100 + 50
    ];
    setDataset(prev => [...prev, anomalyPoint]);
  };

  if (!isReady) {
    return <div>Waiting for Pynomaly client...</div>;
  }

  return (
    <div className="anomaly-detection">
      <h2>Anomaly Detection</h2>

      <div className="controls">
        <div className="form-group">
          <label>Algorithm:</label>
          <select 
            value={algorithm} 
            onChange={(e) => setAlgorithm(e.target.value)}
            disabled={isLoading}
          >
            <option value="isolation_forest">Isolation Forest</option>
            <option value="local_outlier_factor">Local Outlier Factor</option>
            <option value="one_class_svm">One-Class SVM</option>
            <option value="auto">Auto-select</option>
          </select>
        </div>

        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={useAsync}
              onChange={(e) => setUseAsync(e.target.checked)}
              disabled={isLoading}
            />
            Use Async Processing
          </label>
        </div>

        <div className="actions">
          <button onClick={handleDetection} disabled={isLoading || dataset.length === 0}>
            {isLoading ? 'Detecting...' : 'Detect Anomalies'}
          </button>
          <button onClick={addDataPoint} disabled={isLoading}>
            Add Normal Point
          </button>
          <button onClick={addAnomalyPoint} disabled={isLoading}>
            Add Anomaly Point
          </button>
          <button onClick={clear} disabled={isLoading}>
            Clear Results
          </button>
        </div>
      </div>

      {isLoading && progress !== undefined && (
        <div className="progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            />
          </div>
          <span>{progress}% Complete</span>
        </div>
      )}

      <div className="dataset">
        <h3>Dataset ({dataset.length} points)</h3>
        <div className="data-points">
          {dataset.map((point, index) => (
            <div key={index} className="data-point">
              [{point.map(val => val.toFixed(2)).join(', ')}]
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="error">
          <h3>Error</h3>
          <p>{error.message}</p>
        </div>
      )}

      {result && (
        <div className="results">
          <h3>Results</h3>
          <div className="metrics">
            <p><strong>Algorithm:</strong> {result.algorithm}</p>
            <p><strong>Total Points:</strong> {result.metrics.totalPoints}</p>
            <p><strong>Anomalies Found:</strong> {result.metrics.anomalyCount}</p>
            <p><strong>Anomaly Rate:</strong> {(result.metrics.anomalyRate * 100).toFixed(1)}%</p>
            <p><strong>Processing Time:</strong> {result.processingTime}ms</p>
          </div>

          {result.anomalies.length > 0 && (
            <div className="anomalies">
              <h4>Detected Anomalies</h4>
              {result.anomalies.map((anomaly, index) => (
                <div key={index} className="anomaly">
                  <p><strong>Index:</strong> {anomaly.index}</p>
                  <p><strong>Data:</strong> [{anomaly.data.join(', ')}]</p>
                  <p><strong>Score:</strong> {anomaly.score.toFixed(3)}</p>
                  <p><strong>Confidence:</strong> {(anomaly.confidence * 100).toFixed(1)}%</p>
                  <p><strong>Explanation:</strong> {anomaly.explanation}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Example 4: Real-time WebSocket Component
export function RealTimeUpdatesComponent() {
  const { client } = usePynomalyClient();
  const [messages, setMessages] = useState([]);
  const [subscriptions, setSubscriptions] = useState([]);

  const {
    ws,
    isConnected,
    connectionState,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    send
  } = useWebSocket({
    url: 'wss://api.pynomaly.com/ws',
    autoConnect: true,
    maxReconnectAttempts: 5
  });

  useEffect(() => {
    if (ws) {
      ws.on('message', (message) => {
        setMessages(prev => [...prev.slice(-49), {
          id: Date.now(),
          timestamp: new Date(),
          ...message
        }]);
      });

      ws.on('subscribed', (channel) => {
        setSubscriptions(prev => [...prev, channel]);
      });

      ws.on('unsubscribed', (channel) => {
        setSubscriptions(prev => prev.filter(sub => sub !== channel));
      });
    }
  }, [ws]);

  const handleSubscribe = (channel) => {
    if (isConnected) {
      subscribe(channel);
    }
  };

  const handleUnsubscribe = (channel) => {
    if (isConnected) {
      unsubscribe(channel);
    }
  };

  const handleSendMessage = () => {
    const message = prompt('Enter message to send:');
    if (message && isConnected) {
      send({ type: 'user-message', content: message });
    }
  };

  return (
    <div className="realtime-updates">
      <h2>Real-time Updates</h2>

      <div className="connection-status">
        <p>
          <strong>Status:</strong> 
          <span className={`status ${connectionState}`}>
            {connectionState}
          </span>
        </p>
        
        <div className="connection-actions">
          {isConnected ? (
            <button onClick={disconnect} className="btn btn-danger">
              Disconnect
            </button>
          ) : (
            <button onClick={connect} className="btn btn-primary">
              Connect
            </button>
          )}
        </div>
      </div>

      <div className="subscriptions">
        <h3>Subscriptions</h3>
        <div className="subscription-controls">
          <button 
            onClick={() => handleSubscribe('anomaly-alerts')}
            disabled={!isConnected || subscriptions.includes('anomaly-alerts')}
            className="btn btn-secondary"
          >
            Subscribe to Anomaly Alerts
          </button>
          
          <button 
            onClick={() => handleSubscribe('job-updates')}
            disabled={!isConnected || subscriptions.includes('job-updates')}
            className="btn btn-secondary"
          >
            Subscribe to Job Updates
          </button>
          
          <button 
            onClick={handleSendMessage}
            disabled={!isConnected}
            className="btn btn-info"
          >
            Send Message
          </button>
        </div>

        <div className="active-subscriptions">
          <p><strong>Active:</strong> {subscriptions.join(', ') || 'None'}</p>
        </div>
      </div>

      <div className="messages">
        <h3>Messages ({messages.length})</h3>
        <div className="message-list">
          {messages.slice().reverse().map((message) => (
            <div key={message.id} className="message">
              <div className="message-header">
                <span className="message-type">{message.type}</span>
                <span className="message-time">
                  {message.timestamp.toLocaleTimeString()}
                </span>
              </div>
              <div className="message-content">
                {typeof message.data === 'object' 
                  ? JSON.stringify(message.data, null, 2)
                  : message.data || message.content
                }
              </div>
            </div>
          ))}
          
          {messages.length === 0 && (
            <div className="no-messages">
              No messages received yet. Connect and subscribe to channels to see updates.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Example 5: Header and Footer Components
function Header() {
  const { isReady, error } = usePynomalyClient();
  const { authState } = usePynomalyAuth();

  return (
    <header className="header">
      <h1>Pynomaly React Example</h1>
      <div className="status">
        {!isReady && <span className="loading">Loading...</span>}
        {error && <span className="error">Error: {error.message}</span>}
        {isReady && authState.isAuthenticated && (
          <span className="authenticated">
            Logged in as {authState.user?.email}
          </span>
        )}
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <p>Powered by Pynomaly JavaScript SDK</p>
    </footer>
  );
}

function MainContent() {
  const [activeTab, setActiveTab] = useState('auth');

  return (
    <main className="main-content">
      <nav className="tabs">
        <button 
          className={activeTab === 'auth' ? 'active' : ''} 
          onClick={() => setActiveTab('auth')}
        >
          Authentication
        </button>
        <button 
          className={activeTab === 'detection' ? 'active' : ''} 
          onClick={() => setActiveTab('detection')}
        >
          Anomaly Detection
        </button>
        <button 
          className={activeTab === 'realtime' ? 'active' : ''} 
          onClick={() => setActiveTab('realtime')}
        >
          Real-time Updates
        </button>
      </nav>

      <div className="tab-content">
        {activeTab === 'auth' && <AuthComponent />}
        {activeTab === 'detection' && <AnomalyDetectionComponent />}
        {activeTab === 'realtime' && <RealTimeUpdatesComponent />}
      </div>
    </main>
  );
}

// CSS styles (would typically be in a separate .css file)
const styles = `
.app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: Arial, sans-serif;
}

.header {
  border-bottom: 1px solid #ddd;
  padding-bottom: 20px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.status {
  font-size: 14px;
}

.loading { color: #007bff; }
.error { color: #dc3545; }
.authenticated { color: #28a745; }

.tabs {
  display: flex;
  border-bottom: 1px solid #ddd;
  margin-bottom: 20px;
}

.tabs button {
  padding: 10px 20px;
  border: none;
  background: none;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}

.tabs button.active {
  border-bottom-color: #007bff;
  color: #007bff;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-right: 10px;
}

.btn-primary { background: #007bff; color: white; }
.btn-secondary { background: #6c757d; color: white; }
.btn-danger { background: #dc3545; color: white; }
.btn-info { background: #17a2b8; color: white; }

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.progress {
  margin: 20px 0;
}

.progress-bar {
  width: 100%;
  height: 20px;
  background: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #007bff;
  transition: width 0.3s ease;
}

.data-points {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 10px 0;
}

.data-point {
  background: #f8f9fa;
  padding: 5px 10px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 12px;
}

.message-list {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #ddd;
  padding: 10px;
  border-radius: 4px;
}

.message {
  margin-bottom: 15px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 4px;
}

.message-header {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #666;
  margin-bottom: 5px;
}

.message-content {
  font-family: monospace;
  font-size: 12px;
  white-space: pre-wrap;
}

.footer {
  border-top: 1px solid #ddd;
  padding-top: 20px;
  margin-top: 40px;
  text-align: center;
  color: #666;
}
`;

export default PynomalyApp;