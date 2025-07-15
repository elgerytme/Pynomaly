/**
 * React example application using Pynomaly SDK
 */

import React, { useState } from 'react';
import { 
  PynomalyProvider, 
  usePynomaly, 
  usePynomalyAuth, 
  usePynomalyWebSocket,
  AnomalyDetector,
  useAnomalyDetection
} from '../src/frameworks/react';

// Main App with Provider
const App: React.FC = () => {
  return (
    <PynomalyProvider
      config={{
        apiKey: 'demo-api-key',
        baseUrl: 'https://api.pynomaly.com',
        debug: true
      }}
      autoConnect={true}
      onError={(error) => console.error('Pynomaly error:', error)}
      onReady={(client) => console.log('Pynomaly client ready:', client)}
    >
      <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
        <h1>Pynomaly React SDK Example</h1>
        <ClientStatus />
        <AuthenticationExample />
        <AnomalyDetectionExample />
        <WebSocketExample />
        <CustomAnomalyDetector />
      </div>
    </PynomalyProvider>
  );
};

// Component showing client status
const ClientStatus: React.FC = () => {
  const { client, isReady, isLoading, error } = usePynomaly();

  return (
    <div style={{ marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
      <h2>Client Status</h2>
      <p><strong>Ready:</strong> {isReady ? '✅ Yes' : '❌ No'}</p>
      <p><strong>Loading:</strong> {isLoading ? '⏳ Yes' : '✅ No'}</p>
      <p><strong>Client:</strong> {client ? '✅ Available' : '❌ Not Available'}</p>
      {error && <p style={{ color: 'red' }}><strong>Error:</strong> {error.message}</p>}
    </div>
  );
};

// Authentication example
const AuthenticationExample: React.FC = () => {
  const { client } = usePynomaly();
  const {
    authState,
    isAuthenticated,
    isLoading,
    user,
    login,
    loginWithApiKey,
    logout,
    error
  } = usePynomalyAuth({ client: client || undefined });

  const [email, setEmail] = useState('demo@example.com');
  const [password, setPassword] = useState('password');
  const [apiKey, setApiKey] = useState('demo-api-key');

  const handleLogin = async () => {
    try {
      await login({ email, password });
    } catch (err) {
      console.error('Login failed:', err);
    }
  };

  const handleApiKeyLogin = async () => {
    try {
      await loginWithApiKey(apiKey);
    } catch (err) {
      console.error('API key login failed:', err);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch (err) {
      console.error('Logout failed:', err);
    }
  };

  return (
    <div style={{ marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
      <h2>Authentication</h2>
      <p><strong>Authenticated:</strong> {isAuthenticated ? '✅ Yes' : '❌ No'}</p>
      <p><strong>User:</strong> {user ? user.email : 'None'}</p>
      {error && <p style={{ color: 'red' }}><strong>Error:</strong> {error}</p>}

      {!isAuthenticated ? (
        <div>
          <div style={{ marginBottom: '10px' }}>
            <h4>Login with Credentials</h4>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email"
              style={{ marginRight: '10px', padding: '5px' }}
            />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              style={{ marginRight: '10px', padding: '5px' }}
            />
            <button onClick={handleLogin} disabled={isLoading}>
              {isLoading ? 'Logging in...' : 'Login'}
            </button>
          </div>

          <div>
            <h4>Login with API Key</h4>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="API Key"
              style={{ marginRight: '10px', padding: '5px' }}
            />
            <button onClick={handleApiKeyLogin} disabled={isLoading}>
              {isLoading ? 'Logging in...' : 'Login with API Key'}
            </button>
          </div>
        </div>
      ) : (
        <button onClick={handleLogout} disabled={isLoading}>
          {isLoading ? 'Logging out...' : 'Logout'}
        </button>
      )}
    </div>
  );
};

// Anomaly detection example
const AnomalyDetectionExample: React.FC = () => {
  const { client } = usePynomaly();
  const {
    detectAnomalies,
    result,
    isLoading,
    error,
    progress,
    clear
  } = useAnomalyDetection({
    client: client || undefined,
    onSuccess: (result) => console.log('Anomaly detection completed:', result),
    onError: (error) => console.error('Anomaly detection failed:', error)
  });

  const sampleData = [
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [100, 200, 300] // This should be detected as an anomaly
  ];

  const handleDetectAnomalies = async () => {
    try {
      await detectAnomalies({
        data: sampleData,
        algorithm: 'isolation_forest',
        parameters: { contamination: 0.1 }
      });
    } catch (err) {
      console.error('Detection failed:', err);
    }
  };

  return (
    <div style={{ marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
      <h2>Anomaly Detection with Hook</h2>
      
      <div style={{ marginBottom: '10px' }}>
        <button onClick={handleDetectAnomalies} disabled={isLoading || !client}>
          {isLoading ? 'Detecting...' : 'Detect Anomalies'}
        </button>
        <button onClick={clear} disabled={isLoading} style={{ marginLeft: '10px' }}>
          Clear
        </button>
      </div>

      {isLoading && (
        <div style={{ marginBottom: '10px' }}>
          <div style={{ 
            width: '100%', 
            height: '20px', 
            backgroundColor: '#f0f0f0', 
            borderRadius: '10px', 
            overflow: 'hidden' 
          }}>
            <div style={{ 
              width: `${progress}%`, 
              height: '100%', 
              backgroundColor: '#007bff', 
              transition: 'width 0.3s ease' 
            }} />
          </div>
          <p>Progress: {progress}%</p>
        </div>
      )}

      {error && (
        <p style={{ color: 'red' }}>Error: {error.message}</p>
      )}

      {result && (
        <div style={{ marginTop: '10px' }}>
          <h4>Results:</h4>
          <p><strong>Algorithm:</strong> {result.algorithm}</p>
          <p><strong>Anomalies Found:</strong> {result.metrics.anomalyCount}</p>
          <p><strong>Processing Time:</strong> {result.processingTime}ms</p>
        </div>
      )}

      <div style={{ marginTop: '10px' }}>
        <h4>Sample Data:</h4>
        <pre style={{ backgroundColor: '#f8f9fa', padding: '10px', borderRadius: '4px', fontSize: '12px' }}>
          {JSON.stringify(sampleData, null, 2)}
        </pre>
      </div>
    </div>
  );
};

// WebSocket example
const WebSocketExample: React.FC = () => {
  const {
    ws,
    isConnected,
    isConnecting,
    error,
    connect,
    disconnect,
    send,
    connectionState
  } = usePynomalyWebSocket({
    url: 'wss://api.pynomaly.com/ws',
    autoConnect: false,
    onOpen: () => console.log('WebSocket connected'),
    onClose: () => console.log('WebSocket disconnected'),
    onError: (error) => console.error('WebSocket error:', error),
    onMessage: (message) => console.log('WebSocket message:', message)
  });

  const handleConnect = async () => {
    try {
      await connect();
    } catch (err) {
      console.error('WebSocket connection failed:', err);
    }
  };

  const handleSendMessage = () => {
    send({ type: 'ping', timestamp: Date.now() });
  };

  return (
    <div style={{ marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
      <h2>WebSocket Connection</h2>
      
      <p><strong>Connected:</strong> {isConnected ? '✅ Yes' : '❌ No'}</p>
      <p><strong>Connecting:</strong> {isConnecting ? '⏳ Yes' : '✅ No'}</p>
      <p><strong>State:</strong> {connectionState}</p>
      
      {error && <p style={{ color: 'red' }}><strong>Error:</strong> {error.message}</p>}

      <div style={{ marginTop: '10px' }}>
        {!isConnected ? (
          <button onClick={handleConnect} disabled={isConnecting}>
            {isConnecting ? 'Connecting...' : 'Connect'}
          </button>
        ) : (
          <>
            <button onClick={disconnect}>Disconnect</button>
            <button onClick={handleSendMessage} style={{ marginLeft: '10px' }}>
              Send Ping
            </button>
          </>
        )}
      </div>
    </div>
  );
};

// Custom anomaly detector component
const CustomAnomalyDetector: React.FC = () => {
  const [customResult, setCustomResult] = useState(null);

  return (
    <div style={{ marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
      <h2>Anomaly Detector Component</h2>
      <AnomalyDetector
        algorithm="auto"
        autoDetect={false}
        onResult={(result) => {
          console.log('Component result:', result);
          setCustomResult(result);
        }}
        onError={(error) => console.error('Component error:', error)}
      />
    </div>
  );
};

export default App;