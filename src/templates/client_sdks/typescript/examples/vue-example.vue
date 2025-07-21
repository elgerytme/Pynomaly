<template>
  <div class="anomaly_detection-app">
    <header>
      <h1>anomaly_detection Vue.js Example</h1>
      <nav>
        <button
          v-for="tab in tabs"
          :key="tab.id"
          :class="{ active: activeTab === tab.id }"
          @click="activeTab = tab.id"
        >
          {{ tab.label }}
        </button>
      </nav>
    </header>

    <main>
      <!-- Authentication Section -->
      <div v-if="activeTab === 'auth'" class="auth-container">
        <div v-if="isAuthenticated">
          <h2>Welcome, {{ user?.username }}!</h2>
          <button @click="logout" :disabled="loading">
            {{ loading ? 'Logging out...' : 'Logout' }}
          </button>
        </div>
        <form v-else @submit.prevent="login">
          <h2>Login to anomaly_detection</h2>
          
          <div>
            <label>Username:</label>
            <input
              v-model="credentials.username"
              type="text"
              required
            />
          </div>

          <div>
            <label>Password:</label>
            <input
              v-model="credentials.password"
              type="password"
              required
            />
          </div>

          <div>
            <label>MFA Code (optional):</label>
            <input
              v-model="credentials.mfaCode"
              type="text"
            />
          </div>

          <button type="submit" :disabled="loading">
            {{ loading ? 'Logging in...' : 'Login' }}
          </button>
        </form>
      </div>

      <!-- Anomaly Detection Section -->
      <div v-if="activeTab === 'detection'" class="detection-container">
        <h2>Anomaly Detection</h2>
        
        <div>
          <label>Input Data (comma-separated):</label>
          <input
            v-model="inputData"
            type="text"
            placeholder="1,2,3,4,5,100,6,7,8,9"
          />
        </div>

        <div>
          <label>Algorithm:</label>
          <select v-model="algorithm">
            <option value="isolation_forest">Isolation Forest</option>
            <option value="one_class_svm">One-Class SVM</option>
            <option value="local_outlier_factor">Local Outlier Factor</option>
            <option value="elliptic_envelope">Elliptic Envelope</option>
          </select>
        </div>

        <button @click="detectAnomalies" :disabled="loading">
          {{ loading ? 'Detecting...' : 'Detect Anomalies' }}
        </button>

        <div v-if="detectionResult" class="results">
          <h3>Results</h3>
          <div>
            <strong>Anomaly Rate:</strong> {{ (detectionResult.anomalyRate * 100).toFixed(2) }}%
          </div>
          <div>
            <strong>Processing Time:</strong> {{ detectionResult.processingTime }}ms
          </div>
          <div>
            <strong>Algorithm:</strong> {{ detectionResult.algorithm }}
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
              <tr
                v-for="(score, index) in detectionResult.scores"
                :key="index"
                :class="{ anomaly: detectionResult.predictions[index] }"
              >
                <td>{{ index }}</td>
                <td>{{ score.toFixed(4) }}</td>
                <td>{{ detectionResult.predictions[index] ? 'Anomaly' : 'Normal' }}</td>
                <td>{{ detectionResult.confidence[index]?.toFixed(4) || 'N/A' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Streaming Section -->
      <div v-if="activeTab === 'streaming'" class="streaming-container">
        <h2>Real-time Streaming</h2>
        
        <div class="controls">
          <button v-if="!isConnected" @click="connectWebSocket" :disabled="loading">
            {{ loading ? 'Connecting...' : 'Connect WebSocket' }}
          </button>
          <template v-else>
            <button @click="disconnectWebSocket">Disconnect</button>
            <button @click="sendTestData">Send Test Data</button>
          </template>
        </div>

        <div class="status">
          Status: {{ isConnected ? '🟢 Connected' : '🔴 Disconnected' }}
        </div>

        <div v-if="alerts.length > 0" class="alerts">
          <h3>Recent Alerts</h3>
          <div
            v-for="(alert, index) in alerts"
            :key="index"
            :class="`alert alert-${alert.level}`"
          >
            <strong>{{ alert.level.toUpperCase() }}:</strong> {{ alert.message }}
            <span class="score">Score: {{ alert.score.toFixed(4) }}</span>
          </div>
        </div>

        <div v-if="streamData.length > 0" class="stream-data">
          <h3>Real-time Results</h3>
          <div class="data-grid">
            <div
              v-for="(data, index) in streamData"
              :key="index"
              class="data-item"
            >
              <div>Stream: {{ data.streamId }}</div>
              <div>Anomalies: {{ data.result.anomalyRate.toFixed(2) }}%</div>
              <div>Time: {{ formatTime(data.timestamp) }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Health Section -->
      <div v-if="activeTab === 'health'" class="health-container">
        <h2>System Health</h2>
        
        <div class="controls">
          <button @click="fetchHealth" :disabled="loading">
            {{ loading ? 'Refreshing...' : 'Refresh' }}
          </button>
          <label>
            <input
              v-model="autoRefresh"
              type="checkbox"
            />
            Auto-refresh (5s)
          </label>
        </div>

        <div v-if="health" class="health-status">
          <div :class="`overall-status status-${health.status}`">
            <h3>Overall Status: {{ health.status.toUpperCase() }}</h3>
            <div>Version: {{ health.version }}</div>
            <div>Uptime: {{ formatUptime(health.uptime) }}</div>
          </div>

          <div class="services">
            <h4>Services</h4>
            <div
              v-for="(service, index) in health.services"
              :key="index"
              :class="`service status-${service.status}`"
            >
              <strong>{{ service.name }}</strong>
              <span class="status">{{ service.status }}</span>
              <span v-if="service.responseTime" class="response-time">
                {{ service.responseTime }}ms
              </span>
              <div v-if="service.error" class="service-error">
                {{ service.error }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Error Display -->
      <div v-if="error" class="error">
        {{ error }}
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue';
import { AnomalyDetectionClient } from '../src/index';
import type {
  DetectionRequest,
  DetectionResponse,
  StreamDetectionResult,
  StreamAlert,
  HealthStatus,
  UserProfile,
} from '../src/types';

// Reactive state
const activeTab = ref('auth');
const loading = ref(false);
const error = ref<string | null>(null);

// Authentication state
const isAuthenticated = ref(false);
const user = ref<UserProfile | null>(null);
const credentials = reactive({
  username: '',
  password: '',
  mfaCode: '',
});

// Detection state
const inputData = ref('1,2,3,4,5,100,6,7,8,9');
const algorithm = ref('isolation_forest');
const detectionResult = ref<DetectionResponse | null>(null);

// Streaming state
const isConnected = ref(false);
const streamData = ref<StreamDetectionResult[]>([]);
const alerts = ref<StreamAlert[]>([]);

// Health state
const health = ref<HealthStatus | null>(null);
const autoRefresh = ref(false);

// Client instance
const client = new AnomalyDetectionClient({
  baseUrl: import.meta.env.VITE_ANOMALY_DETECTION_API_URL || 'https://api.anomaly_detection.com',
  apiKey: import.meta.env.VITE_ANOMALY_DETECTION_API_KEY,
  debug: import.meta.env.DEV,
  websocket: {
    enabled: true,
    autoReconnect: true,
  },
});

// Tab configuration
const tabs = [
  { id: 'auth', label: 'Authentication' },
  { id: 'detection', label: 'Anomaly Detection' },
  { id: 'streaming', label: 'Real-time Streaming' },
  { id: 'health', label: 'System Health' },
];

// Authentication methods
const login = async () => {
  loading.value = true;
  error.value = null;

  try {
    const response = await client.auth.login(credentials);
    isAuthenticated.value = true;
    user.value = response.user;
  } catch (err: any) {
    error.value = err.message || 'Login failed';
  } finally {
    loading.value = false;
  }
};

const logout = async () => {
  loading.value = true;
  try {
    await client.auth.logout();
    isAuthenticated.value = false;
    user.value = null;
  } catch (err: any) {
    error.value = err.message || 'Logout failed';
  } finally {
    loading.value = false;
  }
};

// Detection methods
const detectAnomalies = async () => {
  loading.value = true;
  error.value = null;

  try {
    const data = inputData.value.split(',').map(Number);
    
    const request: DetectionRequest = {
      data,
      algorithm: algorithm.value as any,
      parameters: {
        contamination: 0.1,
        n_estimators: 100,
      },
      includeExplanations: true,
    };

    detectionResult.value = await client.detection.detect(request);
  } catch (err: any) {
    error.value = err.message || 'Detection failed';
  } finally {
    loading.value = false;
  }
};

// Streaming methods
const connectWebSocket = async () => {
  loading.value = true;
  error.value = null;

  try {
    await client.connectWebSocket({
      onConnect: () => {
        isConnected.value = true;
        console.log('WebSocket connected');
      },
      onDisconnect: () => {
        isConnected.value = false;
        console.log('WebSocket disconnected');
      },
      onData: (data: StreamDetectionResult) => {
        streamData.value = [...streamData.value.slice(-19), data];
      },
      onAlert: (alert: StreamAlert) => {
        alerts.value = [...alerts.value.slice(-9), alert];
      },
      onError: (wsError: Error) => {
        error.value = wsError.message;
      },
    });
  } catch (err: any) {
    error.value = err.message || 'Connection failed';
  } finally {
    loading.value = false;
  }
};

const disconnectWebSocket = () => {
  client.disconnectWebSocket();
  isConnected.value = false;
};

const sendTestData = async () => {
  if (!isConnected.value) return;

  try {
    const streamingManager = client.getStreamingManager();
    if (!streamingManager) return;

    const data = Array.from({ length: 10 }, () => Math.random() * 100);
    await streamingManager.sendStreamData('test-stream', data);
  } catch (err: any) {
    error.value = err.message || 'Failed to send data';
  }
};

// Health methods
const fetchHealth = async () => {
  loading.value = true;
  error.value = null;

  try {
    health.value = await client.health.getHealth();
  } catch (err: any) {
    error.value = err.message || 'Failed to fetch health status';
  } finally {
    loading.value = false;
  }
};

// Utility functions
const formatTime = (timestamp: number) => {
  return new Date(timestamp).toLocaleTimeString();
};

const formatUptime = (uptime: number) => {
  const hours = Math.floor(uptime / 3600);
  const minutes = Math.floor((uptime % 3600) / 60);
  return `${hours}h ${minutes}m`;
};

// Lifecycle hooks
onMounted(() => {
  // Check if already authenticated
  const clientInfo = client.getClientInfo();
  isAuthenticated.value = clientInfo.isAuthenticated;
  user.value = clientInfo.sessionInfo || null;

  // Fetch initial health status
  fetchHealth();
});

// Auto-refresh health status
let healthInterval: NodeJS.Timeout | null = null;

watch(autoRefresh, (newValue) => {
  if (healthInterval) {
    clearInterval(healthInterval);
    healthInterval = null;
  }

  if (newValue) {
    healthInterval = setInterval(fetchHealth, 5000);
  }
});

onUnmounted(() => {
  if (healthInterval) {
    clearInterval(healthInterval);
  }
  client.disconnectWebSocket();
});
</script>

<style scoped>
.anomaly_detection-app {
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
  border-radius: 4px;
}

nav button.active {
  background: #007bff;
  color: white;
}

nav button:hover {
  background: #f8f9fa;
}

nav button.active:hover {
  background: #0056b3;
}

.error {
  color: #dc3545;
  background: #f8d7da;
  padding: 10px;
  border-radius: 4px;
  margin: 10px 0;
  border: 1px solid #f5c6cb;
}

.auth-container,
.detection-container,
.streaming-container,
.health-container {
  padding: 20px;
}

.auth-container form div,
.detection-container div {
  margin-bottom: 15px;
}

.auth-container label,
.detection-container label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.auth-container input,
.detection-container input,
.detection-container select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  box-sizing: border-box;
}

button {
  padding: 10px 20px;
  border: 1px solid #007bff;
  background: #007bff;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  margin-right: 10px;
}

button:hover {
  background: #0056b3;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.controls {
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
}

.status {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 20px;
}

.status-healthy { color: #28a745; }
.status-degraded { color: #ffc107; }
.status-unhealthy { color: #dc3545; }

.alerts {
  margin-bottom: 20px;
}

.alert {
  padding: 10px;
  margin: 5px 0;
  border-radius: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.alert-info { background: #d1ecf1; color: #0c5460; }
.alert-warning { background: #fff3cd; color: #856404; }
.alert-critical { background: #f8d7da; color: #721c24; }

.score {
  font-size: 12px;
  opacity: 0.8;
}

.stream-data {
  margin-top: 20px;
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
  background: #f8f9fa;
}

.results {
  margin-top: 20px;
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #f8f9fa;
}

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

th {
  background: #f2f2f2;
  font-weight: bold;
}

tr.anomaly {
  background-color: #ffe6e6;
}

.health-status {
  margin-top: 20px;
}

.overall-status {
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.overall-status.status-healthy {
  background: #d4edda;
  border: 1px solid #c3e6cb;
}

.overall-status.status-degraded {
  background: #fff3cd;
  border: 1px solid #ffeaa7;
}

.overall-status.status-unhealthy {
  background: #f8d7da;
  border: 1px solid #f5c6cb;
}

.services {
  margin-top: 20px;
}

.service {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  margin: 5px 0;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.service-error {
  color: #dc3545;
  font-size: 12px;
  flex-basis: 100%;
  margin-top: 5px;
}

.response-time {
  font-size: 12px;
  opacity: 0.7;
}
</style>