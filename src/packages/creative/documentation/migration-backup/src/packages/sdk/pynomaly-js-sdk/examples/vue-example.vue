<template>
  <div style="padding: 20px; font-family: Arial, sans-serif;">
    <h1>Pynomaly Vue 3 SDK Example</h1>
    
    <!-- Provider wrapper -->
    <PynomalyProvider 
      :config="config" 
      :auto-connect="true"
      v-slot="{ client, isReady, isLoading, error }"
    >
      <!-- Client Status -->
      <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;">
        <h2>Client Status</h2>
        <p><strong>Ready:</strong> {{ isReady ? '✅ Yes' : '❌ No' }}</p>
        <p><strong>Loading:</strong> {{ isLoading ? '⏳ Yes' : '✅ No' }}</p>
        <p><strong>Client:</strong> {{ client ? '✅ Available' : '❌ Not Available' }}</p>
        <p v-if="error" style="color: red;"><strong>Error:</strong> {{ error.message }}</p>
      </div>

      <!-- Authentication Example -->
      <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;">
        <h2>Authentication</h2>
        <p><strong>Authenticated:</strong> {{ authState.isAuthenticated ? '✅ Yes' : '❌ No' }}</p>
        <p><strong>User:</strong> {{ authState.user ? authState.user.email : 'None' }}</p>
        <p v-if="authError" style="color: red;"><strong>Error:</strong> {{ authError }}</p>

        <div v-if="!authState.isAuthenticated">
          <div style="margin-bottom: 10px;">
            <h4>Login with Credentials</h4>
            <input
              v-model="email"
              type="email"
              placeholder="Email"
              style="margin-right: 10px; padding: 5px;"
            />
            <input
              v-model="password"
              type="password"
              placeholder="Password"
              style="margin-right: 10px; padding: 5px;"
            />
            <button @click="handleLogin" :disabled="authLoading">
              {{ authLoading ? 'Logging in...' : 'Login' }}
            </button>
          </div>

          <div>
            <h4>Login with API Key</h4>
            <input
              v-model="apiKey"
              type="text"
              placeholder="API Key"
              style="margin-right: 10px; padding: 5px;"
            />
            <button @click="handleApiKeyLogin" :disabled="authLoading">
              {{ authLoading ? 'Logging in...' : 'Login with API Key' }}
            </button>
          </div>
        </div>

        <div v-else>
          <button @click="handleLogout" :disabled="authLoading">
            {{ authLoading ? 'Logging out...' : 'Logout' }}
          </button>
        </div>
      </div>

      <!-- Anomaly Detection with Composable -->
      <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;">
        <h2>Anomaly Detection with Composable</h2>
        
        <div style="margin-bottom: 10px;">
          <button @click="handleDetectAnomalies" :disabled="detectionLoading || !client">
            {{ detectionLoading ? 'Detecting...' : 'Detect Anomalies' }}
          </button>
          <button @click="clearDetection" :disabled="detectionLoading" style="margin-left: 10px;">
            Clear
          </button>
        </div>

        <div v-if="detectionLoading" style="margin-bottom: 10px;">
          <div style="width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden;">
            <div :style="{ width: `${detectionProgress}%`, height: '100%', backgroundColor: '#007bff', transition: 'width 0.3s ease' }"></div>
          </div>
          <p>Progress: {{ detectionProgress }}%</p>
        </div>

        <p v-if="detectionError" style="color: red;">Error: {{ detectionError.message }}</p>

        <div v-if="detectionResult" style="margin-top: 10px;">
          <h4>Results:</h4>
          <p><strong>Algorithm:</strong> {{ detectionResult.algorithm }}</p>
          <p><strong>Anomalies Found:</strong> {{ detectionResult.metrics.anomalyCount }}</p>
          <p><strong>Processing Time:</strong> {{ detectionResult.processingTime }}ms</p>
        </div>

        <div style="margin-top: 10px;">
          <h4>Sample Data:</h4>
          <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 12px;">{{ JSON.stringify(sampleData, null, 2) }}</pre>
        </div>
      </div>

      <!-- Anomaly Detector Component -->
      <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;">
        <h2>Anomaly Detector Component</h2>
        <AnomalyDetector
          algorithm="auto"
          :auto-detect="false"
          @result="handleComponentResult"
          @error="handleComponentError"
        />
      </div>
    </PynomalyProvider>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue';
import { 
  PynomalyProvider, 
  AnomalyDetector,
  usePynomalyAuth,
  useAnomalyDetection
} from '../src/frameworks/vue';

// Configuration
const config = reactive({
  apiKey: 'demo-api-key',
  baseUrl: 'https://api.pynomaly.com',
  debug: true
});

// Authentication
const email = ref('demo@example.com');
const password = ref('password');
const apiKey = ref('demo-api-key');

const {
  authState,
  isLoading: authLoading,
  error: authError,
  login,
  loginWithApiKey,
  logout
} = usePynomalyAuth();

// Anomaly Detection
const sampleData = ref([
  [1, 2, 3],
  [2, 3, 4],
  [3, 4, 5],
  [4, 5, 6],
  [100, 200, 300] // This should be detected as an anomaly
]);

const {
  result: detectionResult,
  isLoading: detectionLoading,
  error: detectionError,
  progress: detectionProgress,
  detectAnomalies,
  clear: clearDetection
} = useAnomalyDetection();

// Methods
const handleLogin = async () => {
  try {
    await login({ 
      email: email.value, 
      password: password.value 
    });
  } catch (err) {
    console.error('Login failed:', err);
  }
};

const handleApiKeyLogin = async () => {
  try {
    await loginWithApiKey(apiKey.value);
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

const handleDetectAnomalies = async () => {
  try {
    await detectAnomalies({
      data: sampleData.value,
      algorithm: 'isolation_forest',
      parameters: { contamination: 0.1 }
    });
  } catch (err) {
    console.error('Detection failed:', err);
  }
};

const handleComponentResult = (result: any) => {
  console.log('Component result:', result);
};

const handleComponentError = (error: any) => {
  console.error('Component error:', error);
};
</script>

<style>
/* Component styles would go here */
</style>