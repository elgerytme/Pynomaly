<template>
  <div class="pynomaly-anomaly-detector">
    <div class="input-section">
      <h3>Anomaly Detection</h3>
      
      <div class="algorithm-selection">
        <label>
          Algorithm:
          <select v-model="selectedAlgorithm">
            <option value="auto">Auto</option>
            <option value="isolation_forest">Isolation Forest</option>
            <option value="local_outlier_factor">Local Outlier Factor</option>
            <option value="one_class_svm">One-Class SVM</option>
          </select>
        </label>
      </div>

      <div class="data-input">
        <label>
          Data (JSON array of arrays):
          <textarea
            v-model="inputData"
            placeholder='[[1, 2, 3], [4, 5, 6], [100, 200, 300]]'
            :rows="4"
            style="width: 100%; font-family: monospace;"
          />
        </label>
      </div>

      <div class="actions">
        <button 
          @click="handleAnalyzeInput"
          :disabled="isLoading || !client"
        >
          {{ isLoading ? 'Detecting...' : 'Detect Anomalies' }}
        </button>
        
        <button 
          @click="clear"
          :disabled="isLoading"
        >
          Clear
        </button>
      </div>

      <div v-if="isLoading" class="progress">
        <div class="progress-bar">
          <div 
            class="progress-fill"
            :style="{ width: `${progress}%` }"
          />
        </div>
        <span>Progress: {{ progress }}%</span>
      </div>
    </div>

    <div v-if="error" class="error" style="color: red; margin-top: 10px;">
      Error: {{ error.message }}
    </div>

    <div v-if="result" class="result" style="margin-top: 20px;">
      <h4>Detection Results</h4>
      <div class="result-summary">
        <p><strong>Algorithm:</strong> {{ result.algorithm }}</p>
        <p><strong>Total Points:</strong> {{ result.metrics.totalPoints }}</p>
        <p><strong>Anomalies Found:</strong> {{ result.metrics.anomalyCount }}</p>
        <p><strong>Anomaly Rate:</strong> {{ (result.metrics.anomalyRate * 100).toFixed(2) }}%</p>
        <p><strong>Processing Time:</strong> {{ result.processingTime }}ms</p>
      </div>

      <div v-if="result.anomalies.length > 0" class="anomalies">
        <h5>Detected Anomalies:</h5>
        <table style="width: 100%; border-collapse: collapse;">
          <thead>
            <tr>
              <th style="border: 1px solid #ccc; padding: 8px;">Index</th>
              <th style="border: 1px solid #ccc; padding: 8px;">Score</th>
              <th style="border: 1px solid #ccc; padding: 8px;">Confidence</th>
              <th style="border: 1px solid #ccc; padding: 8px;">Data</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(anomaly, index) in result.anomalies" :key="index">
              <td style="border: 1px solid #ccc; padding: 8px;">{{ anomaly.index }}</td>
              <td style="border: 1px solid #ccc; padding: 8px;">{{ anomaly.score.toFixed(4) }}</td>
              <td style="border: 1px solid #ccc; padding: 8px;">{{ (anomaly.confidence * 100).toFixed(1) }}%</td>
              <td style="border: 1px solid #ccc; padding: 8px;">
                {{ JSON.stringify(anomaly.data) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, inject, watch } from 'vue';
import { PynomalyClient, AnomalyDetectionRequest, AnomalyDetectionResult } from '../../../index';
import { useAnomalyDetection } from '../composables/useAnomalyDetection';

export interface AnomalyDetectorProps {
  data?: number[][];
  algorithm?: 'isolation_forest' | 'local_outlier_factor' | 'one_class_svm' | 'auto';
  parameters?: Record<string, any>;
  autoDetect?: boolean;
}

const props = withDefaults(defineProps<AnomalyDetectorProps>(), {
  algorithm: 'auto',
  parameters: () => ({}),
  autoDetect: false
});

const emit = defineEmits<{
  result: [result: AnomalyDetectionResult];
  error: [error: Error];
}>();

// Inject client from provider
const client = inject<PynomalyClient | null>('pynomaly-client', null);

const selectedAlgorithm = ref(props.algorithm);
const inputData = ref('');

const {
  result,
  isLoading,
  error,
  progress,
  detectAnomalies,
  clear
} = useAnomalyDetection({
  client: client || undefined
});

const parseInputData = (): number[][] | null => {
  try {
    const parsed = JSON.parse(inputData.value);
    if (Array.isArray(parsed) && parsed.every(row => Array.isArray(row))) {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
};

const handleDetectAnomalies = async (dataToAnalyze?: number[][]) => {
  const analysisData = dataToAnalyze || props.data;
  
  if (!analysisData || analysisData.length === 0) {
    return;
  }

  const request: AnomalyDetectionRequest = {
    data: analysisData,
    algorithm: selectedAlgorithm.value,
    parameters: props.parameters
  };

  try {
    const detectionResult = await detectAnomalies(request);
    emit('result', detectionResult);
  } catch (err) {
    emit('error', err as Error);
    console.error('Anomaly detection failed:', err);
  }
};

const handleAnalyzeInput = () => {
  const parsedData = parseInputData();
  if (parsedData) {
    handleDetectAnomalies(parsedData);
  }
};

// Auto-detect when data changes
watch(() => props.data, (newData) => {
  if (props.autoDetect && newData) {
    handleDetectAnomalies();
  }
}, { immediate: true });
</script>

<style scoped>
.pynomaly-anomaly-detector {
  max-width: 800px;
  margin: 0 auto;
}

.input-section {
  margin-bottom: 20px;
}

.algorithm-selection,
.data-input {
  margin-bottom: 15px;
}

.algorithm-selection label,
.data-input label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.algorithm-selection select {
  padding: 5px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.actions {
  margin-bottom: 15px;
}

.actions button {
  padding: 10px 15px;
  margin-right: 10px;
  border: none;
  border-radius: 4px;
  background-color: #007bff;
  color: white;
  cursor: pointer;
}

.actions button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.actions button:not(:disabled):hover {
  background-color: #0056b3;
}

.progress {
  margin-top: 10px;
}

.progress-bar {
  width: 100%;
  height: 20px;
  background-color: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background-color: #007bff;
  transition: width 0.3s ease;
}

.result {
  padding: 15px;
  border: 1px solid #dee2e6;
  border-radius: 4px;
  background-color: #f8f9fa;
}

.result-summary p {
  margin: 5px 0;
}

.anomalies {
  margin-top: 15px;
}

.anomalies table {
  font-size: 14px;
}

.error {
  padding: 10px;
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 4px;
}
</style>