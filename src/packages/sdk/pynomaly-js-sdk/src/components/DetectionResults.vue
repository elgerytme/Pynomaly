<!--
  Detection Results Component (Vue)
  
  Vue component for displaying anomaly detection results with
  visualizations, metrics, and data exploration capabilities.
-->

<template>
  <div :class="`pynomaly-detection-results ${className}`">
    <!-- Header -->
    <div class="results-header">
      <h2>Detection Results</h2>
      <div class="result-summary">
        <span class="anomaly-count">
          {{ result.nAnomalies }} anomalies detected
        </span>
        <span class="sample-count">
          out of {{ result.nSamples }} samples
        </span>
      </div>
    </div>

    <!-- Quick Stats -->
    <div class="quick-stats">
      <div class="stat-card anomaly-rate">
        <div class="stat-value">{{ metrics.anomalyRate.toFixed(1) }}%</div>
        <div class="stat-label">Anomaly Rate</div>
      </div>
      <div class="stat-card threshold">
        <div class="stat-value">{{ result.threshold.toFixed(3) }}</div>
        <div class="stat-label">Threshold</div>
      </div>
      <div class="stat-card execution-time">
        <div class="stat-value">{{ result.executionTime.toFixed(2) }}s</div>
        <div class="stat-label">Execution Time</div>
        <div :class="`performance-badge ${performance.class}`">
          {{ performance.label }}
        </div>
      </div>
      <div class="stat-card score-range">
        <div class="stat-value">{{ metrics.scoreRange.toFixed(3) }}</div>
        <div class="stat-label">Score Range</div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="result-tabs">
      <button
        :class="`tab ${activeTab === 'overview' ? 'active' : ''}`"
        @click="activeTab = 'overview'"
      >
        Overview
      </button>
      <button
        :class="`tab ${activeTab === 'anomalies' ? 'active' : ''}`"
        @click="activeTab = 'anomalies'"
      >
        Anomalies ({{ result.nAnomalies }})
      </button>
      <button
        :class="`tab ${activeTab === 'details' ? 'active' : ''}`"
        @click="activeTab = 'details'"
      >
        Details
      </button>
    </div>

    <!-- Tab Content -->
    <div class="tab-content">
      <!-- Overview Tab -->
      <div v-if="activeTab === 'overview'" class="overview-content">
        <!-- Distribution Chart -->
        <div v-if="showVisualization" class="distribution-chart">
          <h3>Score Distribution</h3>
          <div class="chart-container">
            <ScoreDistributionChart
              :scores="result.anomalyScores"
              :labels="result.anomalyLabels"
              :threshold="result.threshold"
            />
          </div>
        </div>

        <!-- Metrics Summary -->
        <div v-if="showMetrics" class="metrics-summary">
          <h3>Detection Metrics</h3>
          <div class="metrics-grid">
            <div class="metric-item">
              <label>Total Samples:</label>
              <span>{{ result.nSamples.toLocaleString() }}</span>
            </div>
            <div class="metric-item">
              <label>Anomalies Found:</label>
              <span>{{ result.nAnomalies.toLocaleString() }}</span>
            </div>
            <div class="metric-item">
              <label>Contamination Rate:</label>
              <span>{{ (result.contaminationRate * 100).toFixed(1) }}%</span>
            </div>
            <div class="metric-item">
              <label>Average Score:</label>
              <span>{{ metrics.avgScore.toFixed(4) }}</span>
            </div>
            <div class="metric-item">
              <label>Average Anomaly Score:</label>
              <span>{{ metrics.avgAnomalyScore.toFixed(4) }}</span>
            </div>
            <div class="metric-item">
              <label>Score Range:</label>
              <span>{{ metrics.minScore.toFixed(4) }} - {{ metrics.maxScore.toFixed(4) }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Anomalies Tab -->
      <div v-else-if="activeTab === 'anomalies'" class="anomalies-content">
        <!-- Sorting Controls -->
        <div class="sorting-controls">
          <div class="sort-group">
            <label>Sort by:</label>
            <select v-model="sortBy">
              <option value="score">Anomaly Score</option>
              <option value="index">Sample Index</option>
            </select>
          </div>
          <div class="sort-group">
            <label>Order:</label>
            <select v-model="sortOrder">
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </div>
        </div>

        <!-- Anomaly List -->
        <div class="anomaly-list">
          <div v-if="anomalyData.length === 0" class="no-anomalies">
            <p>No anomalies detected in this dataset.</p>
          </div>
          <div v-else class="anomaly-table">
            <div class="table-header">
              <div class="col-index">Index</div>
              <div class="col-score">Score</div>
              <div v-if="dataset?.featureNames" class="col-features">Features</div>
            </div>
            <div
              v-for="anomaly in anomalyData"
              :key="anomaly.index"
              class="table-row"
            >
              <div class="col-index">#{{ anomaly.index }}</div>
              <div class="col-score">
                <span class="score-value">
                  {{ anomaly.score.toFixed(4) }}
                </span>
                <div class="score-bar">
                  <div
                    class="score-fill"
                    :style="{
                      width: `${((anomaly.score - metrics.minScore) / metrics.scoreRange) * 100}%`
                    }"
                  />
                </div>
              </div>
              <div v-if="dataset?.featureNames && anomaly.data" class="col-features">
                <div class="feature-values">
                  <div
                    v-for="(name, featureIdx) in dataset.featureNames"
                    :key="name"
                    class="feature-item"
                  >
                    <span class="feature-name">{{ name }}:</span>
                    <span class="feature-value">
                      {{ 
                        Array.isArray(anomaly.data) 
                          ? anomaly.data[featureIdx]
                          : anomaly.data[name]
                      }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Details Tab -->
      <div v-else-if="activeTab === 'details'" class="details-content">
        <div class="detail-sections">
          <!-- Configuration -->
          <div class="detail-section">
            <h3>Detection Configuration</h3>
            <div class="config-items">
              <div class="config-item">
                <label>Threshold:</label>
                <span>{{ result.threshold }}</span>
              </div>
              <div class="config-item">
                <label>Expected Contamination:</label>
                <span>{{ (result.contaminationRate * 100).toFixed(1) }}%</span>
              </div>
              <div class="config-item">
                <label>Execution Time:</label>
                <span>{{ result.executionTime.toFixed(2) }} seconds</span>
              </div>
            </div>
          </div>

          <!-- Dataset Info -->
          <div v-if="dataset" class="detail-section">
            <h3>Dataset Information</h3>
            <div class="config-items">
              <div class="config-item">
                <label>Dataset Name:</label>
                <span>{{ dataset.name }}</span>
              </div>
              <div class="config-item">
                <label>Sample Count:</label>
                <span>{{ result.nSamples.toLocaleString() }}</span>
              </div>
              <div v-if="dataset.featureNames" class="config-item">
                <label>Features:</label>
                <span>{{ dataset.featureNames.length }} features</span>
              </div>
              <div v-if="dataset.targetColumn" class="config-item">
                <label>Target Column:</label>
                <span>{{ dataset.targetColumn }}</span>
              </div>
            </div>
          </div>

          <!-- Metadata -->
          <div v-if="result.metadata && Object.keys(result.metadata).length > 0" class="detail-section">
            <h3>Additional Metadata</h3>
            <div class="metadata-content">
              <pre>{{ JSON.stringify(result.metadata, null, 2) }}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { DetectionResultsProps, DetectionResult, Dataset } from '../types';
import ScoreDistributionChart from './ScoreDistributionChart.vue';

interface Props extends DetectionResultsProps {
  className?: string;
}

const props = withDefaults(defineProps<Props>(), {
  showVisualization: true,
  showMetrics: true,
  className: ''
});

// Reactive state
const activeTab = ref<'overview' | 'anomalies' | 'details'>('overview');
const sortBy = ref<'index' | 'score'>('score');
const sortOrder = ref<'asc' | 'desc'>('desc');

// Computed properties
const metrics = computed(() => {
  const anomalyRate = (props.result.nAnomalies / props.result.nSamples) * 100;
  const normalRate = 100 - anomalyRate;
  
  const scores = props.result.anomalyScores;
  const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);
  
  const anomalyScores = props.result.anomalyLabels
    .map((label, index) => label === 1 ? scores[index] : null)
    .filter(score => score !== null) as number[];
  
  const avgAnomalyScore = anomalyScores.length > 0 
    ? anomalyScores.reduce((sum, score) => sum + score, 0) / anomalyScores.length 
    : 0;

  return {
    anomalyRate,
    normalRate,
    avgScore,
    maxScore,
    minScore,
    avgAnomalyScore,
    scoreRange: maxScore - minScore
  };
});

const anomalyData = computed(() => {
  const anomalies = props.result.anomalyLabels
    .map((label, index) => ({
      index,
      isAnomaly: label === 1,
      score: props.result.anomalyScores[index],
      data: props.dataset?.data ? props.dataset.data[index] : null
    }))
    .filter(item => item.isAnomaly);

  // Sort anomalies
  return anomalies.sort((a, b) => {
    if (sortBy.value === 'score') {
      return sortOrder.value === 'desc' ? b.score - a.score : a.score - b.score;
    } else {
      return sortOrder.value === 'desc' ? b.index - a.index : a.index - b.index;
    }
  });
});

const performance = computed(() => {
  const executionTime = props.result.executionTime;
  if (executionTime < 1) return { label: 'Excellent', class: 'excellent' };
  if (executionTime < 5) return { label: 'Good', class: 'good' };
  if (executionTime < 15) return { label: 'Fair', class: 'fair' };
  return { label: 'Slow', class: 'slow' };
});
</script>

<style scoped>
/* Add your component styles here */
.pynomaly-detection-results {
  /* Component styles */
}

.quick-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.stat-card {
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  text-align: center;
}

.stat-value {
  font-size: 2em;
  font-weight: bold;
  color: #2c3e50;
}

.stat-label {
  color: #7f8c8d;
  margin-top: 5px;
}

.result-tabs {
  display: flex;
  border-bottom: 1px solid #ddd;
  margin: 20px 0;
}

.tab {
  padding: 10px 20px;
  border: none;
  background: none;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}

.tab.active {
  border-bottom-color: #3498db;
  color: #3498db;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
  margin: 20px 0;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 4px;
}

.anomaly-table {
  border: 1px solid #ddd;
  border-radius: 4px;
  overflow: hidden;
}

.table-header,
.table-row {
  display: grid;
  grid-template-columns: 100px 150px 1fr;
  gap: 20px;
  padding: 15px;
}

.table-header {
  background: #f8f9fa;
  font-weight: bold;
  border-bottom: 1px solid #ddd;
}

.table-row {
  border-bottom: 1px solid #eee;
}

.score-bar {
  width: 100%;
  height: 6px;
  background: #eee;
  border-radius: 3px;
  margin-top: 5px;
}

.score-fill {
  height: 100%;
  background: #e74c3c;
  border-radius: 3px;
}
</style>