/**
 * Detection Results Component (Angular)
 * 
 * Angular component for displaying anomaly detection results with
 * visualizations, metrics, and data exploration capabilities.
 */

import { Component, Input, OnInit } from '@angular/core';
import { DetectionResultsProps, DetectionResult, Dataset } from '../types';

interface AnomalyDataItem {
  index: number;
  isAnomaly: boolean;
  score: number;
  data: any;
}

interface Metrics {
  anomalyRate: number;
  normalRate: number;
  avgScore: number;
  maxScore: number;
  minScore: number;
  avgAnomalyScore: number;
  scoreRange: number;
}

interface Performance {
  label: string;
  class: string;
}

@Component({
  selector: 'pynomaly-detection-results',
  template: `
    <div [class]="'pynomaly-detection-results ' + className">
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
          <div [class]="'performance-badge ' + performance.class">
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
          [class]="'tab ' + (activeTab === 'overview' ? 'active' : '')"
          (click)="activeTab = 'overview'"
        >
          Overview
        </button>
        <button
          [class]="'tab ' + (activeTab === 'anomalies' ? 'active' : '')"
          (click)="activeTab = 'anomalies'"
        >
          Anomalies ({{ result.nAnomalies }})
        </button>
        <button
          [class]="'tab ' + (activeTab === 'details' ? 'active' : '')"
          (click)="activeTab = 'details'"
        >
          Details
        </button>
      </div>

      <!-- Tab Content -->
      <div class="tab-content">
        <!-- Overview Tab -->
        <div *ngIf="activeTab === 'overview'" class="overview-content">
          <!-- Distribution Chart -->
          <div *ngIf="showVisualization" class="distribution-chart">
            <h3>Score Distribution</h3>
            <div class="chart-container">
              <div class="chart-placeholder">
                <!-- In a real implementation, integrate with Angular chart library -->
                <p>Score distribution chart would be rendered here using a charting library like Chart.js or D3.</p>
              </div>
            </div>
          </div>

          <!-- Metrics Summary -->
          <div *ngIf="showMetrics" class="metrics-summary">
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
        <div *ngIf="activeTab === 'anomalies'" class="anomalies-content">
          <!-- Sorting Controls -->
          <div class="sorting-controls">
            <div class="sort-group">
              <label>Sort by:</label>
              <select [(ngModel)]="sortBy" (change)="updateAnomalyData()">
                <option value="score">Anomaly Score</option>
                <option value="index">Sample Index</option>
              </select>
            </div>
            <div class="sort-group">
              <label>Order:</label>
              <select [(ngModel)]="sortOrder" (change)="updateAnomalyData()">
                <option value="desc">Descending</option>
                <option value="asc">Ascending</option>
              </select>
            </div>
          </div>

          <!-- Anomaly List -->
          <div class="anomaly-list">
            <div *ngIf="anomalyData.length === 0" class="no-anomalies">
              <p>No anomalies detected in this dataset.</p>
            </div>
            <div *ngIf="anomalyData.length > 0" class="anomaly-table">
              <div class="table-header">
                <div class="col-index">Index</div>
                <div class="col-score">Score</div>
                <div *ngIf="dataset?.featureNames" class="col-features">Features</div>
              </div>
              <div
                *ngFor="let anomaly of anomalyData"
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
                      [style.width.%]="((anomaly.score - metrics.minScore) / metrics.scoreRange) * 100"
                    ></div>
                  </div>
                </div>
                <div *ngIf="dataset?.featureNames && anomaly.data" class="col-features">
                  <div class="feature-values">
                    <div
                      *ngFor="let name of dataset.featureNames; let featureIdx = index"
                      class="feature-item"
                    >
                      <span class="feature-name">{{ name }}:</span>
                      <span class="feature-value">
                        {{ getFeatureValue(anomaly.data, name, featureIdx) }}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Details Tab -->
        <div *ngIf="activeTab === 'details'" class="details-content">
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
            <div *ngIf="dataset" class="detail-section">
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
                <div *ngIf="dataset.featureNames" class="config-item">
                  <label>Features:</label>
                  <span>{{ dataset.featureNames.length }} features</span>
                </div>
                <div *ngIf="dataset.targetColumn" class="config-item">
                  <label>Target Column:</label>
                  <span>{{ dataset.targetColumn }}</span>
                </div>
              </div>
            </div>

            <!-- Metadata -->
            <div *ngIf="result.metadata && Object.keys(result.metadata).length > 0" class="detail-section">
              <h3>Additional Metadata</h3>
              <div class="metadata-content">
                <pre>{{ JSON.stringify(result.metadata, null, 2) }}</pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .pynomaly-detection-results {
      /* Component styles */
    }

    .results-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }

    .result-summary {
      color: #666;
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

    .performance-badge {
      margin-top: 5px;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 0.7em;
      font-weight: bold;
    }

    .performance-badge.excellent { background: #d4edda; color: #155724; }
    .performance-badge.good { background: #d1ecf1; color: #0c5460; }
    .performance-badge.fair { background: #fff3cd; color: #856404; }
    .performance-badge.slow { background: #f8d7da; color: #721c24; }

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

    .sorting-controls {
      display: flex;
      gap: 20px;
      margin-bottom: 20px;
    }

    .sort-group {
      display: flex;
      align-items: center;
      gap: 10px;
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

    .feature-values {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .feature-item {
      background: #f8f9fa;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 0.8em;
    }

    .feature-name {
      font-weight: bold;
      margin-right: 3px;
    }

    .config-items {
      display: grid;
      gap: 10px;
    }

    .config-item {
      display: flex;
      justify-content: space-between;
      padding: 8px;
      background: #f8f9fa;
      border-radius: 4px;
    }

    .metadata-content {
      background: #f8f9fa;
      padding: 15px;
      border-radius: 4px;
      overflow-x: auto;
    }

    .chart-placeholder {
      padding: 40px;
      text-align: center;
      background: #f8f9fa;
      border-radius: 4px;
      color: #666;
    }
  `]
})
export class DetectionResultsComponent implements OnInit {
  @Input() result!: DetectionResult;
  @Input() dataset?: Dataset;
  @Input() showVisualization: boolean = true;
  @Input() showMetrics: boolean = true;
  @Input() className: string = '';

  // Component state
  activeTab: 'overview' | 'anomalies' | 'details' = 'overview';
  sortBy: 'index' | 'score' = 'score';
  sortOrder: 'asc' | 'desc' = 'desc';
  
  metrics!: Metrics;
  anomalyData: AnomalyDataItem[] = [];
  performance!: Performance;

  ngOnInit() {
    this.calculateMetrics();
    this.updateAnomalyData();
    this.calculatePerformance();
  }

  private calculateMetrics() {
    const anomalyRate = (this.result.nAnomalies / this.result.nSamples) * 100;
    const normalRate = 100 - anomalyRate;
    
    const scores = this.result.anomalyScores;
    const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    
    const anomalyScores = this.result.anomalyLabels
      .map((label, index) => label === 1 ? scores[index] : null)
      .filter(score => score !== null) as number[];
    
    const avgAnomalyScore = anomalyScores.length > 0 
      ? anomalyScores.reduce((sum, score) => sum + score, 0) / anomalyScores.length 
      : 0;

    this.metrics = {
      anomalyRate,
      normalRate,
      avgScore,
      maxScore,
      minScore,
      avgAnomalyScore,
      scoreRange: maxScore - minScore
    };
  }

  updateAnomalyData() {
    const anomalies = this.result.anomalyLabels
      .map((label, index) => ({
        index,
        isAnomaly: label === 1,
        score: this.result.anomalyScores[index],
        data: this.dataset?.data ? this.dataset.data[index] : null
      }))
      .filter(item => item.isAnomaly);

    // Sort anomalies
    this.anomalyData = anomalies.sort((a, b) => {
      if (this.sortBy === 'score') {
        return this.sortOrder === 'desc' ? b.score - a.score : a.score - b.score;
      } else {
        return this.sortOrder === 'desc' ? b.index - a.index : a.index - b.index;
      }
    });
  }

  private calculatePerformance() {
    const executionTime = this.result.executionTime;
    if (executionTime < 1) {
      this.performance = { label: 'Excellent', class: 'excellent' };
    } else if (executionTime < 5) {
      this.performance = { label: 'Good', class: 'good' };
    } else if (executionTime < 15) {
      this.performance = { label: 'Fair', class: 'fair' };
    } else {
      this.performance = { label: 'Slow', class: 'slow' };
    }
  }

  getFeatureValue(data: any, name: string, index: number): any {
    return Array.isArray(data) ? data[index] : data[name];
  }

  // Utility methods for template
  Object = Object;
  JSON = JSON;
}