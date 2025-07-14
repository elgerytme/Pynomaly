/**
 * Detection Results Component
 * 
 * React component for displaying anomaly detection results with
 * visualizations, metrics, and data exploration capabilities.
 */

import React, { useMemo, useState } from 'react';
import { DetectionResultsProps, DetectionResult } from '../types';

/**
 * Detection results component with visualization and metrics.
 * 
 * @param props Component properties
 * @returns JSX element
 */
export const DetectionResults: React.FC<DetectionResultsProps> = ({
  result,
  dataset,
  showVisualization = true,
  showMetrics = true,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'anomalies' | 'details'>('overview');
  const [sortBy, setSortBy] = useState<'index' | 'score'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Calculate derived metrics
  const metrics = useMemo(() => {
    const anomalyRate = (result.nAnomalies / result.nSamples) * 100;
    const normalRate = 100 - anomalyRate;
    
    const scores = result.anomalyScores;
    const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    
    const anomalyScores = result.anomalyLabels
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
  }, [result]);

  // Get anomaly data for display
  const anomalyData = useMemo(() => {
    const anomalies = result.anomalyLabels
      .map((label, index) => ({
        index,
        isAnomaly: label === 1,
        score: result.anomalyScores[index],
        data: dataset?.data ? dataset.data[index] : null
      }))
      .filter(item => item.isAnomaly);

    // Sort anomalies
    return anomalies.sort((a, b) => {
      if (sortBy === 'score') {
        return sortOrder === 'desc' ? b.score - a.score : a.score - b.score;
      } else {
        return sortOrder === 'desc' ? b.index - a.index : a.index - b.index;
      }
    });
  }, [result, dataset, sortBy, sortOrder]);

  // Performance category
  const getPerformanceCategory = (executionTime: number) => {
    if (executionTime < 1) return { label: 'Excellent', class: 'excellent' };
    if (executionTime < 5) return { label: 'Good', class: 'good' };
    if (executionTime < 15) return { label: 'Fair', class: 'fair' };
    return { label: 'Slow', class: 'slow' };
  };

  const performance = getPerformanceCategory(result.executionTime);

  return (
    <div className={`pynomaly-detection-results ${className}`}>
      {/* Header */}
      <div className="results-header">
        <h2>Detection Results</h2>
        <div className="result-summary">
          <span className="anomaly-count">
            {result.nAnomalies} anomalies detected
          </span>
          <span className="sample-count">
            out of {result.nSamples} samples
          </span>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="quick-stats">
        <div className="stat-card anomaly-rate">
          <div className="stat-value">{metrics.anomalyRate.toFixed(1)}%</div>
          <div className="stat-label">Anomaly Rate</div>
        </div>
        <div className="stat-card threshold">
          <div className="stat-value">{result.threshold.toFixed(3)}</div>
          <div className="stat-label">Threshold</div>
        </div>
        <div className="stat-card execution-time">
          <div className="stat-value">{result.executionTime.toFixed(2)}s</div>
          <div className="stat-label">Execution Time</div>
          <div className={`performance-badge ${performance.class}`}>
            {performance.label}
          </div>
        </div>
        <div className="stat-card score-range">
          <div className="stat-value">{metrics.scoreRange.toFixed(3)}</div>
          <div className="stat-label">Score Range</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="result-tabs">
        <button
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={`tab ${activeTab === 'anomalies' ? 'active' : ''}`}
          onClick={() => setActiveTab('anomalies')}
        >
          Anomalies ({result.nAnomalies})
        </button>
        <button
          className={`tab ${activeTab === 'details' ? 'active' : ''}`}
          onClick={() => setActiveTab('details')}
        >
          Details
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <div className="overview-content">
            {/* Distribution Chart */}
            {showVisualization && (
              <div className="distribution-chart">
                <h3>Score Distribution</h3>
                <div className="chart-container">
                  <ScoreDistributionChart
                    scores={result.anomalyScores}
                    labels={result.anomalyLabels}
                    threshold={result.threshold}
                  />
                </div>
              </div>
            )}

            {/* Metrics Summary */}
            {showMetrics && (
              <div className="metrics-summary">
                <h3>Detection Metrics</h3>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <label>Total Samples:</label>
                    <span>{result.nSamples.toLocaleString()}</span>
                  </div>
                  <div className="metric-item">
                    <label>Anomalies Found:</label>
                    <span>{result.nAnomalies.toLocaleString()}</span>
                  </div>
                  <div className="metric-item">
                    <label>Contamination Rate:</label>
                    <span>{(result.contaminationRate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric-item">
                    <label>Average Score:</label>
                    <span>{metrics.avgScore.toFixed(4)}</span>
                  </div>
                  <div className="metric-item">
                    <label>Average Anomaly Score:</label>
                    <span>{metrics.avgAnomalyScore.toFixed(4)}</span>
                  </div>
                  <div className="metric-item">
                    <label>Score Range:</label>
                    <span>{metrics.minScore.toFixed(4)} - {metrics.maxScore.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'anomalies' && (
          <div className="anomalies-content">
            {/* Sorting Controls */}
            <div className="sorting-controls">
              <div className="sort-group">
                <label>Sort by:</label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as 'index' | 'score')}
                >
                  <option value="score">Anomaly Score</option>
                  <option value="index">Sample Index</option>
                </select>
              </div>
              <div className="sort-group">
                <label>Order:</label>
                <select
                  value={sortOrder}
                  onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}
                >
                  <option value="desc">Descending</option>
                  <option value="asc">Ascending</option>
                </select>
              </div>
            </div>

            {/* Anomaly List */}
            <div className="anomaly-list">
              {anomalyData.length === 0 ? (
                <div className="no-anomalies">
                  <p>No anomalies detected in this dataset.</p>
                </div>
              ) : (
                <div className="anomaly-table">
                  <div className="table-header">
                    <div className="col-index">Index</div>
                    <div className="col-score">Score</div>
                    {dataset?.featureNames && (
                      <div className="col-features">Features</div>
                    )}
                  </div>
                  {anomalyData.map((anomaly, idx) => (
                    <div key={anomaly.index} className="table-row">
                      <div className="col-index">#{anomaly.index}</div>
                      <div className="col-score">
                        <span className="score-value">
                          {anomaly.score.toFixed(4)}
                        </span>
                        <div className="score-bar">
                          <div
                            className="score-fill"
                            style={{
                              width: `${((anomaly.score - metrics.minScore) / metrics.scoreRange) * 100}%`
                            }}
                          />
                        </div>
                      </div>
                      {dataset?.featureNames && anomaly.data && (
                        <div className="col-features">
                          <div className="feature-values">
                            {dataset.featureNames.map((name, featureIdx) => (
                              <div key={name} className="feature-item">
                                <span className="feature-name">{name}:</span>
                                <span className="feature-value">
                                  {Array.isArray(anomaly.data) 
                                    ? anomaly.data[featureIdx]
                                    : anomaly.data[name]
                                  }
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'details' && (
          <div className="details-content">
            <div className="detail-sections">
              {/* Configuration */}
              <div className="detail-section">
                <h3>Detection Configuration</h3>
                <div className="config-items">
                  <div className="config-item">
                    <label>Threshold:</label>
                    <span>{result.threshold}</span>
                  </div>
                  <div className="config-item">
                    <label>Expected Contamination:</label>
                    <span>{(result.contaminationRate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="config-item">
                    <label>Execution Time:</label>
                    <span>{result.executionTime.toFixed(2)} seconds</span>
                  </div>
                </div>
              </div>

              {/* Dataset Info */}
              {dataset && (
                <div className="detail-section">
                  <h3>Dataset Information</h3>
                  <div className="config-items">
                    <div className="config-item">
                      <label>Dataset Name:</label>
                      <span>{dataset.name}</span>
                    </div>
                    <div className="config-item">
                      <label>Sample Count:</label>
                      <span>{result.nSamples.toLocaleString()}</span>
                    </div>
                    {dataset.featureNames && (
                      <div className="config-item">
                        <label>Features:</label>
                        <span>{dataset.featureNames.length} features</span>
                      </div>
                    )}
                    {dataset.targetColumn && (
                      <div className="config-item">
                        <label>Target Column:</label>
                        <span>{dataset.targetColumn}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Metadata */}
              {result.metadata && Object.keys(result.metadata).length > 0 && (
                <div className="detail-section">
                  <h3>Additional Metadata</h3>
                  <div className="metadata-content">
                    <pre>{JSON.stringify(result.metadata, null, 2)}</pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Simple score distribution chart component.
 * In a real implementation, you might use a charting library like Chart.js or D3.
 */
const ScoreDistributionChart: React.FC<{
  scores: number[];
  labels: number[];
  threshold: number;
}> = ({ scores, labels, threshold }) => {
  const buckets = useMemo(() => {
    const numBuckets = 20;
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    const bucketSize = (maxScore - minScore) / numBuckets;
    
    const bucketData = Array(numBuckets).fill(0).map((_, i) => ({
      min: minScore + i * bucketSize,
      max: minScore + (i + 1) * bucketSize,
      normal: 0,
      anomaly: 0
    }));
    
    scores.forEach((score, index) => {
      const bucketIndex = Math.min(
        Math.floor((score - minScore) / bucketSize),
        numBuckets - 1
      );
      if (labels[index] === 1) {
        bucketData[bucketIndex].anomaly++;
      } else {
        bucketData[bucketIndex].normal++;
      }
    });
    
    return bucketData;
  }, [scores, labels]);

  const maxCount = Math.max(...buckets.map(b => b.normal + b.anomaly));

  return (
    <div className="score-distribution-chart">
      <div className="chart-bars">
        {buckets.map((bucket, index) => {
          const totalHeight = ((bucket.normal + bucket.anomaly) / maxCount) * 100;
          const anomalyHeight = (bucket.anomaly / (bucket.normal + bucket.anomaly)) * totalHeight;
          const normalHeight = totalHeight - anomalyHeight;
          
          return (
            <div
              key={index}
              className="chart-bar"
              style={{ height: `${totalHeight}%` }}
              title={`Score: ${bucket.min.toFixed(3)} - ${bucket.max.toFixed(3)}\nNormal: ${bucket.normal}\nAnomalies: ${bucket.anomaly}`}
            >
              <div
                className="bar-anomaly"
                style={{ height: `${anomalyHeight}%` }}
              />
              <div
                className="bar-normal"
                style={{ height: `${normalHeight}%` }}
              />
            </div>
          );
        })}
      </div>
      
      {/* Threshold line */}
      <div
        className="threshold-line"
        style={{
          left: `${((threshold - Math.min(...scores)) / (Math.max(...scores) - Math.min(...scores))) * 100}%`
        }}
        title={`Threshold: ${threshold.toFixed(4)}`}
      />
      
      <div className="chart-legend">
        <div className="legend-item">
          <div className="legend-color normal"></div>
          <span>Normal</span>
        </div>
        <div className="legend-item">
          <div className="legend-color anomaly"></div>
          <span>Anomaly</span>
        </div>
        <div className="legend-item">
          <div className="legend-line threshold"></div>
          <span>Threshold</span>
        </div>
      </div>
    </div>
  );
};