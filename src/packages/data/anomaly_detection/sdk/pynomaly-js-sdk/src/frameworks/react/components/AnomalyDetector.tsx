/**
 * React component for anomaly detection
 */

import React, { useState, useCallback } from 'react';
import { AnomalyDetectionRequest, AnomalyDetectionResult } from '../../../index';
import { useAnomalyDetection } from '../hooks/useAnomalyDetection';
import { usePynomaly } from './PynomalyProvider';

export interface AnomalyDetectorProps {
  data?: number[][];
  algorithm?: 'isolation_forest' | 'local_outlier_factor' | 'one_class_svm' | 'auto';
  parameters?: Record<string, any>;
  autoDetect?: boolean;
  onResult?: (result: AnomalyDetectionResult) => void;
  onError?: (error: Error) => void;
  className?: string;
  children?: (props: {
    detectAnomalies: (data: number[][]) => void;
    result: AnomalyDetectionResult | null;
    isLoading: boolean;
    error: Error | null;
    progress: number;
  }) => React.ReactNode;
}

export const AnomalyDetector: React.FC<AnomalyDetectorProps> = ({
  data,
  algorithm = 'auto',
  parameters = {},
  autoDetect = false,
  onResult,
  onError,
  className,
  children
}) => {
  const { client } = usePynomaly();
  const [inputData, setInputData] = useState<string>('');
  
  const {
    detectAnomalies,
    result,
    isLoading,
    error,
    progress,
    clear
  } = useAnomalyDetection({
    client: client || undefined,
    onSuccess: onResult,
    onError
  });

  const handleDetectAnomalies = useCallback(async (dataToAnalyze?: number[][]) => {
    const analysisData = dataToAnalyze || data;
    
    if (!analysisData || analysisData.length === 0) {
      return;
    }

    const request: AnomalyDetectionRequest = {
      data: analysisData,
      algorithm,
      parameters
    };

    try {
      await detectAnomalies(request);
    } catch (err) {
      console.error('Anomaly detection failed:', err);
    }
  }, [data, algorithm, parameters, detectAnomalies]);

  const handleInputDataChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputData(event.target.value);
  };

  const parseInputData = (): number[][] | null => {
    try {
      const parsed = JSON.parse(inputData);
      if (Array.isArray(parsed) && parsed.every(row => Array.isArray(row))) {
        return parsed;
      }
      return null;
    } catch {
      return null;
    }
  };

  const handleAnalyzeInput = () => {
    const parsedData = parseInputData();
    if (parsedData) {
      handleDetectAnomalies(parsedData);
    }
  };

  // Auto-detect when data changes
  React.useEffect(() => {
    if (autoDetect && data) {
      handleDetectAnomalies();
    }
  }, [data, autoDetect, handleDetectAnomalies]);

  if (children) {
    return (
      <div className={className}>
        {children({
          detectAnomalies: handleDetectAnomalies,
          result,
          isLoading,
          error,
          progress
        })}
      </div>
    );
  }

  return (
    <div className={className || 'pynomaly-anomaly-detector'}>
      <div className="input-section">
        <h3>Anomaly Detection</h3>
        
        <div className="algorithm-selection">
          <label>
            Algorithm:
            <select 
              value={algorithm} 
              onChange={(e) => {
                // Note: This would require making algorithm a state variable
                // For this example, we'll keep it as a prop
              }}
            >
              <option value="auto">Auto</option>
              <option value="isolation_forest">Isolation Forest</option>
              <option value="local_outlier_factor">Local Outlier Factor</option>
              <option value="one_class_svm">One-Class SVM</option>
            </select>
          </label>
        </div>

        <div className="data-input">
          <label>
            Data (JSON array of arrays):
            <textarea
              value={inputData}
              onChange={handleInputDataChange}
              placeholder='[[1, 2, 3], [4, 5, 6], [100, 200, 300]]'
              rows={4}
              style={{ width: '100%', fontFamily: 'monospace' }}
            />
          </label>
        </div>

        <div className="actions">
          <button 
            onClick={handleAnalyzeInput}
            disabled={isLoading || !client}
          >
            {isLoading ? 'Detecting...' : 'Detect Anomalies'}
          </button>
          
          <button 
            onClick={clear}
            disabled={isLoading}
          >
            Clear
          </button>
        </div>

        {isLoading && (
          <div className="progress">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span>Progress: {progress}%</span>
          </div>
        )}
      </div>

      {error && (
        <div className="error" style={{ color: 'red', marginTop: '10px' }}>
          Error: {error.message}
        </div>
      )}

      {result && (
        <div className="result" style={{ marginTop: '20px' }}>
          <h4>Detection Results</h4>
          <div className="result-summary">
            <p><strong>Algorithm:</strong> {result.algorithm}</p>
            <p><strong>Total Points:</strong> {result.metrics.totalPoints}</p>
            <p><strong>Anomalies Found:</strong> {result.metrics.anomalyCount}</p>
            <p><strong>Anomaly Rate:</strong> {(result.metrics.anomalyRate * 100).toFixed(2)}%</p>
            <p><strong>Processing Time:</strong> {result.processingTime}ms</p>
          </div>

          {result.anomalies.length > 0 && (
            <div className="anomalies">
              <h5>Detected Anomalies:</h5>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ border: '1px solid #ccc', padding: '8px' }}>Index</th>
                    <th style={{ border: '1px solid #ccc', padding: '8px' }}>Score</th>
                    <th style={{ border: '1px solid #ccc', padding: '8px' }}>Confidence</th>
                    <th style={{ border: '1px solid #ccc', padding: '8px' }}>Data</th>
                  </tr>
                </thead>
                <tbody>
                  {result.anomalies.map((anomaly, index) => (
                    <tr key={index}>
                      <td style={{ border: '1px solid #ccc', padding: '8px' }}>{anomaly.index}</td>
                      <td style={{ border: '1px solid #ccc', padding: '8px' }}>{anomaly.score.toFixed(4)}</td>
                      <td style={{ border: '1px solid #ccc', padding: '8px' }}>{(anomaly.confidence * 100).toFixed(1)}%</td>
                      <td style={{ border: '1px solid #ccc', padding: '8px' }}>
                        {JSON.stringify(anomaly.data)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AnomalyDetector;