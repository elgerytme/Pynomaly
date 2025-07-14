/**
 * React Hook for Anomaly Detection
 * 
 * Custom React hook for managing anomaly detection operations including
 * real-time detection, batch processing, and result management.
 */

import { useState, useCallback, useRef } from 'react';
import { UseDetectionState, Dataset, DetectionResult, PynomalyError } from '../types';
import { PynomalyClient } from '../core/client';

/**
 * Hook for managing anomaly detection operations.
 * 
 * @param client Pynomaly client instance
 * @returns Detection state and management functions
 */
export function useDetection(client: PynomalyClient) {
  const [state, setState] = useState<UseDetectionState>({
    result: null,
    isDetecting: false,
    error: null
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Detect anomalies
  const detectAnomalies = useCallback(async (
    detectorId: string,
    dataset: Dataset,
    options: {
      returnScores?: boolean;
      threshold?: number;
    } = {}
  ) => {
    // Cancel any ongoing detection
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setState(prev => ({ ...prev, isDetecting: true, error: null }));
    
    try {
      const result = await client.dataScience.detectAnomalies(detectorId, dataset, options);
      setState(prev => ({ ...prev, result, isDetecting: false }));
      return result;
    } catch (error) {
      // Don't set error if request was aborted
      if ((error as any).name !== 'AbortError') {
        const pynomalyError = error as PynomalyError;
        setState(prev => ({ ...prev, error: pynomalyError, isDetecting: false }));
      }
      throw error;
    } finally {
      abortControllerRef.current = null;
    }
  }, [client]);

  // Batch detect anomalies
  const batchDetect = useCallback(async (
    detectorId: string,
    datasets: Dataset[],
    jobName?: string
  ) => {
    setState(prev => ({ ...prev, isDetecting: true, error: null }));
    
    try {
      const job = await client.dataScience.batchDetect(detectorId, datasets, jobName);
      setState(prev => ({ ...prev, isDetecting: false }));
      return job;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isDetecting: false }));
      throw error;
    }
  }, [client]);

  // Cancel ongoing detection
  const cancelDetection = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setState(prev => ({ ...prev, isDetecting: false }));
    }
  }, []);

  // Clear results
  const clearResults = useCallback(() => {
    setState(prev => ({ ...prev, result: null }));
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Reset state
  const reset = useCallback(() => {
    cancelDetection();
    setState({
      result: null,
      isDetecting: false,
      error: null
    });
  }, [cancelDetection]);

  return {
    ...state,
    detectAnomalies,
    batchDetect,
    cancelDetection,
    clearResults,
    clearError,
    reset
  };
}

/**
 * Hook for managing multiple detection results with history.
 * 
 * @param client Pynomaly client instance
 * @param maxHistory Maximum number of results to keep in history
 * @returns Detection history state and management functions
 */
export function useDetectionHistory(
  client: PynomalyClient,
  maxHistory: number = 10
) {
  const [state, setState] = useState({
    results: [] as DetectionResult[],
    currentResult: null as DetectionResult | null,
    isDetecting: false,
    error: null as PynomalyError | null
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Detect anomalies and add to history
  const detectAndStore = useCallback(async (
    detectorId: string,
    dataset: Dataset,
    options: {
      returnScores?: boolean;
      threshold?: number;
      label?: string; // Optional label for the detection
    } = {}
  ) => {
    // Cancel any ongoing detection
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setState(prev => ({ ...prev, isDetecting: true, error: null }));
    
    try {
      const result = await client.dataScience.detectAnomalies(detectorId, dataset, options);
      
      // Add label to metadata if provided
      if (options.label) {
        result.metadata = { ...result.metadata, label: options.label };
      }
      
      setState(prev => {
        const newResults = [result, ...prev.results].slice(0, maxHistory);
        return {
          ...prev,
          results: newResults,
          currentResult: result,
          isDetecting: false
        };
      });
      
      return result;
    } catch (error) {
      // Don't set error if request was aborted
      if ((error as any).name !== 'AbortError') {
        const pynomalyError = error as PynomalyError;
        setState(prev => ({ ...prev, error: pynomalyError, isDetecting: false }));
      }
      throw error;
    } finally {
      abortControllerRef.current = null;
    }
  }, [client, maxHistory]);

  // Select a result from history
  const selectResult = useCallback((index: number) => {
    setState(prev => ({
      ...prev,
      currentResult: prev.results[index] || null
    }));
  }, []);

  // Remove a result from history
  const removeResult = useCallback((index: number) => {
    setState(prev => {
      const newResults = prev.results.filter((_, i) => i !== index);
      const currentIndex = prev.results.indexOf(prev.currentResult!);
      
      return {
        ...prev,
        results: newResults,
        currentResult: currentIndex === index ? (newResults[0] || null) : prev.currentResult
      };
    });
  }, []);

  // Clear all results
  const clearHistory = useCallback(() => {
    setState(prev => ({
      ...prev,
      results: [],
      currentResult: null
    }));
  }, []);

  // Cancel ongoing detection
  const cancelDetection = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setState(prev => ({ ...prev, isDetecting: false }));
    }
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  return {
    ...state,
    detectAndStore,
    selectResult,
    removeResult,
    clearHistory,
    cancelDetection,
    clearError
  };
}

/**
 * Hook for comparing multiple detection results.
 * 
 * @param results Array of detection results to compare
 * @returns Comparison metrics and utilities
 */
export function useDetectionComparison(results: DetectionResult[]) {
  const [selectedResults, setSelectedResults] = useState<number[]>([]);

  // Toggle result selection
  const toggleSelection = useCallback((index: number) => {
    setSelectedResults(prev => 
      prev.includes(index) 
        ? prev.filter(i => i !== index)
        : [...prev, index]
    );
  }, []);

  // Select all results
  const selectAll = useCallback(() => {
    setSelectedResults(results.map((_, index) => index));
  }, [results]);

  // Clear selection
  const clearSelection = useCallback(() => {
    setSelectedResults([]);
  }, []);

  // Get comparison metrics
  const getComparisonMetrics = useCallback(() => {
    const selected = selectedResults.map(i => results[i]).filter(Boolean);
    
    if (selected.length === 0) return null;

    const metrics = {
      count: selected.length,
      avgAnomalies: selected.reduce((sum, r) => sum + r.nAnomalies, 0) / selected.length,
      avgContamination: selected.reduce((sum, r) => sum + r.contaminationRate, 0) / selected.length,
      avgThreshold: selected.reduce((sum, r) => sum + r.threshold, 0) / selected.length,
      avgExecutionTime: selected.reduce((sum, r) => sum + r.executionTime, 0) / selected.length,
      minAnomalies: Math.min(...selected.map(r => r.nAnomalies)),
      maxAnomalies: Math.max(...selected.map(r => r.nAnomalies)),
      totalSamples: selected.reduce((sum, r) => sum + r.nSamples, 0)
    };

    return metrics;
  }, [selectedResults, results]);

  return {
    selectedResults,
    toggleSelection,
    selectAll,
    clearSelection,
    getComparisonMetrics
  };
}