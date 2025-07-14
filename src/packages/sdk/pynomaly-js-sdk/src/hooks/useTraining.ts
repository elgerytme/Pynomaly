/**
 * React Hook for Training Management
 * 
 * Custom React hook for managing training operations including
 * job submission, monitoring, and result management.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { UseTrainingState, Dataset, TrainingJob, PynomalyError } from '../types';
import { PynomalyClient } from '../core/client';

/**
 * Hook for managing training operations.
 * 
 * @param client Pynomaly client instance
 * @returns Training state and management functions
 */
export function useTraining(client: PynomalyClient) {
  const [state, setState] = useState<UseTrainingState>({
    job: null,
    isTraining: false,
    error: null
  });

  // Start training
  const startTraining = useCallback(async (
    detectorId: string,
    dataset: Dataset,
    jobName?: string
  ) => {
    setState(prev => ({ ...prev, isTraining: true, error: null }));
    
    try {
      const job = await client.dataScience.trainDetector(detectorId, dataset, jobName);
      setState(prev => ({ ...prev, job, isTraining: false }));
      return job;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isTraining: false }));
      throw error;
    }
  }, [client]);

  // Load training job
  const loadJob = useCallback(async (jobId: string) => {
    setState(prev => ({ ...prev, isTraining: true, error: null }));
    
    try {
      const job = await client.dataScience.getTrainingJob(jobId);
      setState(prev => ({ ...prev, job, isTraining: false }));
      return job;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isTraining: false }));
      throw error;
    }
  }, [client]);

  // Clear job
  const clearJob = useCallback(() => {
    setState(prev => ({ ...prev, job: null }));
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Reset state
  const reset = useCallback(() => {
    setState({
      job: null,
      isTraining: false,
      error: null
    });
  }, []);

  return {
    ...state,
    startTraining,
    loadJob,
    clearJob,
    clearError,
    reset
  };
}

/**
 * Hook for monitoring a training job with automatic polling.
 * 
 * @param client Pynomaly client instance
 * @param jobId Training job ID to monitor
 * @param options Monitoring options
 * @returns Training job state and monitoring controls
 */
export function useTrainingMonitor(
  client: PynomalyClient,
  jobId: string | null,
  options: {
    pollInterval?: number; // Polling interval in milliseconds
    autoStart?: boolean;   // Auto-start monitoring
    onComplete?: (job: TrainingJob) => void;
    onError?: (error: PynomalyError) => void;
  } = {}
) {
  const [state, setState] = useState({
    job: null as TrainingJob | null,
    isMonitoring: false,
    isLoading: false,
    error: null as PynomalyError | null
  });

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const { pollInterval = 5000, autoStart = true, onComplete, onError } = options;

  // Start monitoring
  const startMonitoring = useCallback(() => {
    if (!jobId || state.isMonitoring) return;

    setState(prev => ({ ...prev, isMonitoring: true, error: null }));

    const poll = async () => {
      try {
        setState(prev => ({ ...prev, isLoading: true }));
        const job = await client.dataScience.getTrainingJob(jobId);
        setState(prev => ({ ...prev, job, isLoading: false }));

        // Check if job is completed
        if (job.status === 'completed' || job.status === 'failed') {
          stopMonitoring();
          
          if (job.status === 'completed' && onComplete) {
            onComplete(job);
          } else if (job.status === 'failed' && onError) {
            onError(new Error(job.errorMessage || 'Training failed') as PynomalyError);
          }
        }
      } catch (error) {
        const pynomalyError = error as PynomalyError;
        setState(prev => ({ ...prev, error: pynomalyError, isLoading: false }));
        
        if (onError) {
          onError(pynomalyError);
        }
        
        stopMonitoring();
      }
    };

    // Initial poll
    poll();

    // Set up interval
    intervalRef.current = setInterval(poll, pollInterval);
  }, [client, jobId, state.isMonitoring, pollInterval, onComplete, onError]);

  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setState(prev => ({ ...prev, isMonitoring: false, isLoading: false }));
  }, []);

  // Refresh job status
  const refresh = useCallback(async () => {
    if (!jobId) return;

    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const job = await client.dataScience.getTrainingJob(jobId);
      setState(prev => ({ ...prev, job, isLoading: false }));
      return job;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isLoading: false }));
      throw error;
    }
  }, [client, jobId]);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Auto-start monitoring when jobId changes
  useEffect(() => {
    if (jobId && autoStart) {
      startMonitoring();
    } else {
      stopMonitoring();
    }

    return () => stopMonitoring();
  }, [jobId, autoStart, startMonitoring, stopMonitoring]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopMonitoring();
  }, [stopMonitoring]);

  return {
    ...state,
    startMonitoring,
    stopMonitoring,
    refresh,
    clearError
  };
}

/**
 * Hook for managing multiple training jobs.
 * 
 * @param client Pynomaly client instance
 * @param options List options
 * @returns Training jobs state and management functions
 */
export function useTrainingJobList(
  client: PynomalyClient,
  options: {
    detectorId?: string;
    status?: string;
    pageSize?: number;
    autoLoad?: boolean;
    autoRefresh?: boolean;
    refreshInterval?: number;
  } = {}
) {
  const [state, setState] = useState({
    jobs: [] as TrainingJob[],
    total: 0,
    page: 1,
    pageSize: options.pageSize || 20,
    hasNext: false,
    hasPrevious: false,
    isLoading: false,
    error: null as PynomalyError | null
  });

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const { autoRefresh = false, refreshInterval = 30000 } = options;

  // Load training jobs
  const loadJobs = useCallback(async (page: number = 1) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const result = await client.dataScience.listTrainingJobs({
        page,
        pageSize: state.pageSize,
        detectorId: options.detectorId,
        status: options.status
      });
      
      setState(prev => ({
        ...prev,
        jobs: result.items,
        total: result.total,
        page: result.page,
        hasNext: result.hasNext,
        hasPrevious: result.hasPrevious,
        isLoading: false
      }));
      
      return result;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isLoading: false }));
      throw error;
    }
  }, [client, state.pageSize, options.detectorId, options.status]);

  // Navigate to next page
  const nextPage = useCallback(async () => {
    if (state.hasNext) {
      await loadJobs(state.page + 1);
    }
  }, [loadJobs, state.hasNext, state.page]);

  // Navigate to previous page
  const previousPage = useCallback(async () => {
    if (state.hasPrevious) {
      await loadJobs(state.page - 1);
    }
  }, [loadJobs, state.hasPrevious, state.page]);

  // Go to specific page
  const goToPage = useCallback(async (page: number) => {
    await loadJobs(page);
  }, [loadJobs]);

  // Refresh current page
  const refresh = useCallback(async () => {
    await loadJobs(state.page);
  }, [loadJobs, state.page]);

  // Start auto-refresh
  const startAutoRefresh = useCallback(() => {
    if (intervalRef.current) return;
    
    intervalRef.current = setInterval(() => {
      refresh();
    }, refreshInterval);
  }, [refresh, refreshInterval]);

  // Stop auto-refresh
  const stopAutoRefresh = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Load jobs on mount if autoLoad is true
  useEffect(() => {
    if (options.autoLoad !== false) {
      loadJobs();
    }
  }, [loadJobs, options.autoLoad]);

  // Start auto-refresh if enabled
  useEffect(() => {
    if (autoRefresh) {
      startAutoRefresh();
    }

    return () => stopAutoRefresh();
  }, [autoRefresh, startAutoRefresh, stopAutoRefresh]);

  return {
    ...state,
    loadJobs,
    nextPage,
    previousPage,
    goToPage,
    refresh,
    startAutoRefresh,
    stopAutoRefresh,
    clearError
  };
}