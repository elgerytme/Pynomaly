/**
 * React hook for anomaly detection
 */

import { useState, useCallback, useRef } from 'react';
import { PynomalyClient, AnomalyDetectionRequest, AnomalyDetectionResult } from '../../../index';

export interface UseAnomalyDetectionOptions {
  client?: PynomalyClient;
  onSuccess?: (result: AnomalyDetectionResult) => void;
  onError?: (error: Error) => void;
  autoRetry?: boolean;
  maxRetries?: number;
}

export interface UseAnomalyDetectionReturn {
  detectAnomalies: (request: AnomalyDetectionRequest) => Promise<AnomalyDetectionResult>;
  detectAnomaliesAsync: (request: AnomalyDetectionRequest) => Promise<string>;
  result: AnomalyDetectionResult | null;
  isLoading: boolean;
  error: Error | null;
  progress: number;
  clear: () => void;
}

export function useAnomalyDetection(
  options: UseAnomalyDetectionOptions = {}
): UseAnomalyDetectionReturn {
  const [result, setResult] = useState<AnomalyDetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [progress, setProgress] = useState(0);
  const optionsRef = useRef(options);
  const mountedRef = useRef(true);

  // Update options ref when options change
  useState(() => {
    optionsRef.current = options;
  });

  const clear = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress(0);
  }, []);

  const detectAnomalies = useCallback(async (request: AnomalyDetectionRequest): Promise<AnomalyDetectionResult> => {
    if (!optionsRef.current.client) {
      throw new Error('Pynomaly client is required');
    }

    setIsLoading(true);
    setError(null);
    setProgress(0);

    try {
      setProgress(25);
      const detectionResult = await optionsRef.current.client.detectAnomalies(request);
      
      if (mountedRef.current) {
        setProgress(100);
        setResult(detectionResult);
        optionsRef.current.onSuccess?.(detectionResult);
      }
      
      return detectionResult;
    } catch (err) {
      const error = err as Error;
      if (mountedRef.current) {
        setError(error);
        optionsRef.current.onError?.(error);
      }
      throw error;
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  const detectAnomaliesAsync = useCallback(async (request: AnomalyDetectionRequest): Promise<string> => {
    if (!optionsRef.current.client) {
      throw new Error('Pynomaly client is required');
    }

    setIsLoading(true);
    setError(null);
    setProgress(0);

    try {
      setProgress(10);
      const jobId = await optionsRef.current.client.detectAnomaliesAsync(request);
      
      // Poll for job completion
      const pollJob = async (): Promise<AnomalyDetectionResult> => {
        let attempts = 0;
        const maxAttempts = 300; // 10 minutes max
        
        while (attempts < maxAttempts && mountedRef.current) {
          const status = await optionsRef.current.client!.getJobStatus(jobId);
          
          // Update progress based on job status
          if (status.progress) {
            setProgress(Math.min(status.progress, 95));
          }
          
          switch (status.status) {
            case 'completed':
              const result = await optionsRef.current.client!.getJobResult(jobId);
              setProgress(100);
              setResult(result);
              optionsRef.current.onSuccess?.(result);
              return result;
              
            case 'failed':
              throw new Error(status.error || 'Job failed');
              
            case 'cancelled':
              throw new Error('Job was cancelled');
              
            case 'pending':
            case 'running':
              // Continue polling
              await new Promise(resolve => setTimeout(resolve, 2000));
              attempts++;
              break;
              
            default:
              throw new Error(`Unknown job status: ${status.status}`);
          }
        }
        
        throw new Error('Job polling timeout');
      };
      
      await pollJob();
      return jobId;
    } catch (err) {
      const error = err as Error;
      if (mountedRef.current) {
        setError(error);
        optionsRef.current.onError?.(error);
      }
      throw error;
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  return {
    detectAnomalies,
    detectAnomaliesAsync,
    result,
    isLoading,
    error,
    progress,
    clear
  };
}