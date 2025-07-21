/**
 * Vue 3 composable for anomaly detection
 */

import { ref } from 'vue';
import { PynomalyClient, AnomalyDetectionRequest, AnomalyDetectionResult } from '../../../index';

export interface UseAnomalyDetectionOptions {
  client?: PynomalyClient;
}

export function useAnomalyDetection(options: UseAnomalyDetectionOptions = {}) {
  const result = ref<AnomalyDetectionResult | null>(null);
  const isLoading = ref(false);
  const error = ref<Error | null>(null);
  const progress = ref(0);

  const clear = () => {
    result.value = null;
    error.value = null;
    progress.value = 0;
  };

  const detectAnomalies = async (request: AnomalyDetectionRequest): Promise<AnomalyDetectionResult> => {
    if (!options.client) {
      throw new Error('Pynomaly client is required');
    }

    isLoading.value = true;
    error.value = null;
    progress.value = 0;

    try {
      progress.value = 25;
      const detectionResult = await options.client.detectAnomalies(request);
      
      progress.value = 100;
      result.value = detectionResult;
      
      return detectionResult;
    } catch (err) {
      const detectionError = err as Error;
      error.value = detectionError;
      throw detectionError;
    } finally {
      isLoading.value = false;
    }
  };

  const detectAnomaliesAsync = async (request: AnomalyDetectionRequest): Promise<string> => {
    if (!options.client) {
      throw new Error('Pynomaly client is required');
    }

    isLoading.value = true;
    error.value = null;
    progress.value = 0;

    try {
      progress.value = 10;
      const jobId = await options.client.detectAnomaliesAsync(request);
      
      // Poll for job completion
      const pollJob = async (): Promise<AnomalyDetectionResult> => {
        let attempts = 0;
        const maxAttempts = 300; // 10 minutes max
        
        while (attempts < maxAttempts) {
          const status = await options.client!.getJobStatus(jobId);
          
          // Update progress based on job status
          if (status.progress) {
            progress.value = Math.min(status.progress, 95);
          }
          
          switch (status.status) {
            case 'completed':
              const jobResult = await options.client!.getJobResult(jobId);
              progress.value = 100;
              result.value = jobResult;
              return jobResult;
              
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
      const detectionError = err as Error;
      error.value = detectionError;
      throw detectionError;
    } finally {
      isLoading.value = false;
    }
  };

  return {
    result,
    isLoading,
    error,
    progress,
    detectAnomalies,
    detectAnomaliesAsync,
    clear
  };
}