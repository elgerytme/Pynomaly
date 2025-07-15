/**
 * Async API wrapper with Promise-based operations and async/await support
 */

import { PynomalyClient } from './client';
import {
  AnomalyDetectionRequest,
  AnomalyDetectionResult,
  DataQualityRequest,
  DataQualityResult,
  DataProfilingRequest,
  DataProfilingResult,
  BatchRequest,
  BatchResult,
  JobStatus,
  WebSocketMessage,
  StreamConfig,
  StreamMessage
} from '../types';

export class AsyncAPI {
  private client: PynomalyClient;
  private jobPollingInterval: number = 2000; // 2 seconds
  private maxPollingAttempts: number = 300; // 10 minutes max

  constructor(client: PynomalyClient) {
    this.client = client;
  }

  // Promise-based anomaly detection with automatic job polling
  async detectAnomalies(request: AnomalyDetectionRequest): Promise<AnomalyDetectionResult> {
    const jobId = await this.client.detectAnomaliesAsync(request);
    return this.waitForJobCompletion<AnomalyDetectionResult>(jobId);
  }

  // Promise-based data quality analysis with automatic job polling
  async analyzeDataQuality(request: DataQualityRequest): Promise<DataQualityResult> {
    const jobId = await this.client.analyzeDataQualityAsync(request);
    return this.waitForJobCompletion<DataQualityResult>(jobId);
  }

  // Promise-based data profiling with automatic job polling
  async profileData(request: DataProfilingRequest): Promise<DataProfilingResult> {
    const jobId = await this.client.profileDataAsync(request);
    return this.waitForJobCompletion<DataProfilingResult>(jobId);
  }

  // Promise-based batch processing with automatic job polling
  async processBatch(request: BatchRequest): Promise<BatchResult> {
    const jobId = await this.client.processBatchAsync(request);
    return this.waitForJobCompletion<BatchResult>(jobId);
  }

  // Generic job completion waiter
  private async waitForJobCompletion<T>(jobId: string): Promise<T> {
    return new Promise((resolve, reject) => {
      let attempts = 0;
      
      const poll = async () => {
        try {
          attempts++;
          
          if (attempts > this.maxPollingAttempts) {
            reject(new Error('Job polling timeout exceeded'));
            return;
          }

          const status = await this.client.getJobStatus(jobId);
          
          switch (status.status) {
            case 'completed':
              const result = await this.client.getJobResult(jobId);
              resolve(result);
              return;
              
            case 'failed':
              reject(new Error(status.error || 'Job failed'));
              return;
              
            case 'cancelled':
              reject(new Error('Job was cancelled'));
              return;
              
            case 'pending':
            case 'running':
              // Continue polling
              setTimeout(poll, this.jobPollingInterval);
              break;
              
            default:
              reject(new Error(`Unknown job status: ${status.status}`));
              return;
          }
        } catch (error) {
          reject(error);
        }
      };
      
      poll();
    });
  }

  // Promise-based job status monitoring with progress updates
  async monitorJob(jobId: string): Promise<AsyncIterableIterator<JobStatus>> {
    const self = this;
    let finished = false;
    
    const iterator: AsyncIterableIterator<JobStatus> = {
      [Symbol.asyncIterator](): AsyncIterableIterator<JobStatus> {
        return iterator;
      },
      async next(): Promise<IteratorResult<JobStatus>> {
        if (finished) {
          return { done: true, value: undefined };
        }
        
        try {
          const status = await self.client.getJobStatus(jobId);
          
          if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
            finished = true;
            return { done: true, value: status };
          }
          
          return { done: false, value: status };
        } catch (error) {
          finished = true;
          throw error;
        }
      }
    };
    
    return iterator;
  }

  // Async generator for streaming operations
  async* streamProcessing(
    requests: Array<AnomalyDetectionRequest | DataQualityRequest | DataProfilingRequest>,
    type: 'anomaly' | 'quality' | 'profiling',
    config: StreamConfig = {}
  ): AsyncGenerator<StreamMessage, void, unknown> {
    const bufferSize = config.bufferSize || 10;
    const flushInterval = config.flushInterval || 5000;
    const autoFlush = config.autoFlush !== false;

    const buffer: string[] = [];
    let lastFlush = Date.now();

    for (let i = 0; i < requests.length; i += bufferSize) {
      const batch = requests.slice(i, i + bufferSize);
      const jobs: string[] = [];

      // Start all jobs in the batch
      for (const request of batch) {
        try {
          let jobId: string;
          
          switch (type) {
            case 'anomaly':
              jobId = await this.client.detectAnomaliesAsync(request as AnomalyDetectionRequest);
              break;
            case 'quality':
              jobId = await this.client.analyzeDataQualityAsync(request as DataQualityRequest);
              break;
            case 'profiling':
              jobId = await this.client.profileDataAsync(request as DataProfilingRequest);
              break;
            default:
              throw new Error(`Unknown processing type: ${type}`);
          }
          
          jobs.push(jobId);
          buffer.push(jobId);
        } catch (error) {
          yield {
            type: 'error',
            payload: { error: error instanceof Error ? error.message : String(error), requestIndex: i },
            timestamp: new Date()
          };
        }
      }

      // Monitor jobs and yield results
      const promises = jobs.map(async (jobId) => {
        try {
          const result = await this.waitForJobCompletion(jobId);
          return { jobId, result, error: null };
        } catch (error) {
          return { jobId, result: null, error: error instanceof Error ? error.message : String(error) };
        }
      });

      const results = await Promise.allSettled(promises);
      
      for (const result of results) {
        if (result.status === 'fulfilled') {
          if (result.value.error) {
            yield {
              type: 'error',
              payload: { jobId: result.value.jobId, error: result.value.error },
              timestamp: new Date()
            };
          } else {
            yield {
              type: 'data',
              payload: { jobId: result.value.jobId, result: result.value.result },
              timestamp: new Date()
            };
          }
        } else {
          yield {
            type: 'error',
            payload: { error: result.reason },
            timestamp: new Date()
          };
        }
      }

      // Auto-flush if needed
      if (autoFlush && (Date.now() - lastFlush) > flushInterval) {
        yield {
          type: 'control',
          payload: { action: 'flush', bufferSize: buffer.length },
          timestamp: new Date()
        };
        
        buffer.length = 0;
        lastFlush = Date.now();
      }
    }

    // Final flush
    if (buffer.length > 0) {
      yield {
        type: 'control',
        payload: { action: 'final_flush', bufferSize: buffer.length },
        timestamp: new Date()
      };
    }
  }

  // Promise-based parallel processing with concurrency control
  async processParallel<T>(
    requests: Array<() => Promise<T>>,
    concurrency: number = 5
  ): Promise<Array<{ success: boolean; result?: T; error?: string }>> {
    const results: Array<{ success: boolean; result?: T; error?: string }> = [];
    const executing: Promise<void>[] = [];

    for (let i = 0; i < requests.length; i++) {
      const request = requests[i];
      const promise = request()
        .then(result => {
          results[i] = { success: true, result };
        })
        .catch(error => {
          results[i] = { success: false, error: error.message };
        });

      executing.push(promise);

      if (executing.length >= concurrency) {
        await Promise.race(executing);
        executing.splice(executing.findIndex(p => p === promise), 1);
      }
    }

    await Promise.all(executing);
    return results;
  }

  // Promise-based retry mechanism with exponential backoff
  async withRetry<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
  ): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;
        
        if (attempt === maxRetries) {
          throw lastError;
        }

        const delay = baseDelay * Math.pow(2, attempt - 1);
        await this.delay(delay);
      }
    }

    throw lastError!;
  }

  // Promise-based timeout wrapper
  async withTimeout<T>(
    operation: () => Promise<T>,
    timeoutMs: number,
    timeoutMessage: string = 'Operation timed out'
  ): Promise<T> {
    return Promise.race([
      operation(),
      new Promise<T>((_, reject) => {
        setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
      })
    ]);
  }

  // Promise-based circuit breaker
  async withCircuitBreaker<T>(
    operation: () => Promise<T>,
    failureThreshold: number = 5,
    recoveryTimeout: number = 60000
  ): Promise<T> {
    const key = operation.name || 'anonymous';
    
    if (!this.circuitStates) {
      this.circuitStates = new Map();
    }

    const state = this.circuitStates.get(key) || {
      failures: 0,
      lastFailure: 0,
      isOpen: false
    };

    // Check if circuit is open
    if (state.isOpen) {
      if (Date.now() - state.lastFailure < recoveryTimeout) {
        throw new Error('Circuit breaker is open');
      } else {
        // Reset circuit
        state.isOpen = false;
        state.failures = 0;
      }
    }

    try {
      const result = await operation();
      
      // Reset on success
      state.failures = 0;
      state.isOpen = false;
      this.circuitStates.set(key, state);
      
      return result;
    } catch (error) {
      state.failures++;
      state.lastFailure = Date.now();
      
      if (state.failures >= failureThreshold) {
        state.isOpen = true;
      }
      
      this.circuitStates.set(key, state);
      throw error;
    }
  }

  private circuitStates?: Map<string, {
    failures: number;
    lastFailure: number;
    isOpen: boolean;
  }>;

  // Utility methods
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Configuration methods
  setPollingInterval(interval: number): void {
    this.jobPollingInterval = interval;
  }

  setMaxPollingAttempts(attempts: number): void {
    this.maxPollingAttempts = attempts;
  }

  // Async/await friendly job cancellation
  async cancelJob(jobId: string): Promise<void> {
    return this.client.cancelJob(jobId);
  }

  // Async/await friendly batch job management
  async cancelAllJobs(jobIds: string[]): Promise<void> {
    const promises = jobIds.map(id => this.client.cancelJob(id));
    await Promise.allSettled(promises);
  }

  // Promise-based health check with timeout
  async healthCheck(timeout: number = 5000): Promise<boolean> {
    try {
      await this.withTimeout(
        () => this.client.healthCheck(),
        timeout,
        'Health check timed out'
      );
      return true;
    } catch (error) {
      return false;
    }
  }
}