/**
 * Anomaly Detection API Client
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import {
  ClientConfig,
  DetectionResult,
  ModelInfo,
  ExplanationResult,
  HealthStatus,
  AlgorithmType,
  BatchProcessingRequest,
  TrainingRequest,
  TrainingResult,
  APIError,
  ValidationError,
  ConnectionError,
  TimeoutError,
} from './types';

export class AnomalyDetectionClient {
  private readonly axios: AxiosInstance;
  private readonly config: Required<Omit<ClientConfig, 'apiKey' | 'headers'>> & 
    Pick<ClientConfig, 'apiKey' | 'headers'>;

  constructor(config: ClientConfig) {
    this.config = {
      baseUrl: config.baseUrl.replace(/\/$/, ''),
      timeout: config.timeout ?? 30000,
      maxRetries: config.maxRetries ?? 3,
      apiKey: config.apiKey,
      headers: config.headers,
    };

    // Setup headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...config.headers,
    };

    if (this.config.apiKey) {
      headers.Authorization = `Bearer ${this.config.apiKey}`;
    }

    // Initialize axios instance
    this.axios = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers,
    });

    // Setup response interceptor for error handling
    this.axios.interceptors.response.use(
      (response) => response,
      (error) => this.handleAxiosError(error)
    );
  }

  private handleAxiosError(error: AxiosError): never {
    if (error.code === 'ECONNABORTED') {
      throw new TimeoutError(`Request timed out after ${this.config.timeout}ms`);
    }

    if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
      throw new ConnectionError(`Failed to connect to ${this.config.baseUrl}`);
    }

    if (error.response) {
      const { status, data } = error.response;
      const errorData = typeof data === 'object' ? data : { detail: data };
      const message = errorData.detail || `HTTP ${status}`;

      throw new APIError(message, status, errorData);
    }

    throw new ConnectionError(`Network error: ${error.message}`);
  }

  private async makeRequest<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    data?: any,
    params?: Record<string, any>
  ): Promise<T> {
    let retries = 0;
    
    while (retries <= this.config.maxRetries) {
      try {
        const response: AxiosResponse<T> = await this.axios.request({
          method,
          url: endpoint,
          data,
          params,
        });
        
        return response.data;
      } catch (error) {
        if (retries === this.config.maxRetries) {
          throw error;
        }
        
        // Only retry on certain errors
        if (error instanceof ConnectionError || error instanceof TimeoutError) {
          retries++;
          await this.delay(Math.pow(2, retries) * 1000); // Exponential backoff
        } else {
          throw error;
        }
      }
    }
    
    throw new Error('Max retries exceeded');
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Detect anomalies in the provided data
   */
  async detectAnomalies(
    data: number[][],
    algorithm: AlgorithmType = AlgorithmType.ISOLATION_FOREST,
    parameters?: Record<string, any>,
    returnExplanations = false
  ): Promise<DetectionResult> {
    if (!data || data.length === 0) {
      throw new ValidationError('Data cannot be empty', 'data', data);
    }

    if (!data.every(point => Array.isArray(point))) {
      throw new ValidationError('All data points must be arrays', 'data', data);
    }

    const requestData = {
      data,
      algorithm,
      parameters: parameters || {},
      return_explanations: returnExplanations,
    };

    return this.makeRequest<DetectionResult>('POST', '/api/v1/detect', requestData);
  }

  /**
   * Process a batch detection request
   */
  async batchDetect(request: BatchProcessingRequest): Promise<DetectionResult> {
    return this.makeRequest<DetectionResult>('POST', '/api/v1/batch/detect', request);
  }

  /**
   * Train a new anomaly detection model
   */
  async trainModel(request: TrainingRequest): Promise<TrainingResult> {
    return this.makeRequest<TrainingResult>('POST', '/api/v1/models/train', request);
  }

  /**
   * Get information about a specific model
   */
  async getModel(modelId: string): Promise<ModelInfo> {
    if (!modelId) {
      throw new ValidationError('Model ID is required', 'modelId', modelId);
    }

    return this.makeRequest<ModelInfo>('GET', `/api/v1/models/${modelId}`);
  }

  /**
   * List all available models
   */
  async listModels(): Promise<ModelInfo[]> {
    const response = await this.makeRequest<{ models: ModelInfo[] }>('GET', '/api/v1/models');
    return response.models || [];
  }

  /**
   * Delete a model
   */
  async deleteModel(modelId: string): Promise<{ message: string }> {
    if (!modelId) {
      throw new ValidationError('Model ID is required', 'modelId', modelId);
    }

    return this.makeRequest<{ message: string }>('DELETE', `/api/v1/models/${modelId}`);
  }

  /**
   * Get explanation for why a data point is anomalous
   */
  async explainAnomaly(
    dataPoint: number[],
    options: {
      modelId?: string;
      algorithm?: AlgorithmType;
      method?: string;
    } = {}
  ): Promise<ExplanationResult> {
    if (!Array.isArray(dataPoint)) {
      throw new ValidationError('Data point must be an array', 'dataPoint', dataPoint);
    }

    const requestData = {
      data_point: dataPoint,
      method: options.method || 'shap',
      ...(options.modelId ? { model_id: options.modelId } : {}),
      ...(options.algorithm ? { algorithm: options.algorithm } : {}),
    };

    return this.makeRequest<ExplanationResult>('POST', '/api/v1/explain', requestData);
  }

  /**
   * Get service health status
   */
  async getHealth(): Promise<HealthStatus> {
    return this.makeRequest<HealthStatus>('GET', '/api/v1/health');
  }

  /**
   * Get service metrics
   */
  async getMetrics(): Promise<Record<string, any>> {
    return this.makeRequest<Record<string, any>>('GET', '/api/v1/metrics');
  }

  /**
   * Upload training data to the service
   */
  async uploadData(
    data: number[][],
    datasetName: string,
    description?: string
  ): Promise<{ datasetId: string; message: string }> {
    if (!data || data.length === 0) {
      throw new ValidationError('Data cannot be empty', 'data', data);
    }

    if (!datasetName) {
      throw new ValidationError('Dataset name is required', 'datasetName', datasetName);
    }

    const requestData = {
      data,
      name: datasetName,
      description,
    };

    return this.makeRequest<{ datasetId: string; message: string }>(
      'POST', 
      '/api/v1/data/upload', 
      requestData
    );
  }
}