/**
 * Anomaly Detection service client
 */

import { HttpClient } from '../http';
import { ClientConfig } from '../config';
import {
  DetectionRequest,
  DetectionResponse,
  EnsembleDetectionRequest,
  EnsembleDetectionResponse,
  TrainingRequest,
  TrainingResponse,
  PredictionRequest,
  PredictionResponse,
  ModelInfo,
  AlgorithmInfo,
  HealthResponse,
  MetricsResponse,
  PaginatedResponse
} from '../types';

export class AnomalyDetectionClient {
  private basePath = 'api/v1';
  
  constructor(
    private httpClient: HttpClient,
    private config: ClientConfig
  ) {}
  
  /**
   * Detect anomalies in data using specified algorithm
   */
  async detect(request: DetectionRequest): Promise<DetectionResponse> {
    const response = await this.httpClient.post<DetectionResponse>(
      `${this.basePath}/detect`,
      request
    );
    return response.data;
  }
  
  /**
   * Detect anomalies using ensemble of algorithms
   */
  async detectEnsemble(request: EnsembleDetectionRequest): Promise<EnsembleDetectionResponse> {
    const response = await this.httpClient.post<EnsembleDetectionResponse>(
      `${this.basePath}/ensemble`,
      request
    );
    return response.data;
  }
  
  /**
   * Train a new anomaly detection model
   */
  async trainModel(request: TrainingRequest): Promise<TrainingResponse> {
    const response = await this.httpClient.post<TrainingResponse>(
      `${this.basePath}/models/train`,
      request
    );
    return response.data;
  }
  
  /**
   * Make predictions using a trained model
   */
  async predict(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await this.httpClient.post<PredictionResponse>(
      `${this.basePath}/predict`,
      request
    );
    return response.data;
  }
  
  /**
   * Get information about a specific model
   */
  async getModel(modelId: string): Promise<ModelInfo> {
    const response = await this.httpClient.get<{ data: ModelInfo }>(
      `${this.basePath}/models/${modelId}`
    );
    return response.data.data;
  }
  
  /**
   * List available models
   */
  async listModels(options: {
    algorithm?: string;
    status?: string;
    page?: number;
    pageSize?: number;
  } = {}): Promise<PaginatedResponse<ModelInfo>> {
    const params = new URLSearchParams();
    
    if (options.algorithm) params.set('algorithm', options.algorithm);
    if (options.status) params.set('status', options.status);
    if (options.page) params.set('page', options.page.toString());
    if (options.pageSize) params.set('page_size', options.pageSize.toString());
    
    const response = await this.httpClient.get<PaginatedResponse<ModelInfo>>(
      `${this.basePath}/models?${params.toString()}`
    );
    return response.data;
  }
  
  /**
   * Delete a model
   */
  async deleteModel(modelId: string): Promise<void> {
    await this.httpClient.delete(`${this.basePath}/models/${modelId}`);
  }
  
  /**
   * Get list of available algorithms
   */
  async getAlgorithms(): Promise<AlgorithmInfo[]> {
    const response = await this.httpClient.get<{ data: AlgorithmInfo[] }>(
      `${this.basePath}/algorithms`
    );
    return response.data.data;
  }
  
  /**
   * Process multiple datasets in batch
   */
  async batchDetect(request: {
    datasets: Array<{ id: string; data: number[][] }>;
    algorithm?: string;
    contamination?: number;
    parameters?: Record<string, any>;
    parallelProcessing?: boolean;
  }): Promise<{
    results: Array<{ id: string; result: DetectionResponse; error?: string }>;
    totalDatasets: number;
    successfulCount: number;
    failedCount: number;
    totalProcessingTimeMs: number;
  }> {
    const response = await this.httpClient.post<any>(
      `${this.basePath}/batch/detect`,
      request
    );
    return response.data;
  }
  
  /**
   * Check service health
   */
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.httpClient.get<HealthResponse>('/health');
    return response.data;
  }
  
  /**
   * Get service metrics
   */
  async getMetrics(): Promise<MetricsResponse> {
    const response = await this.httpClient.get<MetricsResponse>(
      `${this.basePath}/metrics`
    );
    return response.data;
  }
  
  /**
   * Stream real-time anomaly detection (if supported)
   */
  async *streamDetection(
    algorithm: string = 'isolation_forest',
    contamination: number = 0.1
  ): AsyncGenerator<DetectionResponse, void, number[][]> {
    // This would implement streaming detection
    // For now, we'll throw as it requires WebSocket or SSE implementation
    throw new Error('Streaming detection not yet implemented');
  }
  
  /**
   * Get model performance metrics over time
   */
  async getModelPerformance(
    modelId: string,
    timeRange: { start: string; end: string }
  ): Promise<{
    metrics: Array<{
      timestamp: string;
      accuracy: number;
      precision: number;
      recall: number;
      f1Score: number;
    }>;
  }> {
    const params = new URLSearchParams({
      start: timeRange.start,
      end: timeRange.end
    });
    
    const response = await this.httpClient.get<any>(
      `${this.basePath}/models/${modelId}/performance?${params.toString()}`
    );
    return response.data;
  }
  
  /**
   * Get feature importance for a model (if supported by algorithm)
   */
  async getFeatureImportance(modelId: string): Promise<{
    features: Array<{
      index: number;
      name?: string;
      importance: number;
    }>;
  }> {
    const response = await this.httpClient.get<any>(
      `${this.basePath}/models/${modelId}/feature-importance`
    );
    return response.data;
  }
}