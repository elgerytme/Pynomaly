/**
 * Streaming client for real-time anomaly detection
 */

import { AxiosInstance } from 'axios';
import { 
  StreamProcessor, 
  CreateProcessorRequest, 
  StreamRecord, 
  StreamMetrics, 
  WindowConfig, 
  BackpressureConfig,
  ListOptions, 
  PaginatedResponse 
} from '../types';
import { PynomalyError, StreamingError } from '../errors';

export class StreamingClient {
  constructor(private httpClient: AxiosInstance) {}

  /**
   * List all stream processors
   */
  async listProcessors(options: ListOptions = {}): Promise<PaginatedResponse<StreamProcessor>> {
    try {
      const response = await this.httpClient.get('/streaming/processors', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError('Failed to list stream processors');
    }
  }

  /**
   * Get processor by ID
   */
  async getProcessor(processorId: string): Promise<StreamProcessor> {
    try {
      const response = await this.httpClient.get(`/streaming/processors/${processorId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to get processor ${processorId}`);
    }
  }

  /**
   * Create new stream processor
   */
  async createProcessor(request: CreateProcessorRequest): Promise<StreamProcessor> {
    try {
      const response = await this.httpClient.post('/streaming/processors', request);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError('Failed to create stream processor');
    }
  }

  /**
   * Update processor configuration
   */
  async updateProcessor(
    processorId: string, 
    updates: Partial<CreateProcessorRequest>
  ): Promise<StreamProcessor> {
    try {
      const response = await this.httpClient.patch(`/streaming/processors/${processorId}`, updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to update processor ${processorId}`);
    }
  }

  /**
   * Delete processor
   */
  async deleteProcessor(processorId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/streaming/processors/${processorId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to delete processor ${processorId}`);
    }
  }

  /**
   * Start stream processor
   */
  async startProcessor(processorId: string): Promise<{ status: string; started_at: string }> {
    try {
      const response = await this.httpClient.post(`/streaming/processors/${processorId}/start`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to start processor ${processorId}`);
    }
  }

  /**
   * Stop stream processor
   */
  async stopProcessor(processorId: string): Promise<{ status: string; stopped_at: string }> {
    try {
      const response = await this.httpClient.post(`/streaming/processors/${processorId}/stop`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to stop processor ${processorId}`);
    }
  }

  /**
   * Pause stream processor
   */
  async pauseProcessor(processorId: string): Promise<{ status: string; paused_at: string }> {
    try {
      const response = await this.httpClient.post(`/streaming/processors/${processorId}/pause`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to pause processor ${processorId}`);
    }
  }

  /**
   * Resume stream processor
   */
  async resumeProcessor(processorId: string): Promise<{ status: string; resumed_at: string }> {
    try {
      const response = await this.httpClient.post(`/streaming/processors/${processorId}/resume`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to resume processor ${processorId}`);
    }
  }

  /**
   * Get processor metrics
   */
  async getProcessorMetrics(processorId: string): Promise<StreamMetrics> {
    try {
      const response = await this.httpClient.get(`/streaming/processors/${processorId}/metrics`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to get metrics for processor ${processorId}`);
    }
  }

  /**
   * Get processor status
   */
  async getProcessorStatus(processorId: string): Promise<{ 
    status: string; 
    uptime: number; 
    last_heartbeat: string; 
    error?: string 
  }> {
    try {
      const response = await this.httpClient.get(`/streaming/processors/${processorId}/status`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to get status for processor ${processorId}`);
    }
  }

  /**
   * Send data to stream processor
   */
  async sendData(processorId: string, records: StreamRecord[]): Promise<{ 
    accepted: number; 
    rejected: number; 
    errors?: string[] 
  }> {
    try {
      const response = await this.httpClient.post(`/streaming/processors/${processorId}/data`, {
        records
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to send data to processor ${processorId}`);
    }
  }

  /**
   * Send single record to stream processor
   */
  async sendRecord(processorId: string, record: StreamRecord): Promise<{ 
    accepted: boolean; 
    anomaly_score?: number; 
    is_anomaly?: boolean 
  }> {
    try {
      const response = await this.httpClient.post(`/streaming/processors/${processorId}/record`, record);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to send record to processor ${processorId}`);
    }
  }

  /**
   * Get recent anomalies from processor
   */
  async getRecentAnomalies(
    processorId: string, 
    limit: number = 100, 
    since?: string
  ): Promise<{
    anomalies: Array<{
      id: string;
      score: number;
      timestamp: string;
      data: Record<string, any>;
      metadata: Record<string, any>;
    }>;
    total: number;
  }> {
    try {
      const response = await this.httpClient.get(`/streaming/processors/${processorId}/anomalies`, {
        params: { limit, since }
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to get recent anomalies for processor ${processorId}`);
    }
  }

  /**
   * Get processor logs
   */
  async getProcessorLogs(
    processorId: string, 
    limit: number = 100, 
    level: 'debug' | 'info' | 'warn' | 'error' = 'info'
  ): Promise<Array<{
    timestamp: string;
    level: string;
    message: string;
    context?: Record<string, any>;
  }>> {
    try {
      const response = await this.httpClient.get(`/streaming/processors/${processorId}/logs`, {
        params: { limit, level }
      });
      return response.data.logs || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to get logs for processor ${processorId}`);
    }
  }

  /**
   * Update window configuration
   */
  async updateWindowConfig(processorId: string, config: WindowConfig): Promise<StreamProcessor> {
    try {
      const response = await this.httpClient.patch(`/streaming/processors/${processorId}/window`, config);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to update window config for processor ${processorId}`);
    }
  }

  /**
   * Update backpressure configuration
   */
  async updateBackpressureConfig(processorId: string, config: BackpressureConfig): Promise<StreamProcessor> {
    try {
      const response = await this.httpClient.patch(`/streaming/processors/${processorId}/backpressure`, config);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to update backpressure config for processor ${processorId}`);
    }
  }

  /**
   * Reset processor metrics
   */
  async resetMetrics(processorId: string): Promise<void> {
    try {
      await this.httpClient.post(`/streaming/processors/${processorId}/metrics/reset`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to reset metrics for processor ${processorId}`);
    }
  }

  /**
   * Get processor health check
   */
  async getProcessorHealth(processorId: string): Promise<{
    status: 'healthy' | 'unhealthy' | 'degraded';
    checks: Array<{
      name: string;
      status: 'pass' | 'fail' | 'warn';
      message?: string;
      details?: Record<string, any>;
    }>;
  }> {
    try {
      const response = await this.httpClient.get(`/streaming/processors/${processorId}/health`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to get health for processor ${processorId}`);
    }
  }

  /**
   * Scale processor resources
   */
  async scaleProcessor(processorId: string, replicas: number): Promise<{ 
    current_replicas: number; 
    target_replicas: number; 
    scaling_status: string 
  }> {
    try {
      const response = await this.httpClient.post(`/streaming/processors/${processorId}/scale`, {
        replicas
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to scale processor ${processorId}`);
    }
  }

  /**
   * Get processor configuration
   */
  async getProcessorConfig(processorId: string): Promise<CreateProcessorRequest> {
    try {
      const response = await this.httpClient.get(`/streaming/processors/${processorId}/config`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to get config for processor ${processorId}`);
    }
  }

  /**
   * Export processor metrics
   */
  async exportMetrics(processorId: string, format: 'csv' | 'json' = 'csv'): Promise<ArrayBuffer> {
    try {
      const response = await this.httpClient.get(`/streaming/processors/${processorId}/metrics/export`, {
        params: { format },
        responseType: 'arraybuffer'
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to export metrics for processor ${processorId}`);
    }
  }

  /**
   * Test processor with sample data
   */
  async testProcessor(processorId: string, sampleData: Record<string, any>[]): Promise<{
    results: Array<{
      input: Record<string, any>;
      anomaly_score: number;
      is_anomaly: boolean;
      processing_time_ms: number;
    }>;
    summary: {
      total_processed: number;
      anomalies_detected: number;
      avg_processing_time_ms: number;
    };
  }> {
    try {
      const response = await this.httpClient.post(`/streaming/processors/${processorId}/test`, {
        sample_data: sampleData
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new StreamingError(`Failed to test processor ${processorId}`);
    }
  }
}