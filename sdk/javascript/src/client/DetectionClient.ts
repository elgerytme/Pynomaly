/**
 * Detection client for managing anomaly detection operations
 */

import { AxiosInstance } from 'axios';
import { 
  Detector, 
  CreateDetectorRequest, 
  TrainDetectorRequest, 
  DetectionRequest, 
  DetectionResult, 
  Dataset, 
  ListOptions, 
  PaginatedResponse 
} from '../types';
import { PynomalyError } from '../errors';

export class DetectionClient {
  constructor(private httpClient: AxiosInstance) {}

  /**
   * List all detectors
   */
  async listDetectors(options: ListOptions = {}): Promise<PaginatedResponse<Detector>> {
    try {
      const response = await this.httpClient.get('/detectors', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to list detectors', 'DETECTION_ERROR');
    }
  }

  /**
   * Get detector by ID
   */
  async getDetector(detectorId: string): Promise<Detector> {
    try {
      const response = await this.httpClient.get(`/detectors/${detectorId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Create new detector
   */
  async createDetector(request: CreateDetectorRequest): Promise<Detector> {
    try {
      const response = await this.httpClient.post('/detectors', request);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to create detector', 'DETECTION_ERROR');
    }
  }

  /**
   * Update detector
   */
  async updateDetector(detectorId: string, updates: Partial<CreateDetectorRequest>): Promise<Detector> {
    try {
      const response = await this.httpClient.patch(`/detectors/${detectorId}`, updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to update detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Delete detector
   */
  async deleteDetector(detectorId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/detectors/${detectorId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to delete detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Train detector with dataset
   */
  async trainDetector(detectorId: string, request: TrainDetectorRequest): Promise<{ job_id: string; status: string }> {
    try {
      const response = await this.httpClient.post(`/detectors/${detectorId}/train`, request);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to train detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Get training status
   */
  async getTrainingStatus(detectorId: string, jobId: string): Promise<{ status: string; progress: number; metrics?: Record<string, any> }> {
    try {
      const response = await this.httpClient.get(`/detectors/${detectorId}/train/${jobId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get training status for detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Stop training
   */
  async stopTraining(detectorId: string, jobId: string): Promise<void> {
    try {
      await this.httpClient.post(`/detectors/${detectorId}/train/${jobId}/stop`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to stop training for detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Detect anomalies in dataset
   */
  async detectAnomalies(request: DetectionRequest): Promise<DetectionResult> {
    try {
      const response = await this.httpClient.post('/detect', request);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to detect anomalies', 'DETECTION_ERROR');
    }
  }

  /**
   * Detect anomalies in real-time data
   */
  async detectAnomaliesRealtime(detectorId: string, data: Record<string, any>[]): Promise<DetectionResult> {
    try {
      const response = await this.httpClient.post(`/detectors/${detectorId}/detect`, { data });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to detect anomalies in real-time', 'DETECTION_ERROR');
    }
  }

  /**
   * Get detection history
   */
  async getDetectionHistory(detectorId: string, options: ListOptions = {}): Promise<PaginatedResponse<DetectionResult>> {
    try {
      const response = await this.httpClient.get(`/detectors/${detectorId}/history`, { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get detection history for detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * List datasets
   */
  async listDatasets(options: ListOptions = {}): Promise<PaginatedResponse<Dataset>> {
    try {
      const response = await this.httpClient.get('/datasets', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to list datasets', 'DETECTION_ERROR');
    }
  }

  /**
   * Get dataset by ID
   */
  async getDataset(datasetId: string): Promise<Dataset> {
    try {
      const response = await this.httpClient.get(`/datasets/${datasetId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get dataset ${datasetId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Create dataset from uploaded file
   */
  async createDataset(
    fileId: string, 
    name: string, 
    description?: string, 
    options?: Record<string, any>
  ): Promise<Dataset> {
    try {
      const response = await this.httpClient.post('/datasets', {
        file_id: fileId,
        name,
        description,
        options
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to create dataset', 'DETECTION_ERROR');
    }
  }

  /**
   * Update dataset
   */
  async updateDataset(datasetId: string, updates: { name?: string; description?: string; metadata?: Record<string, any> }): Promise<Dataset> {
    try {
      const response = await this.httpClient.patch(`/datasets/${datasetId}`, updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to update dataset ${datasetId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Delete dataset
   */
  async deleteDataset(datasetId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/datasets/${datasetId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to delete dataset ${datasetId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Get dataset preview
   */
  async getDatasetPreview(datasetId: string, limit: number = 100): Promise<{ columns: string[]; data: any[][] }> {
    try {
      const response = await this.httpClient.get(`/datasets/${datasetId}/preview`, { params: { limit } });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get dataset preview for ${datasetId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Get dataset statistics
   */
  async getDatasetStats(datasetId: string): Promise<Record<string, any>> {
    try {
      const response = await this.httpClient.get(`/datasets/${datasetId}/stats`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get dataset statistics for ${datasetId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Export detection results
   */
  async exportResults(detectorId: string, format: 'csv' | 'json' | 'xlsx' = 'csv'): Promise<ArrayBuffer> {
    try {
      const response = await this.httpClient.get(`/detectors/${detectorId}/export`, {
        params: { format },
        responseType: 'arraybuffer'
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to export results for detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Get detector performance metrics
   */
  async getDetectorMetrics(detectorId: string): Promise<Record<string, any>> {
    try {
      const response = await this.httpClient.get(`/detectors/${detectorId}/metrics`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get metrics for detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Get feature importance for detector
   */
  async getFeatureImportance(detectorId: string): Promise<{ feature: string; importance: number }[]> {
    try {
      const response = await this.httpClient.get(`/detectors/${detectorId}/feature-importance`);
      return response.data.features || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get feature importance for detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }

  /**
   * Get anomaly explanations
   */
  async getAnomalyExplanations(detectorId: string, anomalyIds: string[]): Promise<Record<string, any>[]> {
    try {
      const response = await this.httpClient.post(`/detectors/${detectorId}/explain`, { anomaly_ids: anomalyIds });
      return response.data.explanations || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get anomaly explanations for detector ${detectorId}`, 'DETECTION_ERROR');
    }
  }
}