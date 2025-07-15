/**
 * Pynomaly JavaScript SDK Data Science API
 * 
 * High-level API for data science operations including detector management,
 * anomaly detection, and experiment tracking for web applications.
 */

import {
  DetectorConfig,
  Dataset,
  DetectionResult,
  ExperimentConfig,
  ModelMetrics,
  TrainingJob,
  PaginatedResponse,
  ApiResponse
} from '../types';
import { ValidationError, APIError, ResourceNotFoundError } from './errors';
import type { PynomalyClient } from './client';

/**
 * High-level API for data science operations in web environments.
 * 
 * Provides convenient methods for common anomaly detection workflows
 * including detector management, data processing, and experiment tracking.
 */
export class DataScienceAPI {
  constructor(private client: PynomalyClient) {}

  // Detector Management

  /**
   * Create a new anomaly detector.
   * 
   * @param name Detector name
   * @param config Detector configuration
   * @param description Optional description
   * @param tags Optional tags for organization
   * @returns Promise resolving to created detector information
   */
  async createDetector(
    name: string,
    config: DetectorConfig,
    description?: string,
    tags?: string[]
  ): Promise<any> {
    const payload = {
      name,
      algorithm_name: config.algorithmName,
      hyperparameters: config.hyperparameters || {},
      contamination_rate: config.contaminationRate || 0.1,
      random_state: config.randomState,
      n_jobs: config.nJobs || 1,
      ...(description && { description }),
      ...(tags && { tags })
    };

    const response = await this.client.request({
      method: 'POST',
      url: '/detectors',
      data: payload
    });

    return response.data;
  }

  /**
   * List available detectors with pagination and filtering.
   * 
   * @param options List options
   * @returns Promise resolving to paginated list of detectors
   */
  async listDetectors(options: {
    page?: number;
    pageSize?: number;
    algorithmName?: string;
    tags?: string[];
  } = {}): Promise<PaginatedResponse> {
    const params: Record<string, any> = {
      page: options.page || 1,
      page_size: options.pageSize || 20
    };

    if (options.algorithmName) {
      params.algorithm_name = options.algorithmName;
    }
    if (options.tags) {
      params.tags = options.tags.join(',');
    }

    const response = await this.client.request({
      method: 'GET',
      url: '/detectors',
      params
    });

    return response.data;
  }

  /**
   * Get detector details by ID.
   * 
   * @param detectorId Detector ID
   * @returns Promise resolving to detector information
   */
  async getDetector(detectorId: string): Promise<any> {
    try {
      const response = await this.client.request({
        method: 'GET',
        url: `/detectors/${detectorId}`
      });
      return response.data;
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new ResourceNotFoundError('Detector', detectorId);
      }
      throw error;
    }
  }

  /**
   * Update detector configuration.
   * 
   * @param detectorId Detector ID
   * @param updates Update options
   * @returns Promise resolving to updated detector information
   */
  async updateDetector(
    detectorId: string,
    updates: {
      name?: string;
      config?: DetectorConfig;
      description?: string;
      tags?: string[];
    }
  ): Promise<any> {
    const payload: Record<string, any> = {};

    if (updates.name) payload.name = updates.name;
    if (updates.config) {
      payload.algorithm_name = updates.config.algorithmName;
      payload.hyperparameters = updates.config.hyperparameters;
      payload.contamination_rate = updates.config.contaminationRate;
      payload.random_state = updates.config.randomState;
      payload.n_jobs = updates.config.nJobs;
    }
    if (updates.description !== undefined) payload.description = updates.description;
    if (updates.tags !== undefined) payload.tags = updates.tags;

    const response = await this.client.request({
      method: 'PUT',
      url: `/detectors/${detectorId}`,
      data: payload
    });

    return response.data;
  }

  /**
   * Delete a detector.
   * 
   * @param detectorId Detector ID
   * @returns Promise resolving to true if deleted successfully
   */
  async deleteDetector(detectorId: string): Promise<boolean> {
    try {
      await this.client.request({
        method: 'DELETE',
        url: `/detectors/${detectorId}`
      });
      return true;
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new ResourceNotFoundError('Detector', detectorId);
      }
      throw error;
    }
  }

  // Training Operations

  /**
   * Train a detector with provided data.
   * 
   * @param detectorId Detector ID to train
   * @param dataset Training dataset
   * @param jobName Optional job name
   * @returns Promise resolving to training job information
   */
  async trainDetector(
    detectorId: string,
    dataset: Dataset,
    jobName?: string
  ): Promise<TrainingJob> {
    const payload = {
      detector_id: detectorId,
      dataset: this.formatDataset(dataset),
      ...(jobName && { job_name: jobName })
    };

    const response = await this.client.request({
      method: 'POST',
      url: '/training/jobs',
      data: payload
    });

    return this.formatTrainingJob(response.data);
  }

  /**
   * Get training job status and results.
   * 
   * @param jobId Training job ID
   * @returns Promise resolving to training job information
   */
  async getTrainingJob(jobId: string): Promise<TrainingJob> {
    try {
      const response = await this.client.request({
        method: 'GET',
        url: `/training/jobs/${jobId}`
      });
      return this.formatTrainingJob(response.data);
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new ResourceNotFoundError('Training Job', jobId);
      }
      throw error;
    }
  }

  /**
   * List training jobs with optional filtering.
   * 
   * @param options Filter and pagination options
   * @returns Promise resolving to paginated list of training jobs
   */
  async listTrainingJobs(options: {
    detectorId?: string;
    status?: string;
    page?: number;
    pageSize?: number;
  } = {}): Promise<PaginatedResponse<TrainingJob>> {
    const params: Record<string, any> = {
      page: options.page || 1,
      page_size: options.pageSize || 20
    };

    if (options.detectorId) params.detector_id = options.detectorId;
    if (options.status) params.status = options.status;

    const response = await this.client.request({
      method: 'GET',
      url: '/training/jobs',
      params
    });

    return {
      ...response.data,
      items: response.data.items.map((item: any) => this.formatTrainingJob(item))
    };
  }

  // Detection Operations

  /**
   * Detect anomalies in data using a trained detector.
   * 
   * @param detectorId ID of trained detector
   * @param dataset Data to analyze
   * @param options Detection options
   * @returns Promise resolving to detection results
   */
  async detectAnomalies(
    detectorId: string,
    dataset: Dataset,
    options: {
      returnScores?: boolean;
      threshold?: number;
    } = {}
  ): Promise<DetectionResult> {
    const payload = {
      detector_id: detectorId,
      dataset: this.formatDataset(dataset),
      return_scores: options.returnScores !== false,
      ...(options.threshold !== undefined && { threshold: options.threshold })
    };

    try {
      const response = await this.client.request({
        method: 'POST',
        url: '/detection/predict',
        data: payload
      });
      return this.formatDetectionResult(response.data);
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new ResourceNotFoundError('Detector', detectorId);
      }
      throw error;
    }
  }

  /**
   * Submit batch detection job for multiple datasets.
   * 
   * @param detectorId ID of trained detector
   * @param datasets List of datasets to process
   * @param jobName Optional job name
   * @returns Promise resolving to batch job information
   */
  async batchDetect(
    detectorId: string,
    datasets: Dataset[],
    jobName?: string
  ): Promise<any> {
    const payload = {
      detector_id: detectorId,
      datasets: datasets.map(dataset => this.formatDataset(dataset)),
      ...(jobName && { job_name: jobName })
    };

    const response = await this.client.request({
      method: 'POST',
      url: '/detection/batch',
      data: payload
    });

    return response.data;
  }

  // Experiment Management

  /**
   * Create and run a new experiment.
   * 
   * @param config Experiment configuration
   * @param dataset Dataset for the experiment
   * @returns Promise resolving to experiment information
   */
  async createExperiment(
    config: ExperimentConfig,
    dataset: Dataset
  ): Promise<any> {
    const payload = {
      name: config.name,
      description: config.description,
      algorithm_configs: config.algorithmConfigs.map(conf => ({
        algorithm_name: conf.algorithmName,
        hyperparameters: conf.hyperparameters || {},
        contamination_rate: conf.contaminationRate || 0.1,
        random_state: conf.randomState,
        n_jobs: conf.nJobs || 1
      })),
      evaluation_metrics: config.evaluationMetrics || ['auc_roc', 'precision', 'recall'],
      cross_validation_folds: config.crossValidationFolds || 5,
      random_state: config.randomState,
      parallel_jobs: config.parallelJobs || 1,
      optimization_enabled: config.optimizationEnabled || false,
      optimization_trials: config.optimizationTrials || 100,
      optimization_timeout: config.optimizationTimeout,
      dataset: this.formatDataset(dataset)
    };

    const response = await this.client.request({
      method: 'POST',
      url: '/experiments',
      data: payload
    });

    return response.data;
  }

  /**
   * Get experiment details and results.
   * 
   * @param experimentId Experiment ID
   * @returns Promise resolving to experiment information
   */
  async getExperiment(experimentId: string): Promise<any> {
    try {
      const response = await this.client.request({
        method: 'GET',
        url: `/experiments/${experimentId}`
      });
      return response.data;
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new ResourceNotFoundError('Experiment', experimentId);
      }
      throw error;
    }
  }

  /**
   * List experiments with optional filtering.
   * 
   * @param options Filter and pagination options
   * @returns Promise resolving to paginated list of experiments
   */
  async listExperiments(options: {
    page?: number;
    pageSize?: number;
    status?: string;
  } = {}): Promise<PaginatedResponse> {
    const params: Record<string, any> = {
      page: options.page || 1,
      page_size: options.pageSize || 20
    };

    if (options.status) params.status = options.status;

    const response = await this.client.request({
      method: 'GET',
      url: '/experiments',
      params
    });

    return response.data;
  }

  // Utility Methods

  /**
   * Validate dataset format and compatibility.
   * 
   * @param dataset Dataset to validate
   * @param algorithmName Optional algorithm to check compatibility
   * @returns Promise resolving to validation results
   */
  async validateDataset(
    dataset: Dataset,
    algorithmName?: string
  ): Promise<any> {
    const payload = {
      dataset: this.formatDataset(dataset),
      ...(algorithmName && { algorithm_name: algorithmName })
    };

    const response = await this.client.request({
      method: 'POST',
      url: '/validation/dataset',
      data: payload
    });

    return response.data;
  }

  /**
   * Get list of supported algorithms and their configurations.
   * 
   * @returns Promise resolving to list of algorithm information
   */
  async getSupportedAlgorithms(): Promise<any[]> {
    const response = await this.client.request({
      method: 'GET',
      url: '/algorithms'
    });

    return response.data.algorithms || [];
  }

  /**
   * Get detailed information about a specific algorithm.
   * 
   * @param algorithmName Name of the algorithm
   * @returns Promise resolving to algorithm details and parameter information
   */
  async getAlgorithmInfo(algorithmName: string): Promise<any> {
    try {
      const response = await this.client.request({
        method: 'GET',
        url: `/algorithms/${algorithmName}`
      });
      return response.data;
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new ResourceNotFoundError('Algorithm', algorithmName);
      }
      throw error;
    }
  }

  // Private helper methods

  /**
   * Format dataset for API consumption.
   */
  private formatDataset(dataset: Dataset): any {
    return {
      name: dataset.name,
      data: dataset.data,
      metadata: dataset.metadata || {},
      feature_names: dataset.featureNames,
      target_column: dataset.targetColumn
    };
  }

  /**
   * Format API training job response to TrainingJob.
   */
  private formatTrainingJob(data: any): TrainingJob {
    return {
      jobId: data.job_id,
      name: data.name,
      status: data.status,
      detectorConfig: {
        algorithmName: data.detector_config.algorithm_name,
        hyperparameters: data.detector_config.hyperparameters,
        contaminationRate: data.detector_config.contamination_rate,
        randomState: data.detector_config.random_state,
        nJobs: data.detector_config.n_jobs
      },
      datasetName: data.dataset_name,
      createdAt: data.created_at,
      startedAt: data.started_at,
      completedAt: data.completed_at,
      metrics: data.metrics ? this.formatModelMetrics(data.metrics) : undefined,
      modelPath: data.model_path,
      logs: data.logs || [],
      errorMessage: data.error_message,
      executionTime: data.execution_time,
      memoryUsage: data.memory_usage,
      cpuUsage: data.cpu_usage
    };
  }

  /**
   * Format API metrics response to ModelMetrics.
   */
  private formatModelMetrics(data: any): ModelMetrics {
    return {
      accuracy: data.accuracy,
      precision: data.precision,
      recall: data.recall,
      f1Score: data.f1_score,
      aucRoc: data.auc_roc,
      aucPr: data.auc_pr,
      contaminationRate: data.contamination_rate,
      anomalyThreshold: data.anomaly_threshold,
      customMetrics: data.custom_metrics || {}
    };
  }

  /**
   * Format API detection response to DetectionResult.
   */
  private formatDetectionResult(data: any): DetectionResult {
    return {
      anomalyScores: data.anomaly_scores,
      anomalyLabels: data.anomaly_labels,
      nAnomalies: data.n_anomalies,
      nSamples: data.n_samples,
      contaminationRate: data.contamination_rate,
      threshold: data.threshold,
      executionTime: data.execution_time,
      metadata: data.metadata || {}
    };
  }
}