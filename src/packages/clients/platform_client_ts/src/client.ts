/**
 * Main platform client that provides access to all services
 */

import { AnomalyDetectionClient } from './services/anomaly-detection';
import { MLOpsClient } from './services/mlops';
import { HttpClient } from './http';
import { ClientConfig } from './config';
import { ClientOptions } from './types';

export class PlatformClient {
  private config: ClientConfig;
  private httpClient: HttpClient;
  
  public readonly anomalyDetection: AnomalyDetectionClient;
  public readonly mlops: MLOpsClient;
  
  constructor(options: ClientOptions) {
    this.config = new ClientConfig(options);
    this.httpClient = new HttpClient(this.config);
    
    // Initialize service clients
    this.anomalyDetection = new AnomalyDetectionClient(this.httpClient, this.config);
    this.mlops = new MLOpsClient(this.httpClient, this.config);
  }
  
  /**
   * Perform a health check across all services
   */
  async healthCheck(): Promise<Record<string, any>> {
    try {
      const [anomalyHealth, mlopsHealth] = await Promise.allSettled([
        this.anomalyDetection.healthCheck(),
        this.mlops.healthCheck()
      ]);
      
      return {
        anomalyDetection: anomalyHealth.status === 'fulfilled' ? anomalyHealth.value : { error: 'Failed' },
        mlops: mlopsHealth.status === 'fulfilled' ? mlopsHealth.value : { error: 'Failed' }
      };
    } catch (error) {
      return { error: 'Health check failed' };
    }
  }
  
  /**
   * Get client configuration
   */
  getConfig(): ClientConfig {
    return this.config;
  }
  
  /**
   * Update client configuration
   */
  updateConfig(options: Partial<ClientOptions>): void {
    this.config.update(options);
  }
  
  /**
   * Close all connections and cleanup resources
   */
  async close(): Promise<void> {
    // Cleanup any persistent connections or resources
    await this.httpClient.close();
  }
}