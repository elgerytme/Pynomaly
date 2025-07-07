/**
 * A/B Testing client for algorithm comparison and experimentation
 */

import { AxiosInstance } from 'axios';
import { 
  ABTest, 
  CreateABTestRequest, 
  TestResult, 
  TestMetric, 
  StatisticalAnalysis, 
  TestVariant,
  ListOptions, 
  PaginatedResponse 
} from '../types';
import { PynomalyError, TestingError } from '../errors';

export class ABTestingClient {
  constructor(private httpClient: AxiosInstance) {}

  /**
   * List all A/B tests
   */
  async listTests(options: ListOptions = {}): Promise<PaginatedResponse<ABTest>> {
    try {
      const response = await this.httpClient.get('/ab-testing/tests', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError('Failed to list A/B tests');
    }
  }

  /**
   * Get A/B test by ID
   */
  async getTest(testId: string): Promise<ABTest> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to get A/B test ${testId}`);
    }
  }

  /**
   * Create new A/B test
   */
  async createTest(request: CreateABTestRequest): Promise<ABTest> {
    try {
      const response = await this.httpClient.post('/ab-testing/tests', request);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError('Failed to create A/B test');
    }
  }

  /**
   * Update A/B test
   */
  async updateTest(testId: string, updates: Partial<CreateABTestRequest>): Promise<ABTest> {
    try {
      const response = await this.httpClient.patch(`/ab-testing/tests/${testId}`, updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to update A/B test ${testId}`);
    }
  }

  /**
   * Delete A/B test
   */
  async deleteTest(testId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/ab-testing/tests/${testId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to delete A/B test ${testId}`);
    }
  }

  /**
   * Start A/B test
   */
  async startTest(testId: string): Promise<{ status: string; started_at: string }> {
    try {
      const response = await this.httpClient.post(`/ab-testing/tests/${testId}/start`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to start A/B test ${testId}`);
    }
  }

  /**
   * Stop A/B test
   */
  async stopTest(testId: string): Promise<{ status: string; stopped_at: string }> {
    try {
      const response = await this.httpClient.post(`/ab-testing/tests/${testId}/stop`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to stop A/B test ${testId}`);
    }
  }

  /**
   * Pause A/B test
   */
  async pauseTest(testId: string): Promise<{ status: string; paused_at: string }> {
    try {
      const response = await this.httpClient.post(`/ab-testing/tests/${testId}/pause`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to pause A/B test ${testId}`);
    }
  }

  /**
   * Resume A/B test
   */
  async resumeTest(testId: string): Promise<{ status: string; resumed_at: string }> {
    try {
      const response = await this.httpClient.post(`/ab-testing/tests/${testId}/resume`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to resume A/B test ${testId}`);
    }
  }

  /**
   * Get test variants
   */
  async getTestVariants(testId: string): Promise<TestVariant[]> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}/variants`);
      return response.data.variants || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to get variants for test ${testId}`);
    }
  }

  /**
   * Add variant to test
   */
  async addVariant(testId: string, variant: Omit<TestVariant, 'id'>): Promise<TestVariant> {
    try {
      const response = await this.httpClient.post(`/ab-testing/tests/${testId}/variants`, variant);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to add variant to test ${testId}`);
    }
  }

  /**
   * Update test variant
   */
  async updateVariant(testId: string, variantId: string, updates: Partial<TestVariant>): Promise<TestVariant> {
    try {
      const response = await this.httpClient.patch(`/ab-testing/tests/${testId}/variants/${variantId}`, updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to update variant ${variantId} in test ${testId}`);
    }
  }

  /**
   * Remove variant from test
   */
  async removeVariant(testId: string, variantId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/ab-testing/tests/${testId}/variants/${variantId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to remove variant ${variantId} from test ${testId}`);
    }
  }

  /**
   * Execute test on dataset
   */
  async executeTest(testId: string, datasetId: string, options?: { 
    sample_size?: number; 
    parallel?: boolean 
  }): Promise<{ job_id: string; status: string }> {
    try {
      const response = await this.httpClient.post(`/ab-testing/tests/${testId}/execute`, {
        dataset_id: datasetId,
        ...options
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to execute test ${testId}`);
    }
  }

  /**
   * Get test execution status
   */
  async getExecutionStatus(testId: string, jobId: string): Promise<{
    status: string;
    progress: number;
    current_variant?: string;
    completed_variants: number;
    total_variants: number;
    estimated_completion?: string;
  }> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}/execute/${jobId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to get execution status for test ${testId}`);
    }
  }

  /**
   * Get test results
   */
  async getTestResults(testId: string, options: ListOptions = {}): Promise<PaginatedResponse<TestResult>> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}/results`, { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to get results for test ${testId}`);
    }
  }

  /**
   * Get test metrics
   */
  async getTestMetrics(testId: string, variantId?: string): Promise<TestMetric[]> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}/metrics`, {
        params: variantId ? { variant_id: variantId } : {}
      });
      return response.data.metrics || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to get metrics for test ${testId}`);
    }
  }

  /**
   * Get statistical analysis
   */
  async getStatisticalAnalysis(testId: string): Promise<StatisticalAnalysis[]> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}/analysis`);
      return response.data.analyses || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to get statistical analysis for test ${testId}`);
    }
  }

  /**
   * Get test summary
   */
  async getTestSummary(testId: string): Promise<{
    test_id: string;
    status: string;
    total_executions: number;
    winner?: string;
    confidence_level: number;
    significance_achieved: boolean;
    variant_performance: Array<{
      variant_id: string;
      variant_name: string;
      executions: number;
      avg_accuracy: number;
      avg_processing_time: number;
      anomaly_detection_rate: number;
    }>;
    recommendations: string[];
  }> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}/summary`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to get summary for test ${testId}`);
    }
  }

  /**
   * Compare variants
   */
  async compareVariants(testId: string, variantIds: string[]): Promise<{
    comparison_id: string;
    variants: Array<{
      variant_id: string;
      variant_name: string;
      metrics: Record<string, number>;
      ranking: number;
    }>;
    statistical_significance: Record<string, StatisticalAnalysis>;
    recommendations: string[];
  }> {
    try {
      const response = await this.httpClient.post(`/ab-testing/tests/${testId}/compare`, {
        variant_ids: variantIds
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to compare variants in test ${testId}`);
    }
  }

  /**
   * Export test results
   */
  async exportResults(testId: string, format: 'csv' | 'json' | 'xlsx' = 'csv'): Promise<ArrayBuffer> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}/export`, {
        params: { format },
        responseType: 'arraybuffer'
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to export results for test ${testId}`);
    }
  }

  /**
   * Clone test
   */
  async cloneTest(testId: string, newName: string): Promise<ABTest> {
    try {
      const response = await this.httpClient.post(`/ab-testing/tests/${testId}/clone`, {
        name: newName
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to clone test ${testId}`);
    }
  }

  /**
   * Archive test
   */
  async archiveTest(testId: string): Promise<void> {
    try {
      await this.httpClient.post(`/ab-testing/tests/${testId}/archive`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to archive test ${testId}`);
    }
  }

  /**
   * Get test templates
   */
  async getTestTemplates(): Promise<Array<{
    id: string;
    name: string;
    description: string;
    variants: CreateABTestRequest['variants'];
    metrics: string[];
  }>> {
    try {
      const response = await this.httpClient.get('/ab-testing/templates');
      return response.data.templates || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError('Failed to get test templates');
    }
  }

  /**
   * Create test from template
   */
  async createTestFromTemplate(templateId: string, customization: Partial<CreateABTestRequest>): Promise<ABTest> {
    try {
      const response = await this.httpClient.post(`/ab-testing/templates/${templateId}/create`, customization);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to create test from template ${templateId}`);
    }
  }

  /**
   * Get test recommendations
   */
  async getTestRecommendations(testId: string): Promise<{
    should_continue: boolean;
    confidence_level: number;
    recommended_actions: string[];
    variant_recommendations: Array<{
      variant_id: string;
      action: 'continue' | 'stop' | 'increase_traffic' | 'decrease_traffic';
      reason: string;
    }>;
  }> {
    try {
      const response = await this.httpClient.get(`/ab-testing/tests/${testId}/recommendations`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new TestingError(`Failed to get recommendations for test ${testId}`);
    }
  }
}