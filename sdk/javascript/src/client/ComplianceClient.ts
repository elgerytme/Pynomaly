/**
 * Compliance client for audit logging and regulatory compliance
 */

import { AxiosInstance } from 'axios';
import { 
  AuditEvent, 
  GDPRRequest, 
  ComplianceFramework, 
  AuditSeverity,
  ListOptions, 
  PaginatedResponse 
} from '../types';
import { PynomalyError, ComplianceError } from '../errors';

export class ComplianceClient {
  constructor(private httpClient: AxiosInstance) {}

  // Audit Events

  /**
   * List audit events
   */
  async listAuditEvents(options: ListOptions & {
    severity?: AuditSeverity;
    action?: string;
    user_id?: string;
    resource_type?: string;
    start_date?: string;
    end_date?: string;
    frameworks?: ComplianceFramework[];
    high_risk_only?: boolean;
  } = {}): Promise<PaginatedResponse<AuditEvent>> {
    try {
      const response = await this.httpClient.get('/compliance/audit-events', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to list audit events');
    }
  }

  /**
   * Get audit event by ID
   */
  async getAuditEvent(eventId: string): Promise<AuditEvent> {
    try {
      const response = await this.httpClient.get(`/compliance/audit-events/${eventId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to get audit event ${eventId}`);
    }
  }

  /**
   * Create audit event
   */
  async createAuditEvent(event: {
    action: string;
    severity: AuditSeverity;
    user_id?: string;
    resource_type?: string;
    resource_id?: string;
    details: Record<string, any>;
    ip_address?: string;
    outcome: string;
    compliance_frameworks?: ComplianceFramework[];
  }): Promise<AuditEvent> {
    try {
      const response = await this.httpClient.post('/compliance/audit-events', event);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to create audit event');
    }
  }

  /**
   * Export audit events
   */
  async exportAuditEvents(options: {
    format?: 'csv' | 'json' | 'xlsx';
    start_date?: string;
    end_date?: string;
    severity?: AuditSeverity;
    frameworks?: ComplianceFramework[];
  } = {}): Promise<ArrayBuffer> {
    try {
      const response = await this.httpClient.get('/compliance/audit-events/export', {
        params: options,
        responseType: 'arraybuffer'
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to export audit events');
    }
  }

  /**
   * Get audit summary
   */
  async getAuditSummary(options: {
    start_date?: string;
    end_date?: string;
    frameworks?: ComplianceFramework[];
  } = {}): Promise<{
    total_events: number;
    events_by_severity: Record<AuditSeverity, number>;
    events_by_framework: Record<ComplianceFramework, number>;
    high_risk_events: number;
    recent_critical_events: AuditEvent[];
    compliance_score: number;
    recommendations: string[];
  }> {
    try {
      const response = await this.httpClient.get('/compliance/audit-summary', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to get audit summary');
    }
  }

  // GDPR Management

  /**
   * List GDPR requests
   */
  async listGDPRRequests(options: ListOptions & {
    request_type?: string;
    status?: string;
    overdue_only?: boolean;
  } = {}): Promise<PaginatedResponse<GDPRRequest>> {
    try {
      const response = await this.httpClient.get('/compliance/gdpr-requests', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to list GDPR requests');
    }
  }

  /**
   * Get GDPR request by ID
   */
  async getGDPRRequest(requestId: string): Promise<GDPRRequest> {
    try {
      const response = await this.httpClient.get(`/compliance/gdpr-requests/${requestId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to get GDPR request ${requestId}`);
    }
  }

  /**
   * Create GDPR request
   */
  async createGDPRRequest(request: {
    request_type: string;
    data_subject_id: string;
    data_subject_email: string;
    request_details: string;
    notes?: string;
  }): Promise<GDPRRequest> {
    try {
      const response = await this.httpClient.post('/compliance/gdpr-requests', request);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to create GDPR request');
    }
  }

  /**
   * Update GDPR request
   */
  async updateGDPRRequest(requestId: string, updates: {
    status?: string;
    assigned_to?: string;
    notes?: string;
    processed_at?: string;
  }): Promise<GDPRRequest> {
    try {
      const response = await this.httpClient.patch(`/compliance/gdpr-requests/${requestId}`, updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to update GDPR request ${requestId}`);
    }
  }

  /**
   * Process GDPR request
   */
  async processGDPRRequest(requestId: string, action: 'approve' | 'reject' | 'complete', notes?: string): Promise<GDPRRequest> {
    try {
      const response = await this.httpClient.post(`/compliance/gdpr-requests/${requestId}/process`, {
        action,
        notes
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to process GDPR request ${requestId}`);
    }
  }

  /**
   * Get GDPR request data
   */
  async getGDPRRequestData(requestId: string): Promise<{
    personal_data: Record<string, any>;
    data_sources: string[];
    processing_activities: string[];
    retention_periods: Record<string, string>;
  }> {
    try {
      const response = await this.httpClient.get(`/compliance/gdpr-requests/${requestId}/data`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to get data for GDPR request ${requestId}`);
    }
  }

  /**
   * Export GDPR request data
   */
  async exportGDPRRequestData(requestId: string, format: 'json' | 'xml' = 'json'): Promise<ArrayBuffer> {
    try {
      const response = await this.httpClient.get(`/compliance/gdpr-requests/${requestId}/export`, {
        params: { format },
        responseType: 'arraybuffer'
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to export data for GDPR request ${requestId}`);
    }
  }

  // Data Subject Rights

  /**
   * Right to be forgotten - delete user data
   */
  async deleteUserData(userId: string, options: {
    hard_delete?: boolean;
    anonymize?: boolean;
    retain_audit_logs?: boolean;
  } = {}): Promise<{
    deleted_records: number;
    anonymized_records: number;
    retained_records: number;
    completion_date: string;
  }> {
    try {
      const response = await this.httpClient.post(`/compliance/data-subjects/${userId}/delete`, options);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to delete data for user ${userId}`);
    }
  }

  /**
   * Right of access - get user data
   */
  async getUserData(userId: string, format: 'json' | 'xml' = 'json'): Promise<{
    personal_data: Record<string, any>;
    processing_activities: Array<{
      activity: string;
      purpose: string;
      legal_basis: string;
      data_categories: string[];
      retention_period: string;
    }>;
    data_sources: string[];
    sharing_activities: Array<{
      recipient: string;
      purpose: string;
      data_categories: string[];
      transfer_date: string;
    }>;
  }> {
    try {
      const response = await this.httpClient.get(`/compliance/data-subjects/${userId}`, {
        params: { format }
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to get data for user ${userId}`);
    }
  }

  /**
   * Right of portability - export user data
   */
  async exportUserData(userId: string, format: 'json' | 'xml' | 'csv' = 'json'): Promise<ArrayBuffer> {
    try {
      const response = await this.httpClient.get(`/compliance/data-subjects/${userId}/export`, {
        params: { format },
        responseType: 'arraybuffer'
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to export data for user ${userId}`);
    }
  }

  // Compliance Frameworks

  /**
   * Get compliance framework requirements
   */
  async getFrameworkRequirements(framework: ComplianceFramework): Promise<{
    framework: ComplianceFramework;
    requirements: Array<{
      id: string;
      title: string;
      description: string;
      category: string;
      mandatory: boolean;
      implementation_status: 'implemented' | 'partial' | 'not_implemented';
      last_assessment: string;
    }>;
    compliance_score: number;
    gaps: string[];
  }> {
    try {
      const response = await this.httpClient.get(`/compliance/frameworks/${framework}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to get requirements for framework ${framework}`);
    }
  }

  /**
   * Get compliance assessment
   */
  async getComplianceAssessment(frameworks?: ComplianceFramework[]): Promise<{
    overall_score: number;
    framework_scores: Record<ComplianceFramework, number>;
    critical_gaps: Array<{
      framework: ComplianceFramework;
      requirement: string;
      description: string;
      risk_level: 'low' | 'medium' | 'high' | 'critical';
      remediation_steps: string[];
    }>;
    recommendations: string[];
    last_assessment: string;
  }> {
    try {
      const response = await this.httpClient.get('/compliance/assessment', {
        params: frameworks ? { frameworks: frameworks.join(',') } : {}
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to get compliance assessment');
    }
  }

  /**
   * Update compliance requirement status
   */
  async updateRequirementStatus(
    framework: ComplianceFramework,
    requirementId: string,
    status: 'implemented' | 'partial' | 'not_implemented',
    notes?: string
  ): Promise<void> {
    try {
      await this.httpClient.patch(`/compliance/frameworks/${framework}/requirements/${requirementId}`, {
        status,
        notes
      });
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to update requirement ${requirementId} for framework ${framework}`);
    }
  }

  // Data Retention

  /**
   * Get data retention policies
   */
  async getRetentionPolicies(): Promise<Array<{
    id: string;
    name: string;
    description: string;
    data_category: string;
    retention_period_days: number;
    auto_delete: boolean;
    compliance_frameworks: ComplianceFramework[];
    is_active: boolean;
    created_at: string;
  }>> {
    try {
      const response = await this.httpClient.get('/compliance/retention-policies');
      return response.data.policies || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to get retention policies');
    }
  }

  /**
   * Create retention policy
   */
  async createRetentionPolicy(policy: {
    name: string;
    description: string;
    data_category: string;
    retention_period_days: number;
    auto_delete?: boolean;
    compliance_frameworks?: ComplianceFramework[];
  }): Promise<{
    id: string;
    name: string;
    description: string;
    data_category: string;
    retention_period_days: number;
    auto_delete: boolean;
    compliance_frameworks: ComplianceFramework[];
    is_active: boolean;
    created_at: string;
  }> {
    try {
      const response = await this.httpClient.post('/compliance/retention-policies', policy);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to create retention policy');
    }
  }

  /**
   * Get data eligible for deletion
   */
  async getDataForDeletion(policyId?: string): Promise<{
    total_records: number;
    data_categories: Record<string, number>;
    oldest_record_date: string;
    estimated_storage_freed_gb: number;
    records: Array<{
      id: string;
      data_category: string;
      created_at: string;
      retention_expires_at: string;
      estimated_size_mb: number;
    }>;
  }> {
    try {
      const response = await this.httpClient.get('/compliance/data-for-deletion', {
        params: policyId ? { policy_id: policyId } : {}
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to get data eligible for deletion');
    }
  }

  /**
   * Execute data deletion
   */
  async executeDataDeletion(policyId: string, options: {
    dry_run?: boolean;
    batch_size?: number;
  } = {}): Promise<{
    job_id: string;
    status: string;
    estimated_completion: string;
    records_to_delete: number;
  }> {
    try {
      const response = await this.httpClient.post(`/compliance/retention-policies/${policyId}/execute`, options);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to execute data deletion for policy ${policyId}`);
    }
  }

  // Compliance Reporting

  /**
   * Generate compliance report
   */
  async generateComplianceReport(options: {
    frameworks?: ComplianceFramework[];
    start_date?: string;
    end_date?: string;
    format?: 'pdf' | 'html' | 'docx';
    include_remediation_plan?: boolean;
  } = {}): Promise<{
    report_id: string;
    status: string;
    download_url?: string;
    generated_at?: string;
  }> {
    try {
      const response = await this.httpClient.post('/compliance/reports', options);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError('Failed to generate compliance report');
    }
  }

  /**
   * Get compliance report status
   */
  async getReportStatus(reportId: string): Promise<{
    status: 'generating' | 'completed' | 'failed';
    progress: number;
    download_url?: string;
    error?: string;
  }> {
    try {
      const response = await this.httpClient.get(`/compliance/reports/${reportId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to get status for report ${reportId}`);
    }
  }

  /**
   * Download compliance report
   */
  async downloadReport(reportId: string): Promise<ArrayBuffer> {
    try {
      const response = await this.httpClient.get(`/compliance/reports/${reportId}/download`, {
        responseType: 'arraybuffer'
      });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new ComplianceError(`Failed to download report ${reportId}`);
    }
  }
}