package pynomaly

import "context"

// ComplianceClient handles compliance and audit operations
type ComplianceClient struct {
	client *Client
}

// ListAuditEvents returns a paginated list of audit events
func (c *ComplianceClient) ListAuditEvents(ctx context.Context, opts *AuditEventListOptions) (*PaginatedResponse[AuditEvent], error) {
	var result PaginatedResponse[AuditEvent]
	err := c.client.makeRequest(ctx, "GET", "/compliance/audit-events", nil, &result)
	return &result, err
}

// CreateAuditEvent creates a new audit event
func (c *ComplianceClient) CreateAuditEvent(ctx context.Context, req *CreateAuditEventRequest) (*AuditEvent, error) {
	var result AuditEvent
	err := c.client.makeRequest(ctx, "POST", "/compliance/audit-events", req, &result)
	return &result, err
}

// GetComplianceAssessment retrieves compliance assessment
func (c *ComplianceClient) GetComplianceAssessment(ctx context.Context, frameworks []ComplianceFramework) (*ComplianceAssessmentResponse, error) {
	request := map[string]interface{}{}
	if len(frameworks) > 0 {
		request["frameworks"] = frameworks
	}

	var result ComplianceAssessmentResponse
	err := c.client.makeRequest(ctx, "GET", "/compliance/assessment", nil, &result)
	return &result, err
}

// DeleteUserData processes right to be forgotten request
func (c *ComplianceClient) DeleteUserData(ctx context.Context, userID string, opts *DeleteUserDataOptions) (*DataDeletionResponse, error) {
	var result DataDeletionResponse
	err := c.client.makeRequest(ctx, "POST", "/compliance/data-subjects/"+userID+"/delete", opts, &result)
	return &result, err
}

// Supporting types
type AuditEventListOptions struct {
	ListOptions
	Severity       *AuditSeverity         `json:"severity,omitempty"`
	Action         *string                `json:"action,omitempty"`
	UserID         *string                `json:"user_id,omitempty"`
	ResourceType   *string                `json:"resource_type,omitempty"`
	StartDate      *string                `json:"start_date,omitempty"`
	EndDate        *string                `json:"end_date,omitempty"`
	Frameworks     []ComplianceFramework  `json:"frameworks,omitempty"`
	HighRiskOnly   *bool                  `json:"high_risk_only,omitempty"`
}

type CreateAuditEventRequest struct {
	Action               string                    `json:"action"`
	Severity             AuditSeverity             `json:"severity"`
	UserID               string                    `json:"user_id,omitempty"`
	ResourceType         string                    `json:"resource_type,omitempty"`
	ResourceID           string                    `json:"resource_id,omitempty"`
	Details              map[string]interface{}    `json:"details"`
	IPAddress            string                    `json:"ip_address,omitempty"`
	Outcome              string                    `json:"outcome"`
	ComplianceFrameworks []ComplianceFramework     `json:"compliance_frameworks,omitempty"`
}

type ComplianceAssessmentResponse struct {
	OverallScore    float64                            `json:"overall_score"`
	FrameworkScores map[ComplianceFramework]float64    `json:"framework_scores"`
	CriticalGaps    []ComplianceGap                    `json:"critical_gaps"`
	Recommendations []string                           `json:"recommendations"`
	LastAssessment  string                             `json:"last_assessment"`
}

type ComplianceGap struct {
	Framework        ComplianceFramework `json:"framework"`
	Requirement      string              `json:"requirement"`
	Description      string              `json:"description"`
	RiskLevel        string              `json:"risk_level"`
	RemediationSteps []string            `json:"remediation_steps"`
}

type DeleteUserDataOptions struct {
	HardDelete       *bool `json:"hard_delete,omitempty"`
	Anonymize        *bool `json:"anonymize,omitempty"`
	RetainAuditLogs  *bool `json:"retain_audit_logs,omitempty"`
}

type DataDeletionResponse struct {
	DeletedRecords    int    `json:"deleted_records"`
	AnonymizedRecords int    `json:"anonymized_records"`
	RetainedRecords   int    `json:"retained_records"`
	CompletionDate    string `json:"completion_date"`
}