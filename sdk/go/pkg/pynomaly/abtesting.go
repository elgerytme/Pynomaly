package pynomaly

import "context"

// ABTestingClient handles A/B testing operations
type ABTestingClient struct {
	client *Client
}

// CreateTest creates a new A/B test
func (a *ABTestingClient) CreateTest(ctx context.Context, req *CreateABTestRequest) (*ABTest, error) {
	var result ABTest
	err := a.client.makeRequest(ctx, "POST", "/ab-testing/tests", req, &result)
	return &result, err
}

// StartTest starts an A/B test
func (a *ABTestingClient) StartTest(ctx context.Context, testID string) (*TestStatusResponse, error) {
	var result TestStatusResponse
	err := a.client.makeRequest(ctx, "POST", "/ab-testing/tests/"+testID+"/start", nil, &result)
	return &result, err
}

// GetTestResults retrieves results for an A/B test
func (a *ABTestingClient) GetTestResults(ctx context.Context, testID string, opts *ListOptions) (*PaginatedResponse[TestResult], error) {
	var result PaginatedResponse[TestResult]
	err := a.client.makeRequest(ctx, "GET", "/ab-testing/tests/"+testID+"/results", nil, &result)
	return &result, err
}

// Supporting types
type CreateABTestRequest struct {
	Name                  string        `json:"name"`
	Description           string        `json:"description"`
	Variants              []TestVariant `json:"variants"`
	SplitStrategy         SplitStrategy `json:"split_strategy,omitempty"`
	MetricsToCollect      []MetricType  `json:"metrics_to_collect,omitempty"`
	MinimumSampleSize     *int          `json:"minimum_sample_size,omitempty"`
	ConfidenceLevel       *float64      `json:"confidence_level,omitempty"`
	SignificanceThreshold *float64      `json:"significance_threshold,omitempty"`
	DurationDays          *int          `json:"duration_days,omitempty"`
}

type TestStatusResponse struct {
	Status    string `json:"status"`
	StartedAt string `json:"started_at,omitempty"`
}

type TestResult struct {
	TestID        string                 `json:"test_id"`
	VariantID     string                 `json:"variant_id"`
	DatasetID     string                 `json:"dataset_id"`
	ExecutionTime float64                `json:"execution_time"`
	Timestamp     string                 `json:"timestamp"`
	Metrics       []TestMetric           `json:"metrics"`
	AnomalyCount  int                    `json:"anomaly_count"`
	Metadata      map[string]interface{} `json:"metadata"`
}

type TestMetric struct {
	Name      string    `json:"name"`
	Type      MetricType `json:"type"`
	Value     float64   `json:"value"`
	Timestamp string    `json:"timestamp"`
	VariantID string    `json:"variant_id"`
	Metadata  map[string]interface{} `json:"metadata"`
}