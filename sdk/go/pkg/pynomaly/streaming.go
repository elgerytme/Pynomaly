package pynomaly

import "context"

// StreamingClient handles real-time stream processing operations
type StreamingClient struct {
	client *Client
}

// CreateProcessor creates a new stream processor
func (s *StreamingClient) CreateProcessor(ctx context.Context, req *CreateProcessorRequest) (*StreamProcessor, error) {
	var result StreamProcessor
	err := s.client.makeRequest(ctx, "POST", "/streaming/processors", req, &result)
	return &result, err
}

// StartProcessor starts a stream processor
func (s *StreamingClient) StartProcessor(ctx context.Context, processorID string) (*ProcessorStatusResponse, error) {
	var result ProcessorStatusResponse
	err := s.client.makeRequest(ctx, "POST", "/streaming/processors/"+processorID+"/start", nil, &result)
	return &result, err
}

// StopProcessor stops a stream processor
func (s *StreamingClient) StopProcessor(ctx context.Context, processorID string) (*ProcessorStatusResponse, error) {
	var result ProcessorStatusResponse
	err := s.client.makeRequest(ctx, "POST", "/streaming/processors/"+processorID+"/stop", nil, &result)
	return &result, err
}

// SendData sends data to a stream processor
func (s *StreamingClient) SendData(ctx context.Context, processorID string, records []StreamRecord) (*SendDataResponse, error) {
	request := map[string]interface{}{
		"records": records,
	}

	var result SendDataResponse
	err := s.client.makeRequest(ctx, "POST", "/streaming/processors/"+processorID+"/data", request, &result)
	return &result, err
}

// GetProcessorMetrics retrieves metrics for a stream processor
func (s *StreamingClient) GetProcessorMetrics(ctx context.Context, processorID string) (*StreamMetrics, error) {
	var result StreamMetrics
	err := s.client.makeRequest(ctx, "GET", "/streaming/processors/"+processorID+"/metrics", nil, &result)
	return &result, err
}

// Supporting types
type ProcessorStatusResponse struct {
	Status    string `json:"status"`
	StartedAt string `json:"started_at,omitempty"`
	StoppedAt string `json:"stopped_at,omitempty"`
}

type SendDataResponse struct {
	Accepted int      `json:"accepted"`
	Rejected int      `json:"rejected"`
	Errors   []string `json:"errors,omitempty"`
}