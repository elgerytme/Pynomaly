package pynomaly

import (
	"context"
	"fmt"
	"strconv"
)

// DetectionClient handles anomaly detection operations
type DetectionClient struct {
	client *Client
}

// ListDetectors returns a paginated list of detectors
func (d *DetectionClient) ListDetectors(ctx context.Context, opts *ListOptions) (*PaginatedResponse[Detector], error) {
	params := make(map[string]string)
	if opts != nil {
		if opts.Page != nil {
			params["page"] = strconv.Itoa(*opts.Page)
		}
		if opts.PageSize != nil {
			params["page_size"] = strconv.Itoa(*opts.PageSize)
		}
		if opts.SortBy != nil {
			params["sort_by"] = *opts.SortBy
		}
		if opts.SortOrder != nil {
			params["sort_order"] = *opts.SortOrder
		}
	}

	var result PaginatedResponse[Detector]
	err := d.client.makeRequestWithQuery(ctx, "GET", "/detectors", params, &result)
	return &result, err
}

// GetDetector retrieves a detector by ID
func (d *DetectionClient) GetDetector(ctx context.Context, detectorID string) (*Detector, error) {
	var result Detector
	err := d.client.makeRequest(ctx, "GET", "/detectors/"+detectorID, nil, &result)
	return &result, err
}

// CreateDetector creates a new detector
func (d *DetectionClient) CreateDetector(ctx context.Context, req *CreateDetectorRequest) (*Detector, error) {
	var result Detector
	err := d.client.makeRequest(ctx, "POST", "/detectors", req, &result)
	return &result, err
}

// UpdateDetector updates an existing detector
func (d *DetectionClient) UpdateDetector(ctx context.Context, detectorID string, updates *CreateDetectorRequest) (*Detector, error) {
	var result Detector
	err := d.client.makeRequest(ctx, "PATCH", "/detectors/"+detectorID, updates, &result)
	return &result, err
}

// DeleteDetector deletes a detector
func (d *DetectionClient) DeleteDetector(ctx context.Context, detectorID string) error {
	return d.client.makeRequest(ctx, "DELETE", "/detectors/"+detectorID, nil, nil)
}

// TrainDetector starts training a detector with a dataset
func (d *DetectionClient) TrainDetector(ctx context.Context, detectorID string, req *TrainDetectorRequest) (*TrainingJobResponse, error) {
	var result TrainingJobResponse
	err := d.client.makeRequest(ctx, "POST", "/detectors/"+detectorID+"/train", req, &result)
	return &result, err
}

// GetTrainingStatus retrieves the training status for a detector
func (d *DetectionClient) GetTrainingStatus(ctx context.Context, detectorID, jobID string) (*TrainingStatusResponse, error) {
	var result TrainingStatusResponse
	err := d.client.makeRequest(ctx, "GET", fmt.Sprintf("/detectors/%s/train/%s", detectorID, jobID), nil, &result)
	return &result, err
}

// StopTraining stops training for a detector
func (d *DetectionClient) StopTraining(ctx context.Context, detectorID, jobID string) error {
	return d.client.makeRequest(ctx, "POST", fmt.Sprintf("/detectors/%s/train/%s/stop", detectorID, jobID), nil, nil)
}

// DetectAnomalies performs anomaly detection on a dataset
func (d *DetectionClient) DetectAnomalies(ctx context.Context, req *DetectionRequest) (*DetectionResult, error) {
	var result DetectionResult
	err := d.client.makeRequest(ctx, "POST", "/detect", req, &result)
	return &result, err
}

// DetectAnomaliesRealtime performs real-time anomaly detection on data
func (d *DetectionClient) DetectAnomaliesRealtime(ctx context.Context, detectorID string, data []map[string]interface{}) (*DetectionResult, error) {
	request := map[string]interface{}{
		"data": data,
	}
	
	var result DetectionResult
	err := d.client.makeRequest(ctx, "POST", "/detectors/"+detectorID+"/detect", request, &result)
	return &result, err
}

// GetDetectionHistory retrieves detection history for a detector
func (d *DetectionClient) GetDetectionHistory(ctx context.Context, detectorID string, opts *ListOptions) (*PaginatedResponse[DetectionResult], error) {
	params := make(map[string]string)
	if opts != nil {
		if opts.Page != nil {
			params["page"] = strconv.Itoa(*opts.Page)
		}
		if opts.PageSize != nil {
			params["page_size"] = strconv.Itoa(*opts.PageSize)
		}
	}

	var result PaginatedResponse[DetectionResult]
	err := d.client.makeRequestWithQuery(ctx, "GET", "/detectors/"+detectorID+"/history", params, &result)
	return &result, err
}

// ListDatasets returns a paginated list of datasets
func (d *DetectionClient) ListDatasets(ctx context.Context, opts *ListOptions) (*PaginatedResponse[Dataset], error) {
	params := make(map[string]string)
	if opts != nil {
		if opts.Page != nil {
			params["page"] = strconv.Itoa(*opts.Page)
		}
		if opts.PageSize != nil {
			params["page_size"] = strconv.Itoa(*opts.PageSize)
		}
		if opts.SortBy != nil {
			params["sort_by"] = *opts.SortBy
		}
		if opts.SortOrder != nil {
			params["sort_order"] = *opts.SortOrder
		}
	}

	var result PaginatedResponse[Dataset]
	err := d.client.makeRequestWithQuery(ctx, "GET", "/datasets", params, &result)
	return &result, err
}

// GetDataset retrieves a dataset by ID
func (d *DetectionClient) GetDataset(ctx context.Context, datasetID string) (*Dataset, error) {
	var result Dataset
	err := d.client.makeRequest(ctx, "GET", "/datasets/"+datasetID, nil, &result)
	return &result, err
}

// CreateDataset creates a dataset from an uploaded file
func (d *DetectionClient) CreateDataset(ctx context.Context, fileID, name, description string, options map[string]interface{}) (*Dataset, error) {
	request := map[string]interface{}{
		"file_id": fileID,
		"name":    name,
	}
	
	if description != "" {
		request["description"] = description
	}
	
	if options != nil {
		request["options"] = options
	}

	var result Dataset
	err := d.client.makeRequest(ctx, "POST", "/datasets", request, &result)
	return &result, err
}

// UpdateDataset updates a dataset
func (d *DetectionClient) UpdateDataset(ctx context.Context, datasetID string, updates *DatasetUpdateRequest) (*Dataset, error) {
	var result Dataset
	err := d.client.makeRequest(ctx, "PATCH", "/datasets/"+datasetID, updates, &result)
	return &result, err
}

// DeleteDataset deletes a dataset
func (d *DetectionClient) DeleteDataset(ctx context.Context, datasetID string) error {
	return d.client.makeRequest(ctx, "DELETE", "/datasets/"+datasetID, nil, nil)
}

// GetDatasetPreview retrieves a preview of dataset data
func (d *DetectionClient) GetDatasetPreview(ctx context.Context, datasetID string, limit int) (*DatasetPreviewResponse, error) {
	params := map[string]string{
		"limit": strconv.Itoa(limit),
	}

	var result DatasetPreviewResponse
	err := d.client.makeRequestWithQuery(ctx, "GET", "/datasets/"+datasetID+"/preview", params, &result)
	return &result, err
}

// GetDatasetStats retrieves statistics for a dataset
func (d *DetectionClient) GetDatasetStats(ctx context.Context, datasetID string) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := d.client.makeRequest(ctx, "GET", "/datasets/"+datasetID+"/stats", nil, &result)
	return result, err
}

// ExportResults exports detection results
func (d *DetectionClient) ExportResults(ctx context.Context, detectorID string, format string) ([]byte, error) {
	params := map[string]string{
		"format": format,
	}

	// For binary responses, we need to handle this differently
	// This is a simplified implementation
	return []byte{}, fmt.Errorf("export functionality not implemented in this example")
}

// GetDetectorMetrics retrieves performance metrics for a detector
func (d *DetectionClient) GetDetectorMetrics(ctx context.Context, detectorID string) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := d.client.makeRequest(ctx, "GET", "/detectors/"+detectorID+"/metrics", nil, &result)
	return result, err
}

// GetFeatureImportance retrieves feature importance for a detector
func (d *DetectionClient) GetFeatureImportance(ctx context.Context, detectorID string) (*FeatureImportanceResponse, error) {
	var result FeatureImportanceResponse
	err := d.client.makeRequest(ctx, "GET", "/detectors/"+detectorID+"/feature-importance", nil, &result)
	return &result, err
}

// GetAnomalyExplanations retrieves explanations for specific anomalies
func (d *DetectionClient) GetAnomalyExplanations(ctx context.Context, detectorID string, anomalyIDs []string) (*AnomalyExplanationResponse, error) {
	request := map[string]interface{}{
		"anomaly_ids": anomalyIDs,
	}

	var result AnomalyExplanationResponse
	err := d.client.makeRequest(ctx, "POST", "/detectors/"+detectorID+"/explain", request, &result)
	return &result, err
}

// Supporting types for detection operations

type TrainingJobResponse struct {
	JobID  string `json:"job_id"`
	Status string `json:"status"`
}

type TrainingStatusResponse struct {
	Status   string                 `json:"status"`
	Progress float64                `json:"progress"`
	Metrics  map[string]interface{} `json:"metrics,omitempty"`
}

type DatasetUpdateRequest struct {
	Name        *string                `json:"name,omitempty"`
	Description *string                `json:"description,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type DatasetPreviewResponse struct {
	Columns []string        `json:"columns"`
	Data    [][]interface{} `json:"data"`
}

type FeatureImportanceResponse struct {
	Features []FeatureImportance `json:"features"`
}

type FeatureImportance struct {
	Feature    string  `json:"feature"`
	Importance float64 `json:"importance"`
}

type AnomalyExplanationResponse struct {
	Explanations []map[string]interface{} `json:"explanations"`
}