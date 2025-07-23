package main

import (
	"context"
	"fmt"
	"log"
	"time"

	anomaly_detection "github.com/monorepo/anomaly-detection-client-go"
)

func main() {
	// Configure the client
	config := &anomaly_detection.ClientConfig{
		BaseURL:    "https://api.anomaly-detection.com",
		APIKey:     "your-api-key-here",
		Timeout:    30 * time.Second,
		MaxRetries: 3,
		Debug:      true,
	}

	// Create client
	client := anomaly_detection.NewClient(config)
	defer client.Close()

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Authenticate with API key
	if err := client.AuthenticateWithAPIKey(ctx, config.APIKey); err != nil {
		log.Fatalf("Authentication failed: %v", err)
	}

	// Basic anomaly detection example
	fmt.Println("=== Basic Anomaly Detection ===")
	basicDetectionExample(ctx, client)

	// Ensemble detection example
	fmt.Println("\n=== Ensemble Anomaly Detection ===")
	ensembleDetectionExample(ctx, client)

	// Model training example
	fmt.Println("\n=== Model Training ===")
	modelTrainingExample(ctx, client)

	// List algorithms example
	fmt.Println("\n=== Available Algorithms ===")
	listAlgorithmsExample(ctx, client)

	// Health check example
	fmt.Println("\n=== Health Check ===")
	healthCheckExample(ctx, client)
}

// basicDetectionExample demonstrates basic anomaly detection
func basicDetectionExample(ctx context.Context, client *anomaly_detection.Client) {
	// Sample data with some anomalies
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{1.1, 2.1, 3.1},
		{0.9, 1.9, 2.9},
		{1.2, 2.2, 3.2},
		{10.0, 20.0, 30.0}, // Anomaly
		{1.0, 2.0, 3.0},
		{0.8, 1.8, 2.8},
		{50.0, 60.0, 70.0}, // Anomaly
	}

	request := &anomaly_detection.DetectionRequest{
		Data:           data,
		Algorithm:      "isolation_forest",
		Contamination:  floatPtr(0.25), // Expect 25% anomalies
		ExplainResults: true,
		Parameters: map[string]interface{}{
			"n_estimators": 100,
			"random_state": 42,
		},
	}

	response, err := client.DetectAnomalies(ctx, request)
	if err != nil {
		log.Printf("Detection failed: %v", err)
		return
	}

	fmt.Printf("Detection completed successfully!\n")
	fmt.Printf("Algorithm: %s\n", response.Algorithm)
	fmt.Printf("Total samples: %d\n", response.TotalSamples)
	fmt.Printf("Anomalies detected: %d\n", response.AnomaliesDetected)
	fmt.Printf("Anomaly rate: %.2f%%\n", response.AnomalyRate*100)
	fmt.Printf("Processing time: %.3f seconds\n", response.ProcessingTime)

	fmt.Printf("Anomaly results: ")
	for i, isAnomaly := range response.Anomalies {
		if isAnomaly {
			fmt.Printf("[%d: ANOMALY (%.3f)] ", i, response.ConfidenceScores[i])
		} else {
			fmt.Printf("[%d: normal] ", i)
		}
	}
	fmt.Println()

	if response.Explanations != nil {
		fmt.Printf("Explanations available: SHAP=%t, LIME=%t\n",
			len(response.Explanations.SHAP) > 0,
			len(response.Explanations.LIME) > 0)
	}
}

// ensembleDetectionExample demonstrates ensemble anomaly detection
func ensembleDetectionExample(ctx context.Context, client *anomaly_detection.Client) {
	// Sample data
	data := [][]float64{
		{1.0, 2.0},
		{1.1, 2.1},
		{0.9, 1.9},
		{10.0, 20.0}, // Anomaly
		{1.2, 2.2},
		{15.0, 25.0}, // Anomaly
	}

	request := &anomaly_detection.EnsembleRequest{
		Data:       data,
		Algorithms: []string{"isolation_forest", "one_class_svm", "local_outlier_factor"},
		Method:     "voting",
		Parameters: map[string]interface{}{
			"contamination": 0.3,
		},
	}

	response, err := client.DetectAnomaliesEnsemble(ctx, request)
	if err != nil {
		log.Printf("Ensemble detection failed: %v", err)
		return
	}

	fmt.Printf("Ensemble detection completed!\n")
	fmt.Printf("Method: %s\n", response.Method)
	fmt.Printf("Algorithms: %v\n", response.Algorithms)
	fmt.Printf("Total samples: %d\n", response.TotalSamples)
	fmt.Printf("Anomalies detected: %d\n", len(getAnomalyIndices(response.Anomalies)))
	fmt.Printf("Processing time: %.3f seconds\n", response.ProcessingTime)

	fmt.Printf("Individual algorithm results:\n")
	for algorithm, results := range response.IndividualResults {
		anomalyCount := len(getAnomalyIndices(results))
		fmt.Printf("  %s: %d anomalies\n", algorithm, anomalyCount)
	}

	fmt.Printf("Final ensemble results: ")
	for i, isAnomaly := range response.Anomalies {
		if isAnomaly {
			fmt.Printf("[%d: ANOMALY] ", i)
		} else {
			fmt.Printf("[%d: normal] ", i)
		}
	}
	fmt.Println()
}

// modelTrainingExample demonstrates model training
func modelTrainingExample(ctx context.Context, client *anomaly_detection.Client) {
	// Training data
	trainingData := [][]float64{
		{1.0, 2.0, 3.0},
		{1.1, 2.1, 3.1},
		{0.9, 1.9, 2.9},
		{1.2, 2.2, 3.2},
		{1.0, 2.0, 3.0},
		{0.8, 1.8, 2.8},
		{1.3, 2.3, 3.3},
		{0.95, 1.95, 2.95},
	}

	request := &anomaly_detection.TrainingRequest{
		Data:            trainingData,
		Algorithm:       "isolation_forest",
		ModelName:       "my_custom_model",
		ValidationSplit: 0.2,
		Parameters: map[string]interface{}{
			"n_estimators":   200,
			"contamination":  0.1,
			"random_state":   42,
			"max_features":   1.0,
		},
	}

	response, err := client.TrainModel(ctx, request)
	if err != nil {
		log.Printf("Model training failed: %v", err)
		return
	}

	fmt.Printf("Model training completed!\n")
	fmt.Printf("Model ID: %s\n", response.ModelID)
	fmt.Printf("Model Name: %s\n", response.ModelName)
	fmt.Printf("Algorithm: %s\n", response.Algorithm)
	fmt.Printf("Training time: %.3f seconds\n", response.TrainingTime)
	if response.ValidationScore > 0 {
		fmt.Printf("Validation score: %.4f\n", response.ValidationScore)
	}
	fmt.Printf("Created at: %s\n", response.CreatedAt.Format(time.RFC3339))

	// List all models
	fmt.Println("\nListing all models:")
	models, err := client.ListModels(ctx, 1, 10)
	if err != nil {
		log.Printf("Failed to list models: %v", err)
		return
	}

	fmt.Printf("Total models: %d\n", models.Total)
	for i, model := range models.Models {
		fmt.Printf("%d. %s (%s) - %s - Status: %s\n",
			i+1, model.Name, model.ID, model.Algorithm, model.Status)
	}
}

// listAlgorithmsExample demonstrates listing available algorithms
func listAlgorithmsExample(ctx context.Context, client *anomaly_detection.Client) {
	response, err := client.ListAlgorithms(ctx)
	if err != nil {
		log.Printf("Failed to list algorithms: %v", err)
		return
	}

	fmt.Printf("Available algorithms (%d total):\n", len(response.Algorithms))
	
	// Group by category
	for category, algorithms := range response.Categories {
		fmt.Printf("\n%s:\n", category)
		for _, algoName := range algorithms {
			// Find the algorithm details
			for _, algo := range response.Algorithms {
				if algo.Name == algoName {
					fmt.Printf("  - %s (%s)\n", algo.DisplayName, algo.Name)
					fmt.Printf("    Description: %s\n", algo.Description)
					fmt.Printf("    Complexity: %s\n", algo.Complexity)
					fmt.Printf("    Suitable for: %v\n", algo.SuitableFor)
					break
				}
			}
		}
	}
}

// healthCheckExample demonstrates health check
func healthCheckExample(ctx context.Context, client *anomaly_detection.Client) {
	response, err := client.HealthCheck(ctx)
	if err != nil {
		log.Printf("Health check failed: %v", err)
		return
	}

	fmt.Printf("Health Status: %s\n", response.Status)
	fmt.Printf("Version: %s\n", response.Version)
	fmt.Printf("Uptime: %.2f seconds\n", response.Uptime)
	fmt.Printf("Timestamp: %s\n", response.Timestamp.Format(time.RFC3339))

	if len(response.Components) > 0 {
		fmt.Println("Component Health:")
		for component, health := range response.Components {
			fmt.Printf("  %s: %s", component, health.Status)
			if health.Message != "" {
				fmt.Printf(" (%s)", health.Message)
			}
			fmt.Println()
		}
	}
}

// Helper functions

func floatPtr(f float64) *float64 {
	return &f
}

func getAnomalyIndices(anomalies []bool) []int {
	var indices []int
	for i, isAnomaly := range anomalies {
		if isAnomaly {
			indices = append(indices, i)
		}
	}
	return indices
}