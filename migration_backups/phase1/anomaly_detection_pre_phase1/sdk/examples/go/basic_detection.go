package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	anomaly "github.com/anomaly-detection/go-sdk"
)

// generateSampleData creates sample data with some anomalies
func generateSampleData() [][]float64 {
	rand.Seed(42)
	var data [][]float64

	// Generate normal data points around [0, 0]
	for i := 0; i < 100; i++ {
		point := []float64{
			rand.Float64()*2 - 1, // Random between -1 and 1
			rand.Float64()*2 - 1,
		}
		data = append(data, point)
	}

	// Generate anomalous data points around [5, 5]
	for i := 0; i < 10; i++ {
		point := []float64{
			rand.Float64()*2 + 4, // Random between 4 and 6
			rand.Float64()*2 + 4,
		}
		data = append(data, point)
	}

	// Shuffle the data
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})

	return data
}

// basicDetectionExample demonstrates basic anomaly detection
func basicDetectionExample() error {
	fmt.Println("=== Basic Detection Example ===")

	// Initialize client
	config := anomaly.ClientConfig{
		BaseURL:    "http://localhost:8000",
		Timeout:    30 * time.Second,
		MaxRetries: 3,
	}

	client := anomaly.NewClient(config)

	// Generate sample data
	data := generateSampleData()
	fmt.Printf("Generated %d data points\n", len(data))

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Detect anomalies using Isolation Forest
	fmt.Println("\nDetecting anomalies with Isolation Forest...")
	parameters := map[string]interface{}{
		"contamination": 0.1,
	}

	result, err := client.DetectAnomalies(
		ctx,
		data,
		anomaly.IsolationForest,
		parameters,
		false, // return explanations
	)
	if err != nil {
		return fmt.Errorf("detection failed: %w", err)
	}

	fmt.Printf("Detection completed in %.3f seconds\n", result.ExecutionTime)
	fmt.Printf("Found %d anomalies out of %d points\n", result.AnomalyCount, result.TotalPoints)

	// Print details of detected anomalies
	displayCount := 5
	if len(result.Anomalies) < displayCount {
		displayCount = len(result.Anomalies)
	}

	for i := 0; i < displayCount; i++ {
		anomaly := result.Anomalies[i]
		fmt.Printf("  Anomaly %d: Index=%d, Score=%.4f\n", i+1, anomaly.Index, anomaly.Score)
	}

	if len(result.Anomalies) > displayCount {
		fmt.Printf("  ... and %d more\n", len(result.Anomalies)-displayCount)
	}

	return nil
}

// algorithmComparisonExample compares different algorithms
func algorithmComparisonExample() error {
	fmt.Println("\n=== Algorithm Comparison Example ===")

	client := anomaly.NewClient(anomaly.ClientConfig{
		BaseURL: "http://localhost:8000",
		Timeout: 30 * time.Second,
	})

	data := generateSampleData()[:50] // Use smaller dataset
	fmt.Printf("Using %d data points for comparison\n", len(data))

	algorithms := []anomaly.AlgorithmType{
		anomaly.IsolationForest,
		anomaly.LocalOutlierFactor,
		anomaly.OneClassSVM,
		anomaly.Ensemble,
	}

	type result struct {
		Algorithm     anomaly.AlgorithmType
		AnomalyCount  int
		ExecutionTime float64
		ClientTime    float64
		Error         error
	}

	var results []result
	ctx := context.Background()

	for _, algorithm := range algorithms {
		fmt.Printf("\nTesting %s...\n", algorithm)
		startTime := time.Now()

		detectionResult, err := client.DetectAnomalies(ctx, data, algorithm, nil, false)
		clientTime := time.Since(startTime).Seconds()

		if err != nil {
			fmt.Printf("  %s: Error - %v\n", algorithm, err)
			results = append(results, result{
				Algorithm: algorithm,
				Error:     err,
			})
		} else {
			fmt.Printf("  %s: %d anomalies (%.3fs server, %.3fs total)\n",
				algorithm, detectionResult.AnomalyCount, detectionResult.ExecutionTime, clientTime)
			results = append(results, result{
				Algorithm:     algorithm,
				AnomalyCount:  detectionResult.AnomalyCount,
				ExecutionTime: detectionResult.ExecutionTime,
				ClientTime:    clientTime,
			})
		}
	}

	// Summary
	fmt.Println("\n--- Algorithm Comparison Summary ---")
	for _, result := range results {
		if result.Error != nil {
			fmt.Printf("%s: Failed - %v\n", result.Algorithm, result.Error)
		} else {
			fmt.Printf("%s: %d anomalies, %.3fs\n",
				result.Algorithm, result.AnomalyCount, result.ExecutionTime)
		}
	}

	return nil
}

// batchProcessingExample demonstrates batch processing
func batchProcessingExample() error {
	fmt.Println("\n=== Batch Processing Example ===")

	client := anomaly.NewClient(anomaly.ClientConfig{
		BaseURL: "http://localhost:8000",
		Timeout: 60 * time.Second, // Longer timeout for batch processing
	})

	// Generate larger dataset
	rand.Seed(42)
	var data [][]float64

	// Normal data
	for i := 0; i < 500; i++ {
		point := []float64{
			rand.Float64()*2 - 1,
			rand.Float64()*2 - 1,
			rand.Float64()*2 - 1,
		}
		data = append(data, point)
	}

	// Anomalous data
	for i := 0; i < 50; i++ {
		point := []float64{
			rand.Float64()*2 + 3,
			rand.Float64()*2 + 3,
			rand.Float64()*2 + 3,
		}
		data = append(data, point)
	}

	// Shuffle
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})

	// Create batch request
	batchRequest := anomaly.BatchProcessingRequest{
		Data:               data,
		Algorithm:          anomaly.IsolationForest,
		Parameters:         map[string]interface{}{"contamination": 0.1},
		ReturnExplanations: true,
	}

	fmt.Printf("Processing batch of %d points...\n", len(data))

	ctx := context.Background()
	result, err := client.BatchDetect(ctx, batchRequest)
	if err != nil {
		return fmt.Errorf("batch processing failed: %w", err)
	}

	fmt.Printf("Batch processing completed in %.3f seconds\n", result.ExecutionTime)
	fmt.Printf("Found %d anomalies\n", result.AnomalyCount)

	// Analyze results
	if len(result.Anomalies) > 0 {
		var minScore, maxScore, totalScore float64
		minScore = result.Anomalies[0].Score
		maxScore = result.Anomalies[0].Score

		for _, anomaly := range result.Anomalies {
			if anomaly.Score < minScore {
				minScore = anomaly.Score
			}
			if anomaly.Score > maxScore {
				maxScore = anomaly.Score
			}
			totalScore += anomaly.Score
		}

		avgScore := totalScore / float64(len(result.Anomalies))
		fmt.Printf("Anomaly scores: min=%.4f, max=%.4f, avg=%.4f\n",
			minScore, maxScore, avgScore)
	}

	return nil
}

// modelManagementExample demonstrates model management
func modelManagementExample() error {
	fmt.Println("\n=== Model Management Example ===")

	client := anomaly.NewClient(anomaly.ClientConfig{
		BaseURL: "http://localhost:8000",
		Timeout: 60 * time.Second,
	})

	ctx := context.Background()

	// Generate training data
	trainingData := generateSampleData()

	// Train a model
	fmt.Println("Training a new model...")
	modelName := "example-model-go"
	trainingRequest := anomaly.TrainingRequest{
		Data:            trainingData,
		Algorithm:       anomaly.IsolationForest,
		Hyperparameters: map[string]interface{}{"contamination": 0.1, "n_estimators": 100},
		ValidationSplit: 0.2,
		ModelName:       &modelName,
	}

	trainingResult, err := client.TrainModel(ctx, trainingRequest)
	if err != nil {
		return fmt.Errorf("model training failed: %w", err)
	}

	fmt.Printf("Model trained: %s\n", trainingResult.ModelID)
	fmt.Printf("Training time: %.2fs\n", trainingResult.TrainingTime)

	// List all models
	fmt.Println("\nListing all models...")
	models, err := client.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("failed to list models: %w", err)
	}

	fmt.Printf("Found %d models:\n", len(models))
	for _, model := range models {
		fmt.Printf("  - %s: %s (%s)\n", model.ModelID, model.Algorithm, model.Status)
		fmt.Printf("    Created: %s\n", model.CreatedAt.Format(time.RFC3339))
	}

	// Get specific model info
	if len(models) > 0 {
		modelID := models[0].ModelID
		fmt.Printf("\nGetting info for model %s...\n", modelID)
		modelInfo, err := client.GetModel(ctx, modelID)
		if err != nil {
			fmt.Printf("Failed to get model info: %v\n", err)
		} else {
			fmt.Printf("  Algorithm: %s\n", modelInfo.Algorithm)
			fmt.Printf("  Version: %s\n", modelInfo.Version)
			fmt.Printf("  Training data size: %d\n", modelInfo.TrainingDataSize)
			fmt.Printf("  Performance metrics: %v\n", modelInfo.PerformanceMetrics)
		}
	}

	return nil
}

// healthCheckExample demonstrates health checking
func healthCheckExample() error {
	fmt.Println("\n=== Health Check Example ===")

	client := anomaly.NewClient(anomaly.ClientConfig{
		BaseURL: "http://localhost:8000",
		Timeout: 10 * time.Second,
	})

	ctx := context.Background()

	// Check service health
	health, err := client.GetHealth(ctx)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}

	fmt.Printf("Service Status: %s\n", health.Status)
	fmt.Printf("Version: %s\n", health.Version)
	fmt.Printf("Uptime: %.1f seconds\n", health.Uptime)

	if len(health.Components) > 0 {
		fmt.Println("Components:")
		for component, status := range health.Components {
			fmt.Printf("  %s: %s\n", component, status)
		}
	}

	// Get metrics
	fmt.Println("\nGetting service metrics...")
	metrics, err := client.GetMetrics(ctx)
	if err != nil {
		return fmt.Errorf("failed to get metrics: %w", err)
	}

	fmt.Println("Service Metrics:")
	for key, value := range metrics {
		if numValue, ok := value.(float64); ok {
			fmt.Printf("  %s: %.2f\n", key, numValue)
		} else {
			fmt.Printf("  %s: %v\n", key, value)
		}
	}

	return nil
}

// errorHandlingExample demonstrates error handling
func errorHandlingExample() {
	fmt.Println("\n=== Error Handling Example ===")

	client := anomaly.NewClient(anomaly.ClientConfig{
		BaseURL: "http://localhost:8000",
		Timeout: 5 * time.Second,
	})

	ctx := context.Background()

	// Test different error scenarios
	errorTests := []struct {
		name string
		test func() error
	}{
		{
			name: "Empty data validation",
			test: func() error {
				_, err := client.DetectAnomalies(ctx, [][]float64{}, anomaly.IsolationForest, nil, false)
				return err
			},
		},
		{
			name: "Non-existent model",
			test: func() error {
				_, err := client.GetModel(ctx, "non-existent-model-id")
				return err
			},
		},
		{
			name: "Invalid model ID",
			test: func() error {
				_, err := client.GetModel(ctx, "")
				return err
			},
		},
	}

	for _, test := range errorTests {
		fmt.Printf("\nTesting: %s\n", test.name)
		err := test.test()
		if err == nil {
			fmt.Println("  ❌ Expected error but got success")
		} else {
			switch e := err.(type) {
			case *anomaly.ValidationError:
				fmt.Printf("  ✅ Validation Error: %s\n", e.Message)
			case *anomaly.APIError:
				fmt.Printf("  ✅ API Error (%d): %s\n", e.StatusCode, e.Message)
			case *anomaly.ConnectionError:
				fmt.Printf("  ✅ Connection Error: %s\n", e.Message)
			case *anomaly.TimeoutError:
				fmt.Printf("  ✅ Timeout Error: %s\n", e.Message)
			default:
				fmt.Printf("  ⚠️  Unknown Error: %v\n", err)
			}
		}
	}
}

// utilityFunctionsExample demonstrates utility functions
func utilityFunctionsExample() error {
	fmt.Println("\n=== Utility Functions Example ===")

	// Generate sample data
	data := generateSampleData()[:50] // Use smaller dataset

	// Validate data format
	fmt.Println("Validating data format...")
	if err := anomaly.ValidateDataFormat(data); err != nil {
		return fmt.Errorf("data validation failed: %w", err)
	}
	fmt.Println("✅ Data format is valid")

	// Normalize data
	fmt.Println("\nNormalizing data...")
	normalizedData, params, err := anomaly.NormalizeData(data)
	if err != nil {
		return fmt.Errorf("normalization failed: %w", err)
	}
	fmt.Printf("✅ Data normalized. Means: %.3f, %.3f\n", params.Means[0], params.Means[1])

	// Apply normalization to new data
	newData := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	normalizedNewData, err := anomaly.ApplyNormalization(newData, params)
	if err != nil {
		return fmt.Errorf("normalization application failed: %w", err)
	}
	fmt.Printf("✅ Applied normalization to new data: %v -> %v\n", newData[0], normalizedNewData[0])

	// Calculate statistics
	fmt.Println("\nCalculating data statistics...")
	stats, err := anomaly.CalculateDataStatistics(data)
	if err != nil {
		return fmt.Errorf("statistics calculation failed: %w", err)
	}
	fmt.Printf("✅ Dataset: %d samples, %d features\n", stats.NumSamples, stats.NumFeatures)
	fmt.Printf("   Feature 1: mean=%.3f, std=%.3f, min=%.3f, max=%.3f\n",
		stats.Means[0], stats.Stds[0], stats.Mins[0], stats.Maxs[0])

	// Generate sample data
	fmt.Println("\nGenerating sample data...")
	sampleData, err := anomaly.GenerateSampleData(100, 3, 0.1)
	if err != nil {
		return fmt.Errorf("sample data generation failed: %w", err)
	}

	anomalyCount := 0
	for _, label := range sampleData.Labels {
		if label {
			anomalyCount++
		}
	}
	fmt.Printf("✅ Generated %d samples with %d anomalies\n", len(sampleData.Data), anomalyCount)

	// Test train/validation split
	fmt.Println("\nSplitting data for training/validation...")
	split, err := anomaly.SplitTrainValidation(data, 0.2, true)
	if err != nil {
		return fmt.Errorf("data split failed: %w", err)
	}
	fmt.Printf("✅ Split data: %d training, %d validation\n",
		len(split.TrainData), len(split.ValidationData))

	// Test available algorithms
	fmt.Println("\nAvailable algorithms:")
	algorithms := anomaly.GetAvailableAlgorithms()
	for _, alg := range algorithms {
		fmt.Printf("  - %s (valid: %t)\n", alg, anomaly.IsValidAlgorithmType(string(alg)))
	}

	// Test execution time formatting
	fmt.Println("\nExecution time formatting examples:")
	times := []float64{0.001, 0.5, 1.5, 65.3, 125.7}
	for _, t := range times {
		fmt.Printf("  %.3f seconds -> %s\n", t, anomaly.FormatExecutionTime(t))
	}

	return nil
}

func main() {
	fmt.Println("Anomaly Detection Go SDK Examples")
	fmt.Println(strings.Repeat("=", 50))

	// List of examples to run
	examples := []struct {
		name string
		fn   func() error
	}{
		{"Basic Detection", basicDetectionExample},
		{"Algorithm Comparison", algorithmComparisonExample},
		{"Batch Processing", batchProcessingExample},
		{"Model Management", modelManagementExample},
		{"Health Check", healthCheckExample},
		{"Utility Functions", utilityFunctionsExample},
	}

	// Run examples
	for _, example := range examples {
		if err := example.fn(); err != nil {
			log.Printf("❌ %s failed: %v", example.name, err)

			// Check for connection errors and provide helpful message
			if connErr, ok := err.(*anomaly.ConnectionError); ok {
				fmt.Printf("\n💡 Connection failed: %s\n", connErr.Message)
				fmt.Println("💡 Make sure the anomaly detection service is running at http://localhost:8000")
				break
			}
		} else {
			fmt.Printf("✅ %s completed successfully\n", example.name)
		}
	}

	// Run error handling example (doesn't return error)
	errorHandlingExample()

	fmt.Println(strings.Repeat("=", 50))
	fmt.Println("Examples completed!")
}