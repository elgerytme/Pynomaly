package anomalydetection

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ValidateDataFormat validates that data is in the correct format for anomaly detection
func ValidateDataFormat(data [][]float64) error {
	if len(data) == 0 {
		return &ValidationError{
			SDKError: &SDKError{
				Message: "Data cannot be empty",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("data"),
			Value: data,
		}
	}

	if len(data[0]) == 0 {
		return &ValidationError{
			SDKError: &SDKError{
				Message: "Data points cannot be empty",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("data"),
			Value: data,
		}
	}

	// Check if all data points have the same length
	firstPointLength := len(data[0])
	for i, point := range data {
		if len(point) != firstPointLength {
			return &ValidationError{
				SDKError: &SDKError{
					Message: fmt.Sprintf("All data points must have the same number of features. Point %d has %d features, expected %d", i, len(point), firstPointLength),
					Code:    stringPtr("VALIDATION_ERROR"),
				},
				Field: stringPtr("data"),
				Value: point,
			}
		}

		// Check for invalid values
		for j, value := range point {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return &ValidationError{
					SDKError: &SDKError{
						Message: fmt.Sprintf("Invalid value at data[%d][%d]: %f", i, j, value),
						Code:    stringPtr("VALIDATION_ERROR"),
					},
					Field: stringPtr("data"),
					Value: value,
				}
			}
		}
	}

	return nil
}

// NormalizationParams holds parameters for data normalization
type NormalizationParams struct {
	Means []float64
	Stds  []float64
}

// NormalizeData normalizes data to zero mean and unit variance
func NormalizeData(data [][]float64) ([][]float64, *NormalizationParams, error) {
	if err := ValidateDataFormat(data); err != nil {
		return nil, nil, err
	}

	numFeatures := len(data[0])
	numSamples := len(data)

	// Calculate means
	means := make([]float64, numFeatures)
	for _, point := range data {
		for j, value := range point {
			means[j] += value
		}
	}
	for j := range means {
		means[j] /= float64(numSamples)
	}

	// Calculate standard deviations
	stds := make([]float64, numFeatures)
	for _, point := range data {
		for j, value := range point {
			diff := value - means[j]
			stds[j] += diff * diff
		}
	}
	for j := range stds {
		stds[j] = math.Sqrt(stds[j] / float64(numSamples))
		// Avoid division by zero
		if stds[j] == 0 {
			stds[j] = 1
		}
	}

	// Normalize data
	normalizedData := make([][]float64, len(data))
	for i, point := range data {
		normalizedData[i] = make([]float64, len(point))
		for j, value := range point {
			normalizedData[i][j] = (value - means[j]) / stds[j]
		}
	}

	params := &NormalizationParams{
		Means: means,
		Stds:  stds,
	}

	return normalizedData, params, nil
}

// ApplyNormalization applies normalization parameters to new data
func ApplyNormalization(data [][]float64, params *NormalizationParams) ([][]float64, error) {
	if err := ValidateDataFormat(data); err != nil {
		return nil, err
	}

	if len(data[0]) != len(params.Means) || len(data[0]) != len(params.Stds) {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Data dimensions must match normalization parameters",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("data"),
			Value: data,
		}
	}

	normalizedData := make([][]float64, len(data))
	for i, point := range data {
		normalizedData[i] = make([]float64, len(point))
		for j, value := range point {
			normalizedData[i][j] = (value - params.Means[j]) / params.Stds[j]
		}
	}

	return normalizedData, nil
}

// TrainValidationSplit splits data into training and validation sets
type TrainValidationSplit struct {
	TrainData      [][]float64
	ValidationData [][]float64
}

// SplitTrainValidation splits data into training and validation sets
func SplitTrainValidation(data [][]float64, validationRatio float64, shuffle bool) (*TrainValidationSplit, error) {
	if err := ValidateDataFormat(data); err != nil {
		return nil, err
	}

	if validationRatio < 0 || validationRatio >= 1 {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Validation ratio must be between 0 and 1",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("validationRatio"),
			Value: validationRatio,
		}
	}

	workingData := make([][]float64, len(data))
	copy(workingData, data)

	// Shuffle if requested
	if shuffle {
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(len(workingData), func(i, j int) {
			workingData[i], workingData[j] = workingData[j], workingData[i]
		})
	}

	splitIndex := int(float64(len(data)) * (1 - validationRatio))

	return &TrainValidationSplit{
		TrainData:      workingData[:splitIndex],
		ValidationData: workingData[splitIndex:],
	}, nil
}

// DataStatistics holds basic statistics for data
type DataStatistics struct {
	NumSamples  int
	NumFeatures int
	Means       []float64
	Stds        []float64
	Mins        []float64
	Maxs        []float64
}

// CalculateDataStatistics calculates basic statistics for data
func CalculateDataStatistics(data [][]float64) (*DataStatistics, error) {
	if err := ValidateDataFormat(data); err != nil {
		return nil, err
	}

	numSamples := len(data)
	numFeatures := len(data[0])

	// Initialize arrays
	means := make([]float64, numFeatures)
	mins := make([]float64, numFeatures)
	maxs := make([]float64, numFeatures)

	// Initialize mins and maxs
	for j := range mins {
		mins[j] = math.Inf(1)
		maxs[j] = math.Inf(-1)
	}

	// Calculate means, mins, and maxs
	for _, point := range data {
		for j, value := range point {
			means[j] += value
			if value < mins[j] {
				mins[j] = value
			}
			if value > maxs[j] {
				maxs[j] = value
			}
		}
	}

	// Finalize means
	for j := range means {
		means[j] /= float64(numSamples)
	}

	// Calculate standard deviations
	stds := make([]float64, numFeatures)
	for _, point := range data {
		for j, value := range point {
			diff := value - means[j]
			stds[j] += diff * diff
		}
	}
	for j := range stds {
		stds[j] = math.Sqrt(stds[j] / float64(numSamples))
	}

	return &DataStatistics{
		NumSamples:  numSamples,
		NumFeatures: numFeatures,
		Means:       means,
		Stds:        stds,
		Mins:        mins,
		Maxs:        maxs,
	}, nil
}

// GenerateSampleData generates sample data for testing
type SampleData struct {
	Data   [][]float64
	Labels []bool // true for anomalies
}

// GenerateSampleData generates sample data with a specified anomaly ratio
func GenerateSampleData(numSamples, numFeatures int, anomalyRatio float64) (*SampleData, error) {
	if numSamples <= 0 || numFeatures <= 0 {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Number of samples and features must be positive",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
		}
	}

	if anomalyRatio < 0 || anomalyRatio > 1 {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Anomaly ratio must be between 0 and 1",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
		}
	}

	rand.Seed(time.Now().UnixNano())

	data := make([][]float64, numSamples)
	labels := make([]bool, numSamples)
	numAnomalies := int(float64(numSamples) * anomalyRatio)

	// Generate normal points (mean=0, std=1)
	for i := 0; i < numSamples-numAnomalies; i++ {
		point := make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			point[j] = rand.NormFloat64()
		}
		data[i] = point
		labels[i] = false
	}

	// Generate anomalous points (mean=3, std=1)
	for i := numSamples - numAnomalies; i < numSamples; i++ {
		point := make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			point[j] = rand.NormFloat64() + 3
		}
		data[i] = point
		labels[i] = true
	}

	// Shuffle the data
	rand.Shuffle(numSamples, func(i, j int) {
		data[i], data[j] = data[j], data[i]
		labels[i], labels[j] = labels[j], labels[i]
	})

	return &SampleData{
		Data:   data,
		Labels: labels,
	}, nil
}

// IsValidAlgorithmType checks if algorithm type is valid
func IsValidAlgorithmType(algorithm string) bool {
	switch AlgorithmType(algorithm) {
	case IsolationForest, LocalOutlierFactor, OneClassSVM, EllipticEnvelope, Autoencoder, Ensemble:
		return true
	default:
		return false
	}
}

// GetAvailableAlgorithms returns all available algorithm types
func GetAvailableAlgorithms() []AlgorithmType {
	return []AlgorithmType{
		IsolationForest,
		LocalOutlierFactor,
		OneClassSVM,
		EllipticEnvelope,
		Autoencoder,
		Ensemble,
	}
}

// FormatExecutionTime formats execution time for display
func FormatExecutionTime(seconds float64) string {
	if seconds < 1 {
		return fmt.Sprintf("%.0fms", seconds*1000)
	} else if seconds < 60 {
		return fmt.Sprintf("%.2fs", seconds)
	} else {
		minutes := int(seconds / 60)
		remainingSeconds := seconds - float64(minutes*60)
		return fmt.Sprintf("%dm %.2fs", minutes, remainingSeconds)
	}
}