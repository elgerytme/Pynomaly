package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	anomaly "github.com/anomaly-detection/go-sdk"
)

// AnomalyDetectionStream wraps the streaming client with additional functionality
type AnomalyDetectionStream struct {
	client       *anomaly.StreamingClient
	anomalyCount int
	totalPoints  int
	running      bool
	mu           sync.RWMutex
}

// NewAnomalyDetectionStream creates a new streaming wrapper
func NewAnomalyDetectionStream(wsURL string) *AnomalyDetectionStream {
	config := anomaly.StreamingClientConfig{
		WSURL: wsURL,
		Config: anomaly.StreamingConfig{
			BufferSize:         50,
			DetectionThreshold: 0.6,
			BatchSize:          5,
			Algorithm:          anomaly.IsolationForest,
			AutoRetrain:        false,
		},
		AutoReconnect:  true,
		ReconnectDelay: 3 * time.Second,
	}

	client := anomaly.NewStreamingClient(config)

	stream := &AnomalyDetectionStream{
		client: client,
	}

	stream.setupHandlers()
	return stream
}

func (s *AnomalyDetectionStream) setupHandlers() {
	handlers := anomaly.StreamingHandlers{
		OnConnect: func() {
			fmt.Println("✅ Connected to streaming service")
			s.mu.Lock()
			s.running = true
			s.mu.Unlock()
		},
		OnDisconnect: func() {
			fmt.Println("❌ Disconnected from streaming service")
			s.mu.Lock()
			s.running = false
			s.mu.Unlock()
		},
		OnAnomaly: func(anomalyData anomaly.AnomalyData) {
			s.mu.Lock()
			s.anomalyCount++
			count := s.anomalyCount
			s.mu.Unlock()

			fmt.Println("🚨 ANOMALY DETECTED:")
			fmt.Printf("   Index: %d\n", anomalyData.Index)
			fmt.Printf("   Score: %.4f\n", anomalyData.Score)
			fmt.Printf("   Data Point: %v\n", formatFloatSlice(anomalyData.DataPoint, 3))
			if anomalyData.Confidence != nil {
				fmt.Printf("   Confidence: %.4f\n", *anomalyData.Confidence)
			}
			fmt.Printf("   Total anomalies so far: %d\n", count)
			fmt.Println(strings.Repeat("-", 50))
		},
		OnError: func(err error) {
			fmt.Printf("❌ Streaming error: %v\n", err)
		},
		OnMessage: func(data map[string]interface{}) {
			// Handle any other message types if needed
			if msgType, ok := data["type"].(string); ok && msgType != "anomaly" && msgType != "ping" {
				fmt.Printf("📨 Received message: %v\n", data)
			}
		},
	}

	s.client.SetHandlers(handlers)
}

func (s *AnomalyDetectionStream) Start() error {
	fmt.Println("Starting streaming anomaly detection...")
	return s.client.Start()
}

func (s *AnomalyDetectionStream) Stop() {
	fmt.Println("Stopping streaming...")
	s.client.Stop()
	s.mu.Lock()
	s.running = false
	s.mu.Unlock()
}

func (s *AnomalyDetectionStream) SendDataPoint(dataPoint []float64) error {
	err := s.client.SendData(dataPoint)
	if err == nil {
		s.mu.Lock()
		s.totalPoints++
		s.mu.Unlock()
	}
	return err
}

func (s *AnomalyDetectionStream) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running
}

func (s *AnomalyDetectionStream) GetStats() (int, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.totalPoints, s.anomalyCount
}

func (s *AnomalyDetectionStream) GenerateAndSendData(duration time.Duration, interval time.Duration) {
	fmt.Printf("Generating sample data for %v...\n", duration)
	fmt.Println("Normal data will be around [0, 0], anomalies around [5, 5]")
	fmt.Println(strings.Repeat("-", 50))

	rand.Seed(42)
	startTime := time.Now()

	for time.Since(startTime) < duration && s.IsRunning() {
		var dataPoint []float64

		// Generate mostly normal data with occasional anomalies
		if rand.Float64() < 0.1 { // 10% chance of anomaly
			// Anomalous data point
			dataPoint = []float64{
				rand.Float64()*1 + 4.5, // Between 4.5 and 5.5
				rand.Float64()*1 + 4.5,
			}
			fmt.Printf("📤 Sending anomalous point: %v\n", formatFloatSlice(dataPoint, 3))
		} else {
			// Normal data point
			dataPoint = []float64{
				(rand.Float64() - 0.5) * 2, // Between -1 and 1
				(rand.Float64() - 0.5) * 2,
			}
			fmt.Printf("📤 Sending normal point: %v\n", formatFloatSlice(dataPoint, 3))
		}

		if err := s.SendDataPoint(dataPoint); err != nil {
			fmt.Printf("Error sending data: %v\n", err)
		}

		time.Sleep(interval)
	}

	totalPoints, anomalyCount := s.GetStats()
	fmt.Printf("\n✅ Sent %d data points\n", totalPoints)
	fmt.Printf("✅ Detected %d anomalies\n", anomalyCount)
}

// Helper function to format float slices
func formatFloatSlice(slice []float64, precision int) []string {
	result := make([]string, len(slice))
	for i, v := range slice {
		result[i] = fmt.Sprintf("%."+strconv.Itoa(precision)+"f", v)
	}
	return result
}

// interactiveStreamingExample allows user to input data interactively
func interactiveStreamingExample() error {
	fmt.Println("=== Interactive Streaming Example ===")

	stream := NewAnomalyDetectionStream("ws://localhost:8000/ws/stream")
	if err := stream.Start(); err != nil {
		return fmt.Errorf("failed to start streaming: %w", err)
	}
	defer stream.Stop()

	// Wait for connection
	time.Sleep(2 * time.Second)

	if !stream.IsRunning() {
		return fmt.Errorf("failed to connect to streaming service")
	}

	fmt.Println("\nInteractive mode - Enter data points as comma-separated values")
	fmt.Println("Example: 1.5,2.3 or 5.0,5.0 (for anomaly)")
	fmt.Println("Type 'quit' to exit")

	scanner := bufio.NewScanner(os.Stdin)

	for stream.IsRunning() {
		fmt.Print("\nEnter data point (x,y): ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())

		if strings.ToLower(input) == "quit" || strings.ToLower(input) == "exit" || input == "q" {
			break
		}

		// Parse input
		parts := strings.Split(input, ",")
		if len(parts) != 2 {
			fmt.Println("Please enter exactly 2 values separated by comma")
			continue
		}

		var values []float64
		valid := true

		for _, part := range parts {
			val, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
			if err != nil {
				fmt.Println("Invalid input. Please enter numeric values separated by comma")
				valid = false
				break
			}
			values = append(values, val)
		}

		if !valid {
			continue
		}

		if err := stream.SendDataPoint(values); err != nil {
			fmt.Printf("Error sending data: %v\n", err)
		} else {
			fmt.Printf("Sent: %v\n", formatFloatSlice(values, 3))
		}
	}

	totalPoints, anomalyCount := stream.GetStats()
	fmt.Printf("\nSession summary:\n")
	fmt.Printf("Total points sent: %d\n", totalPoints)
	fmt.Printf("Anomalies detected: %d\n", anomalyCount)

	return nil
}

// automatedStreamingExample runs automated data generation
func automatedStreamingExample() error {
	fmt.Println("=== Automated Streaming Example ===")

	stream := NewAnomalyDetectionStream("ws://localhost:8000/ws/stream")
	if err := stream.Start(); err != nil {
		return fmt.Errorf("failed to start streaming: %w", err)
	}
	defer stream.Stop()

	// Wait for connection
	time.Sleep(2 * time.Second)

	if !stream.IsRunning() {
		return fmt.Errorf("failed to connect to streaming service")
	}

	// Run automated data generation
	stream.GenerateAndSendData(20*time.Second, 500*time.Millisecond)

	return nil
}

// batchStreamingExample sends data in batches
func batchStreamingExample() error {
	fmt.Println("=== Batch Streaming Example ===")

	// Generate batch data
	rand.Seed(42)
	var normalBatch [][]float64
	var anomalyBatch [][]float64

	for i := 0; i < 20; i++ {
		point := []float64{
			(rand.Float64() - 0.5) * 2,
			(rand.Float64() - 0.5) * 2,
		}
		normalBatch = append(normalBatch, point)
	}

	for i := 0; i < 5; i++ {
		point := []float64{
			rand.Float64()*1 + 4.5,
			rand.Float64()*1 + 4.5,
		}
		anomalyBatch = append(anomalyBatch, point)
	}

	stream := NewAnomalyDetectionStream("ws://localhost:8000/ws/stream")
	if err := stream.Start(); err != nil {
		return fmt.Errorf("failed to start streaming: %w", err)
	}
	defer stream.Stop()

	// Wait for connection
	time.Sleep(2 * time.Second)

	if !stream.IsRunning() {
		return fmt.Errorf("failed to connect to streaming service")
	}

	// Send normal data batch
	fmt.Println("Sending normal data batch...")
	for i, point := range normalBatch {
		if err := stream.SendDataPoint(point); err != nil {
			fmt.Printf("Error sending data: %v\n", err)
			continue
		}
		fmt.Printf("Sent normal point %d/20: %v\n", i+1, formatFloatSlice(point, 3))
		time.Sleep(200 * time.Millisecond)
	}

	// Send anomalous data batch
	fmt.Println("\nSending anomalous data batch...")
	for i, point := range anomalyBatch {
		if err := stream.SendDataPoint(point); err != nil {
			fmt.Printf("Error sending data: %v\n", err)
			continue
		}
		fmt.Printf("Sent anomaly point %d/5: %v\n", i+1, formatFloatSlice(point, 3))
		time.Sleep(200 * time.Millisecond)
	}

	// Wait for processing
	fmt.Println("\nWaiting for processing to complete...")
	time.Sleep(5 * time.Second)

	return nil
}

// performanceStreamingExample tests streaming performance
func performanceStreamingExample() error {
	fmt.Println("=== Performance Streaming Example ===")

	// Configure for high throughput
	config := anomaly.StreamingClientConfig{
		WSURL: "ws://localhost:8000/ws/stream",
		Config: anomaly.StreamingConfig{
			BufferSize:         100,
			DetectionThreshold: 0.5,
			BatchSize:          20, // Larger batches for efficiency
			Algorithm:          anomaly.IsolationForest,
		},
		AutoReconnect: true,
	}

	client := anomaly.NewStreamingClient(config)

	// Performance tracking
	var (
		anomalyCount int
		startTime    time.Time
		pointsSent   int
		mu           sync.Mutex
	)

	handlers := anomaly.StreamingHandlers{
		OnConnect: func() {
			fmt.Println("✅ Connected - Starting performance test")
			startTime = time.Now()
		},
		OnAnomaly: func(anomalyData anomaly.AnomalyData) {
			mu.Lock()
			anomalyCount++
			currentCount := anomalyCount
			currentPoints := pointsSent
			mu.Unlock()

			if currentCount%10 == 0 {
				elapsed := time.Since(startTime).Seconds()
				rate := float64(currentPoints) / elapsed
				fmt.Printf("📊 Performance: %d anomalies detected, %.1f points/sec\n",
					currentCount, rate)
			}
		},
		OnError: func(err error) {
			fmt.Printf("❌ Error: %v\n", err)
		},
	}

	client.SetHandlers(handlers)

	if err := client.Start(); err != nil {
		return fmt.Errorf("failed to start streaming: %w", err)
	}
	defer client.Stop()

	// Wait for connection
	time.Sleep(2 * time.Second)

	// Generate high volume of data
	rand.Seed(42)
	fmt.Println("Sending high volume data stream (1000 points)...")

	for i := 0; i < 1000; i++ {
		if i%100 == 0 {
			fmt.Printf("Sent %d/1000 points...\n", i)
		}

		// Mix of normal and anomalous data
		var dataPoint []float64
		if rand.Float64() < 0.05 { // 5% anomalies
			dataPoint = []float64{
				rand.Float64()*1 + 3.5,
				rand.Float64()*1 + 3.5,
			}
		} else {
			dataPoint = []float64{
				(rand.Float64() - 0.5) * 2,
				(rand.Float64() - 0.5) * 2,
			}
		}

		if err := client.SendData(dataPoint); err != nil {
			fmt.Printf("Error sending data: %v\n", err)
			continue
		}

		mu.Lock()
		pointsSent++
		mu.Unlock()

		// Small delay to avoid overwhelming
		time.Sleep(10 * time.Millisecond)
	}

	// Wait for final processing
	fmt.Println("Waiting for final processing...")
	time.Sleep(5 * time.Second)

	// Calculate final statistics
	totalTime := time.Since(startTime).Seconds()
	throughput := float64(pointsSent) / totalTime

	fmt.Println("\n📊 Performance Results:")
	fmt.Printf("   Total points: %d\n", pointsSent)
	fmt.Printf("   Total time: %.2f seconds\n", totalTime)
	fmt.Printf("   Throughput: %.1f points/second\n", throughput)
	fmt.Printf("   Anomalies detected: %d\n", anomalyCount)
	fmt.Printf("   Anomaly rate: %.2f%%\n", float64(anomalyCount)/float64(pointsSent)*100)

	return nil
}

// streamingErrorHandlingExample demonstrates error handling
func streamingErrorHandlingExample() error {
	fmt.Println("=== Streaming Error Handling Example ===")

	// Test with invalid URL
	invalidConfig := anomaly.StreamingClientConfig{
		WSURL:         "ws://invalid-url:9999/ws/stream",
		AutoReconnect: false,
	}

	invalidClient := anomaly.NewStreamingClient(invalidConfig)
	errorReceived := false

	handlers := anomaly.StreamingHandlers{
		OnError: func(err error) {
			fmt.Printf("✅ Expected connection error: %v\n", err)
			errorReceived = true
		},
	}

	invalidClient.SetHandlers(handlers)

	if err := invalidClient.Start(); err != nil {
		fmt.Printf("✅ Connection failed as expected: %v\n", err)
	} else {
		// Wait a bit to see if error is received
		time.Sleep(5 * time.Second)
		if !errorReceived {
			fmt.Println("❌ Expected error but none received")
		}
	}

	invalidClient.Stop()

	// Test sending invalid data
	fmt.Println("\nTesting invalid data handling...")
	validConfig := anomaly.StreamingClientConfig{
		WSURL: "ws://localhost:8000/ws/stream",
	}

	validClient := anomaly.NewStreamingClient(validConfig)

	if err := validClient.Start(); err != nil {
		fmt.Printf("Connection error (expected if service is down): %v\n", err)
		return nil
	}
	defer validClient.Stop()

	time.Sleep(2 * time.Second)

	// Try to send invalid data
	fmt.Println("Testing empty data point...")
	if err := validClient.SendData([]float64{}); err != nil {
		fmt.Printf("✅ Empty array validation error caught: %v\n", err)
	} else {
		fmt.Println("❌ Expected validation error but got success")
	}

	return nil
}

func main() {
	fmt.Println("Anomaly Detection Go Streaming Examples")
	fmt.Println(strings.Repeat("=", 50))

	examples := map[string]struct {
		name string
		fn   func() error
	}{
		"1": {"Automated Streaming", automatedStreamingExample},
		"2": {"Interactive Streaming", interactiveStreamingExample},
		"3": {"Batch Streaming", batchStreamingExample},
		"4": {"Performance Test", performanceStreamingExample},
		"5": {"Error Handling", streamingErrorHandlingExample},
	}

	// Simple command line interface
	if len(os.Args) > 1 {
		choice := os.Args[1]
		if choice == "all" {
			// Run all examples
			for i := 1; i <= 5; i++ {
				key := strconv.Itoa(i)
				if example, exists := examples[key]; exists {
					fmt.Printf("\n%s %s %s\n",
						strings.Repeat("=", 20), example.name, strings.Repeat("=", 20))

					if err := example.fn(); err != nil {
						log.Printf("❌ Example failed: %v", err)
					}

					if i < 5 { // Don't wait after last example
						fmt.Println("\nPress Enter to continue to next example...")
						bufio.NewScanner(os.Stdin).Scan()
					}
				}
			}
		} else if example, exists := examples[choice]; exists {
			fmt.Printf("\n%s %s %s\n",
				strings.Repeat("=", 20), example.name, strings.Repeat("=", 20))
			if err := example.fn(); err != nil {
				log.Printf("❌ Example failed: %v", err)
			}
		} else {
			fmt.Printf("Invalid choice '%s'. Running automated example...\n", choice)
			if err := automatedStreamingExample(); err != nil {
				log.Printf("❌ Example failed: %v", err)
			}
		}
	} else {
		// Show menu
		fmt.Println("\nAvailable examples:")
		for key, example := range examples {
			fmt.Printf("  %s. %s\n", key, example.name)
		}
		fmt.Println("\nUsage:")
		fmt.Println("  go run streaming_detection.go [1-5|all]")
		fmt.Println("  Or run without arguments to see this menu")
		fmt.Println("\nRunning automated example...")

		if err := automatedStreamingExample(); err != nil {
			log.Printf("❌ Example failed: %v", err)

			if _, ok := err.(*anomaly.StreamingError); ok {
				fmt.Println("\n💡 Make sure the anomaly detection service is running at ws://localhost:8000/ws/stream")
			}
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("Streaming examples completed!")
}