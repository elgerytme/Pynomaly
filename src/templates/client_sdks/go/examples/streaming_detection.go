package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	anomaly_detection "github.com/monorepo/anomaly-detection-client-go"
)

func main() {
	// Configure the client with streaming enabled
	config := &anomaly_detection.ClientConfig{
		BaseURL:         "https://api.anomaly-detection.com",
		APIKey:          "your-api-key-here",
		Timeout:         30 * time.Second,
		MaxRetries:      3,
		EnableStreaming: true,
		StreamingURL:    "wss://api.anomaly-detection.com/ws",
		Debug:           true,
	}

	// Create client
	client := anomaly_detection.NewClient(config)
	defer client.Close()

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Authenticate with API key
	if err := client.AuthenticateWithAPIKey(ctx, config.APIKey); err != nil {
		log.Fatalf("Authentication failed: %v", err)
	}

	fmt.Println("=== Real-time Streaming Anomaly Detection ===")

	// Start streaming connection
	msgChan, err := client.StartStreaming(ctx)
	if err != nil {
		log.Fatalf("Failed to start streaming: %v", err)
	}

	// Start message handler goroutine
	go handleStreamingMessages(msgChan)

	// Simulate real-time data streaming
	simulateDataStream(ctx, client)

	// Wait a bit for final messages
	time.Sleep(2 * time.Second)

	// Stop streaming
	if err := client.StopStreaming(); err != nil {
		log.Printf("Error stopping streaming: %v", err)
	}

	fmt.Println("Streaming demo completed!")
}

// handleStreamingMessages processes incoming streaming messages
func handleStreamingMessages(msgChan <-chan anomaly_detection.StreamingMessage) {
	for msg := range msgChan {
		switch msg.Type {
		case "anomaly":
			if msg.Anomaly {
				fmt.Printf("🚨 ANOMALY DETECTED! Confidence: %.3f, Data: %v\n",
					msg.Confidence, msg.Data)
			} else {
				fmt.Printf("✅ Normal data point: %v (confidence: %.3f)\n",
					msg.Data, msg.Confidence)
			}

		case "status":
			fmt.Printf("📊 Status: %s\n", msg.Message)
			if msg.Metadata != nil {
				for key, value := range msg.Metadata {
					fmt.Printf("   %s: %v\n", key, value)
				}
			}

		case "error":
			fmt.Printf("❌ Error: %s\n", msg.Message)

		default:
			fmt.Printf("📝 Message [%s]: %s\n", msg.Type, msg.Message)
		}
	}
}

// simulateDataStream generates and sends sample data points
func simulateDataStream(ctx context.Context, client *anomaly_detection.Client) {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("Starting data stream simulation...")
	fmt.Println("Sending normal data points with occasional anomalies...")

	for i := 0; i < 50; i++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		var dataPoint []float64

		// Generate mostly normal data with some anomalies
		if i%15 == 0 && i > 0 {
			// Generate anomaly every 15 points
			dataPoint = generateAnomalyDataPoint()
			fmt.Printf("Sending anomaly data point %d: %v\n", i+1, dataPoint)
		} else {
			// Generate normal data point
			dataPoint = generateNormalDataPoint()
			fmt.Printf("Sending normal data point %d: %v\n", i+1, dataPoint)
		}

		// Send data point
		if err := client.SendStreamingData(ctx, dataPoint); err != nil {
			log.Printf("Failed to send data point: %v", err)
			continue
		}

		// Wait between data points
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Println("Data stream simulation completed.")
}

// generateNormalDataPoint creates a normal data point
func generateNormalDataPoint() []float64 {
	// Generate 3D normal data around (0, 0, 0) with small variance
	return []float64{
		rand.NormFloat64() * 0.5,
		rand.NormFloat64() * 0.5,
		rand.NormFloat64() * 0.5,
	}
}

// generateAnomalyDataPoint creates an anomalous data point
func generateAnomalyDataPoint() []float64 {
	// Generate anomalous data that's far from normal distribution
	anomalyType := rand.Intn(3)
	
	switch anomalyType {
	case 0:
		// High magnitude anomaly
		return []float64{
			rand.Float64()*10 + 5,
			rand.Float64()*10 + 5,
			rand.Float64()*10 + 5,
		}
	case 1:
		// Negative anomaly
		return []float64{
			-rand.Float64()*8 - 3,
			-rand.Float64()*8 - 3,
			-rand.Float64()*8 - 3,
		}
	default:
		// Spike anomaly in one dimension
		return []float64{
			rand.NormFloat64() * 0.5,
			rand.Float64()*15 + 8, // Spike
			rand.NormFloat64() * 0.5,
		}
	}
}

// Example showing advanced streaming with custom configuration
func advancedStreamingExample() {
	config := &anomaly_detection.ClientConfig{
		BaseURL:         "https://api.anomaly-detection.com",
		APIKey:          "your-api-key-here",
		Timeout:         30 * time.Second,
		MaxRetries:      3,
		EnableStreaming: true,
		StreamingURL:    "wss://api.anomaly-detection.com/ws",
		RateLimit:       200, // Higher rate limit for streaming
		Debug:           false,
	}

	client := anomaly_detection.NewClient(config)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Authenticate
	if err := client.AuthenticateWithAPIKey(ctx, config.APIKey); err != nil {
		log.Fatalf("Authentication failed: %v", err)
	}

	// Start streaming
	msgChan, err := client.StartStreaming(ctx)
	if err != nil {
		log.Fatalf("Failed to start streaming: %v", err)
	}

	// Track statistics
	stats := &StreamingStats{
		StartTime: time.Now(),
	}

	// Handle messages with statistics tracking
	go handleMessagesWithStats(msgChan, stats)

	// Simulate high-frequency sensor data
	simulateHighFrequencyStream(ctx, client, stats)

	// Print final statistics
	stats.PrintSummary()
}

// StreamingStats tracks streaming statistics
type StreamingStats struct {
	StartTime       time.Time
	DataPointsSent  int
	AnomaliesFound  int
	ErrorsReceived  int
	TotalLatency    time.Duration
	mu              sync.Mutex
}

func (s *StreamingStats) IncrementDataPoints() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.DataPointsSent++
}

func (s *StreamingStats) IncrementAnomalies() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.AnomaliesFound++
}

func (s *StreamingStats) IncrementErrors() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ErrorsReceived++
}

func (s *StreamingStats) AddLatency(latency time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.TotalLatency += latency
}

func (s *StreamingStats) PrintSummary() {
	s.mu.Lock()
	defer s.mu.Unlock()

	duration := time.Since(s.StartTime)
	
	fmt.Println("\n=== Streaming Session Summary ===")
	fmt.Printf("Duration: %v\n", duration)
	fmt.Printf("Data points sent: %d\n", s.DataPointsSent)
	fmt.Printf("Anomalies detected: %d\n", s.AnomaliesFound)
	fmt.Printf("Errors received: %d\n", s.ErrorsReceived)
	
	if s.DataPointsSent > 0 {
		fmt.Printf("Anomaly rate: %.2f%%\n", 
			float64(s.AnomaliesFound)/float64(s.DataPointsSent)*100)
		fmt.Printf("Average throughput: %.1f points/sec\n",
			float64(s.DataPointsSent)/duration.Seconds())
	}
	
	if s.AnomaliesFound > 0 {
		fmt.Printf("Average detection latency: %v\n",
			s.TotalLatency/time.Duration(s.AnomaliesFound))
	}
}

// handleMessagesWithStats handles messages and updates statistics
func handleMessagesWithStats(msgChan <-chan anomaly_detection.StreamingMessage, stats *StreamingStats) {
	for msg := range msgChan {
		switch msg.Type {
		case "anomaly":
			if msg.Anomaly {
				stats.IncrementAnomalies()
				// Calculate latency (simplified - in real use, you'd track send time)
				latency := time.Since(msg.Timestamp)
				stats.AddLatency(latency)
				
				fmt.Printf("🚨 ANOMALY #%d - Confidence: %.3f, Latency: %v\n",
					stats.AnomaliesFound, msg.Confidence, latency)
			}

		case "error":
			stats.IncrementErrors()
			fmt.Printf("❌ Error: %s\n", msg.Message)

		case "status":
			if msg.Message == "processing_complete" {
				fmt.Printf("✅ Batch processed\n")
			}
		}
	}
}

// simulateHighFrequencyStream simulates high-frequency sensor data
func simulateHighFrequencyStream(ctx context.Context, client *anomaly_detection.Client, stats *StreamingStats) {
	ticker := time.NewTicker(100 * time.Millisecond) // 10 Hz
	defer ticker.Stop()

	sensorData := &SensorSimulator{
		Temperature: 20.0,
		Pressure:    1013.25,
		Humidity:    50.0,
	}

	for i := 0; i < 200; i++ {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		// Generate sensor reading
		dataPoint := sensorData.NextReading(i)
		
		// Send data point
		if err := client.SendStreamingData(ctx, dataPoint); err != nil {
			log.Printf("Failed to send data point: %v", err)
			continue
		}

		stats.IncrementDataPoints()

		// Occasionally print progress
		if i%20 == 0 {
			fmt.Printf("Sent %d data points...\n", i+1)
		}
	}
}

// SensorSimulator simulates realistic sensor data
type SensorSimulator struct {
	Temperature float64
	Pressure    float64
	Humidity    float64
}

func (s *SensorSimulator) NextReading(step int) []float64 {
	// Add some drift and noise
	s.Temperature += rand.NormFloat64() * 0.1
	s.Pressure += rand.NormFloat64() * 0.5
	s.Humidity += rand.NormFloat64() * 1.0

	// Occasionally inject anomalies
	if step%50 == 0 && step > 0 {
		// Temperature spike
		s.Temperature += rand.Float64()*10 + 15
	}
	
	if step%73 == 0 && step > 0 {
		// Pressure drop
		s.Pressure -= rand.Float64()*50 + 30
	}

	// Add periodic patterns
	timeComponent := float64(step) * 0.1
	s.Temperature += math.Sin(timeComponent) * 2
	s.Pressure += math.Cos(timeComponent*0.5) * 5

	// Keep values in reasonable ranges
	s.Temperature = math.Max(-10, math.Min(50, s.Temperature))
	s.Pressure = math.Max(900, math.Min(1100, s.Pressure))
	s.Humidity = math.Max(0, math.Min(100, s.Humidity))

	return []float64{s.Temperature, s.Pressure, s.Humidity}
}