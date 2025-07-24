package anomalydetection

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// StreamingClient provides WebSocket-based real-time anomaly detection
type StreamingClient struct {
	config     StreamingClientConfig
	conn       *websocket.Conn
	connected  bool
	running    bool
	buffer     [][]float64
	bufferMu   sync.Mutex
	handlers   *StreamingHandlers
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// StreamingHandlers contains event handlers for the streaming client
type StreamingHandlers struct {
	OnConnect    func()
	OnDisconnect func()
	OnAnomaly    func(AnomalyData)
	OnError      func(error)
	OnMessage    func(map[string]interface{})
}

// NewStreamingClient creates a new streaming client
func NewStreamingClient(config StreamingClientConfig) *StreamingClient {
	// Set defaults
	if config.Config.BufferSize == 0 {
		config.Config.BufferSize = 100
	}
	if config.Config.DetectionThreshold == 0 {
		config.Config.DetectionThreshold = 0.5
	}
	if config.Config.BatchSize == 0 {
		config.Config.BatchSize = 10
	}
	if config.Config.Algorithm == "" {
		config.Config.Algorithm = IsolationForest
	}
	if config.ReconnectDelay == 0 {
		config.ReconnectDelay = 5 * time.Second
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &StreamingClient{
		config:   config,
		handlers: &StreamingHandlers{},
		ctx:      ctx,
		cancel:   cancel,
	}
}

// SetHandlers sets the event handlers
func (sc *StreamingClient) SetHandlers(handlers StreamingHandlers) {
	sc.handlers = &handlers
}

// Start starts the streaming client
func (sc *StreamingClient) Start() error {
	if sc.running {
		return nil
	}

	sc.running = true
	sc.wg.Add(1)
	go sc.connectAndListen()
	
	return nil
}

// Stop stops the streaming client
func (sc *StreamingClient) Stop() {
	if !sc.running {
		return
	}

	sc.running = false
	sc.cancel()
	
	if sc.conn != nil {
		sc.conn.Close()
	}
	
	sc.wg.Wait()
}

// SendData sends a single data point for anomaly detection
func (sc *StreamingClient) SendData(dataPoint []float64) error {
	if len(dataPoint) == 0 {
		return &ValidationError{
			SDKError: &SDKError{
				Message: "Data point cannot be empty",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("dataPoint"),
			Value: dataPoint,
		}
	}

	if !sc.running {
		return &StreamingError{
			SDKError: &SDKError{
				Message: "Client is not running",
				Code:    stringPtr("STREAMING_ERROR"),
			},
		}
	}

	sc.bufferMu.Lock()
	defer sc.bufferMu.Unlock()

	sc.buffer = append(sc.buffer, dataPoint)

	// Process batch when buffer is full
	if len(sc.buffer) >= sc.config.Config.BatchSize {
		batch := make([][]float64, len(sc.buffer))
		copy(batch, sc.buffer)
		sc.buffer = sc.buffer[:0] // Clear buffer
		
		go sc.sendBatch(batch)
	}

	return nil
}

// SendBatch sends multiple data points at once
func (sc *StreamingClient) SendBatch(batch [][]float64) error {
	if !sc.running {
		return &StreamingError{
			SDKError: &SDKError{
				Message: "Client is not running",
				Code:    stringPtr("STREAMING_ERROR"),
			},
		}
	}

	go sc.sendBatch(batch)
	return nil
}

// IsConnected returns true if the client is connected
func (sc *StreamingClient) IsConnected() bool {
	return sc.connected
}

// BufferSize returns the current buffer size
func (sc *StreamingClient) BufferSize() int {
	sc.bufferMu.Lock()
	defer sc.bufferMu.Unlock()
	return len(sc.buffer)
}

func (sc *StreamingClient) connectAndListen() {
	defer sc.wg.Done()

	for sc.running {
		select {
		case <-sc.ctx.Done():
			return
		default:
		}

		if err := sc.connect(); err != nil {
			sc.handleError(err)
			if sc.config.AutoReconnect && sc.running {
				select {
				case <-sc.ctx.Done():
					return
				case <-time.After(sc.config.ReconnectDelay):
					continue
				}
			} else {
				return
			}
		}

		sc.listen()
	}
}

func (sc *StreamingClient) connect() error {
	// Parse WebSocket URL
	u, err := url.Parse(sc.config.WSURL)
	if err != nil {
		return &ConnectionError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Invalid WebSocket URL: %v", err),
				Code:    stringPtr("CONNECTION_ERROR"),
			},
			URL: &sc.config.WSURL,
		}
	}

	// Add API key to query parameters if provided
	if sc.config.APIKey != nil {
		q := u.Query()
		q.Set("token", *sc.config.APIKey)
		u.RawQuery = q.Encode()
	}

	// Set up headers
	headers := http.Header{}
	if sc.config.APIKey != nil {
		headers.Set("Authorization", "Bearer "+*sc.config.APIKey)
	}

	// Connect to WebSocket
	dialer := websocket.DefaultDialer
	dialer.HandshakeTimeout = 30 * time.Second

	conn, _, err := dialer.Dial(u.String(), headers)
	if err != nil {
		return &ConnectionError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to connect to WebSocket: %v", err),
				Code:    stringPtr("CONNECTION_ERROR"),
			},
			URL: &sc.config.WSURL,
		}
	}

	sc.conn = conn
	sc.connected = true

	// Send initial configuration
	configMessage := map[string]interface{}{
		"type":   "config",
		"config": sc.config.Config,
	}

	if err := sc.conn.WriteJSON(configMessage); err != nil {
		sc.conn.Close()
		sc.connected = false
		return &StreamingError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to send configuration: %v", err),
				Code:    stringPtr("STREAMING_ERROR"),
			},
		}
	}

	// Send any buffered data
	sc.bufferMu.Lock()
	if len(sc.buffer) > 0 {
		batch := make([][]float64, len(sc.buffer))
		copy(batch, sc.buffer)
		sc.buffer = sc.buffer[:0]
		sc.bufferMu.Unlock()
		
		go sc.sendBatch(batch)
	} else {
		sc.bufferMu.Unlock()
	}

	sc.handleConnect()
	return nil
}

func (sc *StreamingClient) listen() {
	defer func() {
		sc.connected = false
		if sc.conn != nil {
			sc.conn.Close()
			sc.conn = nil
		}
		sc.handleDisconnect()
	}()

	for sc.running && sc.connected {
		select {
		case <-sc.ctx.Done():
			return
		default:
		}

		var message map[string]interface{}
		if err := sc.conn.ReadJSON(&message); err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				return
			}
			sc.handleError(&StreamingError{
				SDKError: &SDKError{
					Message: fmt.Sprintf("WebSocket read error: %v", err),
					Code:    stringPtr("STREAMING_ERROR"),
				},
			})
			return
		}

		sc.handleMessage(message)
	}
}

func (sc *StreamingClient) handleMessage(message map[string]interface{}) {
	messageType, ok := message["type"].(string)
	if !ok {
		sc.handleError(&StreamingError{
			SDKError: &SDKError{
				Message: "Invalid message type",
				Code:    stringPtr("STREAMING_ERROR"),
			},
		})
		return
	}

	switch messageType {
	case "anomaly":
		sc.handleAnomalyMessage(message)
	case "error":
		if errMsg, ok := message["message"].(string); ok {
			sc.handleError(&StreamingError{
				SDKError: &SDKError{
					Message: errMsg,
					Code:    stringPtr("STREAMING_ERROR"),
				},
			})
		}
	case "ping":
		// Respond to ping with pong
		pongMessage := map[string]interface{}{"type": "pong"}
		if sc.conn != nil {
			sc.conn.WriteJSON(pongMessage)
		}
	default:
		if sc.handlers.OnMessage != nil {
			sc.handlers.OnMessage(message)
		}
	}
}

func (sc *StreamingClient) handleAnomalyMessage(message map[string]interface{}) {
	data, ok := message["data"]
	if !ok {
		sc.handleError(&StreamingError{
			SDKError: &SDKError{
				Message: "Anomaly message missing data",
				Code:    stringPtr("STREAMING_ERROR"),
			},
		})
		return
	}

	// Convert to JSON and back to properly unmarshal
	jsonData, err := json.Marshal(data)
	if err != nil {
		sc.handleError(&StreamingError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to marshal anomaly data: %v", err),
				Code:    stringPtr("STREAMING_ERROR"),
			},
		})
		return
	}

	var anomaly AnomalyData
	if err := json.Unmarshal(jsonData, &anomaly); err != nil {
		sc.handleError(&StreamingError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal anomaly data: %v", err),
				Code:    stringPtr("STREAMING_ERROR"),
			},
		})
		return
	}

	if sc.handlers.OnAnomaly != nil {
		sc.handlers.OnAnomaly(anomaly)
	}
}

func (sc *StreamingClient) sendBatch(batch [][]float64) {
	if !sc.connected || sc.conn == nil {
		// Buffer the data for later sending
		sc.bufferMu.Lock()
		sc.buffer = append(batch, sc.buffer...)
		sc.bufferMu.Unlock()
		return
	}

	message := map[string]interface{}{
		"type":      "batch",
		"data":      batch,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}

	if err := sc.conn.WriteJSON(message); err != nil {
		sc.handleError(&StreamingError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to send batch: %v", err),
				Code:    stringPtr("STREAMING_ERROR"),
			},
		})
	}
}

func (sc *StreamingClient) handleConnect() {
	if sc.handlers.OnConnect != nil {
		sc.handlers.OnConnect()
	}
}

func (sc *StreamingClient) handleDisconnect() {
	if sc.handlers.OnDisconnect != nil {
		sc.handlers.OnDisconnect()
	}
}

func (sc *StreamingClient) handleError(err error) {
	if sc.handlers.OnError != nil {
		sc.handlers.OnError(err)
	}
}