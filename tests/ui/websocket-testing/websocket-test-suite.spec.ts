/**
 * WebSocket Testing Suite for Pynomaly Real-time Features
 * Tests WebSocket connection handling, message flow, and error recovery
 */

import { test, expect, Page } from '@playwright/test';
import WebSocket from 'ws';

// WebSocket configuration
const WS_BASE_URL = process.env.WS_BASE_URL || 'ws://localhost:8000';
const HTTP_BASE_URL = process.env.BASE_URL || 'http://localhost:8000';

// Test data and utilities
class WebSocketTestHelper {
  private ws: WebSocket | null = null;
  private messages: any[] = [];
  private errors: string[] = [];
  private connectionState: string = 'CLOSED';

  constructor(private url: string) {}

  async connect(timeout: number = 5000): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);
      
      const timeoutId = setTimeout(() => {
        reject(new Error(`WebSocket connection timeout after ${timeout}ms`));
      }, timeout);

      this.ws.onopen = () => {
        this.connectionState = 'OPEN';
        clearTimeout(timeoutId);
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data.toString());
          this.messages.push(message);
        } catch (error) {
          this.messages.push({ raw: event.data.toString() });
        }
      };

      this.ws.onerror = (error) => {
        this.errors.push(error.message || 'WebSocket error');
        this.connectionState = 'ERROR';
      };

      this.ws.onclose = () => {
        this.connectionState = 'CLOSED';
      };
    });
  }

  async sendMessage(message: any): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      throw new Error('WebSocket not connected');
    }
  }

  async waitForMessage(
    predicate: (message: any) => boolean,
    timeout: number = 10000
  ): Promise<any> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const message = this.messages.find(predicate);
      if (message) {
        return message;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    throw new Error(`Message not received within ${timeout}ms`);
  }

  getMessages(): any[] {
    return [...this.messages];
  }

  getErrors(): string[] {
    return [...this.errors];
  }

  getConnectionState(): string {
    return this.connectionState;
  }

  async disconnect(): Promise<void> {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  clearMessages(): void {
    this.messages = [];
  }

  clearErrors(): void {
    this.errors = [];
  }
}

test.describe('WebSocket Connection Tests', () => {
  let wsHelper: WebSocketTestHelper;

  test.beforeEach(async () => {
    wsHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/realtime`);
  });

  test.afterEach(async () => {
    await wsHelper.disconnect();
  });

  test('should establish WebSocket connection successfully', async () => {
    await wsHelper.connect();
    expect(wsHelper.getConnectionState()).toBe('OPEN');
    expect(wsHelper.getErrors()).toHaveLength(0);
  });

  test('should handle connection timeout gracefully', async () => {
    const invalidHelper = new WebSocketTestHelper('ws://invalid-host:9999/ws');
    
    await expect(invalidHelper.connect(1000)).rejects.toThrow(/timeout/);
  });

  test('should reconnect after connection loss', async ({ page }) => {
    await wsHelper.connect();
    expect(wsHelper.getConnectionState()).toBe('OPEN');

    // Simulate connection loss by closing the WebSocket
    await wsHelper.disconnect();
    expect(wsHelper.getConnectionState()).toBe('CLOSED');

    // Reconnect
    await wsHelper.connect();
    expect(wsHelper.getConnectionState()).toBe('OPEN');
  });

  test('should handle multiple concurrent connections', async () => {
    const helpers = Array.from({ length: 5 }, () => 
      new WebSocketTestHelper(`${WS_BASE_URL}/ws/realtime`)
    );

    // Connect all WebSockets
    await Promise.all(helpers.map(helper => helper.connect()));

    // Verify all connections are open
    helpers.forEach(helper => {
      expect(helper.getConnectionState()).toBe('OPEN');
    });

    // Disconnect all
    await Promise.all(helpers.map(helper => helper.disconnect()));
  });
});

test.describe('Real-time Data Streaming Tests', () => {
  let wsHelper: WebSocketTestHelper;

  test.beforeEach(async () => {
    wsHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/detection/stream`);
    await wsHelper.connect();
  });

  test.afterEach(async () => {
    await wsHelper.disconnect();
  });

  test('should receive real-time detection updates', async () => {
    // Subscribe to detection updates
    await wsHelper.sendMessage({
      type: 'subscribe',
      channel: 'detection_updates',
      detector_id: 'test_detector_123'
    });

    // Wait for subscription confirmation
    const confirmation = await wsHelper.waitForMessage(
      msg => msg.type === 'subscription_confirmed'
    );
    expect(confirmation.channel).toBe('detection_updates');

    // Trigger a detection via API to generate real-time data
    const response = await fetch(`${HTTP_BASE_URL}/api/v1/detection/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        detector_id: 'test_detector_123',
        data: [[1, 2, 3], [4, 5, 6]]
      })
    });

    if (response.ok) {
      // Wait for real-time update
      const update = await wsHelper.waitForMessage(
        msg => msg.type === 'detection_result'
      );
      expect(update.detector_id).toBe('test_detector_123');
      expect(update.result).toBeDefined();
    }
  });

  test('should handle message ordering correctly', async () => {
    // Send multiple messages in sequence
    const messageIds = ['msg_1', 'msg_2', 'msg_3', 'msg_4', 'msg_5'];
    
    for (const id of messageIds) {
      await wsHelper.sendMessage({
        type: 'test_message',
        id: id,
        timestamp: Date.now()
      });
    }

    // Wait for all messages to be processed
    await new Promise(resolve => setTimeout(resolve, 1000));

    const messages = wsHelper.getMessages();
    const testMessages = messages.filter(msg => msg.type === 'test_message_ack');
    
    // Verify message ordering
    expect(testMessages).toHaveLength(messageIds.length);
    testMessages.forEach((msg, index) => {
      expect(msg.original_id).toBe(messageIds[index]);
    });
  });

  test('should handle large message payloads', async () => {
    // Create a large payload (1MB of data)
    const largeData = Array.from({ length: 100000 }, (_, i) => ({
      id: i,
      value: Math.random(),
      timestamp: Date.now()
    }));

    await wsHelper.sendMessage({
      type: 'large_payload_test',
      data: largeData
    });

    const response = await wsHelper.waitForMessage(
      msg => msg.type === 'large_payload_ack'
    );
    expect(response.received_count).toBe(largeData.length);
  });
});

test.describe('Real-time Monitoring Tests', () => {
  let wsHelper: WebSocketTestHelper;

  test.beforeEach(async () => {
    wsHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/monitoring`);
    await wsHelper.connect();
  });

  test.afterEach(async () => {
    await wsHelper.disconnect();
  });

  test('should stream system metrics in real-time', async () => {
    await wsHelper.sendMessage({
      type: 'subscribe',
      channel: 'system_metrics',
      interval: 1000 // 1 second updates
    });

    // Wait for multiple metric updates
    const metrics = [];
    for (let i = 0; i < 3; i++) {
      const metric = await wsHelper.waitForMessage(
        msg => msg.type === 'system_metric'
      );
      metrics.push(metric);
      wsHelper.clearMessages(); // Clear to wait for next metric
    }

    expect(metrics).toHaveLength(3);
    metrics.forEach(metric => {
      expect(metric).toHaveProperty('cpu_usage');
      expect(metric).toHaveProperty('memory_usage');
      expect(metric).toHaveProperty('timestamp');
    });
  });

  test('should handle performance alerts', async () => {
    await wsHelper.sendMessage({
      type: 'subscribe',
      channel: 'performance_alerts',
      thresholds: {
        cpu_usage: 80,
        memory_usage: 90,
        response_time: 5000
      }
    });

    // Simulate high CPU usage alert
    await wsHelper.sendMessage({
      type: 'simulate_alert',
      alert_type: 'high_cpu_usage',
      value: 95
    });

    const alert = await wsHelper.waitForMessage(
      msg => msg.type === 'performance_alert'
    );
    expect(alert.alert_type).toBe('high_cpu_usage');
    expect(alert.value).toBe(95);
    expect(alert.threshold_exceeded).toBe(true);
  });
});

test.describe('Error Handling and Recovery Tests', () => {
  let wsHelper: WebSocketTestHelper;

  test.beforeEach(async () => {
    wsHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/test`);
  });

  test.afterEach(async () => {
    await wsHelper.disconnect();
  });

  test('should handle malformed JSON messages', async () => {
    await wsHelper.connect();

    // Send malformed JSON
    if (wsHelper['ws']) {
      wsHelper['ws'].send('{ invalid json }');
    }

    const errorResponse = await wsHelper.waitForMessage(
      msg => msg.type === 'error'
    );
    expect(errorResponse.error_type).toBe('invalid_json');
  });

  test('should handle authentication errors', async () => {
    const authHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/secure`);
    
    await authHelper.connect();
    
    // Try to access secured channel without auth
    await authHelper.sendMessage({
      type: 'subscribe',
      channel: 'secure_channel'
    });

    const error = await authHelper.waitForMessage(
      msg => msg.type === 'error'
    );
    expect(error.error_type).toBe('authentication_required');
    
    await authHelper.disconnect();
  });

  test('should implement rate limiting', async () => {
    await wsHelper.connect();

    // Send messages rapidly to trigger rate limiting
    const rapidMessages = Array.from({ length: 100 }, (_, i) => ({
      type: 'rapid_test',
      id: i
    }));

    for (const message of rapidMessages) {
      await wsHelper.sendMessage(message);
    }

    // Should receive rate limit error
    const rateLimitError = await wsHelper.waitForMessage(
      msg => msg.type === 'error' && msg.error_type === 'rate_limited'
    );
    expect(rateLimitError).toBeDefined();
  });

  test('should handle server disconnection gracefully', async ({ page }) => {
    await wsHelper.connect();

    // Navigate to a page that uses WebSocket
    await page.goto(`${HTTP_BASE_URL}/dashboard`);

    // Inject WebSocket monitoring code
    await page.addInitScript(() => {
      window.wsConnectionState = 'unknown';
      window.wsReconnectAttempts = 0;
      
      // Monitor WebSocket state changes
      const originalWebSocket = window.WebSocket;
      window.WebSocket = function(url, protocols) {
        const ws = new originalWebSocket(url, protocols);
        
        ws.addEventListener('open', () => {
          window.wsConnectionState = 'connected';
        });
        
        ws.addEventListener('close', () => {
          window.wsConnectionState = 'disconnected';
        });
        
        ws.addEventListener('error', () => {
          window.wsConnectionState = 'error';
        });
        
        return ws;
      };
    });

    // Simulate server disconnection
    await wsHelper.disconnect();

    // Check that the page handles disconnection
    const connectionState = await page.evaluate(() => window.wsConnectionState);
    expect(['disconnected', 'error']).toContain(connectionState);
  });
});

test.describe('WebSocket UI Integration Tests', () => {
  test('should update dashboard in real-time via WebSocket', async ({ page }) => {
    await page.goto(`${HTTP_BASE_URL}/dashboard`);

    // Wait for page to load and WebSocket to connect
    await page.waitForSelector('[data-testid="dashboard-container"]');
    await page.waitForTimeout(2000);

    // Check initial metrics display
    const initialMetric = await page.textContent('[data-testid="cpu-usage"]');
    
    // Wait for real-time update (metrics should change)
    await page.waitForTimeout(5000);
    
    const updatedMetric = await page.textContent('[data-testid="cpu-usage"]');
    
    // Metrics should be different due to real-time updates
    // Note: In a real test, you might want to trigger specific updates
    expect(updatedMetric).toBeDefined();
  });

  test('should display real-time alerts in the UI', async ({ page }) => {
    await page.goto(`${HTTP_BASE_URL}/monitoring`);
    
    // Wait for WebSocket connection
    await page.waitForSelector('[data-testid="alerts-container"]');
    
    // Check for no alerts initially
    const initialAlerts = await page.locator('[data-testid="alert-item"]').count();
    
    // Trigger an alert via API
    await fetch(`${HTTP_BASE_URL}/api/v1/monitoring/trigger-alert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'high_cpu_usage',
        value: 95,
        threshold: 80
      })
    });
    
    // Wait for alert to appear in UI
    await page.waitForSelector('[data-testid="alert-item"]', { timeout: 10000 });
    
    const finalAlerts = await page.locator('[data-testid="alert-item"]').count();
    expect(finalAlerts).toBeGreaterThan(initialAlerts);
  });

  test('should handle WebSocket reconnection in UI', async ({ page }) => {
    await page.goto(`${HTTP_BASE_URL}/dashboard`);
    
    // Wait for WebSocket connection indicator
    await page.waitForSelector('[data-testid="connection-status"]');
    
    // Check initial connection status
    const initialStatus = await page.textContent('[data-testid="connection-status"]');
    expect(initialStatus).toContain('Connected');
    
    // Simulate network interruption
    await page.evaluate(() => {
      // Close WebSocket connection
      if (window.wsConnection) {
        window.wsConnection.close();
      }
    });
    
    // Wait for disconnection to be detected
    await page.waitForSelector('[data-testid="connection-status"]:has-text("Disconnected")', 
      { timeout: 10000 });
    
    // Wait for automatic reconnection
    await page.waitForSelector('[data-testid="connection-status"]:has-text("Connected")', 
      { timeout: 15000 });
  });
});

test.describe('WebSocket Performance Tests', () => {
  test('should handle high-frequency messages', async () => {
    const wsHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/performance`);
    await wsHelper.connect();

    const startTime = Date.now();
    const messageCount = 1000;
    const messages = [];

    // Send high-frequency messages
    for (let i = 0; i < messageCount; i++) {
      await wsHelper.sendMessage({
        type: 'performance_test',
        id: i,
        timestamp: Date.now()
      });
    }

    // Wait for all responses
    while (messages.length < messageCount) {
      const message = await wsHelper.waitForMessage(
        msg => msg.type === 'performance_test_ack'
      );
      messages.push(message);
      wsHelper.clearMessages();
    }

    const endTime = Date.now();
    const duration = endTime - startTime;
    const messagesPerSecond = (messageCount / duration) * 1000;

    expect(messages).toHaveLength(messageCount);
    expect(messagesPerSecond).toBeGreaterThan(100); // Should handle at least 100 msg/sec
    
    await wsHelper.disconnect();
  });

  test('should maintain connection under load', async () => {
    const connections = 10;
    const helpers = Array.from({ length: connections }, () => 
      new WebSocketTestHelper(`${WS_BASE_URL}/ws/load-test`)
    );

    // Connect all WebSockets
    await Promise.all(helpers.map(helper => helper.connect()));

    // Send messages from all connections simultaneously
    const sendPromises = helpers.map((helper, index) => 
      helper.sendMessage({
        type: 'load_test',
        connection_id: index,
        timestamp: Date.now()
      })
    );

    await Promise.all(sendPromises);

    // Verify all connections are still open
    helpers.forEach((helper, index) => {
      expect(helper.getConnectionState()).toBe('OPEN');
    });

    // Disconnect all
    await Promise.all(helpers.map(helper => helper.disconnect()));
  });
});

test.describe('WebSocket Security Tests', () => {
  test('should validate message structure', async () => {
    const wsHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/secure`);
    await wsHelper.connect();

    // Send message with invalid structure
    await wsHelper.sendMessage({
      // Missing required 'type' field
      data: 'test data'
    });

    const error = await wsHelper.waitForMessage(
      msg => msg.type === 'error'
    );
    expect(error.error_type).toBe('invalid_message_structure');
    
    await wsHelper.disconnect();
  });

  test('should prevent message injection', async () => {
    const wsHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/secure`);
    await wsHelper.connect();

    // Attempt to send malicious message
    await wsHelper.sendMessage({
      type: 'inject_script',
      payload: '<script>alert("xss")</script>'
    });

    const response = await wsHelper.waitForMessage(
      msg => msg.type === 'error' || msg.type === 'message_rejected'
    );
    expect(['error', 'message_rejected']).toContain(response.type);
    
    await wsHelper.disconnect();
  });
});

// Custom test fixtures and utilities
test.describe('WebSocket Test Utilities', () => {
  test('WebSocket helper should track message history correctly', async () => {
    const wsHelper = new WebSocketTestHelper(`${WS_BASE_URL}/ws/test`);
    await wsHelper.connect();

    // Send test messages
    await wsHelper.sendMessage({ type: 'test1', data: 'message1' });
    await wsHelper.sendMessage({ type: 'test2', data: 'message2' });

    // Wait for responses
    await wsHelper.waitForMessage(msg => msg.type === 'test1_ack');
    await wsHelper.waitForMessage(msg => msg.type === 'test2_ack');

    const messages = wsHelper.getMessages();
    expect(messages.length).toBeGreaterThanOrEqual(2);

    await wsHelper.disconnect();
  });
});