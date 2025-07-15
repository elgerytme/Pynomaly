/**
 * React hook for Pynomaly WebSocket connection
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { PynomalyWebSocket, WebSocketConfig, JobStatus, WebSocketMessage } from '../../../index';

export interface UsePynomalyWebSocketOptions extends Partial<WebSocketConfig> {
  autoConnect?: boolean;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Error) => void;
  onMessage?: (message: WebSocketMessage) => void;
  onJobStatus?: (status: JobStatus) => void;
}

export interface UsePynomalyWebSocketReturn {
  ws: PynomalyWebSocket | null;
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  connect: () => Promise<void>;
  disconnect: () => void;
  subscribeToJob: (jobId: string) => void;
  unsubscribeFromJob: (jobId: string) => void;
  subscribeToNotifications: (userId: string) => void;
  unsubscribeFromNotifications: (userId: string) => void;
  send: (data: any) => void;
  connectionState: string;
  queuedMessageCount: number;
}

export function usePynomalyWebSocket(
  options: UsePynomalyWebSocketOptions
): UsePynomalyWebSocketReturn {
  const [ws, setWs] = useState<PynomalyWebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [connectionState, setConnectionState] = useState('DISCONNECTED');
  const [queuedMessageCount, setQueuedMessageCount] = useState(0);
  const optionsRef = useRef(options);
  const mountedRef = useRef(true);

  // Update options ref when options change
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (ws) {
        ws.destroy();
      }
    };
  }, [ws]);

  const createWebSocket = useCallback(() => {
    const wsInstance = new PynomalyWebSocket({
      url: 'wss://api.pynomaly.com/ws',
      ...optionsRef.current
    });

    // Set up event listeners
    wsInstance.on('connection:open', () => {
      if (mountedRef.current) {
        setIsConnected(true);
        setIsConnecting(false);
        setError(null);
        setConnectionState('OPEN');
        optionsRef.current.onOpen?.();
      }
    });

    wsInstance.on('connection:close', () => {
      if (mountedRef.current) {
        setIsConnected(false);
        setIsConnecting(false);
        setConnectionState('CLOSED');
        optionsRef.current.onClose?.();
      }
    });

    wsInstance.on('connection:error', (error: Error) => {
      if (mountedRef.current) {
        setError(error);
        setIsConnecting(false);
        setConnectionState('ERROR');
        optionsRef.current.onError?.(error);
      }
    });

    wsInstance.on('message', (message: WebSocketMessage) => {
      if (mountedRef.current) {
        optionsRef.current.onMessage?.(message);
      }
    });

    wsInstance.on('job:status', (status: JobStatus) => {
      if (mountedRef.current) {
        optionsRef.current.onJobStatus?.(status);
      }
    });

    return wsInstance;
  }, []);

  const connect = useCallback(async () => {
    if (isConnecting || isConnected) {
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      const wsInstance = createWebSocket();
      await wsInstance.connect();
      
      if (mountedRef.current) {
        setWs(wsInstance);
      }
    } catch (err) {
      const error = err as Error;
      if (mountedRef.current) {
        setError(error);
        setIsConnecting(false);
      }
      throw error;
    }
  }, [isConnecting, isConnected, createWebSocket]);

  const disconnect = useCallback(() => {
    if (ws) {
      ws.disconnect();
      setWs(null);
      setIsConnected(false);
      setIsConnecting(false);
      setConnectionState('DISCONNECTED');
    }
  }, [ws]);

  const subscribeToJob = useCallback((jobId: string) => {
    if (ws && isConnected) {
      ws.subscribeToJob(jobId);
    }
  }, [ws, isConnected]);

  const unsubscribeFromJob = useCallback((jobId: string) => {
    if (ws && isConnected) {
      ws.unsubscribeFromJob(jobId);
    }
  }, [ws, isConnected]);

  const subscribeToNotifications = useCallback((userId: string) => {
    if (ws && isConnected) {
      ws.subscribeToNotifications(userId);
    }
  }, [ws, isConnected]);

  const unsubscribeFromNotifications = useCallback((userId: string) => {
    if (ws && isConnected) {
      ws.unsubscribeFromNotifications(userId);
    }
  }, [ws, isConnected]);

  const send = useCallback((data: any) => {
    if (ws) {
      ws.send(data);
      setQueuedMessageCount(ws.getQueuedMessageCount());
    }
  }, [ws]);

  // Auto-connect if enabled
  useEffect(() => {
    if (optionsRef.current.autoConnect && !ws && !isConnecting) {
      connect();
    }
  }, [connect, ws, isConnecting]);

  // Update connection state and queued message count
  useEffect(() => {
    if (ws) {
      const updateState = () => {
        if (mountedRef.current) {
          setConnectionState(ws.getConnectionState());
          setQueuedMessageCount(ws.getQueuedMessageCount());
        }
      };

      const interval = setInterval(updateState, 1000);
      return () => clearInterval(interval);
    }
  }, [ws]);

  return {
    ws,
    isConnected,
    isConnecting,
    error,
    connect,
    disconnect,
    subscribeToJob,
    unsubscribeFromJob,
    subscribeToNotifications,
    unsubscribeFromNotifications,
    send,
    connectionState,
    queuedMessageCount
  };
}