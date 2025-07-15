/**
 * React hook for Pynomaly client
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { PynomalyClient, PynomalyConfig } from '../../../index';

export interface UsePynomalyClientOptions extends Partial<PynomalyConfig> {
  autoConnect?: boolean;
  onError?: (error: Error) => void;
  onReady?: (client: PynomalyClient) => void;
}

export interface UsePynomalyClientReturn {
  client: PynomalyClient | null;
  isReady: boolean;
  isLoading: boolean;
  error: Error | null;
  reconnect: () => void;
  disconnect: () => void;
}

export function usePynomalyClient(
  config: UsePynomalyClientOptions
): UsePynomalyClientReturn {
  const [client, setClient] = useState<PynomalyClient | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const configRef = useRef(config);
  const mountedRef = useRef(true);

  // Update config ref when config changes
  useEffect(() => {
    configRef.current = config;
  }, [config]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (client) {
        client.destroy();
      }
    };
  }, [client]);

  const initializeClient = useCallback(async () => {
    if (!mountedRef.current) return;

    setIsLoading(true);
    setError(null);

    try {
      const newClient = new PynomalyClient({
        apiKey: '',
        baseUrl: 'https://api.pynomaly.com',
        ...configRef.current
      });

      // Test connection if autoConnect is enabled
      if (configRef.current.autoConnect) {
        try {
          await newClient.healthCheck();
        } catch (healthError) {
          throw new Error(`Failed to connect to Pynomaly API: ${healthError.message}`);
        }
      }

      if (mountedRef.current) {
        setClient(newClient);
        setIsReady(true);
        configRef.current.onReady?.(newClient);
      }
    } catch (err) {
      const error = err as Error;
      if (mountedRef.current) {
        setError(error);
        configRef.current.onError?.(error);
      }
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  const reconnect = useCallback(() => {
    if (client) {
      client.destroy();
      setClient(null);
      setIsReady(false);
    }
    initializeClient();
  }, [client, initializeClient]);

  const disconnect = useCallback(() => {
    if (client) {
      client.destroy();
      setClient(null);
      setIsReady(false);
    }
  }, [client]);

  // Initialize client on mount
  useEffect(() => {
    initializeClient();
  }, [initializeClient]);

  return {
    client,
    isReady,
    isLoading,
    error,
    reconnect,
    disconnect
  };
}