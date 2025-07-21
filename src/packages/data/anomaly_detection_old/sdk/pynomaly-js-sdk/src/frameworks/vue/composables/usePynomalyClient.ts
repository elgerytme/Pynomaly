/**
 * Vue 3 composable for Pynomaly client
 */

import { ref, onUnmounted, watch } from 'vue';
import { PynomalyClient, PynomalyConfig } from '../../../index';

export interface UsePynomalyClientOptions extends Partial<PynomalyConfig> {
  autoConnect?: boolean;
}

export function usePynomalyClient(options: UsePynomalyClientOptions) {
  const client = ref<PynomalyClient | null>(null);
  const isReady = ref(false);
  const isLoading = ref(false);
  const error = ref<Error | null>(null);

  const initializeClient = async () => {
    isLoading.value = true;
    error.value = null;

    try {
      const newClient = new PynomalyClient({
        apiKey: '',
        baseUrl: 'https://api.pynomaly.com',
        ...options
      });

      // Test connection if autoConnect is enabled
      if (options.autoConnect) {
        try {
          await newClient.healthCheck();
        } catch (healthError) {
          throw new Error(`Failed to connect to Pynomaly API: ${healthError.message}`);
        }
      }

      client.value = newClient;
      isReady.value = true;
    } catch (err) {
      error.value = err as Error;
    } finally {
      isLoading.value = false;
    }
  };

  const reconnect = async () => {
    if (client.value) {
      client.value.destroy();
      client.value = null;
      isReady.value = false;
    }
    await initializeClient();
  };

  const disconnect = () => {
    if (client.value) {
      client.value.destroy();
      client.value = null;
      isReady.value = false;
    }
  };

  // Initialize client
  initializeClient();

  // Cleanup on unmount
  onUnmounted(() => {
    if (client.value) {
      client.value.destroy();
    }
  });

  return {
    client,
    isReady,
    isLoading,
    error,
    reconnect,
    disconnect
  };
}