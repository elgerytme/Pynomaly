<template>
  <slot :client="client" :isReady="isReady" :isLoading="isLoading" :error="error" />
</template>

<script setup lang="ts">
import { provide } from 'vue';
import { PynomalyConfig } from '../../../index';
import { usePynomalyClient } from '../composables/usePynomalyClient';

export interface PynomalyProviderProps {
  config: PynomalyConfig;
  autoConnect?: boolean;
}

const props = withDefaults(defineProps<PynomalyProviderProps>(), {
  autoConnect: true
});

const { client, isReady, isLoading, error, reconnect, disconnect } = usePynomalyClient({
  ...props.config,
  autoConnect: props.autoConnect
});

// Provide client to child components
provide('pynomaly-client', client);
provide('pynomaly-ready', isReady);
provide('pynomaly-loading', isLoading);
provide('pynomaly-error', error);
provide('pynomaly-reconnect', reconnect);
provide('pynomaly-disconnect', disconnect);
</script>