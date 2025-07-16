/**
 * Framework integrations exports
 */

// React exports
export * as React from './react';

// Vue exports
export * as Vue from './vue';

// Angular exports
export * as Angular from './angular';

// Re-export commonly used items for convenience
export {
  // React
  usePynomalyClient as useReactPynomalyClient,
  usePynomalyAuth as useReactPynomalyAuth,
  useAnomalyDetection as useReactAnomalyDetection,
  usePynomalyWebSocket as useReactPynomalyWebSocket,
  PynomalyProvider as ReactPynomalyProvider,
  AnomalyDetector as ReactAnomalyDetector
} from './react';

export {
  // Vue
  usePynomalyClient as useVuePynomalyClient,
  usePynomalyAuth as useVuePynomalyAuth,
  useAnomalyDetection as useVueAnomalyDetection,
  PynomalyProvider as VuePynomalyProvider,
  AnomalyDetector as VueAnomalyDetector
} from './vue';

export {
  // Angular
  PynomalyModule,
  PynomalyService,
  PynomalyAuthService,
  AnomalyDetectorComponent as AngularAnomalyDetector
} from './angular';