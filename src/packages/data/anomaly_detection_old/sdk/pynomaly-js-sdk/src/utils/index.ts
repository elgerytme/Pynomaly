/**
 * Utility exports
 */

export { Environment } from './environment';
export { 
  StorageFactory, 
  TimerUtils, 
  CryptoUtils, 
  HTTPUtils, 
  EventUtils, 
  CompatibilityChecker 
} from './compatibility';
export { setupPolyfills, checkPolyfillsNeeded } from './polyfills';
export { CompatibilityTest } from './compatibility-test';

export type { EnvironmentInfo } from './environment';
export type { UniversalStorage } from './compatibility';
export type { CompatibilityTestResult } from './compatibility-test';