# Progressive Web App API Reference

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📖 [Reference](README.md) > 📁 Api > 📄 Pwa Api Reference

---


Complete API reference for Pynomaly's Progressive Web App components, including offline capabilities, synchronization, and advanced visualization features.

## 📋 Table of Contents

- [PWA Manager API](#pwa-manager-api)
- [Sync Manager API](#sync-manager-api)
- [Offline Detector API](#offline-detector-api)
- [Offline Visualizer API](#offline-visualizer-api)
- [Offline Dashboard API](#offline-dashboard-api)
- [Service Worker API](#service-worker-api)
- [Error Handling](#error-handling)
- [Type Definitions](#type-definitions)

---

## 🚀 PWA Manager API

### Class: `PWAManager`

Main controller for Progressive Web App functionality including installation, updates, and offline management.

#### Constructor

```javascript
const pwaManager = new PWAManager();
```

#### Methods

##### **Installation Management**

###### `isAppInstalled(): boolean`

Check if the PWA is currently installed.

```javascript
const isInstalled = pwaManager.isAppInstalled();
console.log(`App installed: ${isInstalled}`);
```

**Returns:** `boolean` - True if app is installed

---

###### `async installPWA(): Promise<InstallResult>`

Trigger the PWA installation prompt.

```javascript
try {
  const result = await pwaManager.installPWA();
  if (result.outcome === 'accepted') {
    console.log('App installed successfully');
  }
} catch (error) {
  console.error('Installation failed:', error);
}
```

**Returns:** `Promise<InstallResult>`

```typescript
interface InstallResult {
  outcome: 'accepted' | 'dismissed';
  platform?: string;
}
```

---

##### **Status and Information**

###### `async getAppStatus(): Promise<AppStatus>`

Get comprehensive PWA status information.

```javascript
const status = await pwaManager.getAppStatus();
console.log('PWA Status:', status);
```

**Returns:** `Promise<AppStatus>`

```typescript
interface AppStatus {
  installed: boolean;
  online: boolean;
  serviceWorkerActive: boolean;
  cacheInfo: CacheInfo;
  syncStatus: SyncStatus;
}
```

---

###### `isAppOnline(): boolean`

Check current online/offline status.

```javascript
const isOnline = pwaManager.isAppOnline();
if (!isOnline) {
  console.log('App is offline - using cached data');
}
```

**Returns:** `boolean` - Current connectivity status

---

##### **Data Management**

###### `async saveDataOffline(type: string, data: any): Promise<void>`

Save data for offline access.

```javascript
await pwaManager.saveDataOffline('dataset', {
  id: 'ds_001',
  name: 'Production Data',
  data: analysisData
});
```

**Parameters:**
- `type: string` - Data type ('dataset', 'result', 'preference')
- `data: any` - Data to cache offline

---

###### `async clearCache(cacheName?: string): Promise<boolean>`

Clear cached data.

```javascript
// Clear specific cache
await pwaManager.clearCache('datasets');

// Clear all caches
await pwaManager.clearCache();
```

**Parameters:**
- `cacheName?: string` - Optional specific cache to clear

**Returns:** `Promise<boolean>` - Success status

---

##### **Event Handling**

###### `on(event: string, callback: Function): void`

Subscribe to PWA events.

```javascript
pwaManager.on('connectionchange', (isOnline) => {
  console.log(`Connection changed: ${isOnline ? 'online' : 'offline'}`);
});

pwaManager.on('updateavailable', () => {
  console.log('App update available');
});
```

**Events:**
- `connectionchange` - Online/offline status change
- `updateavailable` - New version available
- `installed` - App successfully installed
- `syncstatus` - Sync status change

---

## 🔄 Sync Manager API

### Class: `SyncManager`

Handles data synchronization between offline and online modes with conflict resolution.

#### Constructor

```javascript
const syncManager = new SyncManager();
```

#### Methods

##### **Queue Management**

###### `async queueForSync(operation: string, data: SyncData, priority?: Priority): Promise<string>`

Queue an operation for background synchronization.

```javascript
const syncId = await syncManager.queueForSync('create', {
  entityType: 'dataset',
  entityId: 'ds_001',
  payload: datasetData
}, 'high');
```

**Parameters:**
- `operation: string` - 'create', 'update', 'delete'
- `data: SyncData` - Data to synchronize
- `priority?: Priority` - 'high', 'normal', 'low' (default: 'normal')

**Returns:** `Promise<string>` - Unique sync ID

---

###### `async processSyncQueue(): Promise<SyncResult>`

Manually trigger sync queue processing.

```javascript
const result = await syncManager.processSyncQueue();
console.log(`Synced: ${result.completed}, Failed: ${result.failed}`);
```

**Returns:** `Promise<SyncResult>`

```typescript
interface SyncResult {
  completed: number;
  failed: number;
  conflicts: number;
  details: SyncItemResult[];
}
```

---

##### **Sync Configuration**

###### `setSyncStrategy(strategy: SyncStrategy): void`

Configure synchronization behavior.

```javascript
syncManager.setSyncStrategy('smart');
```

**Parameters:**
- `strategy: SyncStrategy` - 'immediate', 'smart', 'manual'

---

###### `getSyncStatus(): SyncStatus`

Get current synchronization status.

```javascript
const status = syncManager.getSyncStatus();
console.log(`Pending items: ${status.pending}`);
```

**Returns:** `SyncStatus`

```typescript
interface SyncStatus {
  isOnline: boolean;
  isSyncing: boolean;
  pending: number;
  syncing: number;
  failed: number;
  conflicts: number;
  strategy: SyncStrategy;
  lastSyncAt?: number;
}
```

---

##### **Conflict Resolution**

###### `async resolveConflict(conflictId: string, strategy: ResolutionStrategy, resolution?: any): Promise<ResolutionResult>`

Resolve synchronization conflicts.

```javascript
// Server wins
await syncManager.resolveConflict(conflictId, 'server_wins');

// Manual merge
await syncManager.resolveConflict(conflictId, 'merge', {
  mergedData: customMergedData
});
```

**Parameters:**
- `conflictId: string` - Conflict identifier
- `strategy: ResolutionStrategy` - 'server_wins', 'client_wins', 'merge', 'manual'
- `resolution?: any` - Custom resolution data for merge strategy

---

###### `getConflicts(): ConflictInfo[]`

Get list of unresolved conflicts.

```javascript
const conflicts = syncManager.getConflicts();
conflicts.forEach(conflict => {
  console.log(`Conflict in ${conflict.entityType}:${conflict.entityId}`);
});
```

**Returns:** `ConflictInfo[]`

---

##### **Entity-Specific Methods**

###### `async queueDatasetSync(operation: string, dataset: Dataset, priority?: Priority): Promise<string>`

Queue dataset synchronization.

```javascript
const syncId = await syncManager.queueDatasetSync('update', dataset, 'high');
```

###### `async queueResultSync(operation: string, result: AnalysisResult, priority?: Priority): Promise<string>`

Queue analysis result synchronization.

```javascript
const syncId = await syncManager.queueResultSync('create', analysisResult);
```

###### `async queueModelSync(operation: string, model: Model, priority?: Priority): Promise<string>`

Queue model synchronization.

```javascript
const syncId = await syncManager.queueModelSync('update', trainedModel, 'high');
```

---

## 🔍 Offline Detector API

### Class: `OfflineDetector`

Browser-based anomaly detection algorithms for offline analysis.

#### Constructor

```javascript
const offlineDetector = new OfflineDetector();
```

#### Methods

##### **Algorithm Management**

###### `getAlgorithms(): AlgorithmInfo[]`

Get list of available offline algorithms.

```javascript
const algorithms = offlineDetector.getAlgorithms();
algorithms.forEach(algo => {
  console.log(`${algo.id}: ${algo.name} - ${algo.description}`);
});
```

**Returns:** `AlgorithmInfo[]`

```typescript
interface AlgorithmInfo {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
}
```

---

##### **Data Management**

###### `async loadCachedDatasets(): Promise<Dataset[]>`

Load datasets from offline cache.

```javascript
const datasets = await offlineDetector.loadCachedDatasets();
console.log(`${datasets.length} datasets available offline`);
```

**Returns:** `Promise<Dataset[]>`

---

###### `getCachedDatasets(): Dataset[]`

Get synchronously available cached datasets.

```javascript
const datasets = offlineDetector.getCachedDatasets();
```

**Returns:** `Dataset[]`

---

##### **Anomaly Detection**

###### `async detectAnomalies(datasetId: string, algorithmId: string, parameters?: Record<string, any>): Promise<DetectionResult>`

Run anomaly detection on cached data.

```javascript
const result = await offlineDetector.detectAnomalies(
  'dataset_001',
  'zscore',
  { threshold: 3.0 }
);

console.log(`Found ${result.anomalies.length} anomalies`);
```

**Parameters:**
- `datasetId: string` - Cached dataset identifier
- `algorithmId: string` - Algorithm to use ('zscore', 'iqr', 'isolation', 'mad')
- `parameters?: Record<string, any>` - Algorithm-specific parameters

**Returns:** `Promise<DetectionResult>`

```typescript
interface DetectionResult {
  id: string;
  datasetId: string;
  algorithmId: string;
  timestamp: string;
  processingTimeMs: number;
  anomalies: AnomalyPoint[];
  scores: number[];
  statistics: DetectionStatistics;
  parameters: Record<string, any>;
  isOffline: boolean;
}
```

---

##### **History Management**

###### `async getDetectionHistory(): Promise<DetectionResult[]>`

Get offline detection history.

```javascript
const history = await offlineDetector.getDetectionHistory();
const recentResults = history.slice(0, 10); // Last 10 results
```

**Returns:** `Promise<DetectionResult[]>`

---

###### `async saveResult(result: DetectionResult): Promise<void>`

Save detection result to offline storage.

```javascript
await offlineDetector.saveResult(detectionResult);
```

**Parameters:**
- `result: DetectionResult` - Detection result to save

---

## 📊 Offline Visualizer API

### Class: `OfflineVisualizer`

Advanced data visualization using cached data and ECharts integration.

#### Constructor

```javascript
const offlineVisualizer = new OfflineVisualizer();
```

#### Methods

##### **Dataset Visualization**

###### `async selectDataset(datasetId: string): Promise<void>`

Select dataset for visualization.

```javascript
await offlineVisualizer.selectDataset('dataset_001');
```

**Parameters:**
- `datasetId: string` - Dataset identifier

---

###### `async renderDatasetVisualization(): Promise<void>`

Render comprehensive dataset visualizations.

```javascript
await offlineVisualizer.renderDatasetVisualization();
// Renders: distribution charts, correlation matrix, statistics table, quality metrics
```

---

##### **Result Visualization**

###### `async selectResult(resultId: string): Promise<void>`

Select analysis result for visualization.

```javascript
await offlineVisualizer.selectResult('result_001');
```

**Parameters:**
- `resultId: string` - Analysis result identifier

---

###### `async renderResultVisualization(): Promise<void>`

Render anomaly detection result visualizations.

```javascript
await offlineVisualizer.renderResultVisualization();
// Renders: anomaly distribution, score distribution, scatter plots, summary
```

---

##### **Chart Management**

###### `exportChart(chartId: string): void`

Export chart as image.

```javascript
offlineVisualizer.exportChart('anomaly-scatter-chart');
// Downloads PNG file
```

**Parameters:**
- `chartId: string` - Chart container ID

---

###### `changeVisualizationType(type: string): void`

Change visualization type for dynamic charts.

```javascript
offlineVisualizer.changeVisualizationType('histogram'); // or 'boxplot', 'violin'
```

**Parameters:**
- `type: string` - Visualization type

---

##### **Data Access**

###### `getAvailableDatasets(): Dataset[]`

Get datasets available for visualization.

```javascript
const datasets = offlineVisualizer.getAvailableDatasets();
```

**Returns:** `Dataset[]`

---

###### `getAvailableResults(): AnalysisResult[]`

Get analysis results available for visualization.

```javascript
const results = offlineVisualizer.getAvailableResults();
```

**Returns:** `AnalysisResult[]`

---

## 📋 Offline Dashboard API

### Class: `OfflineDashboard`

Interactive dashboard with cached data and real-time statistics.

#### Constructor

```javascript
const offlineDashboard = new OfflineDashboard();
```

#### Methods

##### **Dashboard Management**

###### `async refreshDashboard(): Promise<void>`

Refresh dashboard with latest cached data.

```javascript
await offlineDashboard.refreshDashboard();
```

---

###### `renderDashboard(): void`

Render complete dashboard interface.

```javascript
offlineDashboard.renderDashboard();
// Renders: overview cards, charts, recent activity
```

---

##### **Chart Rendering**

###### `renderOverviewCards(): void`

Render statistical overview cards.

```javascript
offlineDashboard.renderOverviewCards();
// Shows: total datasets, detections run, anomalies found, cache size
```

---

###### `renderDatasetChart(): void`

Render dataset distribution chart.

```javascript
offlineDashboard.renderDatasetChart();
```

---

###### `renderAlgorithmPerformanceChart(): void`

Render algorithm performance comparison.

```javascript
offlineDashboard.renderAlgorithmPerformanceChart();
```

---

###### `renderAnomalyTimelineChart(): void`

Render detection activity timeline.

```javascript
offlineDashboard.renderAnomalyTimelineChart();
```

---

###### `renderRecentActivity(): void`

Render recent activity feed.

```javascript
offlineDashboard.renderRecentActivity();
```

---

##### **Event Handlers**

###### `onDatasetChange(datasetId: string): void`

Handle dataset selection change.

```javascript
offlineDashboard.onDatasetChange('dataset_001');
```

###### `onAlgorithmChange(algorithmId: string): void`

Handle algorithm selection change.

```javascript
offlineDashboard.onAlgorithmChange('isolation');
```

---

## 🔧 Service Worker API

### Service Worker Message Interface

Communication interface with the service worker for offline functionality.

#### Message Types

##### **Cache Management**

```javascript
// Get cache status
navigator.serviceWorker.ready.then(registration => {
  registration.active.postMessage({ type: 'GET_CACHE_STATUS' });
});

// Clear specific cache
navigator.serviceWorker.ready.then(registration => {
  registration.active.postMessage({
    type: 'CLEAR_CACHE',
    payload: { cacheName: 'datasets' }
  });
});
```

##### **Data Operations**

```javascript
// Get offline dashboard data
navigator.serviceWorker.ready.then(registration => {
  registration.active.postMessage({ type: 'GET_OFFLINE_DASHBOARD_DATA' });
});

// Save detection result
navigator.serviceWorker.ready.then(registration => {
  registration.active.postMessage({
    type: 'SAVE_DETECTION_RESULT',
    payload: detectionResult
  });
});
```

##### **Sync Management**

```javascript
// Get sync queue
navigator.serviceWorker.ready.then(registration => {
  registration.active.postMessage({ type: 'GET_SYNC_QUEUE' });
});

// Trigger background sync
navigator.serviceWorker.ready.then(registration => {
  registration.active.postMessage({ type: 'SYNC_ALL_QUEUES' });
});
```

#### Event Listeners

```javascript
// Listen for service worker messages
navigator.serviceWorker.addEventListener('message', (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'OFFLINE_DASHBOARD_DATA':
      updateDashboard(data);
      break;
    case 'SYNC_QUEUE_STATUS':
      updateSyncStatus(data);
      break;
    case 'CACHE_STATUS':
      updateCacheInfo(data);
      break;
  }
});
```

---

## ⚠️ Error Handling

### Error Types

#### **PWAError**

Base error class for PWA-related issues.

```typescript
class PWAError extends Error {
  code: string;
  details?: any;
}
```

#### **SyncError**

Synchronization-specific errors.

```typescript
class SyncError extends PWAError {
  conflictData?: ConflictInfo;
  retryable: boolean;
}
```

#### **OfflineError**

Offline operation errors.

```typescript
class OfflineError extends PWAError {
  requiredOnline: boolean;
  fallbackAvailable: boolean;
}
```

### Common Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `PWA_NOT_SUPPORTED` | Browser doesn't support PWA | Use regular web interface |
| `STORAGE_QUOTA_EXCEEDED` | Insufficient storage space | Clear cache or free space |
| `SYNC_CONFLICT` | Data synchronization conflict | Resolve conflict manually |
| `OFFLINE_ALGORITHM_ERROR` | Offline algorithm execution failed | Check data format and parameters |
| `CACHE_ACCESS_DENIED` | Cannot access offline cache | Check browser permissions |

### Error Handling Patterns

```javascript
try {
  const result = await offlineDetector.detectAnomalies(datasetId, algorithmId);
  // Handle success
} catch (error) {
  if (error instanceof OfflineError) {
    if (error.requiredOnline) {
      showMessage('This feature requires internet connection');
    } else if (error.fallbackAvailable) {
      // Use fallback method
    }
  } else if (error instanceof SyncError) {
    if (error.conflictData) {
      showConflictResolutionDialog(error.conflictData);
    }
  } else {
    showGenericError(error.message);
  }
}
```

---

## 📘 Type Definitions

### Core Types

```typescript
// Dataset structure
interface Dataset {
  id: string;
  name: string;
  type: 'tabular' | 'time_series' | 'graph' | 'text' | 'image';
  data: any[];
  metadata: DatasetMetadata;
  timestamp: number;
  size: number;
}

// Analysis result
interface AnalysisResult {
  id: string;
  datasetId: string;
  algorithmId: string;
  timestamp: string;
  processingTimeMs: number;
  anomalies: AnomalyPoint[];
  scores: number[];
  statistics: DetectionStatistics;
  parameters: Record<string, any>;
  isOffline?: boolean;
}

// Anomaly point
interface AnomalyPoint {
  index: number;
  score: number;
  values: number[];
  explanation?: string;
}

// Detection statistics
interface DetectionStatistics {
  totalSamples: number;
  totalAnomalies: number;
  anomalyRate: number;
  averageScore: number;
  maxScore: number;
  threshold?: number;
  processingTime: number;
}
```

### PWA-Specific Types

```typescript
// Sync data structure
interface SyncData {
  entityType: 'dataset' | 'result' | 'model' | 'preference';
  entityId: string;
  payload: any;
  version?: string;
}

// Conflict information
interface ConflictInfo {
  id: string;
  entityType: string;
  entityId: string;
  operation: 'create' | 'update' | 'delete';
  conflicts: ConflictDetail[];
  timestamp: number;
}

// Cache information
interface CacheInfo {
  caches: number;
  names: string[];
  totalSize: number;
  lastUpdated: number;
}

// App installation result
interface InstallResult {
  outcome: 'accepted' | 'dismissed';
  platform?: string;
  timestamp: number;
}
```

### Algorithm Types

```typescript
// Algorithm configuration
interface AlgorithmConfig {
  id: string;
  name: string;
  description: string;
  parameters: ParameterDefinition[];
  category: 'statistical' | 'ml' | 'ensemble' | 'specialized';
  complexity: 'low' | 'medium' | 'high';
  supportsStreaming: boolean;
  offlineCapable: boolean;
}

// Parameter definition
interface ParameterDefinition {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  default: any;
  min?: number;
  max?: number;
  options?: string[];
  description: string;
  required: boolean;
}
```

---

## 🔗 Related Documentation

- **[Progressive Web App User Guide](../../user-guides/progressive-web-app.md)** - Complete PWA usage guide
- **[Web Interface Quickstart](../../getting-started/web-interface-quickstart.md)** - Getting started with web UI
- **[REST API Reference](./rest-api.md)** - Server-side API documentation
- **[Python SDK Reference](./python-sdk.md)** - Python API documentation

---

*For additional help, see our [troubleshooting guide](../../user-guides/troubleshooting/) or [contact support](https://github.com/your-org/pynomaly/discussions).*
