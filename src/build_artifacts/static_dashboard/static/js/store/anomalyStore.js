/**
 * Pynomaly Anomaly Detection State Management
 *
 * Comprehensive state management using Zustand for:
 * - Global application state
 * - Real-time data management
 * - User preferences and settings
 * - Component synchronization
 * - Offline state handling
 */

// Zustand-like implementation (vanilla JS version)
class ZustandStore {
  constructor(createState) {
    this.state = {};
    this.listeners = new Set();
    this.middlewares = [];

    const setState = (partial, replace = false) => {
      const nextState =
        typeof partial === "function" ? partial(this.state) : partial;

      if (replace) {
        this.state = nextState;
      } else {
        this.state = { ...this.state, ...nextState };
      }

      this.listeners.forEach((listener) => listener(this.state));
    };

    const getState = () => this.state;

    this.state = createState(setState, getState, this);
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getState() {
    return this.state;
  }

  setState(partial, replace = false) {
    const nextState =
      typeof partial === "function" ? partial(this.state) : partial;

    if (replace) {
      this.state = nextState;
    } else {
      this.state = { ...this.state, ...nextState };
    }

    this.listeners.forEach((listener) => listener(this.state));
  }

  destroy() {
    this.listeners.clear();
    this.middlewares = [];
  }
}

// Main Anomaly Detection Store
const createAnomalyStore = (set, get) => ({
  // Data State
  datasets: [],
  currentDataset: null,
  timeSeries: [],
  anomalies: [],
  features: [],
  models: [],

  // UI State
  selectedTimeRange: { start: null, end: null },
  selectedFeatures: [],
  selectedAnomalies: [],
  brushedData: [],
  zoomLevel: 1,

  // Filter State
  filters: {
    timeRange: null,
    severityThreshold: 0,
    anomalyTypes: [],
    featureFilters: {},
    dateRange: { start: null, end: null },
  },

  // Real-time State
  realTime: {
    enabled: false,
    connected: false,
    lastUpdate: null,
    updateCount: 0,
    errors: [],
  },

  // User Preferences
  preferences: {
    theme: "light",
    accessibility: {
      highContrast: false,
      reducedMotion: false,
      screenReader: false,
    },
    notifications: {
      enabled: true,
      anomalyAlerts: true,
      systemAlerts: false,
    },
    dashboard: {
      layout: "grid",
      autoRefresh: true,
      refreshInterval: 30000,
    },
  },

  // Loading and Error States
  loading: {
    datasets: false,
    anomalies: false,
    models: false,
    global: false,
  },

  errors: {
    datasets: null,
    anomalies: null,
    models: null,
    global: null,
  },

  // Performance Metrics
  performance: {
    renderTimes: {},
    dataUpdateTimes: {},
    memoryUsage: 0,
    errorCount: 0,
  },

  // Actions - Data Management
  setDatasets: (datasets) => set({ datasets }),

  setCurrentDataset: (dataset) =>
    set({
      currentDataset: dataset,
      timeSeries: dataset?.timeSeries || [],
      anomalies: dataset?.anomalies || [],
      features: dataset?.features || [],
    }),

  addTimeSeries: (data) =>
    set((state) => ({
      timeSeries: [...state.timeSeries, ...data],
    })),

  updateTimeSeries: (data) => set({ timeSeries: data }),

  addAnomalies: (anomalies) =>
    set((state) => ({
      anomalies: [...state.anomalies, ...anomalies],
    })),

  updateAnomalies: (anomalies) => set({ anomalies }),

  setFeatures: (features) => set({ features }),

  setModels: (models) => set({ models }),

  // Actions - Selection Management
  setSelectedTimeRange: (range) => set({ selectedTimeRange: range }),

  setSelectedFeatures: (features) => set({ selectedFeatures: features }),

  addSelectedFeature: (feature) =>
    set((state) => ({
      selectedFeatures: [...state.selectedFeatures, feature],
    })),

  removeSelectedFeature: (feature) =>
    set((state) => ({
      selectedFeatures: state.selectedFeatures.filter((f) => f !== feature),
    })),

  setSelectedAnomalies: (anomalies) => set({ selectedAnomalies: anomalies }),

  addSelectedAnomaly: (anomaly) =>
    set((state) => ({
      selectedAnomalies: [...state.selectedAnomalies, anomaly],
    })),

  removeSelectedAnomaly: (anomaly) =>
    set((state) => ({
      selectedAnomalies: state.selectedAnomalies.filter(
        (a) => a.id !== anomaly.id,
      ),
    })),

  setBrushedData: (data) => set({ brushedData: data }),

  clearSelections: () =>
    set({
      selectedFeatures: [],
      selectedAnomalies: [],
      brushedData: [],
      selectedTimeRange: { start: null, end: null },
    }),

  // Actions - Filter Management
  setFilters: (filters) =>
    set((state) => ({
      filters: { ...state.filters, ...filters },
    })),

  setSeverityThreshold: (threshold) =>
    set((state) => ({
      filters: { ...state.filters, severityThreshold: threshold },
    })),

  setAnomalyTypes: (types) =>
    set((state) => ({
      filters: { ...state.filters, anomalyTypes: types },
    })),

  setFeatureFilters: (featureFilters) =>
    set((state) => ({
      filters: { ...state.filters, featureFilters },
    })),

  setDateRange: (range) =>
    set((state) => ({
      filters: { ...state.filters, dateRange: range },
    })),

  clearFilters: () =>
    set((state) => ({
      filters: {
        timeRange: null,
        severityThreshold: 0,
        anomalyTypes: [],
        featureFilters: {},
        dateRange: { start: null, end: null },
      },
    })),

  // Actions - Real-time Management
  setRealTimeEnabled: (enabled) =>
    set((state) => ({
      realTime: { ...state.realTime, enabled },
    })),

  setRealTimeConnected: (connected) =>
    set((state) => ({
      realTime: { ...state.realTime, connected, lastUpdate: Date.now() },
    })),

  incrementUpdateCount: () =>
    set((state) => ({
      realTime: {
        ...state.realTime,
        updateCount: state.realTime.updateCount + 1,
        lastUpdate: Date.now(),
      },
    })),

  addRealTimeError: (error) =>
    set((state) => ({
      realTime: {
        ...state.realTime,
        errors: [
          ...state.realTime.errors.slice(-9),
          {
            // Keep last 10 errors
            message: error.message,
            timestamp: Date.now(),
            type: error.type || "unknown",
          },
        ],
      },
    })),

  clearRealTimeErrors: () =>
    set((state) => ({
      realTime: { ...state.realTime, errors: [] },
    })),

  // Actions - User Preferences
  setTheme: (theme) =>
    set((state) => ({
      preferences: { ...state.preferences, theme },
    })),

  setAccessibilityPreference: (key, value) =>
    set((state) => ({
      preferences: {
        ...state.preferences,
        accessibility: { ...state.preferences.accessibility, [key]: value },
      },
    })),

  setNotificationPreference: (key, value) =>
    set((state) => ({
      preferences: {
        ...state.preferences,
        notifications: { ...state.preferences.notifications, [key]: value },
      },
    })),

  setDashboardPreference: (key, value) =>
    set((state) => ({
      preferences: {
        ...state.preferences,
        dashboard: { ...state.preferences.dashboard, [key]: value },
      },
    })),

  // Actions - Loading States
  setLoading: (key, value) =>
    set((state) => ({
      loading: { ...state.loading, [key]: value },
    })),

  setGlobalLoading: (loading) =>
    set((state) => ({
      loading: { ...state.loading, global: loading },
    })),

  // Actions - Error Handling
  setError: (key, error) =>
    set((state) => ({
      errors: { ...state.errors, [key]: error },
    })),

  clearError: (key) =>
    set((state) => ({
      errors: { ...state.errors, [key]: null },
    })),

  clearAllErrors: () =>
    set({
      errors: {
        datasets: null,
        anomalies: null,
        models: null,
        global: null,
      },
    }),

  // Actions - Performance Tracking
  recordRenderTime: (component, time) =>
    set((state) => ({
      performance: {
        ...state.performance,
        renderTimes: { ...state.performance.renderTimes, [component]: time },
      },
    })),

  recordDataUpdateTime: (component, time) =>
    set((state) => ({
      performance: {
        ...state.performance,
        dataUpdateTimes: {
          ...state.performance.dataUpdateTimes,
          [component]: time,
        },
      },
    })),

  updateMemoryUsage: (usage) =>
    set((state) => ({
      performance: { ...state.performance, memoryUsage: usage },
    })),

  incrementErrorCount: () =>
    set((state) => ({
      performance: {
        ...state.performance,
        errorCount: state.performance.errorCount + 1,
      },
    })),

  // Computed Getters
  getFilteredData: () => {
    const state = get();
    let filtered = state.timeSeries;

    // Apply severity threshold
    if (state.filters.severityThreshold > 0) {
      filtered = filtered.filter(
        (point) =>
          !point.isAnomaly ||
          point.anomalyScore >= state.filters.severityThreshold,
      );
    }

    // Apply time range filter
    if (state.filters.dateRange.start && state.filters.dateRange.end) {
      filtered = filtered.filter((point) => {
        const timestamp = new Date(point.timestamp);
        return (
          timestamp >= state.filters.dateRange.start &&
          timestamp <= state.filters.dateRange.end
        );
      });
    }

    // Apply anomaly type filters
    if (state.filters.anomalyTypes.length > 0) {
      filtered = filtered.filter(
        (point) =>
          !point.isAnomaly ||
          state.filters.anomalyTypes.includes(point.anomalyType || "unknown"),
      );
    }

    return filtered;
  },

  getSelectedData: () => {
    const state = get();
    if (state.brushedData.length > 0) {
      return state.brushedData;
    }

    if (state.selectedTimeRange.start && state.selectedTimeRange.end) {
      return state.timeSeries.filter((point) => {
        const timestamp = new Date(point.timestamp);
        return (
          timestamp >= state.selectedTimeRange.start &&
          timestamp <= state.selectedTimeRange.end
        );
      });
    }

    return state.timeSeries;
  },

  getAnomalyStatistics: () => {
    const state = get();
    const data = state.getFilteredData();
    const anomalies = data.filter((point) => point.isAnomaly);

    return {
      total: data.length,
      anomalies: anomalies.length,
      anomalyRate: data.length > 0 ? anomalies.length / data.length : 0,
      severityDistribution: anomalies.reduce((acc, anomaly) => {
        const severity =
          anomaly.anomalyScore > 0.8
            ? "high"
            : anomaly.anomalyScore > 0.6
              ? "medium"
              : "low";
        acc[severity] = (acc[severity] || 0) + 1;
        return acc;
      }, {}),
      timeRange:
        data.length > 0
          ? {
              start: Math.min(...data.map((d) => new Date(d.timestamp))),
              end: Math.max(...data.map((d) => new Date(d.timestamp))),
            }
          : null,
    };
  },

  getPerformanceMetrics: () => {
    const state = get();
    return {
      ...state.performance,
      averageRenderTime:
        Object.values(state.performance.renderTimes).length > 0
          ? Object.values(state.performance.renderTimes).reduce(
              (a, b) => a + b,
              0,
            ) / Object.values(state.performance.renderTimes).length
          : 0,
      averageDataUpdateTime:
        Object.values(state.performance.dataUpdateTimes).length > 0
          ? Object.values(state.performance.dataUpdateTimes).reduce(
              (a, b) => a + b,
              0,
            ) / Object.values(state.performance.dataUpdateTimes).length
          : 0,
    };
  },
});

// Create the main store instance
const anomalyStore = new ZustandStore(createAnomalyStore);

// Persistence Middleware
class StorePersistence {
  constructor(store, options = {}) {
    this.store = store;
    this.options = {
      key: "pynomaly-store",
      storage: localStorage,
      partialize: (state) => ({
        preferences: state.preferences,
        filters: state.filters,
      }),
      ...options,
    };

    this.init();
  }

  init() {
    // Load persisted state
    this.loadState();

    // Subscribe to changes and persist
    this.store.subscribe((state) => {
      this.saveState(state);
    });
  }

  loadState() {
    try {
      const stored = this.options.storage.getItem(this.options.key);
      if (stored) {
        const parsed = JSON.parse(stored);
        this.store.setState(parsed);
      }
    } catch (error) {
      console.warn("Failed to load persisted state:", error);
    }
  }

  saveState(state) {
    try {
      const toSave = this.options.partialize(state);
      this.options.storage.setItem(this.options.key, JSON.stringify(toSave));
    } catch (error) {
      console.warn("Failed to persist state:", error);
    }
  }

  clearPersistedState() {
    try {
      this.options.storage.removeItem(this.options.key);
    } catch (error) {
      console.warn("Failed to clear persisted state:", error);
    }
  }
}

// DevTools Middleware
class StoreDevTools {
  constructor(store, options = {}) {
    this.store = store;
    this.options = {
      enabled:
        typeof window !== "undefined" && window.__REDUX_DEVTOOLS_EXTENSION__,
      name: "Pynomaly Store",
      ...options,
    };

    if (this.options.enabled) {
      this.init();
    }
  }

  init() {
    this.devTools = window.__REDUX_DEVTOOLS_EXTENSION__.connect({
      name: this.options.name,
    });

    this.devTools.init(this.store.getState());

    this.store.subscribe((state) => {
      this.devTools.send("state_change", state);
    });
  }
}

// Initialize middleware
const persistence = new StorePersistence(anomalyStore);
const devTools = new StoreDevTools(anomalyStore);

// Store utilities and selectors
const useAnomalyStore = {
  // Get current state
  getState: () => anomalyStore.getState(),

  // Subscribe to changes
  subscribe: (selector, callback) => {
    let currentValue = selector(anomalyStore.getState());

    return anomalyStore.subscribe((state) => {
      const nextValue = selector(state);
      if (nextValue !== currentValue) {
        currentValue = nextValue;
        callback(nextValue, state);
      }
    });
  },

  // Batch actions
  batch: (fn) => {
    const state = anomalyStore.getState();
    fn(state);
  },

  // Reset store to initial state
  reset: () => {
    anomalyStore.setState(
      createAnomalyStore(
        () => {},
        () => {},
      ),
      true,
    );
  },

  // Export state for debugging
  export: () => {
    return JSON.stringify(anomalyStore.getState(), null, 2);
  },

  // Import state from JSON
  import: (stateJson) => {
    try {
      const state = JSON.parse(stateJson);
      anomalyStore.setState(state, true);
    } catch (error) {
      console.error("Failed to import state:", error);
    }
  },
};

// Common selectors
const selectors = {
  // Data selectors
  getTimeSeries: (state) => state.timeSeries,
  getAnomalies: (state) => state.anomalies,
  getFeatures: (state) => state.features,
  getCurrentDataset: (state) => state.currentDataset,

  // Filter selectors
  getFilters: (state) => state.filters,
  getFilteredData: (state) => state.getFilteredData(),
  getSelectedData: (state) => state.getSelectedData(),

  // UI selectors
  getSelectedFeatures: (state) => state.selectedFeatures,
  getSelectedAnomalies: (state) => state.selectedAnomalies,
  getBrushedData: (state) => state.brushedData,

  // Real-time selectors
  getRealTimeStatus: (state) => state.realTime,
  isRealTimeEnabled: (state) => state.realTime.enabled,
  isRealTimeConnected: (state) => state.realTime.connected,

  // Preference selectors
  getTheme: (state) => state.preferences.theme,
  getAccessibilityPreferences: (state) => state.preferences.accessibility,
  getNotificationPreferences: (state) => state.preferences.notifications,
  getDashboardPreferences: (state) => state.preferences.dashboard,

  // Loading and error selectors
  getLoadingState: (state) => state.loading,
  getErrors: (state) => state.errors,
  isLoading: (key) => (state) => state.loading[key] || state.loading.global,
  hasError: (key) => (state) => state.errors[key] !== null,

  // Performance selectors
  getPerformanceMetrics: (state) => state.getPerformanceMetrics(),
  getAnomalyStatistics: (state) => state.getAnomalyStatistics(),
};

// Export everything
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    anomalyStore,
    useAnomalyStore,
    selectors,
    StorePersistence,
    StoreDevTools,
  };
}

// Global access
window.anomalyStore = anomalyStore;
window.useAnomalyStore = useAnomalyStore;
window.selectors = selectors;
