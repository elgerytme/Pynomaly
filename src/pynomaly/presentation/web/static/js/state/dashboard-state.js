/**
 * Dashboard State Management
 * Centralized state management for complex dashboard interactions
 */

// Simple state management inspired by Redux but lightweight
export class DashboardStateManager {
  constructor(initialState = {}) {
    this.state = {
      // UI State
      ui: {
        sidebarCollapsed: false,
        theme: 'light',
        layout: 'grid',
        loading: false,
        errors: [],
        notifications: []
      },
      
      // Data State
      data: {
        datasets: [],
        currentDataset: null,
        anomalies: [],
        alerts: [],
        metrics: {
          totalDataPoints: 0,
          anomalyCount: 0,
          anomalyRate: 0,
          lastUpdate: null
        }
      },
      
      // Filter State
      filters: {
        dateRange: {
          start: null,
          end: null
        },
        confidence: {
          min: 0,
          max: 1
        },
        features: [],
        algorithms: [],
        status: 'all' // 'all', 'normal', 'anomaly'
      },
      
      // Chart State
      charts: {
        timeline: {
          zoom: 1,
          pan: { x: 0, y: 0 },
          selectedRange: null,
          showLegend: true,
          showTooltip: true
        },
        heatmap: {
          colorScheme: 'RdYlBu',
          aggregation: '15min',
          showValues: false
        }
      },
      
      // Real-time State
      realTime: {
        connected: false,
        paused: false,
        updateRate: 1000,
        bufferSize: 1000,
        autoScroll: true
      },
      
      // User Preferences
      preferences: {
        autoRefresh: true,
        refreshInterval: 30000,
        alertsEnabled: true,
        soundEnabled: false,
        compactMode: false
      },
      
      ...initialState
    };
    
    this.listeners = new Set();
    this.middleware = [];
    this.history = [];
    this.maxHistorySize = 50;
    
    // Load saved state from localStorage
    this.loadFromStorage();
  }
  
  // Subscribe to state changes
  subscribe(listener) {
    this.listeners.add(listener);
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener);
    };
  }
  
  // Add middleware for state transformations
  addMiddleware(middleware) {
    this.middleware.push(middleware);
  }
  
  // Get current state (immutable)
  getState() {
    return JSON.parse(JSON.stringify(this.state));
  }
  
  // Get specific state slice
  getStateSlice(path) {
    return this.getNestedValue(this.state, path);
  }
  
  // Dispatch action to update state
  dispatch(action) {
    const prevState = this.getState();
    
    // Apply middleware
    let processedAction = action;
    for (const middleware of this.middleware) {
      processedAction = middleware(processedAction, prevState, this);
      if (!processedAction) return; // Middleware can cancel action
    }
    
    // Apply action to state
    const newState = this.reducer(prevState, processedAction);
    
    // Update state if changed
    if (JSON.stringify(newState) !== JSON.stringify(prevState)) {
      this.state = newState;
      
      // Add to history
      this.addToHistory(prevState, processedAction);
      
      // Save to localStorage
      this.saveToStorage();
      
      // Notify listeners
      this.notifyListeners(processedAction, prevState, newState);
    }
  }
  
  // Main state reducer
  reducer(state, action) {
    switch (action.type) {
      // UI Actions
      case 'SET_LOADING':
        return this.updateState(state, 'ui.loading', action.payload);
        
      case 'SET_THEME':
        return this.updateState(state, 'ui.theme', action.payload);
        
      case 'TOGGLE_SIDEBAR':
        return this.updateState(state, 'ui.sidebarCollapsed', !state.ui.sidebarCollapsed);
        
      case 'SET_LAYOUT':
        return this.updateState(state, 'ui.layout', action.payload);
        
      case 'ADD_ERROR':
        return this.updateState(state, 'ui.errors', [...state.ui.errors, action.payload]);
        
      case 'REMOVE_ERROR':
        return this.updateState(state, 'ui.errors', 
          state.ui.errors.filter(error => error.id !== action.payload));
        
      case 'ADD_NOTIFICATION':
        return this.updateState(state, 'ui.notifications', [...state.ui.notifications, {
          id: Date.now(),
          timestamp: new Date(),
          ...action.payload
        }]);
        
      case 'REMOVE_NOTIFICATION':
        return this.updateState(state, 'ui.notifications',
          state.ui.notifications.filter(notification => notification.id !== action.payload));
        
      // Data Actions
      case 'SET_DATASETS':
        return this.updateState(state, 'data.datasets', action.payload);
        
      case 'SET_CURRENT_DATASET':
        return this.updateState(state, 'data.currentDataset', action.payload);
        
      case 'ADD_ANOMALY':
        return this.updateState(state, 'data.anomalies', [...state.data.anomalies, action.payload]);
        
      case 'ADD_ANOMALIES':
        return this.updateState(state, 'data.anomalies', [...state.data.anomalies, ...action.payload]);
        
      case 'UPDATE_ANOMALY':
        return this.updateState(state, 'data.anomalies',
          state.data.anomalies.map(anomaly => 
            anomaly.id === action.payload.id ? { ...anomaly, ...action.payload.updates } : anomaly
          ));
        
      case 'REMOVE_ANOMALY':
        return this.updateState(state, 'data.anomalies',
          state.data.anomalies.filter(anomaly => anomaly.id !== action.payload));
        
      case 'CLEAR_ANOMALIES':
        return this.updateState(state, 'data.anomalies', []);
        
      case 'ADD_ALERT':
        return this.updateState(state, 'data.alerts', [...state.data.alerts, {
          id: Date.now(),
          timestamp: new Date(),
          ...action.payload
        }]);
        
      case 'REMOVE_ALERT':
        return this.updateState(state, 'data.alerts',
          state.data.alerts.filter(alert => alert.id !== action.payload));
        
      case 'CLEAR_ALERTS':
        return this.updateState(state, 'data.alerts', []);
        
      case 'UPDATE_METRICS':
        return this.updateState(state, 'data.metrics', {
          ...state.data.metrics,
          ...action.payload,
          lastUpdate: new Date()
        });
        
      // Filter Actions
      case 'SET_DATE_RANGE':
        return this.updateState(state, 'filters.dateRange', action.payload);
        
      case 'SET_CONFIDENCE_RANGE':
        return this.updateState(state, 'filters.confidence', action.payload);
        
      case 'SET_FEATURE_FILTER':
        return this.updateState(state, 'filters.features', action.payload);
        
      case 'SET_ALGORITHM_FILTER':
        return this.updateState(state, 'filters.algorithms', action.payload);
        
      case 'SET_STATUS_FILTER':
        return this.updateState(state, 'filters.status', action.payload);
        
      case 'RESET_FILTERS':
        return this.updateState(state, 'filters', {
          dateRange: { start: null, end: null },
          confidence: { min: 0, max: 1 },
          features: [],
          algorithms: [],
          status: 'all'
        });
        
      // Chart Actions
      case 'SET_CHART_CONFIG':
        return this.updateState(state, `charts.${action.payload.chart}`, {
          ...state.charts[action.payload.chart],
          ...action.payload.config
        });
        
      case 'SET_TIMELINE_ZOOM':
        return this.updateState(state, 'charts.timeline.zoom', action.payload);
        
      case 'SET_TIMELINE_PAN':
        return this.updateState(state, 'charts.timeline.pan', action.payload);
        
      case 'SET_TIMELINE_SELECTION':
        return this.updateState(state, 'charts.timeline.selectedRange', action.payload);
        
      case 'SET_HEATMAP_COLOR_SCHEME':
        return this.updateState(state, 'charts.heatmap.colorScheme', action.payload);
        
      case 'SET_HEATMAP_AGGREGATION':
        return this.updateState(state, 'charts.heatmap.aggregation', action.payload);
        
      // Real-time Actions
      case 'SET_CONNECTION_STATUS':
        return this.updateState(state, 'realTime.connected', action.payload);
        
      case 'TOGGLE_REAL_TIME_PAUSE':
        return this.updateState(state, 'realTime.paused', !state.realTime.paused);
        
      case 'SET_UPDATE_RATE':
        return this.updateState(state, 'realTime.updateRate', action.payload);
        
      case 'SET_BUFFER_SIZE':
        return this.updateState(state, 'realTime.bufferSize', action.payload);
        
      case 'TOGGLE_AUTO_SCROLL':
        return this.updateState(state, 'realTime.autoScroll', !state.realTime.autoScroll);
        
      // Preference Actions
      case 'SET_PREFERENCE':
        return this.updateState(state, `preferences.${action.payload.key}`, action.payload.value);
        
      case 'TOGGLE_PREFERENCE':
        return this.updateState(state, `preferences.${action.payload}`, 
          !this.getNestedValue(state, `preferences.${action.payload}`));
        
      default:
        console.warn(`Unknown action type: ${action.type}`);
        return state;
    }
  }
  
  // Helper method to update nested state
  updateState(state, path, value) {
    const newState = JSON.parse(JSON.stringify(state));
    this.setNestedValue(newState, path, value);
    return newState;
  }
  
  // Get nested object value by path string
  getNestedValue(obj, path) {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }
  
  // Set nested object value by path string
  setNestedValue(obj, path, value) {
    const keys = path.split('.');
    const lastKey = keys.pop();
    const target = keys.reduce((current, key) => current[key], obj);
    target[lastKey] = value;
  }
  
  // Add state change to history
  addToHistory(prevState, action) {
    this.history.unshift({
      timestamp: new Date(),
      action,
      prevState,
      newState: this.getState()
    });
    
    // Limit history size
    if (this.history.length > this.maxHistorySize) {
      this.history = this.history.slice(0, this.maxHistorySize);
    }
  }
  
  // Notify all listeners of state change
  notifyListeners(action, prevState, newState) {
    this.listeners.forEach(listener => {
      try {
        listener(action, prevState, newState);
      } catch (error) {
        console.error('Error in state listener:', error);
      }
    });
  }
  
  // Action creators for common operations
  actions = {
    // UI Actions
    setLoading: (loading) => ({ type: 'SET_LOADING', payload: loading }),
    setTheme: (theme) => ({ type: 'SET_THEME', payload: theme }),
    toggleSidebar: () => ({ type: 'TOGGLE_SIDEBAR' }),
    setLayout: (layout) => ({ type: 'SET_LAYOUT', payload: layout }),
    
    addError: (error) => ({ type: 'ADD_ERROR', payload: { id: Date.now(), ...error } }),
    removeError: (id) => ({ type: 'REMOVE_ERROR', payload: id }),
    
    addNotification: (notification) => ({ type: 'ADD_NOTIFICATION', payload: notification }),
    removeNotification: (id) => ({ type: 'REMOVE_NOTIFICATION', payload: id }),
    
    // Data Actions
    setDatasets: (datasets) => ({ type: 'SET_DATASETS', payload: datasets }),
    setCurrentDataset: (dataset) => ({ type: 'SET_CURRENT_DATASET', payload: dataset }),
    
    addAnomaly: (anomaly) => ({ type: 'ADD_ANOMALY', payload: anomaly }),
    addAnomalies: (anomalies) => ({ type: 'ADD_ANOMALIES', payload: anomalies }),
    updateAnomaly: (id, updates) => ({ type: 'UPDATE_ANOMALY', payload: { id, updates } }),
    removeAnomaly: (id) => ({ type: 'REMOVE_ANOMALY', payload: id }),
    clearAnomalies: () => ({ type: 'CLEAR_ANOMALIES' }),
    
    addAlert: (alert) => ({ type: 'ADD_ALERT', payload: alert }),
    removeAlert: (id) => ({ type: 'REMOVE_ALERT', payload: id }),
    clearAlerts: () => ({ type: 'CLEAR_ALERTS' }),
    
    updateMetrics: (metrics) => ({ type: 'UPDATE_METRICS', payload: metrics }),
    
    // Filter Actions
    setDateRange: (start, end) => ({ type: 'SET_DATE_RANGE', payload: { start, end } }),
    setConfidenceRange: (min, max) => ({ type: 'SET_CONFIDENCE_RANGE', payload: { min, max } }),
    setFeatureFilter: (features) => ({ type: 'SET_FEATURE_FILTER', payload: features }),
    setAlgorithmFilter: (algorithms) => ({ type: 'SET_ALGORITHM_FILTER', payload: algorithms }),
    setStatusFilter: (status) => ({ type: 'SET_STATUS_FILTER', payload: status }),
    resetFilters: () => ({ type: 'RESET_FILTERS' }),
    
    // Chart Actions
    setChartConfig: (chart, config) => ({ type: 'SET_CHART_CONFIG', payload: { chart, config } }),
    setTimelineZoom: (zoom) => ({ type: 'SET_TIMELINE_ZOOM', payload: zoom }),
    setTimelinePan: (pan) => ({ type: 'SET_TIMELINE_PAN', payload: pan }),
    setTimelineSelection: (range) => ({ type: 'SET_TIMELINE_SELECTION', payload: range }),
    setHeatmapColorScheme: (scheme) => ({ type: 'SET_HEATMAP_COLOR_SCHEME', payload: scheme }),
    setHeatmapAggregation: (aggregation) => ({ type: 'SET_HEATMAP_AGGREGATION', payload: aggregation }),
    
    // Real-time Actions
    setConnectionStatus: (connected) => ({ type: 'SET_CONNECTION_STATUS', payload: connected }),
    toggleRealTimePause: () => ({ type: 'TOGGLE_REAL_TIME_PAUSE' }),
    setUpdateRate: (rate) => ({ type: 'SET_UPDATE_RATE', payload: rate }),
    setBufferSize: (size) => ({ type: 'SET_BUFFER_SIZE', payload: size }),
    toggleAutoScroll: () => ({ type: 'TOGGLE_AUTO_SCROLL' }),
    
    // Preference Actions
    setPreference: (key, value) => ({ type: 'SET_PREFERENCE', payload: { key, value } }),
    togglePreference: (key) => ({ type: 'TOGGLE_PREFERENCE', payload: key })
  };
  
  // Computed properties (derived state)
  getters = {
    // Filtered anomalies based on current filters
    getFilteredAnomalies: () => {
      const state = this.getState();
      const { anomalies } = state.data;
      const filters = state.filters;
      
      return anomalies.filter(anomaly => {
        // Date range filter
        if (filters.dateRange.start && anomaly.timestamp < filters.dateRange.start) return false;
        if (filters.dateRange.end && anomaly.timestamp > filters.dateRange.end) return false;
        
        // Confidence filter
        if (anomaly.confidence < filters.confidence.min || anomaly.confidence > filters.confidence.max) return false;
        
        // Feature filter
        if (filters.features.length > 0 && !filters.features.includes(anomaly.feature)) return false;
        
        // Algorithm filter
        if (filters.algorithms.length > 0 && !filters.algorithms.includes(anomaly.algorithm)) return false;
        
        // Status filter
        if (filters.status !== 'all') {
          if (filters.status === 'anomaly' && !anomaly.isAnomaly) return false;
          if (filters.status === 'normal' && anomaly.isAnomaly) return false;
        }
        
        return true;
      });
    },
    
    // Get available features from datasets
    getAvailableFeatures: () => {
      const datasets = this.getStateSlice('data.datasets');
      const features = new Set();
      datasets.forEach(dataset => {
        if (dataset.features) {
          dataset.features.forEach(feature => features.add(feature));
        }
      });
      return Array.from(features);
    },
    
    // Get available algorithms
    getAvailableAlgorithms: () => {
      const anomalies = this.getStateSlice('data.anomalies');
      const algorithms = new Set();
      anomalies.forEach(anomaly => {
        if (anomaly.algorithm) {
          algorithms.add(anomaly.algorithm);
        }
      });
      return Array.from(algorithms);
    },
    
    // Check if filters are active
    hasActiveFilters: () => {
      const filters = this.getStateSlice('filters');
      return (
        filters.dateRange.start !== null ||
        filters.dateRange.end !== null ||
        filters.confidence.min > 0 ||
        filters.confidence.max < 1 ||
        filters.features.length > 0 ||
        filters.algorithms.length > 0 ||
        filters.status !== 'all'
      );
    }
  };
  
  // Save state to localStorage
  saveToStorage() {
    try {
      const stateToPersist = {
        ui: {
          theme: this.state.ui.theme,
          sidebarCollapsed: this.state.ui.sidebarCollapsed,
          layout: this.state.ui.layout
        },
        charts: this.state.charts,
        preferences: this.state.preferences,
        filters: this.state.filters
      };
      
      localStorage.setItem('dashboardState', JSON.stringify(stateToPersist));
    } catch (error) {
      console.warn('Failed to save state to localStorage:', error);
    }
  }
  
  // Load state from localStorage
  loadFromStorage() {
    try {
      const savedState = localStorage.getItem('dashboardState');
      if (savedState) {
        const parsedState = JSON.parse(savedState);
        
        // Merge with current state
        this.state = {
          ...this.state,
          ui: { ...this.state.ui, ...parsedState.ui },
          charts: { ...this.state.charts, ...parsedState.charts },
          preferences: { ...this.state.preferences, ...parsedState.preferences },
          filters: { ...this.state.filters, ...parsedState.filters }
        };
      }
    } catch (error) {
      console.warn('Failed to load state from localStorage:', error);
    }
  }
  
  // Clear persisted state
  clearStorage() {
    localStorage.removeItem('dashboardState');
  }
  
  // Get state history
  getHistory() {
    return [...this.history];
  }
  
  // Time travel debugging (go back to previous state)
  timeTravel(historyIndex) {
    if (historyIndex >= 0 && historyIndex < this.history.length) {
      const historicalState = this.history[historyIndex].prevState;
      this.state = historicalState;
      this.notifyListeners({ type: 'TIME_TRAVEL' }, this.state, historicalState);
    }
  }
  
  // Debug helpers
  debug = {
    logState: () => console.log('Current State:', this.getState()),
    logHistory: () => console.log('State History:', this.getHistory()),
    logListeners: () => console.log('Active Listeners:', this.listeners.size),
    exportState: () => JSON.stringify(this.getState(), null, 2),
    importState: (stateJson) => {
      try {
        const importedState = JSON.parse(stateJson);
        this.state = importedState;
        this.notifyListeners({ type: 'STATE_IMPORTED' }, {}, importedState);
      } catch (error) {
        console.error('Failed to import state:', error);
      }
    }
  };
}

// Middleware for logging actions (development)
export const loggingMiddleware = (action, prevState, store) => {
  console.group(`Action: ${action.type}`);
  console.log('Previous State:', prevState);
  console.log('Action:', action);
  console.log('New State:', store.getState());
  console.groupEnd();
  return action;
};

// Middleware for async actions
export const asyncMiddleware = (action, prevState, store) => {
  if (typeof action === 'function') {
    // Thunk-style async action
    action(store.dispatch, store.getState);
    return null; // Cancel normal action processing
  }
  return action;
};

// Create global state manager instance
export const dashboardState = new DashboardStateManager();

// Add development middleware in non-production
if (process.env.NODE_ENV !== 'production') {
  dashboardState.addMiddleware(loggingMiddleware);
}

// Add async middleware
dashboardState.addMiddleware(asyncMiddleware);

// Export for external use
export default dashboardState;