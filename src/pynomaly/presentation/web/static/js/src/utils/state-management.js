/**
 * State Management System for Pynomaly
 * Zustand-based state management for complex component interactions
 * and data flow with persistence, middleware, and real-time updates
 */

/**
 * Simple Zustand-like state management implementation
 * Provides reactive state management with subscriptions and middleware
 */
class StateStore {
  constructor(createState) {
    this.state = {};
    this.listeners = new Set();
    this.middlewares = [];
    this.slices = new Map();
    this.persistConfig = null;
    
    // Initialize state
    const setState = this.createSetState();
    const getState = () => this.state;
    this.state = createState(setState, getState, this);
    
    // Load persisted state if configured
    this.loadPersistedState();
  }

  createSetState() {
    return (partial, replace = false) => {
      const nextState = typeof partial === 'function' 
        ? partial(this.state) 
        : partial;
        
      const prevState = this.state;
      this.state = replace ? nextState : { ...this.state, ...nextState };
      
      // Apply middleware
      this.middlewares.forEach(middleware => {
        middleware(this.state, prevState, this);
      });
      
      // Persist state if configured
      this.persistState();
      
      // Notify subscribers
      this.listeners.forEach(listener => listener(this.state, prevState));
    };
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getState() {
    return this.state;
  }

  addMiddleware(middleware) {
    this.middlewares.push(middleware);
  }

  addSlice(name, slice) {
    this.slices.set(name, slice);
  }

  getSlice(name) {
    return this.slices.get(name);
  }

  configurePersistence(config) {
    this.persistConfig = config;
  }

  persistState() {
    if (!this.persistConfig) return;
    
    try {
      const { key, storage = localStorage, whitelist, blacklist } = this.persistConfig;
      let stateToSave = this.state;
      
      if (whitelist) {
        stateToSave = Object.keys(this.state)
          .filter(key => whitelist.includes(key))
          .reduce((obj, key) => {
            obj[key] = this.state[key];
            return obj;
          }, {});
      }
      
      if (blacklist) {
        stateToSave = Object.keys(this.state)
          .filter(key => !blacklist.includes(key))
          .reduce((obj, key) => {
            obj[key] = this.state[key];
            return obj;
          }, {});
      }
      
      storage.setItem(key, JSON.stringify(stateToSave));
    } catch (error) {
      console.warn('Failed to persist state:', error);
    }
  }

  loadPersistedState() {
    if (!this.persistConfig) return;
    
    try {
      const { key, storage = localStorage } = this.persistConfig;
      const persistedState = storage.getItem(key);
      
      if (persistedState) {
        const parsed = JSON.parse(persistedState);
        this.state = { ...this.state, ...parsed };
      }
    } catch (error) {
      console.warn('Failed to load persisted state:', error);
    }
  }
}

/**
 * Main Application Store
 * Central state management for the entire Pynomaly application
 */
const createAppStore = (set, get) => ({
  // UI State
  ui: {
    theme: 'light',
    sidebarOpen: true,
    loading: false,
    notifications: [],
    modal: null,
    activeView: 'dashboard',
    layout: 'default'
  },

  // User State
  user: {
    isAuthenticated: false,
    profile: null,
    preferences: {
      chartAnimations: true,
      realTimeUpdates: true,
      accessibilityMode: false,
      dataRefreshInterval: 30000
    },
    permissions: []
  },

  // Data State
  data: {
    datasets: [],
    models: [],
    detectionResults: [],
    performanceMetrics: [],
    realTimeData: {
      isConnected: false,
      lastUpdate: null,
      buffer: []
    }
  },

  // Chart State
  charts: {
    instances: new Map(),
    configurations: {},
    themes: {
      light: { /* theme config */ },
      dark: { /* theme config */ }
    },
    activeFilters: {},
    selectedData: null
  },

  // Dashboard State
  dashboard: {
    layout: [],
    widgets: {},
    filters: {},
    refreshInterval: 30000,
    autoRefresh: false
  },

  // Actions
  setTheme: (theme) => set((state) => ({
    ui: { ...state.ui, theme }
  })),

  toggleSidebar: () => set((state) => ({
    ui: { ...state.ui, sidebarOpen: !state.ui.sidebarOpen }
  })),

  setLoading: (loading) => set((state) => ({
    ui: { ...state.ui, loading }
  })),

  addNotification: (notification) => set((state) => ({
    ui: {
      ...state.ui,
      notifications: [...state.ui.notifications, {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        ...notification
      }]
    }
  })),

  removeNotification: (id) => set((state) => ({
    ui: {
      ...state.ui,
      notifications: state.ui.notifications.filter(n => n.id !== id)
    }
  })),

  showModal: (modal) => set((state) => ({
    ui: { ...state.ui, modal }
  })),

  hideModal: () => set((state) => ({
    ui: { ...state.ui, modal: null }
  })),

  setActiveView: (view) => set((state) => ({
    ui: { ...state.ui, activeView: view }
  })),

  // User Actions
  setUser: (user) => set((state) => ({
    user: { ...state.user, ...user }
  })),

  updateUserPreferences: (preferences) => set((state) => ({
    user: {
      ...state.user,
      preferences: { ...state.user.preferences, ...preferences }
    }
  })),

  // Data Actions
  setDatasets: (datasets) => set((state) => ({
    data: { ...state.data, datasets }
  })),

  addDataset: (dataset) => set((state) => ({
    data: {
      ...state.data,
      datasets: [...state.data.datasets, dataset]
    }
  })),

  updateDataset: (id, updates) => set((state) => ({
    data: {
      ...state.data,
      datasets: state.data.datasets.map(d => 
        d.id === id ? { ...d, ...updates } : d
      )
    }
  })),

  removeDataset: (id) => set((state) => ({
    data: {
      ...state.data,
      datasets: state.data.datasets.filter(d => d.id !== id)
    }
  })),

  setModels: (models) => set((state) => ({
    data: { ...state.data, models }
  })),

  addDetectionResult: (result) => set((state) => ({
    data: {
      ...state.data,
      detectionResults: [...state.data.detectionResults, result]
    }
  })),

  updatePerformanceMetrics: (metrics) => set((state) => ({
    data: {
      ...state.data,
      performanceMetrics: [...state.data.performanceMetrics, metrics]
    }
  })),

  // Real-time Data Actions
  setRealTimeConnection: (isConnected) => set((state) => ({
    data: {
      ...state.data,
      realTimeData: {
        ...state.data.realTimeData,
        isConnected,
        lastUpdate: isConnected ? new Date().toISOString() : state.data.realTimeData.lastUpdate
      }
    }
  })),

  addRealTimeData: (data) => set((state) => {
    const buffer = [...state.data.realTimeData.buffer, data];
    // Keep only last 1000 items
    const trimmedBuffer = buffer.slice(-1000);
    
    return {
      data: {
        ...state.data,
        realTimeData: {
          ...state.data.realTimeData,
          buffer: trimmedBuffer,
          lastUpdate: new Date().toISOString()
        }
      }
    };
  }),

  clearRealTimeBuffer: () => set((state) => ({
    data: {
      ...state.data,
      realTimeData: {
        ...state.data.realTimeData,
        buffer: []
      }
    }
  })),

  // Chart Actions
  registerChart: (id, chart) => set((state) => {
    const newInstances = new Map(state.charts.instances);
    newInstances.set(id, chart);
    return {
      charts: { ...state.charts, instances: newInstances }
    };
  }),

  unregisterChart: (id) => set((state) => {
    const newInstances = new Map(state.charts.instances);
    newInstances.delete(id);
    return {
      charts: { ...state.charts, instances: newInstances }
    };
  }),

  updateChartConfiguration: (id, config) => set((state) => ({
    charts: {
      ...state.charts,
      configurations: {
        ...state.charts.configurations,
        [id]: { ...state.charts.configurations[id], ...config }
      }
    }
  })),

  setActiveFilters: (filters) => set((state) => ({
    charts: { ...state.charts, activeFilters: filters }
  })),

  setSelectedData: (data) => set((state) => ({
    charts: { ...state.charts, selectedData: data }
  })),

  // Dashboard Actions
  updateDashboardLayout: (layout) => set((state) => ({
    dashboard: { ...state.dashboard, layout }
  })),

  addWidget: (widget) => set((state) => ({
    dashboard: {
      ...state.dashboard,
      widgets: { ...state.dashboard.widgets, [widget.id]: widget }
    }
  })),

  updateWidget: (id, updates) => set((state) => ({
    dashboard: {
      ...state.dashboard,
      widgets: {
        ...state.dashboard.widgets,
        [id]: { ...state.dashboard.widgets[id], ...updates }
      }
    }
  })),

  removeWidget: (id) => set((state) => {
    const newWidgets = { ...state.dashboard.widgets };
    delete newWidgets[id];
    return {
      dashboard: { ...state.dashboard, widgets: newWidgets }
    };
  }),

  setDashboardFilters: (filters) => set((state) => ({
    dashboard: { ...state.dashboard, filters }
  })),

  setAutoRefresh: (autoRefresh) => set((state) => ({
    dashboard: { ...state.dashboard, autoRefresh }
  })),

  setRefreshInterval: (interval) => set((state) => ({
    dashboard: { ...state.dashboard, refreshInterval: interval }
  }))
});

// Create the main application store
const appStore = new StateStore(createAppStore);

// Configure persistence
appStore.configurePersistence({
  key: 'pynomaly-app-state',
  storage: localStorage,
  whitelist: ['user', 'dashboard', 'ui'],
  blacklist: ['data.realTimeData']
});

/**
 * Middleware for logging state changes
 */
const loggingMiddleware = (currentState, previousState, store) => {
  if (process.env.NODE_ENV === 'development') {
    console.group('State Update');
    console.log('Previous State:', previousState);
    console.log('Current State:', currentState);
    console.groupEnd();
  }
};

/**
 * Middleware for analytics tracking
 */
const analyticsMiddleware = (currentState, previousState, store) => {
  // Track significant state changes
  if (currentState.ui.activeView !== previousState.ui.activeView) {
    // Track view changes
    if (window.gtag) {
      window.gtag('event', 'page_view', {
        page_title: currentState.ui.activeView,
        page_location: window.location.href
      });
    }
  }

  if (currentState.data.detectionResults.length > previousState.data.detectionResults.length) {
    // Track new detections
    if (window.gtag) {
      window.gtag('event', 'anomaly_detected', {
        custom_parameter: currentState.data.detectionResults.length
      });
    }
  }
};

/**
 * Middleware for accessibility announcements
 */
const accessibilityMiddleware = (currentState, previousState, store) => {
  const announcer = document.getElementById('state-announcer') || 
                   document.querySelector('[aria-live="polite"]');
  
  if (!announcer) return;

  // Announce loading state changes
  if (currentState.ui.loading !== previousState.ui.loading) {
    if (currentState.ui.loading) {
      announcer.textContent = 'Loading data, please wait';
    } else {
      announcer.textContent = 'Data loaded successfully';
    }
  }

  // Announce new notifications
  if (currentState.ui.notifications.length > previousState.ui.notifications.length) {
    const newNotifications = currentState.ui.notifications.slice(previousState.ui.notifications.length);
    newNotifications.forEach(notification => {
      announcer.textContent = `${notification.type}: ${notification.message}`;
    });
  }

  // Announce real-time connection changes
  if (currentState.data.realTimeData.isConnected !== previousState.data.realTimeData.isConnected) {
    announcer.textContent = currentState.data.realTimeData.isConnected 
      ? 'Real-time data connection established'
      : 'Real-time data connection lost';
  }
};

// Add middleware to store
appStore.addMiddleware(loggingMiddleware);
appStore.addMiddleware(analyticsMiddleware);
appStore.addMiddleware(accessibilityMiddleware);

/**
 * Chart State Slice
 * Specialized state management for chart interactions
 */
const createChartSlice = (set, get) => ({
  selectedPoints: [],
  hoveredPoint: null,
  brushSelection: null,
  zoomLevel: 1,
  filters: {
    timeRange: null,
    confidenceThreshold: 0,
    anomalyTypes: []
  },
  interactions: {
    brushEnabled: true,
    zoomEnabled: true,
    tooltipsEnabled: true
  },

  // Actions
  selectPoints: (points) => set((state) => ({
    selectedPoints: points
  })),

  setHoveredPoint: (point) => set((state) => ({
    hoveredPoint: point
  })),

  setBrushSelection: (selection) => set((state) => ({
    brushSelection: selection
  })),

  setZoomLevel: (level) => set((state) => ({
    zoomLevel: level
  })),

  updateFilters: (filters) => set((state) => ({
    filters: { ...state.filters, ...filters }
  })),

  resetFilters: () => set((state) => ({
    filters: {
      timeRange: null,
      confidenceThreshold: 0,
      anomalyTypes: []
    }
  })),

  setInteractions: (interactions) => set((state) => ({
    interactions: { ...state.interactions, ...interactions }
  }))
});

const chartStore = new StateStore(createChartSlice);
appStore.addSlice('charts', chartStore);

/**
 * Form State Slice
 * Specialized state management for complex forms
 */
const createFormSlice = (set, get) => ({
  forms: {},
  validations: {},
  submissions: {},

  // Actions
  createForm: (formId, initialData = {}) => set((state) => ({
    forms: {
      ...state.forms,
      [formId]: {
        id: formId,
        data: initialData,
        touched: {},
        errors: {},
        isValid: true,
        isSubmitting: false,
        isDirty: false
      }
    }
  })),

  updateFormField: (formId, field, value) => set((state) => {
    const form = state.forms[formId];
    if (!form) return state;

    const newData = { ...form.data, [field]: value };
    const touched = { ...form.touched, [field]: true };
    
    // Run validation
    const validator = state.validations[formId];
    const errors = validator ? validator(newData) : {};
    const isValid = Object.keys(errors).length === 0;
    
    return {
      forms: {
        ...state.forms,
        [formId]: {
          ...form,
          data: newData,
          touched,
          errors,
          isValid,
          isDirty: true
        }
      }
    };
  }),

  setFormValidation: (formId, validator) => set((state) => ({
    validations: {
      ...state.validations,
      [formId]: validator
    }
  })),

  setFormSubmitting: (formId, isSubmitting) => set((state) => ({
    forms: {
      ...state.forms,
      [formId]: {
        ...state.forms[formId],
        isSubmitting
      }
    }
  })),

  resetForm: (formId) => set((state) => {
    const form = state.forms[formId];
    if (!form) return state;

    return {
      forms: {
        ...state.forms,
        [formId]: {
          ...form,
          touched: {},
          errors: {},
          isValid: true,
          isSubmitting: false,
          isDirty: false
        }
      }
    };
  }),

  removeForm: (formId) => set((state) => {
    const newForms = { ...state.forms };
    const newValidations = { ...state.validations };
    const newSubmissions = { ...state.submissions };
    
    delete newForms[formId];
    delete newValidations[formId];
    delete newSubmissions[formId];
    
    return {
      forms: newForms,
      validations: newValidations,
      submissions: newSubmissions
    };
  })
});

const formStore = new StateStore(createFormSlice);
appStore.addSlice('forms', formStore);

/**
 * Real-time Data Manager
 * Handles WebSocket connections and real-time updates
 */
class RealTimeManager {
  constructor(store) {
    this.store = store;
    this.websocket = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.heartbeatInterval = null;
  }

  connect(url = 'ws://localhost:8000/ws') {
    try {
      this.websocket = new WebSocket(url);
      this.setupEventListeners();
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      this.store.getState().setRealTimeConnection(false);
    }
  }

  setupEventListeners() {
    this.websocket.onopen = () => {
      console.log('WebSocket connected');
      this.store.getState().setRealTimeConnection(true);
      this.reconnectAttempts = 0;
      this.startHeartbeat();
    };

    this.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.websocket.onclose = () => {
      console.log('WebSocket disconnected');
      this.store.getState().setRealTimeConnection(false);
      this.stopHeartbeat();
      this.attemptReconnect();
    };

    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.store.getState().addNotification({
        type: 'error',
        message: 'Real-time connection error',
        duration: 5000
      });
    };
  }

  handleMessage(data) {
    const { type, payload } = data;

    switch (type) {
      case 'anomaly_detected':
        this.store.getState().addDetectionResult(payload);
        this.store.getState().addNotification({
          type: 'warning',
          message: `Anomaly detected: ${payload.type}`,
          duration: 10000
        });
        break;

      case 'performance_update':
        this.store.getState().updatePerformanceMetrics(payload);
        this.store.getState().addRealTimeData(payload);
        break;

      case 'system_alert':
        this.store.getState().addNotification({
          type: payload.severity,
          message: payload.message,
          duration: payload.severity === 'error' ? 0 : 5000
        });
        break;

      case 'data_update':
        this.store.getState().addRealTimeData(payload);
        break;

      default:
        console.warn('Unknown message type:', type);
    }
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  }

  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
      this.store.getState().addNotification({
        type: 'error',
        message: 'Real-time connection failed. Please refresh the page.',
        duration: 0
      });
    }
  }

  disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    this.stopHeartbeat();
  }

  send(message) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
    }
  }
}

/**
 * React-like hooks for state management
 */
const useStore = (selector) => {
  const state = appStore.getState();
  return selector ? selector(state) : state;
};

const useStoreSubscription = (selector, callback) => {
  const unsubscribe = appStore.subscribe((state, prevState) => {
    const current = selector(state);
    const previous = selector(prevState);
    
    if (current !== previous) {
      callback(current, previous);
    }
  });
  
  return unsubscribe;
};

/**
 * Computed state selectors
 */
const selectors = {
  // UI Selectors
  getTheme: (state) => state.ui.theme,
  getLoading: (state) => state.ui.loading,
  getNotifications: (state) => state.ui.notifications,
  getActiveView: (state) => state.ui.activeView,

  // User Selectors
  getUser: (state) => state.user,
  getUserPreferences: (state) => state.user.preferences,
  isAuthenticated: (state) => state.user.isAuthenticated,

  // Data Selectors
  getDatasets: (state) => state.data.datasets,
  getModels: (state) => state.data.models,
  getDetectionResults: (state) => state.data.detectionResults,
  getPerformanceMetrics: (state) => state.data.performanceMetrics,
  getRealTimeData: (state) => state.data.realTimeData,

  // Computed Selectors
  getRecentAnomalies: (state) => {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    return state.data.detectionResults.filter(result => 
      new Date(result.timestamp) > oneDayAgo
    );
  },

  getHighConfidenceAnomalies: (state) => {
    return state.data.detectionResults.filter(result => 
      result.confidence > 0.8
    );
  },

  getAnomalyTypeDistribution: (state) => {
    const distribution = {};
    state.data.detectionResults.forEach(result => {
      const type = result.type || 'Unknown';
      distribution[type] = (distribution[type] || 0) + 1;
    });
    return distribution;
  },

  getSystemHealth: (state) => {
    const recentAnomalies = selectors.getRecentAnomalies(state);
    const criticalCount = recentAnomalies.filter(a => a.confidence > 0.95).length;
    
    if (criticalCount > 5) return 'critical';
    if (criticalCount > 2) return 'warning';
    if (recentAnomalies.length > 10) return 'caution';
    return 'good';
  },

  // Chart Selectors
  getChartInstances: (state) => state.charts.instances,
  getActiveFilters: (state) => state.charts.activeFilters,
  getSelectedData: (state) => state.charts.selectedData,

  // Dashboard Selectors
  getDashboardLayout: (state) => state.dashboard.layout,
  getDashboardWidgets: (state) => state.dashboard.widgets,
  getDashboardFilters: (state) => state.dashboard.filters,
  isAutoRefreshEnabled: (state) => state.dashboard.autoRefresh
};

// Initialize real-time manager
const realTimeManager = new RealTimeManager(appStore);

// Auto-connect on page load if user preferences allow
document.addEventListener('DOMContentLoaded', () => {
  const userPrefs = useStore(selectors.getUserPreferences);
  if (userPrefs.realTimeUpdates) {
    realTimeManager.connect();
  }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    StateStore,
    appStore,
    chartStore,
    formStore,
    RealTimeManager,
    realTimeManager,
    useStore,
    useStoreSubscription,
    selectors
  };
} else {
  // Browser environment
  window.StateStore = StateStore;
  window.appStore = appStore;
  window.chartStore = chartStore;
  window.formStore = formStore;
  window.RealTimeManager = RealTimeManager;
  window.realTimeManager = realTimeManager;
  window.useStore = useStore;
  window.useStoreSubscription = useStoreSubscription;
  window.selectors = selectors;
}