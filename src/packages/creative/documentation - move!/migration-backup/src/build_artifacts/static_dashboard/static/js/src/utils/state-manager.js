/**
 * Advanced State Management System
 *
 * Lightweight state management inspired by Redux with reactive subscriptions
 * Features middleware support, time-travel debugging, and async action handling
 */

export class StateManager {
  constructor(initialState = {}, options = {}) {
    this.state = { ...initialState };
    this.listeners = new Set();
    this.middleware = [];
    this.history = [];
    this.historyIndex = -1;
    this.options = {
      maxHistorySize: 50,
      enableTimeTravel: true,
      enableLogging: false,
      persistState: false,
      persistKey: "pynomaly-state",
      ...options,
    };

    this.actionQueue = [];
    this.isDispatching = false;

    this.init();
  }

  init() {
    // Load persisted state
    if (this.options.persistState) {
      this.loadPersistedState();
    }

    // Add default middleware
    if (this.options.enableLogging) {
      this.use(this.createLoggingMiddleware());
    }

    // Save initial state to history
    if (this.options.enableTimeTravel) {
      this.saveToHistory({
        type: "@@INIT",
        payload: null,
      });
    }
  }

  /**
   * Dispatch an action to update state
   */
  dispatch(action) {
    if (typeof action !== "object" || action === null) {
      throw new Error("Action must be a plain object");
    }

    if (typeof action.type !== "string") {
      throw new Error("Action must have a type property");
    }

    // Handle async actions
    if (typeof action === "function") {
      return action(this.dispatch.bind(this), this.getState.bind(this));
    }

    // Queue actions if currently dispatching
    if (this.isDispatching) {
      this.actionQueue.push(action);
      return;
    }

    this.isDispatching = true;

    try {
      // Apply middleware
      const middlewareChain = this.middleware.slice();
      let dispatch = this.dispatchAction.bind(this);

      // Compose middleware
      for (let i = middlewareChain.length - 1; i >= 0; i--) {
        const middleware = middlewareChain[i];
        dispatch = middleware(this)(dispatch);
      }

      // Execute dispatch
      const result = dispatch(action);

      // Process queued actions
      this.processActionQueue();

      return result;
    } finally {
      this.isDispatching = false;
    }
  }

  dispatchAction(action) {
    const prevState = this.state;

    // Create new state
    this.state = this.reduce(this.state, action);

    // Save to history
    if (this.options.enableTimeTravel) {
      this.saveToHistory(action);
    }

    // Persist state
    if (this.options.persistState) {
      this.persistState();
    }

    // Notify listeners
    this.notifyListeners(prevState, this.state, action);

    return action;
  }

  processActionQueue() {
    while (this.actionQueue.length > 0) {
      const action = this.actionQueue.shift();
      this.dispatchAction(action);
    }
  }

  /**
   * Get current state
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener) {
    if (typeof listener !== "function") {
      throw new Error("Listener must be a function");
    }

    this.listeners.add(listener);

    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Add middleware
   */
  use(middleware) {
    if (typeof middleware !== "function") {
      throw new Error("Middleware must be a function");
    }

    this.middleware.push(middleware);
    return this;
  }

  /**
   * Main reducer function
   */
  reduce(state, action) {
    // Handle built-in actions
    switch (action.type) {
      case "@@STATE/RESET":
        return action.payload || {};

      case "@@STATE/MERGE":
        return { ...state, ...action.payload };

      case "@@STATE/SET_PROPERTY":
        return this.setNestedProperty(
          state,
          action.payload.path,
          action.payload.value,
        );

      case "@@STATE/DELETE_PROPERTY":
        return this.deleteNestedProperty(state, action.payload.path);

      default:
        return this.handleCustomAction(state, action);
    }
  }

  /**
   * Handle custom actions - override in subclasses
   */
  handleCustomAction(state, action) {
    // Default: return state unchanged
    return state;
  }

  setNestedProperty(obj, path, value) {
    const keys = Array.isArray(path) ? path : path.split(".");
    const result = { ...obj };
    let current = result;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      current[key] = { ...current[key] };
      current = current[key];
    }

    current[keys[keys.length - 1]] = value;
    return result;
  }

  deleteNestedProperty(obj, path) {
    const keys = Array.isArray(path) ? path : path.split(".");
    const result = { ...obj };
    let current = result;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      current[key] = { ...current[key] };
      current = current[key];
    }

    delete current[keys[keys.length - 1]];
    return result;
  }

  notifyListeners(prevState, nextState, action) {
    this.listeners.forEach((listener) => {
      try {
        listener(nextState, prevState, action);
      } catch (error) {
        console.error("Error in state listener:", error);
      }
    });
  }

  saveToHistory(action) {
    // Remove future history if we're not at the end
    if (this.historyIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.historyIndex + 1);
    }

    // Add new state to history
    this.history.push({
      state: { ...this.state },
      action: { ...action },
      timestamp: Date.now(),
    });

    // Limit history size
    if (this.history.length > this.options.maxHistorySize) {
      this.history.shift();
    } else {
      this.historyIndex++;
    }
  }

  // Time travel methods
  undo() {
    if (!this.options.enableTimeTravel || this.historyIndex <= 0) {
      return false;
    }

    this.historyIndex--;
    const historyEntry = this.history[this.historyIndex];
    const prevState = this.state;

    this.state = { ...historyEntry.state };
    this.notifyListeners(prevState, this.state, {
      type: "@@HISTORY/UNDO",
      payload: { historyIndex: this.historyIndex },
    });

    return true;
  }

  redo() {
    if (
      !this.options.enableTimeTravel ||
      this.historyIndex >= this.history.length - 1
    ) {
      return false;
    }

    this.historyIndex++;
    const historyEntry = this.history[this.historyIndex];
    const prevState = this.state;

    this.state = { ...historyEntry.state };
    this.notifyListeners(prevState, this.state, {
      type: "@@HISTORY/REDO",
      payload: { historyIndex: this.historyIndex },
    });

    return true;
  }

  jumpToHistory(index) {
    if (
      !this.options.enableTimeTravel ||
      index < 0 ||
      index >= this.history.length
    ) {
      return false;
    }

    const historyEntry = this.history[index];
    const prevState = this.state;

    this.historyIndex = index;
    this.state = { ...historyEntry.state };
    this.notifyListeners(prevState, this.state, {
      type: "@@HISTORY/JUMP",
      payload: { historyIndex: index },
    });

    return true;
  }

  getHistory() {
    return this.history.map((entry, index) => ({
      ...entry,
      index,
      isCurrent: index === this.historyIndex,
    }));
  }

  // State persistence
  persistState() {
    try {
      localStorage.setItem(this.options.persistKey, JSON.stringify(this.state));
    } catch (error) {
      console.warn("Failed to persist state:", error);
    }
  }

  loadPersistedState() {
    try {
      const saved = localStorage.getItem(this.options.persistKey);
      if (saved) {
        this.state = { ...this.state, ...JSON.parse(saved) };
      }
    } catch (error) {
      console.warn("Failed to load persisted state:", error);
    }
  }

  clearPersistedState() {
    try {
      localStorage.removeItem(this.options.persistKey);
    } catch (error) {
      console.warn("Failed to clear persisted state:", error);
    }
  }

  // Middleware creators
  createLoggingMiddleware() {
    return (store) => (next) => (action) => {
      const prevState = store.getState();
      console.group(`🎯 Action: ${action.type}`);
      console.log("📤 Dispatching:", action);
      console.log("📋 Previous state:", prevState);

      const result = next(action);

      console.log("📋 Next state:", store.getState());
      console.groupEnd();

      return result;
    };
  }

  createAsyncMiddleware() {
    return (store) => (next) => (action) => {
      if (typeof action === "function") {
        return action(store.dispatch, store.getState);
      }

      if (action && typeof action.then === "function") {
        return action.then(store.dispatch);
      }

      return next(action);
    };
  }

  createValidationMiddleware(validators = {}) {
    return (store) => (next) => (action) => {
      const validator = validators[action.type];
      if (validator && !validator(action, store.getState())) {
        console.warn(`Validation failed for action: ${action.type}`);
        return action;
      }

      return next(action);
    };
  }

  // Action creators
  createActions() {
    return {
      reset: (state = {}) => ({
        type: "@@STATE/RESET",
        payload: state,
      }),

      merge: (updates) => ({
        type: "@@STATE/MERGE",
        payload: updates,
      }),

      set: (path, value) => ({
        type: "@@STATE/SET_PROPERTY",
        payload: { path, value },
      }),

      delete: (path) => ({
        type: "@@STATE/DELETE_PROPERTY",
        payload: { path },
      }),
    };
  }

  // Utility methods
  select(selector) {
    if (typeof selector === "string") {
      return this.getNestedProperty(this.state, selector);
    }

    if (typeof selector === "function") {
      return selector(this.state);
    }

    return this.state;
  }

  getNestedProperty(obj, path) {
    const keys = Array.isArray(path) ? path : path.split(".");
    return keys.reduce((current, key) => current && current[key], obj);
  }

  // Batching support
  batch(actions) {
    const prevState = this.state;
    let finalAction = {
      type: "@@BATCH",
      payload: actions,
    };

    // Apply all actions without notifying listeners
    const originalNotify = this.notifyListeners;
    this.notifyListeners = () => {}; // Temporarily disable notifications

    try {
      actions.forEach((action) => this.dispatch(action));
      finalAction.payload = { actions, count: actions.length };
    } finally {
      this.notifyListeners = originalNotify;
    }

    // Notify listeners once with all changes
    this.notifyListeners(prevState, this.state, finalAction);
  }

  // DevTools integration
  connectDevTools() {
    if (typeof window !== "undefined" && window.__REDUX_DEVTOOLS_EXTENSION__) {
      this.devTools = window.__REDUX_DEVTOOLS_EXTENSION__.connect({
        name: "Pynomaly State Manager",
      });

      this.devTools.init(this.state);

      this.subscribe((state, prevState, action) => {
        this.devTools.send(action, state);
      });

      this.devTools.subscribe((message) => {
        if (message.type === "DISPATCH") {
          switch (message.payload.type) {
            case "JUMP_TO_STATE":
            case "JUMP_TO_ACTION":
              this.state = JSON.parse(message.state);
              this.notifyListeners({}, this.state, { type: "@@DEVTOOLS" });
              break;
          }
        }
      });
    }
  }

  destroy() {
    this.listeners.clear();
    this.middleware = [];
    this.history = [];

    if (this.devTools) {
      this.devTools.disconnect();
    }
  }
}

/**
 * React-like hooks for state management
 */
export class StateHooks {
  constructor(stateManager) {
    this.stateManager = stateManager;
    this.components = new Map();
  }

  useState(component, selector = (state) => state) {
    if (!this.components.has(component)) {
      this.components.set(component, {
        selectors: new Set(),
        unsubscribe: null,
      });
    }

    const componentData = this.components.get(component);
    componentData.selectors.add(selector);

    // Subscribe to state changes if not already subscribed
    if (!componentData.unsubscribe) {
      componentData.unsubscribe = this.stateManager.subscribe(
        (state, prevState, action) => {
          // Check if any selector results changed
          const hasChanges = Array.from(componentData.selectors).some((sel) => {
            const current = sel(state);
            const previous = sel(prevState);
            return !this.shallowEqual(current, previous);
          });

          if (hasChanges && typeof component.forceUpdate === "function") {
            component.forceUpdate();
          } else if (hasChanges && typeof component.render === "function") {
            component.render();
          }
        },
      );
    }

    return [
      this.stateManager.select(selector),
      this.stateManager.dispatch.bind(this.stateManager),
    ];
  }

  useDispatch() {
    return this.stateManager.dispatch.bind(this.stateManager);
  }

  useSelector(selector) {
    return this.stateManager.select(selector);
  }

  useActions(actionCreators) {
    const dispatch = this.stateManager.dispatch.bind(this.stateManager);

    if (typeof actionCreators === "function") {
      return (...args) => dispatch(actionCreators(...args));
    }

    if (typeof actionCreators === "object") {
      const boundActions = {};
      Object.keys(actionCreators).forEach((key) => {
        boundActions[key] = (...args) => dispatch(actionCreators[key](...args));
      });
      return boundActions;
    }

    return dispatch;
  }

  cleanup(component) {
    const componentData = this.components.get(component);
    if (componentData && componentData.unsubscribe) {
      componentData.unsubscribe();
      this.components.delete(component);
    }
  }

  shallowEqual(obj1, obj2) {
    if (obj1 === obj2) return true;

    if (
      typeof obj1 !== "object" ||
      typeof obj2 !== "object" ||
      obj1 === null ||
      obj2 === null
    ) {
      return false;
    }

    const keys1 = Object.keys(obj1);
    const keys2 = Object.keys(obj2);

    if (keys1.length !== keys2.length) return false;

    for (let key of keys1) {
      if (obj1[key] !== obj2[key]) return false;
    }

    return true;
  }
}

/**
 * Anomaly Detection specific state manager
 */
export class AnomalyStateManager extends StateManager {
  constructor(initialState = {}) {
    const defaultState = {
      datasets: [],
      models: [],
      results: [],
      currentDataset: null,
      currentModel: null,
      isLoading: false,
      error: null,
      ui: {
        activeTab: "overview",
        sidebarOpen: true,
        notifications: [],
        theme: "light",
      },
      realTime: {
        connected: false,
        streamingData: [],
        alerts: [],
      },
      ...initialState,
    };

    super(defaultState, {
      enableTimeTravel: true,
      enableLogging: process.env.NODE_ENV === "development",
      persistState: true,
      persistKey: "pynomaly-app-state",
    });

    // Add async middleware
    this.use(this.createAsyncMiddleware());
  }

  handleCustomAction(state, action) {
    switch (action.type) {
      case "DATASETS/LOAD_REQUEST":
        return { ...state, isLoading: true, error: null };

      case "DATASETS/LOAD_SUCCESS":
        return {
          ...state,
          datasets: action.payload,
          isLoading: false,
          error: null,
        };

      case "DATASETS/LOAD_FAILURE":
        return {
          ...state,
          datasets: [],
          isLoading: false,
          error: action.payload,
        };

      case "DATASETS/SELECT":
        return { ...state, currentDataset: action.payload };

      case "MODELS/LOAD_SUCCESS":
        return { ...state, models: action.payload };

      case "MODELS/SELECT":
        return { ...state, currentModel: action.payload };

      case "RESULTS/ADD":
        return {
          ...state,
          results: [...state.results, action.payload],
        };

      case "RESULTS/UPDATE":
        return {
          ...state,
          results: state.results.map((result) =>
            result.id === action.payload.id
              ? { ...result, ...action.payload }
              : result,
          ),
        };

      case "UI/SET_ACTIVE_TAB":
        return {
          ...state,
          ui: { ...state.ui, activeTab: action.payload },
        };

      case "UI/TOGGLE_SIDEBAR":
        return {
          ...state,
          ui: { ...state.ui, sidebarOpen: !state.ui.sidebarOpen },
        };

      case "UI/ADD_NOTIFICATION":
        return {
          ...state,
          ui: {
            ...state.ui,
            notifications: [...state.ui.notifications, action.payload],
          },
        };

      case "UI/REMOVE_NOTIFICATION":
        return {
          ...state,
          ui: {
            ...state.ui,
            notifications: state.ui.notifications.filter(
              (notification) => notification.id !== action.payload,
            ),
          },
        };

      case "REALTIME/CONNECT":
        return {
          ...state,
          realTime: { ...state.realTime, connected: true },
        };

      case "REALTIME/DISCONNECT":
        return {
          ...state,
          realTime: { ...state.realTime, connected: false },
        };

      case "REALTIME/ADD_DATA":
        return {
          ...state,
          realTime: {
            ...state.realTime,
            streamingData: [...state.realTime.streamingData, action.payload],
          },
        };

      case "REALTIME/ADD_ALERT":
        return {
          ...state,
          realTime: {
            ...state.realTime,
            alerts: [...state.realTime.alerts, action.payload],
          },
        };

      default:
        return state;
    }
  }
}

/**
 * Action creators for anomaly detection
 */
export const anomalyActions = {
  // Dataset actions
  loadDatasets: () => async (dispatch, getState) => {
    dispatch({ type: "DATASETS/LOAD_REQUEST" });

    try {
      const response = await fetch("/api/datasets");
      const datasets = await response.json();
      dispatch({ type: "DATASETS/LOAD_SUCCESS", payload: datasets });
    } catch (error) {
      dispatch({ type: "DATASETS/LOAD_FAILURE", payload: error.message });
    }
  },

  selectDataset: (dataset) => ({
    type: "DATASETS/SELECT",
    payload: dataset,
  }),

  // Model actions
  loadModels: () => async (dispatch) => {
    try {
      const response = await fetch("/api/models");
      const models = await response.json();
      dispatch({ type: "MODELS/LOAD_SUCCESS", payload: models });
    } catch (error) {
      console.error("Failed to load models:", error);
    }
  },

  selectModel: (model) => ({
    type: "MODELS/SELECT",
    payload: model,
  }),

  // Result actions
  addResult: (result) => ({
    type: "RESULTS/ADD",
    payload: { ...result, id: Date.now(), timestamp: new Date().toISOString() },
  }),

  updateResult: (id, updates) => ({
    type: "RESULTS/UPDATE",
    payload: { id, ...updates },
  }),

  // UI actions
  setActiveTab: (tab) => ({
    type: "UI/SET_ACTIVE_TAB",
    payload: tab,
  }),

  toggleSidebar: () => ({
    type: "UI/TOGGLE_SIDEBAR",
  }),

  addNotification: (message, type = "info", duration = 5000) => ({
    type: "UI/ADD_NOTIFICATION",
    payload: {
      id: Date.now(),
      message,
      type,
      timestamp: Date.now(),
      duration,
    },
  }),

  removeNotification: (id) => ({
    type: "UI/REMOVE_NOTIFICATION",
    payload: id,
  }),

  // Real-time actions
  connectRealTime: () => ({
    type: "REALTIME/CONNECT",
  }),

  disconnectRealTime: () => ({
    type: "REALTIME/DISCONNECT",
  }),

  addStreamingData: (data) => ({
    type: "REALTIME/ADD_DATA",
    payload: data,
  }),

  addAlert: (alert) => ({
    type: "REALTIME/ADD_ALERT",
    payload: {
      ...alert,
      id: Date.now(),
      timestamp: new Date().toISOString(),
    },
  }),
};

// Default export
export default StateManager;
