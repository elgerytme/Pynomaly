import { AnomalyTimeline } from '../src/pynomaly/presentation/web/static/js/src/components/anomaly-timeline.js';
import { DashboardLayout, WidgetRegistry, defaultWidgets } from '../src/pynomaly/presentation/web/static/js/src/components/dashboard-layout.js';
import { MultiStepForm, anomalyDetectionFormSteps } from '../src/pynomaly/presentation/web/static/js/src/components/multi-step-form.js';
import { AnomalyStateManager, anomalyActions } from '../src/pynomaly/presentation/web/static/js/src/utils/state-manager.js';

export default {
  title: 'Advanced Components',
  parameters: {
    docs: {
      description: {
        component: 'Advanced interactive components for anomaly detection workflows, featuring real-time data visualization, configurable dashboards, and complex form management.'
      }
    }
  }
};

// Anomaly Timeline Component
export const AnomalyTimelineVisualization = () => {
  const container = document.createElement('div');
  container.style.width = '100%';
  container.style.height = '500px';
  container.style.border = '1px solid #e2e8f0';
  container.style.borderRadius = '8px';
  container.style.padding = '20px';
  container.style.backgroundColor = 'white';
  
  // Generate sample anomaly data
  const generateAnomalyData = () => {
    const data = [];
    const now = new Date();
    const startTime = new Date(now.getTime() - 24 * 60 * 60 * 1000); // 24 hours ago
    
    for (let i = 0; i < 100; i++) {
      const timestamp = new Date(startTime.getTime() + (i * 14.4 * 60 * 1000)); // Every 14.4 minutes
      const baseScore = Math.random() * 0.3; // Normal baseline
      const isAnomaly = Math.random() < 0.1; // 10% chance of anomaly
      const score = isAnomaly ? baseScore + Math.random() * 0.7 + 0.3 : baseScore;
      
      data.push({
        id: `anomaly_${i}`,
        timestamp: timestamp,
        score: Math.min(1, score),
        severity: score > 0.8 ? 'critical' : score > 0.6 ? 'high' : score > 0.4 ? 'medium' : 'low',
        description: isAnomaly ? `Anomaly detected at ${timestamp.toLocaleTimeString()}` : `Normal behavior`,
        features: isAnomaly ? ['cpu_usage', 'memory_consumption', 'network_traffic'] : ['cpu_usage']
      });
    }
    
    return data;
  };
  
  // Initialize timeline
  setTimeout(() => {
    const timeline = new AnomalyTimeline(container, {
      width: container.clientWidth - 40,
      height: 400,
      enableZoom: true,
      enableBrush: true,
      showTooltip: true
    });
    
    timeline.setData(generateAnomalyData());
    
    // Add event listeners for demo
    container.addEventListener('anomalySelected', (e) => {
      console.log('Anomaly selected:', e.detail.anomaly);
    });
    
    container.addEventListener('timeRangeFiltered', (e) => {
      console.log('Time range filtered:', e.detail.range);
    });
    
    // Add demo controls
    const controls = document.createElement('div');
    controls.style.marginTop = '20px';
    controls.style.display = 'flex';
    controls.style.gap = '10px';
    controls.style.flexWrap = 'wrap';
    
    const addDataBtn = document.createElement('button');
    addDataBtn.textContent = 'Add Real-time Data';
    addDataBtn.className = 'btn btn-primary btn-sm';
    addDataBtn.onclick = () => {
      const newData = [{
        id: `rt_${Date.now()}`,
        timestamp: new Date(),
        score: Math.random(),
        severity: Math.random() > 0.7 ? 'high' : 'medium'
      }];
      timeline.addRealTimeData(newData);
    };
    
    const exportBtn = document.createElement('button');
    exportBtn.textContent = 'Export Data';
    exportBtn.className = 'btn btn-secondary btn-sm';
    exportBtn.onclick = () => {
      const data = timeline.exportData();
      console.log('Exported data:', data);
      alert('Data exported to console');
    };
    
    const filterBtn = document.createElement('button');
    filterBtn.textContent = 'Filter High Severity';
    filterBtn.className = 'btn btn-warning btn-sm';
    filterBtn.onclick = () => {
      timeline.filterBySeverity(['high', 'critical']);
    };
    
    const resetBtn = document.createElement('button');
    resetBtn.textContent = 'Reset Filters';
    resetBtn.className = 'btn btn-ghost btn-sm';
    resetBtn.onclick = () => {
      timeline.setData(generateAnomalyData());
    };
    
    controls.appendChild(addDataBtn);
    controls.appendChild(exportBtn);
    controls.appendChild(filterBtn);
    controls.appendChild(resetBtn);
    container.appendChild(controls);
    
    // Store timeline instance for cleanup
    container._timelineInstance = timeline;
  }, 100);
  
  return container;
};

AnomalyTimelineVisualization.parameters = {
  docs: {
    storyDescription: 'Interactive timeline visualization for anomaly detection events with real-time updates, zooming, filtering, and detailed tooltips. Features D3.js-powered charts with brush selection and time-travel capabilities.'
  }
};

// Dashboard Layout System
export const ConfigurableDashboard = () => {
  const container = document.createElement('div');
  container.style.width = '100%';
  container.style.height = '600px';
  container.style.border = '1px solid #e2e8f0';
  container.style.borderRadius = '8px';
  container.style.backgroundColor = '#f8fafc';
  
  // Initialize dashboard
  setTimeout(() => {
    const dashboard = new DashboardLayout(container, {
      columns: 12,
      rowHeight: 60,
      margin: [10, 10],
      isDraggable: true,
      isResizable: true,
      compactType: 'vertical'
    });
    
    // Add sample widgets
    const widgets = [
      {
        id: 'metrics_overview',
        title: 'Metrics Overview',
        type: 'metric',
        x: 0, y: 0, w: 3, h: 2,
        component: () => {
          const div = document.createElement('div');
          div.innerHTML = `
            <div style="text-align: center; padding: 20px;">
              <div style="font-size: 2rem; font-weight: bold; color: #1f2937;">1,234</div>
              <div style="color: #6b7280; margin: 8px 0;">Total Datasets</div>
              <div style="color: #059669; font-weight: 500;">+12.5%</div>
            </div>
          `;
          return div;
        }
      },
      {
        id: 'anomaly_chart',
        title: 'Anomaly Detection Timeline',
        type: 'chart',
        x: 3, y: 0, w: 6, h: 4,
        component: () => {
          const div = document.createElement('div');
          div.style.height = '100%';
          div.style.display = 'flex';
          div.style.alignItems = 'center';
          div.style.justifyContent = 'center';
          div.style.backgroundColor = '#f9fafb';
          div.innerHTML = `
            <div style="text-align: center; color: #6b7280;">
              <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
              <p>Interactive anomaly timeline would render here</p>
            </div>
          `;
          return div;
        }
      },
      {
        id: 'recent_alerts',
        title: 'Recent Alerts',
        type: 'table',
        x: 9, y: 0, w: 3, h: 4,
        component: () => {
          const div = document.createElement('div');
          div.innerHTML = `
            <div style="padding: 16px;">
              <div style="margin-bottom: 12px; padding: 8px; background: #fef2f2; border-left: 4px solid #dc2626; border-radius: 4px;">
                <div style="font-weight: 600; color: #dc2626;">Critical Alert</div>
                <div style="font-size: 0.875rem; color: #6b7280;">High anomaly score detected</div>
              </div>
              <div style="margin-bottom: 12px; padding: 8px; background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 4px;">
                <div style="font-weight: 600; color: #f59e0b;">Warning</div>
                <div style="font-size: 0.875rem; color: #6b7280;">Unusual pattern detected</div>
              </div>
              <div style="padding: 8px; background: #f0f9ff; border-left: 4px solid #3b82f6; border-radius: 4px;">
                <div style="font-weight: 600; color: #3b82f6;">Info</div>
                <div style="font-size: 0.875rem; color: #6b7280;">Processing completed</div>
              </div>
            </div>
          `;
          return div;
        }
      },
      {
        id: 'system_health',
        title: 'System Health',
        type: 'metric',
        x: 0, y: 2, w: 3, h: 2,
        component: () => {
          const div = document.createElement('div');
          div.innerHTML = `
            <div style="text-align: center; padding: 20px;">
              <div style="font-size: 2rem; font-weight: bold; color: #059669;">98.5%</div>
              <div style="color: #6b7280; margin: 8px 0;">Uptime</div>
              <div style="color: #059669; font-weight: 500;">Excellent</div>
            </div>
          `;
          return div;
        }
      },
      {
        id: 'processing_queue',
        title: 'Processing Queue',
        type: 'metric',
        x: 0, y: 4, w: 3, h: 2,
        component: () => {
          const div = document.createElement('div');
          div.innerHTML = `
            <div style="text-align: center; padding: 20px;">
              <div style="font-size: 2rem; font-weight: bold; color: #3b82f6;">23</div>
              <div style="color: #6b7280; margin: 8px 0;">Jobs in Queue</div>
              <div style="color: #6b7280; font-weight: 500;">~5 min wait</div>
            </div>
          `;
          return div;
        }
      }
    ];
    
    widgets.forEach(widget => dashboard.addWidget(widget));
    
    // Add controls
    const controls = document.createElement('div');
    controls.style.position = 'absolute';
    controls.style.top = '10px';
    controls.style.right = '10px';
    controls.style.display = 'flex';
    controls.style.gap = '8px';
    controls.style.zIndex = '1000';
    
    const addWidgetBtn = document.createElement('button');
    addWidgetBtn.textContent = 'Add Widget';
    addWidgetBtn.className = 'btn btn-primary btn-sm';
    addWidgetBtn.onclick = () => {
      dashboard.addWidget({
        title: `New Widget ${Date.now()}`,
        type: 'metric',
        w: 2, h: 2,
        component: () => {
          const div = document.createElement('div');
          div.style.textAlign = 'center';
          div.style.padding = '20px';
          div.innerHTML = `<div>New Widget Content</div>`;
          return div;
        }
      });
    };
    
    const exportLayoutBtn = document.createElement('button');
    exportLayoutBtn.textContent = 'Export Layout';
    exportLayoutBtn.className = 'btn btn-secondary btn-sm';
    exportLayoutBtn.onclick = () => {
      const layout = dashboard.exportLayout();
      console.log('Exported layout:', layout);
      alert('Layout exported to console');
    };
    
    const resetBtn = document.createElement('button');
    resetBtn.textContent = 'Reset Layout';
    resetBtn.className = 'btn btn-ghost btn-sm';
    resetBtn.onclick = () => {
      dashboard.setLayout([]);
      widgets.forEach(widget => dashboard.addWidget(widget));
    };
    
    controls.appendChild(addWidgetBtn);
    controls.appendChild(exportLayoutBtn);
    controls.appendChild(resetBtn);
    container.appendChild(controls);
    
    // Add event listeners
    container.addEventListener('dashboard:widgetAdded', (e) => {
      console.log('Widget added:', e.detail.widget);
    });
    
    container.addEventListener('dashboard:layoutSaved', (e) => {
      console.log('Layout saved:', e.detail.layout);
    });
    
    // Store dashboard instance for cleanup
    container._dashboardInstance = dashboard;
  }, 100);
  
  return container;
};

ConfigurableDashboard.parameters = {
  docs: {
    storyDescription: 'Drag-and-drop dashboard configuration system with responsive grid layouts. Features widget management, layout persistence, real-time updates, and customizable components for anomaly detection monitoring.'
  }
};

// Multi-Step Form Component
export const AnomalyDetectionForm = () => {
  const container = document.createElement('div');
  container.style.width = '100%';
  container.style.maxWidth = '800px';
  container.style.margin = '0 auto';
  container.style.border = '1px solid #e2e8f0';
  container.style.borderRadius = '8px';
  container.style.backgroundColor = 'white';
  container.style.padding = '20px';
  
  // Initialize form
  setTimeout(() => {
    const form = new MultiStepForm(container, {
      showProgress: true,
      showStepNumbers: true,
      allowStepNavigation: true,
      validateOnStepChange: true,
      saveProgress: true,
      progressKey: 'demo-anomaly-form'
    });
    
    // Add steps
    anomalyDetectionFormSteps.forEach(step => form.addStep(step));
    
    // Add event listeners
    container.addEventListener('multiStepForm:stepChanged', (e) => {
      console.log('Step changed:', e.detail);
    });
    
    container.addEventListener('multiStepForm:submitSuccess', (e) => {
      console.log('Form submitted successfully:', e.detail);
      
      // Show success message
      const success = document.createElement('div');
      success.className = 'alert alert-success';
      success.style.marginTop = '20px';
      success.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px;">
          <div style="color: #059669; font-size: 1.5rem;">✓</div>
          <div>
            <div style="font-weight: 600;">Form Submitted Successfully!</div>
            <div style="font-size: 0.875rem; color: #6b7280;">Your anomaly detection job has been queued for processing.</div>
          </div>
        </div>
      `;
      container.appendChild(success);
      
      setTimeout(() => {
        form.reset();
        success.remove();
      }, 3000);
    });
    
    container.addEventListener('multiStepForm:submitError', (e) => {
      console.error('Form submission error:', e.detail);
    });
    
    // Start the form
    form.start();
    
    // Add demo controls
    const controls = document.createElement('div');
    controls.style.marginTop = '20px';
    controls.style.display = 'flex';
    controls.style.gap = '10px';
    controls.style.justifyContent = 'center';
    controls.style.flexWrap = 'wrap';
    
    const fillDemoDataBtn = document.createElement('button');
    fillDemoDataBtn.textContent = 'Fill Demo Data';
    fillDemoDataBtn.className = 'btn btn-outline btn-sm';
    fillDemoDataBtn.onclick = () => {
      form.setData({
        dataset_name: 'Customer Transaction Data',
        has_header: true,
        algorithm: 'isolation_forest',
        contamination: 0.1,
        features: 'all',
        execution_mode: 'immediate',
        save_model: true,
        generate_report: true
      });
    };
    
    const resetFormBtn = document.createElement('button');
    resetFormBtn.textContent = 'Reset Form';
    resetFormBtn.className = 'btn btn-ghost btn-sm';
    resetFormBtn.onclick = () => {
      form.reset();
    };
    
    const getDataBtn = document.createElement('button');
    getDataBtn.textContent = 'View Form Data';
    getDataBtn.className = 'btn btn-secondary btn-sm';
    getDataBtn.onclick = () => {
      const data = form.getData();
      console.log('Current form data:', data);
      alert('Form data logged to console');
    };
    
    controls.appendChild(fillDemoDataBtn);
    controls.appendChild(getDataBtn);
    controls.appendChild(resetFormBtn);
    container.appendChild(controls);
    
    // Store form instance for cleanup
    container._formInstance = form;
  }, 100);
  
  return container;
};

AnomalyDetectionForm.parameters = {
  docs: {
    storyDescription: 'Advanced multi-step form component with validation, file upload, progress tracking, and dynamic field generation. Specifically designed for anomaly detection workflows with dataset configuration, algorithm selection, and execution settings.'
  }
};

// State Management Demo
export const StateManagementDemo = () => {
  const container = document.createElement('div');
  container.style.padding = '20px';
  container.style.border = '1px solid #e2e8f0';
  container.style.borderRadius = '8px';
  container.style.backgroundColor = 'white';
  
  // Create state manager
  const stateManager = new AnomalyStateManager({
    datasets: [
      { id: 1, name: 'Customer Data', records: 10000, status: 'active' },
      { id: 2, name: 'Transaction Log', records: 50000, status: 'processing' }
    ],
    currentDataset: null,
    ui: {
      activeTab: 'overview',
      sidebarOpen: true,
      notifications: []
    }
  });
  
  // Create UI
  container.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h3>State Management Demo</h3>
      <p style="color: #6b7280;">Interactive demo of the anomaly detection state management system</p>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
      <div>
        <h4>Current State</h4>
        <pre id="current-state" style="background: #f8fafc; padding: 12px; border-radius: 4px; font-size: 0.875rem; overflow: auto; max-height: 200px;"></pre>
      </div>
      <div>
        <h4>Actions</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <button class="btn btn-primary btn-sm" id="select-dataset">Select Dataset</button>
          <button class="btn btn-secondary btn-sm" id="add-notification">Add Notification</button>
          <button class="btn btn-success btn-sm" id="toggle-sidebar">Toggle Sidebar</button>
          <button class="btn btn-warning btn-sm" id="add-result">Add Result</button>
          <button class="btn btn-danger btn-sm" id="clear-notifications">Clear Notifications</button>
        </div>
      </div>
    </div>
    
    <div style="margin-bottom: 20px;">
      <h4>Time Travel (History)</h4>
      <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <button class="btn btn-ghost btn-sm" id="undo">↶ Undo</button>
        <button class="btn btn-ghost btn-sm" id="redo">↷ Redo</button>
        <button class="btn btn-outline btn-sm" id="show-history">View History</button>
      </div>
    </div>
    
    <div id="history-display" style="display: none;">
      <h4>State History</h4>
      <div id="history-list" style="max-height: 150px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 4px; padding: 8px;"></div>
    </div>
  `;
  
  // Update state display
  const updateStateDisplay = () => {
    const stateEl = container.querySelector('#current-state');
    stateEl.textContent = JSON.stringify(stateManager.getState(), null, 2);
  };
  
  // Subscribe to state changes
  stateManager.subscribe((state, prevState, action) => {
    updateStateDisplay();
    console.log('State changed:', { action, state });
  });
  
  // Initial state display
  updateStateDisplay();
  
  // Bind event handlers
  container.querySelector('#select-dataset').onclick = () => {
    const datasets = stateManager.getState().datasets;
    const randomDataset = datasets[Math.floor(Math.random() * datasets.length)];
    stateManager.dispatch(anomalyActions.selectDataset(randomDataset));
  };
  
  container.querySelector('#add-notification').onclick = () => {
    const messages = [
      'New anomaly detected in dataset',
      'Model training completed',
      'Analysis results available',
      'System maintenance scheduled'
    ];
    const types = ['info', 'success', 'warning', 'error'];
    const message = messages[Math.floor(Math.random() * messages.length)];
    const type = types[Math.floor(Math.random() * types.length)];
    
    stateManager.dispatch(anomalyActions.addNotification(message, type));
  };
  
  container.querySelector('#toggle-sidebar').onclick = () => {
    stateManager.dispatch(anomalyActions.toggleSidebar());
  };
  
  container.querySelector('#add-result').onclick = () => {
    const result = {
      datasetId: Math.floor(Math.random() * 1000),
      algorithm: 'isolation_forest',
      score: Math.random(),
      anomalies: Math.floor(Math.random() * 50)
    };
    stateManager.dispatch(anomalyActions.addResult(result));
  };
  
  container.querySelector('#clear-notifications').onclick = () => {
    const notifications = stateManager.getState().ui.notifications;
    notifications.forEach(notification => {
      stateManager.dispatch(anomalyActions.removeNotification(notification.id));
    });
  };
  
  container.querySelector('#undo').onclick = () => {
    const success = stateManager.undo();
    if (!success) {
      alert('Nothing to undo');
    }
  };
  
  container.querySelector('#redo').onclick = () => {
    const success = stateManager.redo();
    if (!success) {
      alert('Nothing to redo');
    }
  };
  
  container.querySelector('#show-history').onclick = () => {
    const historyDisplay = container.querySelector('#history-display');
    const historyList = container.querySelector('#history-list');
    
    if (historyDisplay.style.display === 'none') {
      const history = stateManager.getHistory();
      historyList.innerHTML = history.map((entry, index) => `
        <div style="padding: 4px 8px; margin: 2px 0; background: ${entry.isCurrent ? '#dbeafe' : '#f9fafb'}; border-radius: 4px; font-size: 0.75rem;">
          <strong>${entry.action.type}</strong>
          <span style="color: #6b7280; margin-left: 8px;">${new Date(entry.timestamp).toLocaleTimeString()}</span>
          ${entry.isCurrent ? '<span style="color: #3b82f6; margin-left: 8px;">← Current</span>' : ''}
        </div>
      `).join('');
      historyDisplay.style.display = 'block';
      container.querySelector('#show-history').textContent = 'Hide History';
    } else {
      historyDisplay.style.display = 'none';
      container.querySelector('#show-history').textContent = 'View History';
    }
  };
  
  // Store state manager instance for cleanup
  container._stateManagerInstance = stateManager;
  
  return container;
};

StateManagementDemo.parameters = {
  docs: {
    storyDescription: 'Advanced state management system with Redux-like architecture, time-travel debugging, middleware support, and persistent state. Specifically designed for anomaly detection applications with reactive subscriptions and async action handling.'
  }
};

// Component Integration Demo
export const ComponentIntegration = () => {
  const container = document.createElement('div');
  container.style.padding = '20px';
  container.style.border = '1px solid #e2e8f0';
  container.style.borderRadius = '8px';
  container.style.backgroundColor = 'white';
  
  container.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h3>Component Integration Demo</h3>
      <p style="color: #6b7280;">Demonstration of how advanced components work together in a real anomaly detection workflow</p>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
      <div>
        <h4>Quick Actions</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <button class="btn btn-primary btn-sm" id="simulate-detection">Simulate Detection</button>
          <button class="btn btn-secondary btn-sm" id="add-widget">Add Dashboard Widget</button>
          <button class="btn btn-success btn-sm" id="export-timeline">Export Timeline Data</button>
          <button class="btn btn-warning btn-sm" id="reset-demo">Reset Demo</button>
        </div>
      </div>
      <div>
        <h4>Component Status</h4>
        <div id="component-status" style="font-size: 0.875rem;">
          <div>🟢 Timeline: Ready</div>
          <div>🟢 Dashboard: Ready</div>
          <div>🟢 Forms: Ready</div>
          <div>🟢 State Manager: Ready</div>
        </div>
      </div>
    </div>
    
    <div style="margin-top: 20px; padding: 16px; background: #f8fafc; border-radius: 8px;">
      <h4>Integration Notes</h4>
      <ul style="margin: 0; padding-left: 20px; color: #6b7280; font-size: 0.875rem;">
        <li>All components share state through the centralized state manager</li>
        <li>Timeline component can trigger dashboard updates in real-time</li>
        <li>Form submissions automatically update the data pipeline</li>
        <li>Dashboard widgets are dynamically configurable and persistent</li>
        <li>State changes are tracked with time-travel debugging capabilities</li>
      </ul>
    </div>
  `;
  
  // Demo actions
  container.querySelector('#simulate-detection').onclick = () => {
    console.log('Simulating anomaly detection...');
    // Simulate data flow between components
    const event = new CustomEvent('demo:detectionSimulated', {
      detail: {
        timestamp: new Date(),
        anomalies: Math.floor(Math.random() * 5) + 1,
        score: Math.random()
      }
    });
    container.dispatchEvent(event);
  };
  
  container.querySelector('#add-widget').onclick = () => {
    console.log('Adding dashboard widget...');
    const event = new CustomEvent('demo:widgetAdded', {
      detail: {
        type: 'metric',
        title: `Widget ${Date.now()}`,
        data: { value: Math.floor(Math.random() * 1000) }
      }
    });
    container.dispatchEvent(event);
  };
  
  container.querySelector('#export-timeline').onclick = () => {
    console.log('Exporting timeline data...');
    alert('Timeline data would be exported (check console for demo events)');
  };
  
  container.querySelector('#reset-demo').onclick = () => {
    console.log('Resetting demo state...');
    const event = new CustomEvent('demo:reset');
    container.dispatchEvent(event);
  };
  
  // Listen for demo events
  container.addEventListener('demo:detectionSimulated', (e) => {
    console.log('Detection simulated:', e.detail);
    const status = container.querySelector('#component-status');
    status.innerHTML += `<div style="color: #059669;">📊 Detection: ${e.detail.anomalies} anomalies detected</div>`;
  });
  
  container.addEventListener('demo:widgetAdded', (e) => {
    console.log('Widget added:', e.detail);
    const status = container.querySelector('#component-status');
    status.innerHTML += `<div style="color: #3b82f6;">📈 Widget: "${e.detail.title}" added</div>`;
  });
  
  container.addEventListener('demo:reset', () => {
    const status = container.querySelector('#component-status');
    status.innerHTML = `
      <div>🟢 Timeline: Ready</div>
      <div>🟢 Dashboard: Ready</div>
      <div>🟢 Forms: Ready</div>
      <div>🟢 State Manager: Ready</div>
    `;
  });
  
  return container;
};

ComponentIntegration.parameters = {
  docs: {
    storyDescription: 'Demonstration of how all advanced components integrate together in a cohesive anomaly detection workflow. Shows real-time data flow, shared state management, and component communication patterns.'
  }
};

// Cleanup function for story instances
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    // Cleanup any running instances when navigating away
    document.querySelectorAll('[class*="Instance"]').forEach(el => {
      if (el._timelineInstance) el._timelineInstance.destroy();
      if (el._dashboardInstance) el._dashboardInstance.destroy();
      if (el._formInstance) el._formInstance.destroy();
      if (el._stateManagerInstance) el._stateManagerInstance.destroy();
    });
  });
}
