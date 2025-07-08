/**
 * Real-time progress tracking for long-running operations
 * Supports WebSocket and Server-Sent Events for live updates
 */

class ProgressTracker {
  constructor() {
    this.connections = new Map();
    this.eventSources = new Map();
    this.progressBars = new Map();
    this.statusElements = new Map();
  }

  /**
   * Track a task using WebSocket connection
   * @param {string} taskId - Task identifier
   * @param {string} elementId - DOM element to update
   * @param {Object} options - Configuration options
   */
  trackTask(taskId, elementId, options = {}) {
    const config = {
      useWebSocket: true,
      showProgressBar: true,
      showDetails: true,
      autoClose: true,
      onUpdate: null,
      onComplete: null,
      onError: null,
      ...options,
    };

    if (config.useWebSocket) {
      this.trackWithWebSocket(taskId, elementId, config);
    } else {
      this.trackWithSSE(taskId, elementId, config);
    }
  }

  /**
   * Track task using WebSocket
   */
  trackWithWebSocket(taskId, elementId, config) {
    const wsUrl = `ws://${window.location.host}/api/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`WebSocket connected for task ${taskId}`);

      // Subscribe to task updates
      ws.send(
        JSON.stringify({
          type: "subscribe_task",
          task_id: taskId,
        }),
      );
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleTaskUpdate(message, taskId, elementId, config);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      if (config.onError) {
        config.onError(error);
      }
    };

    ws.onclose = () => {
      console.log(`WebSocket closed for task ${taskId}`);
    };

    this.connections.set(taskId, ws);
  }

  /**
   * Track task using Server-Sent Events
   */
  trackWithSSE(taskId, elementId, config) {
    const sseUrl = `/api/tasks/${taskId}/sse`;
    const eventSource = new EventSource(sseUrl);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type !== "keepalive") {
        this.handleTaskUpdate(
          {
            type: "task_update",
            data: data,
          },
          taskId,
          elementId,
          config,
        );
      }
    };

    eventSource.onerror = (error) => {
      console.error("SSE error:", error);
      if (config.onError) {
        config.onError(error);
      }
    };

    this.eventSources.set(taskId, eventSource);
  }

  /**
   * Handle task update messages
   */
  handleTaskUpdate(message, taskId, elementId, config) {
    if (message.type === "task_update" || message.type === "task_status") {
      const taskData = message.data;
      this.updateTaskDisplay(taskData, elementId, config);

      // Call custom update handler
      if (config.onUpdate) {
        config.onUpdate(taskData);
      }

      // Handle completion
      if (["completed", "failed", "cancelled"].includes(taskData.status)) {
        if (taskData.status === "completed" && config.onComplete) {
          config.onComplete(taskData);
        } else if (taskData.status === "failed" && config.onError) {
          config.onError(taskData.error);
        }

        if (config.autoClose) {
          setTimeout(() => this.stopTracking(taskId), 3000);
        }
      }
    }
  }

  /**
   * Update the visual display of task progress
   */
  updateTaskDisplay(taskData, elementId, config) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const progress = taskData.progress;
    const status = taskData.status;

    // Update progress bar if enabled
    if (config.showProgressBar) {
      this.updateProgressBar(element, progress, status);
    }

    // Update status text
    this.updateStatusText(element, taskData, config);

    // Update details if enabled
    if (config.showDetails) {
      this.updateTaskDetails(element, taskData);
    }
  }

  /**
   * Update progress bar
   */
  updateProgressBar(element, progress, status) {
    let progressBarContainer = element.querySelector(".progress-container");

    if (!progressBarContainer) {
      progressBarContainer = document.createElement("div");
      progressBarContainer.className = "progress-container mb-4";
      progressBarContainer.innerHTML = `
                <div class="flex justify-between text-sm text-gray-600 mb-1">
                    <span class="progress-label">Progress</span>
                    <span class="progress-percentage">0%</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div class="progress-bar bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
                </div>
            `;
      element.insertBefore(progressBarContainer, element.firstChild);
    }

    const progressBar = progressBarContainer.querySelector(".progress-bar");
    const progressPercentage = progressBarContainer.querySelector(
      ".progress-percentage",
    );

    if (progressBar && progressPercentage) {
      const percentage = Math.round(progress.percentage || 0);
      progressBar.style.width = `${percentage}%`;
      progressPercentage.textContent = `${percentage}%`;

      // Update color based on status
      if (status === "completed") {
        progressBar.className =
          "progress-bar bg-green-600 h-2 rounded-full transition-all duration-300";
      } else if (status === "failed") {
        progressBar.className =
          "progress-bar bg-red-600 h-2 rounded-full transition-all duration-300";
      } else if (status === "running") {
        progressBar.className =
          "progress-bar bg-blue-600 h-2 rounded-full transition-all duration-300";
      }
    }
  }

  /**
   * Update status text
   */
  updateStatusText(element, taskData, config) {
    let statusElement = element.querySelector(".task-status");

    if (!statusElement) {
      statusElement = document.createElement("div");
      statusElement.className = "task-status mb-2";
      element.appendChild(statusElement);
    }

    const progress = taskData.progress;
    const status = taskData.status;
    const statusIcons = {
      pending: "⏳",
      running: "🚀",
      completed: "✅",
      failed: "❌",
      cancelled: "⏸️",
    };

    statusElement.innerHTML = `
            <div class="flex items-center space-x-2">
                <span class="text-lg">${statusIcons[status] || "❓"}</span>
                <span class="font-medium">${this.capitalizeFirst(status)}</span>
                <span class="text-gray-600">- ${progress.message || "No message"}</span>
            </div>
        `;
  }

  /**
   * Update task details
   */
  updateTaskDetails(element, taskData) {
    let detailsElement = element.querySelector(".task-details");

    if (!detailsElement) {
      detailsElement = document.createElement("div");
      detailsElement.className =
        "task-details mt-4 p-3 bg-gray-50 rounded text-sm";
      element.appendChild(detailsElement);
    }

    const progress = taskData.progress;
    const details = progress.details || {};

    let detailsHtml = `
            <div class="grid grid-cols-2 gap-2">
                <div><strong>Task:</strong> ${taskData.name}</div>
                <div><strong>Started:</strong> ${new Date(taskData.started_at).toLocaleTimeString()}</div>
        `;

    // Add custom details
    Object.entries(details).forEach(([key, value]) => {
      if (typeof value === "object") {
        value = JSON.stringify(value);
      }
      detailsHtml += `<div><strong>${this.formatKey(key)}:</strong> ${value}</div>`;
    });

    detailsHtml += "</div>";
    detailsElement.innerHTML = detailsHtml;
  }

  /**
   * Stop tracking a task
   */
  stopTracking(taskId) {
    // Close WebSocket connection
    if (this.connections.has(taskId)) {
      this.connections.get(taskId).close();
      this.connections.delete(taskId);
    }

    // Close SSE connection
    if (this.eventSources.has(taskId)) {
      this.eventSources.get(taskId).close();
      this.eventSources.delete(taskId);
    }

    console.log(`Stopped tracking task ${taskId}`);
  }

  /**
   * Stop all tracking
   */
  stopAllTracking() {
    this.connections.forEach((ws, taskId) => {
      ws.close();
    });
    this.connections.clear();

    this.eventSources.forEach((es, taskId) => {
      es.close();
    });
    this.eventSources.clear();
  }

  /**
   * Create a progress tracking modal
   */
  createProgressModal(taskId, taskName) {
    const modalId = `progress-modal-${taskId}`;

    const modalHtml = `
            <div id="${modalId}" class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
                <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                    <div class="mt-3">
                        <h3 class="text-lg font-medium text-gray-900 mb-4">${taskName}</h3>
                        <div id="progress-content-${taskId}" class="space-y-4">
                            <!-- Progress content will be inserted here -->
                        </div>
                        <div class="mt-6 flex justify-end space-x-3">
                            <button onclick="progressTracker.cancelTask('${taskId}')"
                                    class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
                                Cancel
                            </button>
                            <button onclick="progressTracker.closeModal('${modalId}')"
                                    class="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400">
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

    document.body.insertAdjacentHTML("beforeend", modalHtml);

    // Start tracking in the modal
    this.trackTask(taskId, `progress-content-${taskId}`, {
      showProgressBar: true,
      showDetails: true,
      onComplete: () => {
        const cancelBtn = document.querySelector(`#${modalId} button`);
        if (cancelBtn) cancelBtn.style.display = "none";
      },
    });

    return modalId;
  }

  /**
   * Close progress modal
   */
  closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
      modal.remove();
    }
  }

  /**
   * Cancel a task
   */
  cancelTask(taskId) {
    const ws = this.connections.get(taskId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "cancel_task",
          task_id: taskId,
        }),
      );
    }
  }

  /**
   * Utility methods
   */
  capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  formatKey(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  }
}

// Global instance
const progressTracker = new ProgressTracker();

// Clean up connections when page unloads
window.addEventListener("beforeunload", () => {
  progressTracker.stopAllTracking();
});

// Utility functions for easy use in templates
window.trackOptimization = function (taskId, elementId) {
  progressTracker.trackTask(taskId, elementId, {
    showProgressBar: true,
    showDetails: true,
    onComplete: (taskData) => {
      // Refresh the page or update UI
      setTimeout(() => {
        location.reload();
      }, 2000);
    },
  });
};

window.showProgressModal = function (taskId, taskName) {
  return progressTracker.createProgressModal(taskId, taskName);
};

window.startAutoMLOptimization = function (formData) {
  // This would be called from the AutoML form
  // to start optimization and show progress modal
  const taskId = "opt_" + Date.now();
  const modalId = progressTracker.createProgressModal(
    taskId,
    "AutoML Optimization",
  );

  // Start the optimization via HTMX or fetch
  // The server would return the actual task ID
  return { taskId, modalId };
};
