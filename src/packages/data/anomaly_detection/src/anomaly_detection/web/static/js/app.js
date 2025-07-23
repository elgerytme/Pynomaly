// Anomaly Detection Dashboard JavaScript

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize charts
    initializeCharts();
    
    // Initialize form handling
    initializeFormHandling();
    
    // Initialize real-time updates
    initializeRealTimeUpdates();
    
    console.log('Anomaly Detection Dashboard initialized');
}

// Tooltip initialization
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(event) {
    const text = event.target.getAttribute('data-tooltip');
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip absolute bg-gray-800 text-white px-2 py-1 rounded text-sm z-50';
    tooltip.textContent = text;
    tooltip.id = 'tooltip';
    
    document.body.appendChild(tooltip);
    
    const rect = event.target.getBoundingClientRect();
    tooltip.style.left = rect.left + 'px';
    tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
}

function hideTooltip() {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// Chart initialization (using Chart.js if available)
function initializeCharts() {
    // Performance chart
    const performanceCtx = document.getElementById('performanceChart');
    if (performanceCtx && typeof Chart !== 'undefined') {
        new Chart(performanceCtx, {
            type: 'line',
            data: {
                labels: ['1h ago', '45m ago', '30m ago', '15m ago', 'Now'],
                datasets: [{
                    label: 'Detection Time (ms)',
                    data: [120, 135, 110, 125, 118],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // Anomaly distribution chart
    const distributionCtx = document.getElementById('distributionChart');
    if (distributionCtx && typeof Chart !== 'undefined') {
        new Chart(distributionCtx, {
            type: 'doughnut',
            data: {
                labels: ['Normal', 'Anomalies'],
                datasets: [{
                    data: [85, 15],
                    backgroundColor: [
                        'rgb(34, 197, 94)',
                        'rgb(239, 68, 68)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
}

// Form handling utilities
function initializeFormHandling() {
    // Auto-resize textareas
    const textareas = document.querySelectorAll('textarea[data-auto-resize]');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', autoResizeTextarea);
    });
    
    // File upload handling
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', handleFileUpload);
    });
    
    // Form validation
    const forms = document.querySelectorAll('form[data-validate]');
    forms.forEach(form => {
        form.addEventListener('submit', validateForm);
    });
}

function autoResizeTextarea(event) {
    const textarea = event.target;
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const data = JSON.parse(e.target.result);
                const dataInput = document.getElementById('sample_data');
                if (dataInput) {
                    dataInput.value = JSON.stringify(data, null, 2);
                    autoResizeTextarea({ target: dataInput });
                }
            } catch (error) {
                showNotification('Invalid JSON file', 'error');
            }
        };
        reader.readAsText(file);
    }
}

function validateForm(event) {
    const form = event.target;
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showFieldError(field, 'This field is required');
            isValid = false;
        } else {
            clearFieldError(field);
        }
    });
    
    // Validate contamination rate
    const contaminationField = form.querySelector('input[name="contamination"]');
    if (contaminationField) {
        const value = parseFloat(contaminationField.value);
        if (value < 0.001 || value > 0.5) {
            showFieldError(contaminationField, 'Contamination must be between 0.001 and 0.5');
            isValid = false;
        }
    }
    
    if (!isValid) {
        event.preventDefault();
    }
}

function showFieldError(field, message) {
    clearFieldError(field);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'form-error';
    errorDiv.textContent = message;
    errorDiv.setAttribute('data-error-for', field.name);
    
    field.parentNode.appendChild(errorDiv);
    field.classList.add('border-red-500');
}

function clearFieldError(field) {
    const existingError = field.parentNode.querySelector(`[data-error-for="${field.name}"]`);
    if (existingError) {
        existingError.remove();
    }
    field.classList.remove('border-red-500');
}

// Real-time updates
function initializeRealTimeUpdates() {
    // Update dashboard stats every 30 seconds
    setInterval(() => {
        updateDashboardStats();
    }, 30000);
    
    // Update model list every minute
    setInterval(() => {
        updateModelList();
    }, 60000);
}

function updateDashboardStats() {
    const statsContainer = document.getElementById('dashboard-stats');
    if (statsContainer) {
        htmx.ajax('GET', '/htmx/dashboard/stats', {
            target: '#dashboard-stats',
            swap: 'innerHTML'
        });
    }
}

function updateModelList() {
    const modelList = document.getElementById('model-list');
    if (modelList) {
        htmx.ajax('GET', '/htmx/models/list', {
            target: '#model-list',
            swap: 'innerHTML'
        });
    }
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} fixed top-4 right-4 z-50 animate-fade-in`;
    notification.innerHTML = `
        <div class="flex items-center">
            <span class="flex-1">${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-lg">&times;</button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard', 'success');
    }).catch(() => {
        showNotification('Failed to copy to clipboard', 'error');
    });
}

function downloadData(data, filename, type = 'application/json') {
    const blob = new Blob([data], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

// HTMX event listeners
document.addEventListener('htmx:beforeRequest', function(event) {
    // Show loading indicator
    const loadingEl = document.querySelector('.htmx-indicator');
    if (loadingEl) {
        loadingEl.style.display = 'inline';
    }
});

document.addEventListener('htmx:afterRequest', function(event) {
    // Hide loading indicator
    const loadingEl = document.querySelector('.htmx-indicator');
    if (loadingEl) {
        loadingEl.style.display = 'none';
    }
    
    // Show notification based on response
    if (event.detail.successful) {
        const contentType = event.detail.xhr.getResponseHeader('content-type');
        if (contentType && contentType.includes('application/json')) {
            try {
                const response = JSON.parse(event.detail.xhr.responseText);
                if (response.message) {
                    showNotification(response.message, 'success');
                }
            } catch (e) {
                // Not JSON, ignore
            }
        }
    } else {
        showNotification('Request failed', 'error');
    }
});

document.addEventListener('htmx:responseError', function(event) {
    showNotification('Server error occurred', 'error');
});

// Export functions for global use
window.AnomalyDetection = {
    showNotification,
    copyToClipboard,
    downloadData,
    formatNumber,
    formatTimestamp,
    updateDashboardStats,
    updateModelList
};