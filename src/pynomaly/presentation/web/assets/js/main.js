/**
 * Pynomaly PWA Main JavaScript Module
 * Integrates HTMX, D3.js, and ECharts for anomaly detection visualizations
 */

// Import dependencies
import * as d3 from 'd3';
import * as echarts from 'echarts';

// HTMX is loaded via CDN in templates, but we can configure it here
document.addEventListener('DOMContentLoaded', function() {
    // Initialize HTMX configuration
    if (typeof htmx !== 'undefined') {
        // Configure HTMX defaults
        htmx.config.defaultSwapStyle = 'outerHTML';
        htmx.config.defaultSwapDelay = 100;
        htmx.config.defaultSettleDelay = 100;
        
        // Add global request headers
        document.body.addEventListener('htmx:configRequest', function(evt) {
            evt.detail.headers['X-Requested-With'] = 'HTMX';
        });
        
        // Global error handling
        document.body.addEventListener('htmx:responseError', function(evt) {
            console.error('HTMX Request Error:', evt.detail);
            showNotification('Request failed. Please try again.', 'error');
        });
        
        // Loading indicators
        document.body.addEventListener('htmx:beforeRequest', function(evt) {
            showLoading(evt.target);
        });
        
        document.body.addEventListener('htmx:afterRequest', function(evt) {
            hideLoading(evt.target);
        });
    }
    
    // Initialize PWA features
    initializePWA();
    
    // Initialize chart handlers
    initializeChartHandlers();
    
    // Initialize form enhancements
    initializeFormEnhancements();
    
    // Initialize notifications
    initializeNotificationSystem();
});

/**
 * PWA Initialization
 */
function initializePWA() {
    // Register service worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/static/sw.js')
            .then(registration => {
                console.log('SW registered:', registration);
            })
            .catch(error => {
                console.log('SW registration failed:', error);
            });
    }
    
    // Handle install prompt
    let deferredPrompt;
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        showInstallPrompt();
    });
    
    function showInstallPrompt() {
        const installBanner = document.createElement('div');
        installBanner.className = 'pwa-install-prompt';
        installBanner.innerHTML = `
            <div class="flex items-center justify-between">
                <div>
                    <h4 class="font-semibold">Install Pynomaly</h4>
                    <p class="text-sm opacity-90">Get quick access to anomaly detection</p>
                </div>
                <div class="flex gap-2">
                    <button id="install-btn" class="bg-white text-blue-600 px-3 py-1 rounded text-sm font-medium">
                        Install
                    </button>
                    <button id="dismiss-install" class="text-white opacity-75 hover:opacity-100">
                        ×
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(installBanner);
        
        document.getElementById('install-btn').addEventListener('click', async () => {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                const { outcome } = await deferredPrompt.userChoice;
                console.log('Install prompt outcome:', outcome);
                deferredPrompt = null;
                installBanner.remove();
            }
        });
        
        document.getElementById('dismiss-install').addEventListener('click', () => {
            installBanner.remove();
        });
    }
}

/**
 * Chart Initialization and Management
 */
function initializeChartHandlers() {
    // Auto-initialize charts on page load
    initializeChartsOnPage();
    
    // Re-initialize charts after HTMX swaps
    document.body.addEventListener('htmx:afterSwap', function(evt) {
        initializeChartsOnPage();
    });
}

function initializeChartsOnPage() {
    // Initialize D3 charts
    document.querySelectorAll('[data-chart="d3"]').forEach(initializeD3Chart);
    
    // Initialize ECharts
    document.querySelectorAll('[data-chart="echarts"]').forEach(initializeEChart);
}

function initializeD3Chart(element) {
    const chartType = element.dataset.chartType || 'scatter';
    const dataUrl = element.dataset.dataUrl;
    
    if (!dataUrl) {
        console.warn('No data URL provided for D3 chart');
        return;
    }
    
    // Clear existing chart
    d3.select(element).selectAll('*').remove();
    
    // Load data and create chart
    d3.json(dataUrl).then(data => {
        switch (chartType) {
            case 'scatter':
                createScatterPlot(element, data);
                break;
            case 'line':
                createLinePlot(element, data);
                break;
            case 'heatmap':
                createHeatmap(element, data);
                break;
            default:
                console.warn('Unknown D3 chart type:', chartType);
        }
    }).catch(error => {
        console.error('Error loading chart data:', error);
        showChartError(element, 'Failed to load chart data');
    });
}

function createScatterPlot(element, data) {
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const width = element.clientWidth - margin.left - margin.right;
    const height = element.clientHeight - margin.top - margin.bottom;
    
    const svg = d3.select(element)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);
    
    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scales
    const xScale = d3.scaleLinear()
        .domain(d3.extent(data, d => d.x))
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain(d3.extent(data, d => d.y))
        .range([height, 0]);
    
    const colorScale = d3.scaleOrdinal()
        .domain(['normal', 'anomaly'])
        .range(['#10b981', '#ef4444']);
    
    // Axes
    g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale));
    
    g.append('g')
        .call(d3.axisLeft(yScale));
    
    // Points
    g.selectAll('.point')
        .data(data)
        .enter().append('circle')
        .attr('class', 'point')
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', 4)
        .attr('fill', d => colorScale(d.label || 'normal'))
        .attr('opacity', 0.7)
        .on('mouseover', function(event, d) {
            // Tooltip
            const tooltip = d3.select('body').append('div')
                .attr('class', 'tooltip bg-gray-800 text-white p-2 rounded text-sm')
                .style('position', 'absolute')
                .style('opacity', 0);
            
            tooltip.transition().duration(200).style('opacity', .9);
            tooltip.html(`X: ${d.x.toFixed(2)}<br>Y: ${d.y.toFixed(2)}<br>Score: ${(d.score || 0).toFixed(3)}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function() {
            d3.selectAll('.tooltip').remove();
        });
}

function initializeEChart(element) {
    const chartType = element.dataset.chartType || 'line';
    const dataUrl = element.dataset.dataUrl;
    
    if (!dataUrl) {
        console.warn('No data URL provided for EChart');
        return;
    }
    
    // Initialize ECharts instance
    const chart = echarts.init(element);
    
    // Load data and configure chart
    fetch(dataUrl)
        .then(response => response.json())
        .then(data => {
            let option;
            
            switch (chartType) {
                case 'line':
                    option = createLineChartOption(data);
                    break;
                case 'bar':
                    option = createBarChartOption(data);
                    break;
                case 'pie':
                    option = createPieChartOption(data);
                    break;
                case 'heatmap':
                    option = createHeatmapOption(data);
                    break;
                default:
                    console.warn('Unknown EChart type:', chartType);
                    return;
            }
            
            chart.setOption(option);
            
            // Handle window resize
            window.addEventListener('resize', () => {
                chart.resize();
            });
        })
        .catch(error => {
            console.error('Error loading EChart data:', error);
            showChartError(element, 'Failed to load chart data');
        });
}

function createLineChartOption(data) {
    return {
        title: {
            text: data.title || 'Anomaly Scores Over Time',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            formatter: function (params) {
                return `${params[0].name}<br/>Score: ${params[0].value}`;
            }
        },
        xAxis: {
            type: 'category',
            data: data.x || []
        },
        yAxis: {
            type: 'value',
            name: 'Anomaly Score'
        },
        series: [{
            data: data.y || [],
            type: 'line',
            smooth: true,
            lineStyle: {
                color: '#3b82f6'
            },
            areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                    { offset: 1, color: 'rgba(59, 130, 246, 0.1)' }
                ])
            }
        }]
    };
}

/**
 * Form Enhancements
 */
function initializeFormEnhancements() {
    // Auto-save form data
    document.querySelectorAll('form[data-autosave]').forEach(form => {
        const formId = form.dataset.autosave;
        
        // Load saved data
        const savedData = localStorage.getItem(`form_${formId}`);
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                Object.keys(data).forEach(key => {
                    const input = form.querySelector(`[name="${key}"]`);
                    if (input) input.value = data[key];
                });
            } catch (e) {
                console.warn('Failed to load saved form data:', e);
            }
        }
        
        // Save data on input
        form.addEventListener('input', debounce(() => {
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            localStorage.setItem(`form_${formId}`, JSON.stringify(data));
        }, 1000));
    });
    
    // File upload progress
    document.querySelectorAll('input[type="file"]').forEach(input => {
        input.addEventListener('change', function(e) {
            const files = e.target.files;
            if (files.length > 0) {
                showFileUploadProgress(files[0]);
            }
        });
    });
}

/**
 * Notification System
 */
function initializeNotificationSystem() {
    // Create notification container if it doesn't exist
    if (!document.getElementById('notification-container')) {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'fixed top-4 right-4 z-50 space-y-2';
        document.body.appendChild(container);
    }
}

function showNotification(message, type = 'info', duration = 5000) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    
    const typeClasses = {
        info: 'bg-blue-500',
        success: 'bg-green-500',
        warning: 'bg-yellow-500',
        error: 'bg-red-500'
    };
    
    notification.className = `${typeClasses[type]} text-white p-4 rounded-lg shadow-lg transform translate-x-full transition-transform duration-300`;
    notification.innerHTML = `
        <div class="flex items-center justify-between">
            <span>${message}</span>
            <button class="ml-4 text-white hover:text-gray-200" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    container.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 10);
    
    // Auto-remove
    if (duration > 0) {
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }
}

/**
 * Utility Functions
 */
function showLoading(element) {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'htmx-indicator absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center';
    loadingDiv.innerHTML = '<div class="loading-spinner"></div>';
    
    if (element.style.position !== 'absolute' && element.style.position !== 'relative') {
        element.style.position = 'relative';
    }
    
    element.appendChild(loadingDiv);
}

function hideLoading(element) {
    const loadingDiv = element.querySelector('.htmx-indicator');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function showChartError(element, message) {
    element.innerHTML = `
        <div class="flex items-center justify-center h-full text-gray-500">
            <div class="text-center">
                <div class="text-4xl mb-2">📊</div>
                <div>${message}</div>
            </div>
        </div>
    `;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showFileUploadProgress(file) {
    // This would integrate with the actual upload mechanism
    console.log('File selected for upload:', file.name);
    showNotification(`Preparing to upload: ${file.name}`, 'info');
}

// Export functions for global access
window.Pynomaly = {
    showNotification,
    initializeD3Chart,
    initializeEChart,
    createScatterPlot
};
