// Custom JavaScript for Anomaly Detection documentation

document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize interactive features
    initializeInteractiveElements();
    initializeProgressTracking();
    initializeCodeExamples();
    initializeDemoComponents();
    
});

/**
 * Initialize interactive UI elements
 */
function initializeInteractiveElements() {
    // Add smooth scrolling to internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add copy buttons to code blocks that don't already have them
    document.querySelectorAll('pre code').forEach(block => {
        if (!block.parentElement.querySelector('.md-clipboard')) {
            const button = document.createElement('button');
            button.className = 'md-clipboard md-icon';
            button.title = 'Copy to clipboard';
            button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"/></svg>';
            
            button.addEventListener('click', function() {
                navigator.clipboard.writeText(block.textContent).then(() => {
                    button.setAttribute('data-clipboard-copied', '');
                    setTimeout(() => button.removeAttribute('data-clipboard-copied'), 2000);
                });
            });
            
            block.parentElement.style.position = 'relative';
            block.parentElement.appendChild(button);
        }
    });
}

/**
 * Initialize progress tracking for learning paths
 */
function initializeProgressTracking() {
    const progressKey = 'anomaly-detection-docs-progress';
    
    // Load saved progress
    let progress = {};
    try {
        progress = JSON.parse(localStorage.getItem(progressKey) || '{}');
    } catch (e) {
        console.warn('Failed to load progress from localStorage');
    }
    
    // Add checkboxes to learning path items
    document.querySelectorAll('.learning-path-steps li').forEach((item, index) => {
        const pathId = item.closest('.learning-path-card')?.id || 'default';
        const itemId = `${pathId}-${index}`;
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = itemId;
        checkbox.checked = progress[itemId] || false;
        checkbox.style.marginRight = '0.5rem';
        
        checkbox.addEventListener('change', function() {
            progress[itemId] = this.checked;
            localStorage.setItem(progressKey, JSON.stringify(progress));
            updateProgressBar(pathId);
        });
        
        item.insertBefore(checkbox, item.firstChild);
        updateProgressBar(pathId);
    });
}

/**
 * Update progress bar for a learning path
 */
function updateProgressBar(pathId) {
    const pathCard = document.getElementById(pathId);
    if (!pathCard) return;
    
    const checkboxes = pathCard.querySelectorAll('input[type="checkbox"]');
    const checked = pathCard.querySelectorAll('input[type="checkbox"]:checked');
    const percentage = checkboxes.length > 0 ? (checked.length / checkboxes.length) * 100 : 0;
    
    let progressBar = pathCard.querySelector('.progress-bar');
    if (!progressBar) {
        progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        progressBar.innerHTML = '<div class="progress-fill"></div>';
        pathCard.querySelector('.learning-path-description').after(progressBar);
    }
    
    const progressFill = progressBar.querySelector('.progress-fill');
    progressFill.style.width = `${percentage}%`;
}

/**
 * Initialize interactive code examples
 */
function initializeCodeExamples() {
    // Add "Try it" buttons to Python code examples
    document.querySelectorAll('pre code.language-python').forEach(block => {
        if (block.textContent.includes('from anomaly_detection')) {
            const tryButton = document.createElement('button');
            tryButton.className = 'demo-button';
            tryButton.textContent = 'Try in Browser';
            tryButton.style.marginTop = '0.5rem';
            
            tryButton.addEventListener('click', function() {
                // Open in a Python REPL or notebook environment
                const code = encodeURIComponent(block.textContent);
                const url = `https://replit.com/@new/python?code=${code}`;
                window.open(url, '_blank');
            });
            
            block.parentElement.after(tryButton);
        }
    });
}

/**
 * Initialize demo components
 */
function initializeDemoComponents() {
    // Algorithm comparison interactive demo
    const algoDemo = document.getElementById('algorithm-demo');
    if (algoDemo) {
        createAlgorithmDemo(algoDemo);
    }
    
    // Performance metrics simulator
    const perfDemo = document.getElementById('performance-demo');
    if (perfDemo) {
        createPerformanceDemo(perfDemo);
    }
    
    // Data visualization demos
    document.querySelectorAll('.data-viz-demo').forEach(createDataVizDemo);
}

/**
 * Create algorithm comparison demo
 */
function createAlgorithmDemo(container) {
    const algorithms = [
        { name: 'Isolation Forest', speed: 'Fast', accuracy: 'High', memory: 'Medium' },
        { name: 'LOF', speed: 'Medium', accuracy: 'High', memory: 'High' },
        { name: 'One-Class SVM', speed: 'Slow', accuracy: 'Medium', memory: 'Low' },
        { name: 'Autoencoder', speed: 'Slow', accuracy: 'Very High', memory: 'High' }
    ];
    
    container.innerHTML = `
        <div class="demo-controls">
            <select id="algo-select">
                <option value="">Select an algorithm...</option>
                ${algorithms.map(a => `<option value="${a.name}">${a.name}</option>`).join('')}
            </select>
            <button class="demo-button" onclick="runAlgorithmDemo()">Run Demo</button>
        </div>
        <div class="demo-output" id="algo-output">Select an algorithm and click "Run Demo" to see performance characteristics.</div>
    `;
    
    window.runAlgorithmDemo = function() {
        const select = document.getElementById('algo-select');
        const output = document.getElementById('algo-output');
        const selected = algorithms.find(a => a.name === select.value);
        
        if (!selected) {
            output.textContent = 'Please select an algorithm first.';
            return;
        }
        
        output.innerHTML = `
            <strong>${selected.name} Performance Characteristics:</strong><br>
            Speed: ${selected.speed}<br>
            Accuracy: ${selected.accuracy}<br>
            Memory Usage: ${selected.memory}<br><br>
            <em>Running simulation...</em>
        `;
        
        setTimeout(() => {
            const anomalyRate = (Math.random() * 10 + 1).toFixed(1);
            const processingTime = (Math.random() * 2 + 0.5).toFixed(2);
            
            output.innerHTML += `<br><br>
                <strong>Simulation Results:</strong><br>
                Anomaly Rate: ${anomalyRate}%<br>
                Processing Time: ${processingTime}s<br>
                Status: ✅ Complete
            `;
        }, 2000);
    };
}

/**
 * Create performance metrics demo
 */
function createPerformanceDemo(container) {
    container.innerHTML = `
        <div class="demo-controls">
            <button class="demo-button" onclick="generateMetrics()">Generate Sample Metrics</button>
            <button class="demo-button" onclick="clearMetrics()">Clear</button>
        </div>
        <div class="demo-output" id="metrics-output">Click "Generate Sample Metrics" to see example performance data.</div>
    `;
    
    window.generateMetrics = function() {
        const output = document.getElementById('metrics-output');
        const metrics = {
            total_detections: Math.floor(Math.random() * 1000 + 100),
            anomalies_found: Math.floor(Math.random() * 50 + 10),
            avg_processing_time: (Math.random() * 2 + 0.5).toFixed(2),
            success_rate: (Math.random() * 10 + 90).toFixed(1),
            throughput: (Math.random() * 20 + 5).toFixed(1)
        };
        
        output.innerHTML = `
            <strong>Sample Performance Metrics:</strong><br>
            Total Detections: ${metrics.total_detections}<br>
            Anomalies Found: ${metrics.anomalies_found}<br>
            Anomaly Rate: ${(metrics.anomalies_found / metrics.total_detections * 100).toFixed(1)}%<br>
            Avg Processing Time: ${metrics.avg_processing_time}s<br>
            Success Rate: ${metrics.success_rate}%<br>
            Throughput: ${metrics.throughput} ops/sec<br><br>
            <em>Generated at ${new Date().toLocaleTimeString()}</em>
        `;
    };
    
    window.clearMetrics = function() {
        document.getElementById('metrics-output').textContent = 'Click "Generate Sample Metrics" to see example performance data.';
    };
}

/**
 * Create data visualization demo
 */
function createDataVizDemo(container) {
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 200;
    canvas.style.border = '1px solid #ccc';
    canvas.style.marginTop = '1rem';
    
    const controls = document.createElement('div');
    controls.className = 'demo-controls';
    controls.innerHTML = `
        <button class="demo-button" onclick="generateVisualization('${container.id}')">Generate Data</button>
        <button class="demo-button" onclick="detectAnomalies('${container.id}')">Detect Anomalies</button>
    `;
    
    container.appendChild(controls);
    container.appendChild(canvas);
    
    // Store canvas reference
    container.canvas = canvas;
    container.data = [];
}

/**
 * Generate sample data visualization
 */
window.generateVisualization = function(containerId) {
    const container = document.getElementById(containerId);
    const canvas = container.canvas;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Generate sample data
    const data = [];
    for (let i = 0; i < 50; i++) {
        const isAnomaly = Math.random() < 0.1;
        data.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            isAnomaly: isAnomaly,
            detected: false
        });
    }
    
    container.data = data;
    
    // Draw data points
    data.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.isAnomaly ? 6 : 3, 0, 2 * Math.PI);
        ctx.fillStyle = point.isAnomaly ? '#ff4444' : '#4444ff';
        ctx.fill();
    });
};

/**
 * Detect anomalies in visualization
 */
window.detectAnomalies = function(containerId) {
    const container = document.getElementById(containerId);
    const canvas = container.canvas;
    const ctx = canvas.getContext('2d');
    const data = container.data;
    
    if (!data || data.length === 0) {
        alert('Generate data first!');
        return;
    }
    
    // Simulate anomaly detection
    data.forEach(point => {
        if (point.isAnomaly && Math.random() < 0.8) { // 80% detection rate
            point.detected = true;
            
            // Draw detection circle
            ctx.beginPath();
            ctx.arc(point.x, point.y, 12, 0, 2 * Math.PI);
            ctx.strokeStyle = '#ff8800';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
    
    const detected = data.filter(p => p.detected).length;
    const actual = data.filter(p => p.isAnomaly).length;
    
    alert(`Detection Complete!\nDetected: ${detected} anomalies\nActual: ${actual} anomalies\nAccuracy: ${(detected/actual*100).toFixed(1)}%`);
};

/**
 * Utility function to format code examples
 */
function formatCodeExample(code) {
    return code
        .split('\n')
        .map(line => line.trim())
        .filter(line => line)
        .join('\n');
}

/**
 * Add keyboard shortcuts
 */
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[data-md-component="search-query"]');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Escape to close modals/overlays
    if (e.key === 'Escape') {
        document.querySelectorAll('.md-search__overlay').forEach(overlay => {
            if (overlay.style.display !== 'none') {
                overlay.click();
            }
        });
    }
});

// Initialize theme persistence
const themeToggle = document.querySelector('[data-md-component="palette"]');
if (themeToggle) {
    const savedTheme = localStorage.getItem('anomaly-detection-theme');
    if (savedTheme) {
        document.querySelector(`input[data-md-color-scheme="${savedTheme}"]`)?.click();
    }
    
    document.querySelectorAll('input[data-md-color-scheme]').forEach(input => {
        input.addEventListener('change', function() {
            localStorage.setItem('anomaly-detection-theme', this.dataset.mdColorScheme);
        });
    });
}