// Custom JavaScript for Anomaly Detection Platform Documentation

document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize all custom features
    initializeInteractiveElements();
    initializeCodeExamples();
    initializeMetricsAnimations();
    initializeSearchEnhancements();
    initializeNavigationEnhancements();
    
});

/**
 * Initialize interactive elements like demos and tutorials
 */
function initializeInteractiveElements() {
    
    // Interactive demo buttons
    const demoBtns = document.querySelectorAll('.interactive-demo');
    demoBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const demoType = this.dataset.demo;
            launchInteractiveDemo(demoType);
        });
    });
    
    // Algorithm comparison selectors
    const algorithmSelectors = document.querySelectorAll('.algorithm-selector');
    algorithmSelectors.forEach(selector => {
        selector.addEventListener('change', function() {
            updateAlgorithmComparison(this.value);
        });
    });
    
}

/**
 * Enhance code examples with copy buttons and syntax highlighting
 */
function initializeCodeExamples() {
    
    // Add copy buttons to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const container = block.parentElement;
        const copyBtn = createCopyButton();
        
        copyBtn.addEventListener('click', function() {
            copyToClipboard(block.textContent);
            showCopyFeedback(this);
        });
        
        container.style.position = 'relative';
        container.appendChild(copyBtn);
    });
    
    // Enhanced code execution simulation
    const executableCode = document.querySelectorAll('.executable-code');
    executableCode.forEach(code => {
        const runBtn = createRunButton();
        runBtn.addEventListener('click', function() {
            simulateCodeExecution(code);
        });
        code.parentElement.appendChild(runBtn);
    });
    
}

/**
 * Animate metrics and performance indicators
 */
function initializeMetricsAnimations() {
    
    const metricCards = document.querySelectorAll('.metric-card');
    
    // Intersection Observer for metric animations
    const metricObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateMetricValue(entry.target);
            }
        });
    });
    
    metricCards.forEach(card => {
        metricObserver.observe(card);
    });
    
}

/**
 * Enhance search functionality
 */
function initializeSearchEnhancements() {
    
    // Add search suggestions based on current page
    const searchInput = document.querySelector('[data-md-component="search-query"]');
    if (searchInput) {
        
        searchInput.addEventListener('focus', function() {
            showSearchSuggestions();
        });
        
        searchInput.addEventListener('input', function() {
            enhancedSearch(this.value);
        });
        
    }
    
}

/**
 * Enhance navigation with context awareness
 */
function initializeNavigationEnhancements() {
    
    // Highlight current section in navigation
    updateNavigationContext();
    
    // Add progress indicator for learning paths
    if (window.location.pathname.includes('learning-paths')) {
        initializeLearningPathProgress();
    }
    
    // Enhanced mobile navigation
    enhanceMobileNavigation();
    
}

/**
 * Create a copy button for code blocks
 */
function createCopyButton() {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16">
            <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
            <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
        </svg>
    `;
    btn.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        background: rgba(255,255,255,0.8);
        border: none;
        border-radius: 4px;
        padding: 4px;
        cursor: pointer;
        opacity: 0.7;
        transition: opacity 0.2s;
    `;
    btn.addEventListener('mouseenter', () => btn.style.opacity = '1');
    btn.addEventListener('mouseleave', () => btn.style.opacity = '0.7');
    
    return btn;
}

/**
 * Create a run button for executable code
 */
function createRunButton() {
    const btn = document.createElement('button');
    btn.className = 'run-btn';
    btn.textContent = 'Run Example';
    btn.style.cssText = `
        background: var(--ad-primary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        margin-top: 8px;
        cursor: pointer;
        font-size: 0.875rem;
        transition: background-color 0.2s;
    `;
    btn.addEventListener('mouseenter', () => {
        btn.style.backgroundColor = 'var(--ad-primary-dark)';
    });
    btn.addEventListener('mouseleave', () => {
        btn.style.backgroundColor = 'var(--ad-primary)';
    });
    
    return btn;
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        return true;
    }
}

/**
 * Show copy feedback
 */
function showCopyFeedback(button) {
    const originalContent = button.innerHTML;
    button.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" style="color: green;">
            <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/>
        </svg>
    `;
    
    setTimeout(() => {
        button.innerHTML = originalContent;
    }, 2000);
}

/**
 * Simulate code execution with visual feedback
 */
function simulateCodeExecution(codeElement) {
    const outputDiv = document.createElement('div');
    outputDiv.className = 'code-output';
    outputDiv.style.cssText = `
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-top: none;
        padding: 12px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.875rem;
        white-space: pre-wrap;
    `;
    
    // Simulate loading
    outputDiv.textContent = 'Running...';
    codeElement.parentElement.appendChild(outputDiv);
    
    setTimeout(() => {
        // Simulate results based on code content
        const code = codeElement.textContent;
        let output = '';
        
        if (code.includes('detect_anomalies')) {
            output = `✅ Anomaly detection completed!
📊 Found 12 anomalies out of 1000 samples
📈 Confidence scores: [0.95, 0.87, 0.92, ...]
⏱️  Processing time: 0.23 seconds`;
        } else if (code.includes('create_detector')) {
            output = `✅ Detector created successfully!
🔧 Algorithm: Isolation Forest
⚙️  Parameters: contamination=0.1, n_estimators=100
📋 Status: Ready for detection`;
        } else {
            output = `✅ Code executed successfully!
📝 Output generated
⏱️  Execution time: 0.05 seconds`;
        }
        
        outputDiv.textContent = output;
        outputDiv.style.background = '#f0f9ff';
        outputDiv.style.borderColor = '#bfdbfe';
    }, 1500);
}

/**
 * Animate metric values with counting effect
 */
function animateMetricValue(metricCard) {
    const valueElement = metricCard.querySelector('.metric-value');
    if (!valueElement) return;
    
    const finalValue = valueElement.textContent;
    const numericValue = parseFloat(finalValue.replace(/[^\d.]/g, ''));
    
    if (isNaN(numericValue)) return;
    
    let currentValue = 0;
    const increment = numericValue / 50; // 50 steps
    const timer = setInterval(() => {
        currentValue += increment;
        if (currentValue >= numericValue) {
            valueElement.textContent = finalValue;
            clearInterval(timer);
        } else {
            const suffix = finalValue.replace(/[\d.]/g, '');
            valueElement.textContent = Math.floor(currentValue) + suffix;
        }
    }, 30);
}

/**
 * Show contextual search suggestions
 */
function showSearchSuggestions() {
    const currentPath = window.location.pathname;
    let suggestions = [];
    
    if (currentPath.includes('anomaly-detection')) {
        suggestions = [
            'isolation forest algorithm',
            'ensemble methods',
            'streaming detection',
            'explainable AI'
        ];
    } else if (currentPath.includes('machine-learning')) {
        suggestions = [
            'AutoML optimization',
            'active learning',
            'model management',
            'A/B testing'
        ];
    } else if (currentPath.includes('getting-started')) {
        suggestions = [
            'installation guide',
            'first detection',
            'learning paths',
            'quick start'
        ];
    }
    
    // Implementation would show these suggestions in the search interface
}

/**
 * Enhanced search with context awareness
 */
function enhancedSearch(query) {
    if (query.length < 3) return;
    
    // Add search analytics
    if (typeof gtag !== 'undefined') {
        gtag('event', 'search', {
            'search_term': query,
            'page_path': window.location.pathname
        });
    }
    
    // Context-aware search suggestions
    const context = getCurrentPageContext();
    // Implementation would enhance search results based on context
}

/**
 * Get current page context for enhanced features
 */
function getCurrentPageContext() {
    const path = window.location.pathname;
    
    if (path.includes('anomaly-detection')) return 'anomaly-detection';
    if (path.includes('machine-learning')) return 'machine-learning';
    if (path.includes('data-platform')) return 'data-platform';
    if (path.includes('enterprise')) return 'enterprise';
    if (path.includes('getting-started')) return 'getting-started';
    if (path.includes('guides')) return 'guides';
    if (path.includes('api')) return 'api';
    
    return 'general';
}

/**
 * Update navigation context highlighting
 */
function updateNavigationContext() {
    const navItems = document.querySelectorAll('.md-nav__item');
    const currentPath = window.location.pathname;
    
    navItems.forEach(item => {
        const link = item.querySelector('.md-nav__link');
        if (link && link.href && currentPath.includes(link.pathname)) {
            item.classList.add('md-nav__item--active');
        }
    });
}

/**
 * Initialize learning path progress tracking
 */
function initializeLearningPathProgress() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    let completedCount = 0;
    
    checkboxes.forEach((checkbox, index) => {
        // Load saved progress
        const saved = localStorage.getItem(`learning-progress-${index}`);
        if (saved === 'true') {
            checkbox.checked = true;
            completedCount++;
        }
        
        // Save progress on change
        checkbox.addEventListener('change', function() {
            localStorage.setItem(`learning-progress-${index}`, this.checked);
            updateProgressIndicator();
        });
    });
    
    // Show progress indicator
    updateProgressIndicator();
}

/**
 * Update learning path progress indicator
 */
function updateProgressIndicator() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    const completed = Array.from(checkboxes).filter(cb => cb.checked).length;
    const total = checkboxes.length;
    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    // Create or update progress bar
    let progressBar = document.querySelector('.learning-progress');
    if (!progressBar) {
        progressBar = document.createElement('div');
        progressBar.className = 'learning-progress';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: rgba(25, 118, 210, 0.1);
            z-index: 1000;
        `;
        
        const progressFill = document.createElement('div');
        progressFill.className = 'progress-fill';
        progressFill.style.cssText = `
            height: 100%;
            background: var(--ad-primary);
            transition: width 0.3s ease;
            width: 0%;
        `;
        
        progressBar.appendChild(progressFill);
        document.body.appendChild(progressBar);
    }
    
    const progressFill = progressBar.querySelector('.progress-fill');
    progressFill.style.width = `${percentage}%`;
}

/**
 * Enhance mobile navigation
 */
function enhanceMobileNavigation() {
    // Add swipe gesture support for mobile
    let startX = 0;
    let startY = 0;
    
    document.addEventListener('touchstart', function(e) {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
    });
    
    document.addEventListener('touchend', function(e) {
        const endX = e.changedTouches[0].clientX;
        const endY = e.changedTouches[0].clientY;
        
        const diffX = startX - endX;
        const diffY = startY - endY;
        
        // Horizontal swipe detection
        if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 50) {
            if (diffX > 0) {
                // Swipe left - next page
                const nextLink = document.querySelector('.md-footer__link--next');
                if (nextLink) nextLink.click();
            } else {
                // Swipe right - previous page
                const prevLink = document.querySelector('.md-footer__link--prev');
                if (prevLink) prevLink.click();
            }
        }
    });
}

/**
 * Launch interactive demo based on type
 */
function launchInteractiveDemo(demoType) {
    switch (demoType) {
        case 'anomaly-detection':
            launchAnomalyDetectionDemo();
            break;
        case 'algorithm-comparison':
            launchAlgorithmComparisonDemo();
            break;
        case 'streaming':
            launchStreamingDemo();
            break;
        default:
            showGenericDemo();
    }
}

/**
 * Launch anomaly detection interactive demo
 */
function launchAnomalyDetectionDemo() {
    // This would typically open a modal or navigate to an interactive demo
    const modal = createDemoModal('Anomaly Detection Demo', `
        <div class="demo-container">
            <h3>Interactive Anomaly Detection</h3>
            <p>Try different algorithms on sample data:</p>
            <select id="algorithm-select">
                <option value="isolation_forest">Isolation Forest</option>
                <option value="one_class_svm">One-Class SVM</option>
                <option value="lof">Local Outlier Factor</option>
            </select>
            <button onclick="runDemoDetection()">Run Detection</button>
            <div id="demo-results"></div>
        </div>
    `);
    
    document.body.appendChild(modal);
}

/**
 * Create demo modal
 */
function createDemoModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'demo-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10000;
    `;
    
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background: white;
        border-radius: 8px;
        padding: 24px;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
        position: relative;
    `;
    
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: 8px;
        right: 16px;
        background: none;
        border: none;
        font-size: 24px;
        cursor: pointer;
    `;
    closeBtn.addEventListener('click', () => modal.remove());
    
    modalContent.innerHTML = `<h2>${title}</h2>${content}`;
    modalContent.appendChild(closeBtn);
    modal.appendChild(modalContent);
    
    // Close on background click
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    });
    
    return modal;
}

// Global functions for demo interactions
window.runDemoDetection = function() {
    const results = document.getElementById('demo-results');
    results.innerHTML = '<div class="loading">Running detection...</div>';
    
    setTimeout(() => {
        results.innerHTML = `
            <div class="demo-results">
                <h4>Detection Results:</h4>
                <ul>
                    <li>Anomalies detected: 8 out of 100 samples</li>
                    <li>Average confidence: 0.87</li>
                    <li>Processing time: 0.15 seconds</li>
                </ul>
                <div style="margin-top: 16px;">
                    <strong>Anomaly indices:</strong> [12, 34, 45, 67, 78, 89, 92, 95]
                </div>
            </div>
        `;
    }, 1500);
};

// Export functions for external use
window.AnomalyDetectionDocs = {
    copyToClipboard,
    showCopyFeedback,
    launchInteractiveDemo,
    createDemoModal
};