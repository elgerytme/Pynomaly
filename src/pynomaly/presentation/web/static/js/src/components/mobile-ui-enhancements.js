/**
 * Mobile UI Enhancements for Issue #18
 * Advanced mobile-responsive UI features for tablets and mobile devices
 */

class MobileUIEnhancements {
    constructor() {
        this.viewport = {
            width: window.innerWidth,
            height: window.innerHeight,
            orientation: window.screen.orientation?.type || 'portrait-primary',
            isTouch: 'ontouchstart' in window,
            isMobile: this.detectMobile(),
            isTablet: this.detectTablet()
        };
        
        this.breakpoints = {
            mobile: 768,
            tablet: 1024,
            desktop: 1200
        };
        
        this.init();
    }
    
    init() {
        this.setupResponsiveLayout();
        this.setupAdvancedGestures();
        this.setupMobileNavigation();
        this.setupTouchOptimizations();
        this.setupOfflineSupport();
        this.setupMobileAlerts();
        this.setupAdaptiveCharts();
        this.setupMobileAccessibility();
        
        // Listen for orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => this.handleOrientationChange(), 100);
        });
        
        // Listen for resize events
        window.addEventListener('resize', this.debounce(() => {
            this.handleResize();
        }, 250));
    }
    
    detectMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
               window.innerWidth <= 768;
    }
    
    detectTablet() {
        return (/iPad|Android/i.test(navigator.userAgent) && window.innerWidth > 768 && window.innerWidth <= 1024) ||
               (window.innerWidth > 768 && window.innerWidth <= 1024 && this.viewport.isTouch);
    }
    
    setupResponsiveLayout() {
        const layout = this.getCurrentLayout();
        document.documentElement.setAttribute('data-layout', layout);
        
        // Apply layout-specific styles
        this.applyLayoutStyles(layout);
        
        // Setup responsive containers
        this.setupResponsiveContainers();
    }
    
    getCurrentLayout() {
        if (this.viewport.width <= this.breakpoints.mobile) return 'mobile';
        if (this.viewport.width <= this.breakpoints.tablet) return 'tablet';
        return 'desktop';
    }
    
    applyLayoutStyles(layout) {
        const root = document.documentElement;
        
        // Remove existing layout classes
        root.classList.remove('layout-mobile', 'layout-tablet', 'layout-desktop');
        
        // Add current layout class
        root.classList.add(`layout-${layout}`);
        
        // Apply layout-specific CSS custom properties
        const layoutStyles = {
            mobile: {
                '--sidebar-width': '0px',
                '--main-padding': '16px',
                '--card-gap': '12px',
                '--font-size-base': '16px',
                '--touch-target-size': '44px'
            },
            tablet: {
                '--sidebar-width': '240px',
                '--main-padding': '24px',
                '--card-gap': '16px',
                '--font-size-base': '16px',
                '--touch-target-size': '44px'
            },
            desktop: {
                '--sidebar-width': '280px',
                '--main-padding': '32px',
                '--card-gap': '24px',
                '--font-size-base': '14px',
                '--touch-target-size': '32px'
            }
        };
        
        Object.entries(layoutStyles[layout]).forEach(([property, value]) => {
            root.style.setProperty(property, value);
        });
    }
    
    setupResponsiveContainers() {
        const containers = document.querySelectorAll('[data-responsive-container]');
        
        containers.forEach(container => {
            const breakpoint = container.dataset.responsiveContainer;
            const currentLayout = this.getCurrentLayout();
            
            // Apply responsive visibility
            if (breakpoint === 'mobile-only' && currentLayout !== 'mobile') {
                container.style.display = 'none';
            } else if (breakpoint === 'tablet-up' && currentLayout === 'mobile') {
                container.style.display = 'none';
            } else if (breakpoint === 'desktop-only' && currentLayout !== 'desktop') {
                container.style.display = 'none';
            } else {
                container.style.display = '';
            }
        });
    }
    
    setupAdvancedGestures() {
        // Enhanced gesture recognition beyond basic touch
        this.gestureManager = new AdvancedGestureManager();
        
        // Setup swipe-to-navigate
        this.setupSwipeNavigation();
        
        // Setup pinch-to-zoom for charts
        this.setupPinchZoom();
        
        // Setup long-press actions
        this.setupLongPress();
    }
    
    setupSwipeNavigation() {
        const tabContainer = document.querySelector('.tab-container');
        if (!tabContainer) return;
        
        let startX = 0;
        let currentX = 0;
        let isDragging = false;
        
        tabContainer.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            isDragging = true;
        });
        
        tabContainer.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            
            currentX = e.touches[0].clientX;
            const deltaX = currentX - startX;
            
            // Provide visual feedback
            const activeTab = tabContainer.querySelector('.tab-active');
            if (activeTab) {
                activeTab.style.transform = `translateX(${deltaX * 0.1}px)`;
            }
        });
        
        tabContainer.addEventListener('touchend', (e) => {
            if (!isDragging) return;
            
            const deltaX = currentX - startX;
            const threshold = 50;
            
            if (Math.abs(deltaX) > threshold) {
                const direction = deltaX > 0 ? 'next' : 'prev';
                this.navigateTab(direction);
            }
            
            // Reset transform
            const activeTab = tabContainer.querySelector('.tab-active');
            if (activeTab) {
                activeTab.style.transform = '';
            }
            
            isDragging = false;
        });
    }
    
    setupPinchZoom() {
        const charts = document.querySelectorAll('[data-chart]');
        
        charts.forEach(chart => {
            let initialDistance = 0;
            let currentScale = 1;
            
            chart.addEventListener('touchstart', (e) => {
                if (e.touches.length === 2) {
                    initialDistance = this.getDistance(e.touches[0], e.touches[1]);
                }
            });
            
            chart.addEventListener('touchmove', (e) => {
                if (e.touches.length === 2) {
                    e.preventDefault();
                    
                    const currentDistance = this.getDistance(e.touches[0], e.touches[1]);
                    const scale = currentDistance / initialDistance;
                    
                    currentScale = Math.max(0.5, Math.min(3, scale));
                    chart.style.transform = `scale(${currentScale})`;
                }
            });
            
            chart.addEventListener('touchend', (e) => {
                if (e.touches.length === 0) {
                    // Smooth transition back to normal scale if needed
                    chart.style.transition = 'transform 0.3s ease';
                    setTimeout(() => {
                        chart.style.transition = '';
                    }, 300);
                }
            });
        });
    }
    
    setupLongPress() {
        const longPressElements = document.querySelectorAll('[data-long-press]');
        
        longPressElements.forEach(element => {
            let longPressTimer;
            
            element.addEventListener('touchstart', (e) => {
                longPressTimer = setTimeout(() => {
                    this.handleLongPress(element, e);
                }, 500);
            });
            
            element.addEventListener('touchend', () => {
                clearTimeout(longPressTimer);
            });
            
            element.addEventListener('touchmove', () => {
                clearTimeout(longPressTimer);
            });
        });
    }
    
    setupMobileNavigation() {
        // Enhanced bottom navigation with haptic feedback
        const bottomNav = document.querySelector('.bottom-nav');
        if (!bottomNav) return;
        
        const navItems = bottomNav.querySelectorAll('.nav-item');
        
        navItems.forEach(item => {
            item.addEventListener('touchstart', () => {
                // Haptic feedback
                if (navigator.vibrate) {
                    navigator.vibrate(10);
                }
                
                // Visual feedback
                item.classList.add('nav-item-pressed');
            });
            
            item.addEventListener('touchend', () => {
                item.classList.remove('nav-item-pressed');
            });
        });
        
        // Setup slide-up modal navigation for mobile
        this.setupSlideUpModal();
    }
    
    setupSlideUpModal() {
        const modalTriggers = document.querySelectorAll('[data-modal-trigger="slide-up"]');
        
        modalTriggers.forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                const targetModal = document.querySelector(trigger.dataset.target);
                if (targetModal) {
                    this.showSlideUpModal(targetModal);
                }
            });
        });
    }
    
    showSlideUpModal(modal) {
        modal.classList.add('modal-active');
        document.body.style.overflow = 'hidden';
        
        // Add backdrop
        const backdrop = document.createElement('div');
        backdrop.className = 'modal-backdrop';
        backdrop.addEventListener('click', () => {
            this.hideSlideUpModal(modal);
        });
        
        document.body.appendChild(backdrop);
        
        // Handle swipe down to close
        this.setupSwipeToClose(modal);
    }
    
    hideSlideUpModal(modal) {
        modal.classList.remove('modal-active');
        document.body.style.overflow = '';
        
        const backdrop = document.querySelector('.modal-backdrop');
        if (backdrop) {
            backdrop.remove();
        }
    }
    
    setupSwipeToClose(modal) {
        let startY = 0;
        let currentY = 0;
        let isDragging = false;
        
        modal.addEventListener('touchstart', (e) => {
            startY = e.touches[0].clientY;
            isDragging = true;
        });
        
        modal.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            
            currentY = e.touches[0].clientY;
            const deltaY = currentY - startY;
            
            if (deltaY > 0) {
                modal.style.transform = `translateY(${deltaY}px)`;
            }
        });
        
        modal.addEventListener('touchend', () => {
            if (!isDragging) return;
            
            const deltaY = currentY - startY;
            const threshold = 100;
            
            if (deltaY > threshold) {
                this.hideSlideUpModal(modal);
            } else {
                modal.style.transform = '';
            }
            
            isDragging = false;
        });
    }
    
    setupTouchOptimizations() {
        // Enhanced touch target sizing
        const touchTargets = document.querySelectorAll('button, [role="button"], input, select, textarea');
        
        touchTargets.forEach(target => {
            const computedStyle = window.getComputedStyle(target);
            const minSize = this.viewport.isMobile ? 44 : 32;
            
            if (parseInt(computedStyle.height) < minSize) {
                target.style.minHeight = `${minSize}px`;
            }
            
            if (parseInt(computedStyle.width) < minSize) {
                target.style.minWidth = `${minSize}px`;
            }
        });
        
        // Setup touch feedback
        this.setupTouchFeedback();
    }
    
    setupTouchFeedback() {
        const interactiveElements = document.querySelectorAll('button, [role="button"], .clickable');
        
        interactiveElements.forEach(element => {
            element.addEventListener('touchstart', (e) => {
                element.classList.add('touch-active');
                
                // Create ripple effect
                const ripple = document.createElement('span');
                ripple.className = 'touch-ripple';
                
                const rect = element.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.touches[0].clientX - rect.left - size / 2;
                const y = e.touches[0].clientY - rect.top - size / 2;
                
                ripple.style.width = ripple.style.height = `${size}px`;
                ripple.style.left = `${x}px`;
                ripple.style.top = `${y}px`;
                
                element.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
            
            element.addEventListener('touchend', () => {
                element.classList.remove('touch-active');
            });
        });
    }
    
    setupOfflineSupport() {
        // Enhanced offline capabilities
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.ready.then((registration) => {
                // Show offline indicator
                window.addEventListener('online', () => {
                    this.showOfflineStatus(false);
                });
                
                window.addEventListener('offline', () => {
                    this.showOfflineStatus(true);
                });
            });
        }
    }
    
    showOfflineStatus(isOffline) {
        const offlineIndicator = document.querySelector('.offline-indicator') || 
                                document.createElement('div');
        
        offlineIndicator.className = 'offline-indicator';
        offlineIndicator.textContent = isOffline ? 'Offline' : 'Online';
        offlineIndicator.style.display = isOffline ? 'block' : 'none';
        
        if (!document.querySelector('.offline-indicator')) {
            document.body.appendChild(offlineIndicator);
        }
    }
    
    setupMobileAlerts() {
        // Mobile-optimized alert system
        const alerts = document.querySelectorAll('.alert');
        
        alerts.forEach(alert => {
            // Make alerts swipeable to dismiss
            this.makeSwipeable(alert, 'horizontal', (direction) => {
                if (Math.abs(direction) > 100) {
                    this.dismissAlert(alert);
                }
            });
        });
    }
    
    dismissAlert(alert) {
        alert.style.transform = 'translateX(100%)';
        alert.style.opacity = '0';
        
        setTimeout(() => {
            alert.remove();
        }, 300);
    }
    
    setupAdaptiveCharts() {
        // Make charts responsive and touch-friendly
        const charts = document.querySelectorAll('[data-chart]');
        
        charts.forEach(chart => {
            this.makeChartResponsive(chart);
            this.addChartTouchControls(chart);
        });
    }
    
    makeChartResponsive(chart) {
        const observer = new ResizeObserver((entries) => {
            entries.forEach(entry => {
                const width = entry.contentRect.width;
                const height = entry.contentRect.height;
                
                // Trigger chart resize
                if (chart.chartInstance && chart.chartInstance.resize) {
                    chart.chartInstance.resize(width, height);
                }
            });
        });
        
        observer.observe(chart);
    }
    
    addChartTouchControls(chart) {
        // Add touch controls for chart interaction
        let isTouch = false;
        
        chart.addEventListener('touchstart', (e) => {
            isTouch = true;
            
            // Show touch controls
            const controls = chart.querySelector('.chart-touch-controls');
            if (controls) {
                controls.style.display = 'block';
            }
        });
        
        chart.addEventListener('touchend', () => {
            setTimeout(() => {
                if (isTouch) {
                    const controls = chart.querySelector('.chart-touch-controls');
                    if (controls) {
                        controls.style.display = 'none';
                    }
                    isTouch = false;
                }
            }, 3000);
        });
    }
    
    setupMobileAccessibility() {
        // Enhanced mobile accessibility
        
        // Setup focus management for mobile
        this.setupMobileFocusManagement();
        
        // Setup voice commands (if supported)
        this.setupVoiceCommands();
        
        // Setup gesture alternatives
        this.setupGestureAlternatives();
    }
    
    setupMobileFocusManagement() {
        // Enhanced focus management for mobile devices
        const focusableElements = document.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        focusableElements.forEach(element => {
            element.addEventListener('focus', (e) => {
                // Ensure focused element is visible
                e.target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
            });
        });
    }
    
    setupVoiceCommands() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();
            
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onresult = (event) => {
                const command = event.results[0][0].transcript.toLowerCase();
                this.handleVoiceCommand(command);
            };
            
            // Add voice activation button
            const voiceButton = document.createElement('button');
            voiceButton.className = 'voice-command-button';
            voiceButton.innerHTML = '🎤';
            voiceButton.addEventListener('click', () => {
                recognition.start();
            });
            
            document.body.appendChild(voiceButton);
        }
    }
    
    handleVoiceCommand(command) {
        const commands = {
            'dashboard': () => this.navigateToTab('dashboard'),
            'analytics': () => this.navigateToTab('analytics'),
            'alerts': () => this.navigateToTab('alerts'),
            'models': () => this.navigateToTab('models'),
            'refresh': () => location.reload(),
            'help': () => this.showHelp()
        };
        
        Object.keys(commands).forEach(key => {
            if (command.includes(key)) {
                commands[key]();
            }
        });
    }
    
    setupGestureAlternatives() {
        // Provide keyboard/voice alternatives to gestures
        document.addEventListener('keydown', (e) => {
            if (e.altKey) {
                switch (e.code) {
                    case 'ArrowLeft':
                        this.navigateTab('prev');
                        break;
                    case 'ArrowRight':
                        this.navigateTab('next');
                        break;
                    case 'KeyR':
                        this.refreshCurrentView();
                        break;
                }
            }
        });
    }
    
    // Utility methods
    getDistance(touch1, touch2) {
        const dx = touch1.clientX - touch2.clientX;
        const dy = touch1.clientY - touch2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    debounce(func, wait) {
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
    
    makeSwipeable(element, direction, callback) {
        let startX = 0, startY = 0, currentX = 0, currentY = 0;
        
        element.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        });
        
        element.addEventListener('touchmove', (e) => {
            currentX = e.touches[0].clientX;
            currentY = e.touches[0].clientY;
        });
        
        element.addEventListener('touchend', () => {
            const deltaX = currentX - startX;
            const deltaY = currentY - startY;
            
            if (direction === 'horizontal') {
                callback(deltaX);
            } else {
                callback(deltaY);
            }
        });
    }
    
    handleOrientationChange() {
        this.viewport.orientation = window.screen.orientation?.type || 'portrait-primary';
        this.viewport.width = window.innerWidth;
        this.viewport.height = window.innerHeight;
        
        // Re-apply responsive layout
        this.setupResponsiveLayout();
        
        // Trigger chart resize
        const charts = document.querySelectorAll('[data-chart]');
        charts.forEach(chart => {
            if (chart.chartInstance && chart.chartInstance.resize) {
                chart.chartInstance.resize();
            }
        });
    }
    
    handleResize() {
        this.viewport.width = window.innerWidth;
        this.viewport.height = window.innerHeight;
        this.setupResponsiveLayout();
    }
    
    navigateTab(direction) {
        const tabs = document.querySelectorAll('.tab-item');
        const activeTab = document.querySelector('.tab-item.active');
        
        if (!activeTab) return;
        
        const currentIndex = Array.from(tabs).indexOf(activeTab);
        let newIndex;
        
        if (direction === 'next') {
            newIndex = (currentIndex + 1) % tabs.length;
        } else {
            newIndex = (currentIndex - 1 + tabs.length) % tabs.length;
        }
        
        tabs[newIndex].click();
    }
    
    navigateToTab(tabName) {
        const tab = document.querySelector(`[data-tab="${tabName}"]`);
        if (tab) {
            tab.click();
        }
    }
    
    refreshCurrentView() {
        const activeTab = document.querySelector('.tab-item.active');
        if (activeTab) {
            // Trigger refresh for active tab
            const event = new CustomEvent('refresh-tab', { detail: { tab: activeTab } });
            document.dispatchEvent(event);
        }
    }
    
    showHelp() {
        // Show mobile help modal
        const helpModal = document.querySelector('#help-modal');
        if (helpModal) {
            this.showSlideUpModal(helpModal);
        }
    }
    
    handleLongPress(element, event) {
        const action = element.dataset.longPress;
        
        // Haptic feedback
        if (navigator.vibrate) {
            navigator.vibrate(50);
        }
        
        switch (action) {
            case 'context-menu':
                this.showContextMenu(element, event);
                break;
            case 'quick-action':
                this.showQuickActions(element);
                break;
            case 'info':
                this.showElementInfo(element);
                break;
        }
    }
    
    showContextMenu(element, event) {
        const menu = document.createElement('div');
        menu.className = 'context-menu mobile-context-menu';
        
        // Add menu items based on element type
        const menuItems = this.getContextMenuItems(element);
        menuItems.forEach(item => {
            const menuItem = document.createElement('div');
            menuItem.className = 'context-menu-item';
            menuItem.textContent = item.label;
            menuItem.addEventListener('click', item.action);
            menu.appendChild(menuItem);
        });
        
        // Position menu
        const rect = element.getBoundingClientRect();
        menu.style.top = `${rect.bottom + 10}px`;
        menu.style.left = `${rect.left}px`;
        
        document.body.appendChild(menu);
        
        // Remove menu on outside click
        setTimeout(() => {
            document.addEventListener('click', () => {
                menu.remove();
            }, { once: true });
        }, 100);
    }
    
    getContextMenuItems(element) {
        const items = [];
        
        if (element.dataset.exportable) {
            items.push({
                label: 'Export',
                action: () => this.exportElement(element)
            });
        }
        
        if (element.dataset.shareable) {
            items.push({
                label: 'Share',
                action: () => this.shareElement(element)
            });
        }
        
        items.push({
            label: 'Refresh',
            action: () => this.refreshElement(element)
        });
        
        return items;
    }
    
    exportElement(element) {
        // Implementation for exporting element data
        console.log('Exporting element:', element);
    }
    
    shareElement(element) {
        // Implementation for sharing element
        if (navigator.share) {
            navigator.share({
                title: 'Pynomaly Data',
                text: 'Check out this anomaly detection result',
                url: window.location.href
            });
        }
    }
    
    refreshElement(element) {
        // Implementation for refreshing element
        element.dispatchEvent(new CustomEvent('refresh'));
    }
    
    showQuickActions(element) {
        // Implementation for quick actions
        console.log('Showing quick actions for:', element);
    }
    
    showElementInfo(element) {
        // Implementation for element info
        console.log('Showing info for:', element);
    }
}

// Advanced gesture manager for complex interactions
class AdvancedGestureManager {
    constructor() {
        this.gestures = new Map();
        this.isTracking = false;
        this.touchHistory = [];
    }
    
    addGesture(name, pattern, callback) {
        this.gestures.set(name, { pattern, callback });
    }
    
    startTracking(element) {
        if (this.isTracking) return;
        
        this.isTracking = true;
        this.touchHistory = [];
        
        element.addEventListener('touchstart', this.onTouchStart.bind(this));
        element.addEventListener('touchmove', this.onTouchMove.bind(this));
        element.addEventListener('touchend', this.onTouchEnd.bind(this));
    }
    
    onTouchStart(e) {
        this.touchHistory = [];
        this.recordTouch(e);
    }
    
    onTouchMove(e) {
        this.recordTouch(e);
    }
    
    onTouchEnd(e) {
        this.recognizeGesture();
    }
    
    recordTouch(e) {
        const touch = {
            x: e.touches[0].clientX,
            y: e.touches[0].clientY,
            timestamp: Date.now()
        };
        this.touchHistory.push(touch);
    }
    
    recognizeGesture() {
        // Implement gesture recognition logic
        this.gestures.forEach((gesture, name) => {
            if (this.matchesPattern(gesture.pattern)) {
                gesture.callback(this.touchHistory);
            }
        });
    }
    
    matchesPattern(pattern) {
        // Implement pattern matching logic
        return false; // Placeholder
    }
}

// Initialize mobile UI enhancements when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new MobileUIEnhancements();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MobileUIEnhancements, AdvancedGestureManager };
}