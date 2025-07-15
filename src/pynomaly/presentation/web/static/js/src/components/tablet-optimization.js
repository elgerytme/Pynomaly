/**
 * Tablet Optimization Component for Issue #18
 * Specialized optimizations for tablet devices and hybrid interfaces
 */

class TabletOptimization {
    constructor() {
        this.isTablet = this.detectTablet();
        this.orientation = this.getOrientation();
        this.hasKeyboard = this.detectKeyboard();
        this.hasMouse = this.detectMouse();
        this.supportsSplitScreen = this.detectSplitScreen();
        
        this.init();
    }
    
    init() {
        if (!this.isTablet) return;
        
        this.setupTabletLayout();
        this.setupSplitScreenSupport();
        this.setupKeyboardSupport();
        this.setupMouseSupport();
        this.setupOrientationHandling();
        this.setupTabletNavigation();
        this.setupTabletGestures();
        this.setupTabletAccessibility();
        
        // Listen for orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => this.handleOrientationChange(), 100);
        });
        
        // Listen for keyboard attachment/detachment
        this.setupKeyboardDetection();
    }
    
    detectTablet() {
        const userAgent = navigator.userAgent;
        const isTabletUA = /iPad|Android.*(?:tablet|pad)|Windows.*Touch/i.test(userAgent);
        const isTabletSize = window.screen.width >= 768 && window.screen.width <= 1024;
        const isTouchDevice = 'ontouchstart' in window;
        
        return isTabletUA || (isTabletSize && isTouchDevice);
    }
    
    getOrientation() {
        return window.screen.orientation?.type || 
               (window.innerWidth > window.innerHeight ? 'landscape' : 'portrait');
    }
    
    detectKeyboard() {
        // Detect if physical keyboard is connected
        return window.navigator.keyboard !== undefined ||
               window.screen.width > 1024 || // Assume keyboard in desktop mode
               this.hasKeyboard;
    }
    
    detectMouse() {
        // Detect if mouse/trackpad is available
        return window.matchMedia('(pointer: fine)').matches ||
               window.matchMedia('(hover: hover)').matches;
    }
    
    detectSplitScreen() {
        // Detect if device supports split screen
        return window.screen.width >= 1024 && 
               ('splitView' in window || 'slideOver' in window);
    }
    
    setupTabletLayout() {
        document.documentElement.classList.add('tablet-mode');
        
        // Set tablet-specific CSS properties
        document.documentElement.style.setProperty('--tablet-columns', 
            this.orientation === 'landscape' ? '2' : '1');
        document.documentElement.style.setProperty('--tablet-sidebar-width', 
            this.orientation === 'landscape' ? '320px' : '0px');
        
        // Setup responsive grid for tablet
        this.setupTabletGrid();
    }
    
    setupTabletGrid() {
        const grids = document.querySelectorAll('.tablet-grid');
        
        grids.forEach(grid => {
            const columns = this.orientation === 'landscape' ? 2 : 1;
            grid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
            grid.style.gap = '24px';
            grid.style.padding = '24px';
        });
    }
    
    setupSplitScreenSupport() {
        if (!this.supportsSplitScreen) return;
        
        // Add split screen button to toolbar
        const toolbar = document.querySelector('.toolbar');
        if (toolbar) {
            const splitScreenButton = document.createElement('button');
            splitScreenButton.className = 'tablet-split-screen-btn';
            splitScreenButton.innerHTML = '⧉';
            splitScreenButton.title = 'Toggle Split Screen';
            splitScreenButton.addEventListener('click', () => {
                this.toggleSplitScreen();
            });
            toolbar.appendChild(splitScreenButton);
        }
        
        // Setup split screen panels
        this.setupSplitScreenPanels();
    }
    
    setupSplitScreenPanels() {
        const mainContainer = document.querySelector('.main-container');
        if (!mainContainer) return;
        
        // Create split screen container
        const splitContainer = document.createElement('div');
        splitContainer.className = 'tablet-split-container';
        splitContainer.innerHTML = `
            <div class="tablet-split-panel primary-panel">
                <div class="tablet-split-header">
                    <h3>Primary View</h3>
                    <button class="tablet-split-close">×</button>
                </div>
                <div class="tablet-split-content"></div>
            </div>
            <div class="tablet-split-divider"></div>
            <div class="tablet-split-panel secondary-panel">
                <div class="tablet-split-header">
                    <h3>Secondary View</h3>
                    <button class="tablet-split-close">×</button>
                </div>
                <div class="tablet-split-content"></div>
            </div>
        `;
        
        mainContainer.appendChild(splitContainer);
        
        // Setup split screen interactions
        this.setupSplitScreenInteractions();
    }
    
    setupSplitScreenInteractions() {
        const divider = document.querySelector('.tablet-split-divider');
        const primaryPanel = document.querySelector('.primary-panel');
        const secondaryPanel = document.querySelector('.secondary-panel');
        
        if (!divider || !primaryPanel || !secondaryPanel) return;
        
        let isDragging = false;
        let startX = 0;
        let startWidth = 0;
        
        divider.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX;
            startWidth = primaryPanel.offsetWidth;
            document.body.style.cursor = 'col-resize';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const deltaX = e.clientX - startX;
            const newWidth = Math.max(200, Math.min(window.innerWidth - 200, startWidth + deltaX));
            const percentage = (newWidth / window.innerWidth) * 100;
            
            primaryPanel.style.width = `${percentage}%`;
            secondaryPanel.style.width = `${100 - percentage}%`;
        });
        
        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                document.body.style.cursor = '';
            }
        });
    }
    
    setupKeyboardSupport() {
        if (!this.hasKeyboard) return;
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case '1':
                        e.preventDefault();
                        this.switchToTab(0);
                        break;
                    case '2':
                        e.preventDefault();
                        this.switchToTab(1);
                        break;
                    case '3':
                        e.preventDefault();
                        this.switchToTab(2);
                        break;
                    case '4':
                        e.preventDefault();
                        this.switchToTab(3);
                        break;
                    case 'k':
                        e.preventDefault();
                        this.openCommandPalette();
                        break;
                    case 'f':
                        e.preventDefault();
                        this.openSearch();
                        break;
                    case 'r':
                        e.preventDefault();
                        this.refreshCurrentView();
                        break;
                }
            }
        });
        
        // Setup command palette
        this.setupCommandPalette();
    }
    
    setupCommandPalette() {
        const palette = document.createElement('div');
        palette.className = 'tablet-command-palette';
        palette.innerHTML = `
            <div class="tablet-command-input-container">
                <input type="text" class="tablet-command-input" placeholder="Type a command...">
                <div class="tablet-command-results"></div>
            </div>
        `;
        
        document.body.appendChild(palette);
        
        const input = palette.querySelector('.tablet-command-input');
        const results = palette.querySelector('.tablet-command-results');
        
        input.addEventListener('input', (e) => {
            this.searchCommands(e.target.value, results);
        });
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const selectedCommand = results.querySelector('.command-result.selected');
                if (selectedCommand) {
                    this.executeCommand(selectedCommand.dataset.command);
                    this.closeCommandPalette();
                }
            } else if (e.key === 'Escape') {
                this.closeCommandPalette();
            }
        });
    }
    
    setupMouseSupport() {
        if (!this.hasMouse) return;
        
        // Add hover states and right-click context menus
        document.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.showContextMenu(e.clientX, e.clientY, e.target);
        });
        
        // Add hover effects for interactive elements
        const interactiveElements = document.querySelectorAll(
            'button, [role="button"], .clickable, .card, .list-item'
        );
        
        interactiveElements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                element.classList.add('tablet-hover');
            });
            
            element.addEventListener('mouseleave', () => {
                element.classList.remove('tablet-hover');
            });
        });
    }
    
    setupOrientationHandling() {
        // Handle orientation changes
        this.handleOrientationChange();
    }
    
    handleOrientationChange() {
        this.orientation = this.getOrientation();
        
        // Update layout based on orientation
        document.documentElement.classList.remove('tablet-portrait', 'tablet-landscape');
        document.documentElement.classList.add(`tablet-${this.orientation}`);
        
        // Update grid layout
        this.setupTabletGrid();
        
        // Update navigation
        this.updateNavigationForOrientation();
        
        // Update split screen layout
        this.updateSplitScreenLayout();
    }
    
    updateNavigationForOrientation() {
        const nav = document.querySelector('.tablet-nav');
        if (!nav) return;
        
        if (this.orientation === 'landscape') {
            nav.classList.add('nav-sidebar');
            nav.classList.remove('nav-bottom');
        } else {
            nav.classList.add('nav-bottom');
            nav.classList.remove('nav-sidebar');
        }
    }
    
    updateSplitScreenLayout() {
        const splitContainer = document.querySelector('.tablet-split-container');
        if (!splitContainer) return;
        
        if (this.orientation === 'landscape') {
            splitContainer.classList.add('split-horizontal');
            splitContainer.classList.remove('split-vertical');
        } else {
            splitContainer.classList.add('split-vertical');
            splitContainer.classList.remove('split-horizontal');
        }
    }
    
    setupTabletNavigation() {
        // Create tablet-optimized navigation
        const nav = document.createElement('nav');
        nav.className = 'tablet-nav';
        
        const navItems = [
            { icon: '📊', label: 'Dashboard', id: 'dashboard' },
            { icon: '📈', label: 'Analytics', id: 'analytics' },
            { icon: '🔔', label: 'Alerts', id: 'alerts' },
            { icon: '🤖', label: 'Models', id: 'models' },
            { icon: '⚙️', label: 'Settings', id: 'settings' }
        ];
        
        navItems.forEach(item => {
            const navItem = document.createElement('button');
            navItem.className = 'tablet-nav-item';
            navItem.dataset.tab = item.id;
            navItem.innerHTML = `
                <span class="tablet-nav-icon">${item.icon}</span>
                <span class="tablet-nav-label">${item.label}</span>
            `;
            
            navItem.addEventListener('click', () => {
                this.switchToTab(item.id);
            });
            
            nav.appendChild(navItem);
        });
        
        document.body.appendChild(nav);
        
        // Update navigation for current orientation
        this.updateNavigationForOrientation();
    }
    
    setupTabletGestures() {
        // Three-finger swipe for tab switching
        let touchCount = 0;
        let startX = 0;
        let startY = 0;
        
        document.addEventListener('touchstart', (e) => {
            touchCount = e.touches.length;
            if (touchCount === 3) {
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            }
        });
        
        document.addEventListener('touchmove', (e) => {
            if (touchCount === 3) {
                e.preventDefault();
                const deltaX = e.touches[0].clientX - startX;
                const deltaY = e.touches[0].clientY - startY;
                
                // Horizontal swipe for tab switching
                if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 100) {
                    const direction = deltaX > 0 ? 'next' : 'prev';
                    this.switchTab(direction);
                    touchCount = 0; // Reset to prevent multiple triggers
                }
            }
        });
        
        // Four-finger pinch for app switcher
        this.setupAppSwitcher();
    }
    
    setupAppSwitcher() {
        let fourFingerGesture = false;
        
        document.addEventListener('touchstart', (e) => {
            if (e.touches.length === 4) {
                fourFingerGesture = true;
            }
        });
        
        document.addEventListener('touchmove', (e) => {
            if (fourFingerGesture && e.touches.length === 4) {
                // Check for pinch gesture
                const touches = Array.from(e.touches);
                const distance = this.calculateAverageDistance(touches);
                
                if (distance < 100) { // Threshold for pinch
                    this.showAppSwitcher();
                    fourFingerGesture = false;
                }
            }
        });
        
        document.addEventListener('touchend', () => {
            fourFingerGesture = false;
        });
    }
    
    setupTabletAccessibility() {
        // Enhanced focus management for tablet
        const focusableElements = document.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        focusableElements.forEach(element => {
            element.addEventListener('focus', (e) => {
                // Ensure focused element is visible
                e.target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'nearest'
                });
            });
        });
        
        // Add focus indicators
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
        });
        
        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-navigation');
        });
    }
    
    setupKeyboardDetection() {
        // Listen for virtual keyboard
        const initialViewportHeight = window.innerHeight;
        
        window.addEventListener('resize', () => {
            const currentViewportHeight = window.innerHeight;
            const heightDifference = initialViewportHeight - currentViewportHeight;
            
            if (heightDifference > 150) { // Keyboard is likely visible
                document.body.classList.add('virtual-keyboard-open');
                this.handleVirtualKeyboard(true);
            } else {
                document.body.classList.remove('virtual-keyboard-open');
                this.handleVirtualKeyboard(false);
            }
        });
    }
    
    handleVirtualKeyboard(isOpen) {
        const fixedElements = document.querySelectorAll('.tablet-nav, .tablet-header');
        
        fixedElements.forEach(element => {
            if (isOpen) {
                element.style.position = 'absolute';
            } else {
                element.style.position = 'fixed';
            }
        });
    }
    
    // Utility methods
    switchToTab(tabId) {
        const tabs = document.querySelectorAll('.tablet-nav-item');
        const tabPanels = document.querySelectorAll('.tab-panel');
        
        tabs.forEach(tab => {
            if (tab.dataset.tab === tabId) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
        
        tabPanels.forEach(panel => {
            if (panel.dataset.tab === tabId) {
                panel.classList.add('active');
            } else {
                panel.classList.remove('active');
            }
        });
    }
    
    switchTab(direction) {
        const tabs = document.querySelectorAll('.tablet-nav-item');
        const activeTab = document.querySelector('.tablet-nav-item.active');
        
        if (!activeTab) return;
        
        const currentIndex = Array.from(tabs).indexOf(activeTab);
        let newIndex;
        
        if (direction === 'next') {
            newIndex = (currentIndex + 1) % tabs.length;
        } else {
            newIndex = (currentIndex - 1 + tabs.length) % tabs.length;
        }
        
        this.switchToTab(tabs[newIndex].dataset.tab);
    }
    
    toggleSplitScreen() {
        const splitContainer = document.querySelector('.tablet-split-container');
        if (splitContainer) {
            splitContainer.classList.toggle('active');
        }
    }
    
    openCommandPalette() {
        const palette = document.querySelector('.tablet-command-palette');
        if (palette) {
            palette.classList.add('active');
            palette.querySelector('.tablet-command-input').focus();
        }
    }
    
    closeCommandPalette() {
        const palette = document.querySelector('.tablet-command-palette');
        if (palette) {
            palette.classList.remove('active');
            palette.querySelector('.tablet-command-input').value = '';
        }
    }
    
    searchCommands(query, resultsContainer) {
        const commands = [
            { name: 'Switch to Dashboard', command: 'dashboard' },
            { name: 'Switch to Analytics', command: 'analytics' },
            { name: 'Switch to Alerts', command: 'alerts' },
            { name: 'Switch to Models', command: 'models' },
            { name: 'Refresh View', command: 'refresh' },
            { name: 'Toggle Split Screen', command: 'split' },
            { name: 'Open Settings', command: 'settings' },
            { name: 'Export Data', command: 'export' }
        ];
        
        const filteredCommands = commands.filter(cmd => 
            cmd.name.toLowerCase().includes(query.toLowerCase())
        );
        
        resultsContainer.innerHTML = '';
        
        filteredCommands.forEach((cmd, index) => {
            const result = document.createElement('div');
            result.className = 'command-result';
            result.dataset.command = cmd.command;
            result.textContent = cmd.name;
            
            if (index === 0) {
                result.classList.add('selected');
            }
            
            result.addEventListener('click', () => {
                this.executeCommand(cmd.command);
                this.closeCommandPalette();
            });
            
            resultsContainer.appendChild(result);
        });
    }
    
    executeCommand(command) {
        switch (command) {
            case 'dashboard':
            case 'analytics':
            case 'alerts':
            case 'models':
            case 'settings':
                this.switchToTab(command);
                break;
            case 'refresh':
                this.refreshCurrentView();
                break;
            case 'split':
                this.toggleSplitScreen();
                break;
            case 'export':
                this.exportCurrentView();
                break;
        }
    }
    
    showContextMenu(x, y, target) {
        const menu = document.createElement('div');
        menu.className = 'tablet-context-menu';
        
        const menuItems = this.getContextMenuItems(target);
        
        menuItems.forEach(item => {
            const menuItem = document.createElement('button');
            menuItem.className = 'tablet-context-menu-item';
            menuItem.textContent = item.label;
            menuItem.addEventListener('click', () => {
                item.action();
                menu.remove();
            });
            menu.appendChild(menuItem);
        });
        
        menu.style.position = 'fixed';
        menu.style.left = `${x}px`;
        menu.style.top = `${y}px`;
        menu.style.zIndex = '1000';
        
        document.body.appendChild(menu);
        
        // Remove menu on outside click
        setTimeout(() => {
            document.addEventListener('click', (e) => {
                if (!menu.contains(e.target)) {
                    menu.remove();
                }
            });
        }, 100);
    }
    
    getContextMenuItems(target) {
        const items = [];
        
        if (target.closest('.chart')) {
            items.push({
                label: 'Export Chart',
                action: () => this.exportChart(target)
            });
        }
        
        if (target.closest('.data-table')) {
            items.push({
                label: 'Export Table',
                action: () => this.exportTable(target)
            });
        }
        
        items.push({
            label: 'Refresh',
            action: () => this.refreshElement(target)
        });
        
        if (navigator.share) {
            items.push({
                label: 'Share',
                action: () => this.shareElement(target)
            });
        }
        
        return items;
    }
    
    calculateAverageDistance(touches) {
        let totalDistance = 0;
        let pairCount = 0;
        
        for (let i = 0; i < touches.length; i++) {
            for (let j = i + 1; j < touches.length; j++) {
                const dx = touches[i].clientX - touches[j].clientX;
                const dy = touches[i].clientY - touches[j].clientY;
                totalDistance += Math.sqrt(dx * dx + dy * dy);
                pairCount++;
            }
        }
        
        return totalDistance / pairCount;
    }
    
    showAppSwitcher() {
        const switcher = document.createElement('div');
        switcher.className = 'tablet-app-switcher';
        switcher.innerHTML = `
            <div class="app-switcher-header">
                <h3>App Switcher</h3>
                <button class="app-switcher-close">×</button>
            </div>
            <div class="app-switcher-content">
                <div class="app-item active">
                    <div class="app-preview">Current App</div>
                    <div class="app-title">Pynomaly</div>
                </div>
            </div>
        `;
        
        document.body.appendChild(switcher);
        
        switcher.querySelector('.app-switcher-close').addEventListener('click', () => {
            switcher.remove();
        });
        
        setTimeout(() => {
            document.addEventListener('click', (e) => {
                if (!switcher.contains(e.target)) {
                    switcher.remove();
                }
            });
        }, 100);
    }
    
    refreshCurrentView() {
        const activeTab = document.querySelector('.tablet-nav-item.active');
        if (activeTab) {
            const event = new CustomEvent('refresh-view', { 
                detail: { tab: activeTab.dataset.tab } 
            });
            document.dispatchEvent(event);
        }
    }
    
    exportCurrentView() {
        const activeTab = document.querySelector('.tablet-nav-item.active');
        if (activeTab) {
            const event = new CustomEvent('export-view', { 
                detail: { tab: activeTab.dataset.tab } 
            });
            document.dispatchEvent(event);
        }
    }
    
    exportChart(target) {
        // Implementation for chart export
        console.log('Exporting chart:', target);
    }
    
    exportTable(target) {
        // Implementation for table export
        console.log('Exporting table:', target);
    }
    
    refreshElement(target) {
        // Implementation for element refresh
        target.dispatchEvent(new CustomEvent('refresh'));
    }
    
    shareElement(target) {
        // Implementation for element sharing
        if (navigator.share) {
            navigator.share({
                title: 'Pynomaly Data',
                text: 'Check out this data visualization',
                url: window.location.href
            });
        }
    }
    
    openSearch() {
        const searchModal = document.querySelector('#search-modal');
        if (searchModal) {
            searchModal.classList.add('active');
            searchModal.querySelector('input').focus();
        }
    }
}

// Initialize tablet optimization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new TabletOptimization();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TabletOptimization;
}