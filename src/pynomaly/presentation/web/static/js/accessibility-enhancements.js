/**
 * Pynomaly Accessibility Enhancement JavaScript
 * 
 * Provides comprehensive accessibility features including:
 * - Voice command support
 * - Enhanced keyboard navigation
 * - Screen reader announcements
 * - Motor accessibility features
 * - Focus management and trapping
 */

(function() {
    'use strict';

    // === Voice Command Support ===
    class VoiceCommandHandler {
        constructor() {
            this.recognition = null;
            this.isListening = false;
            this.commands = new Map();
            this.setupVoiceRecognition();
            this.registerCommands();
        }

        setupVoiceRecognition() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                this.recognition = new SpeechRecognition();
                
                this.recognition.continuous = false;
                this.recognition.interimResults = false;
                this.recognition.lang = 'en-US';
                
                this.recognition.onresult = (event) => {
                    const command = event.results[0][0].transcript.toLowerCase().trim();
                    this.processCommand(command);
                };
                
                this.recognition.onerror = (event) => {
                    console.warn('Voice recognition error:', event.error);
                    announceToScreenReader('Voice command failed. Please try again.');
                };
                
                this.recognition.onend = () => {
                    this.isListening = false;
                    this.updateVoiceButton();
                };
            }
        }

        registerCommands() {
            // Navigation commands
            this.commands.set('go to dashboard', () => window.location.href = '/');
            this.commands.set('go to datasets', () => window.location.href = '/datasets');
            this.commands.set('go to detectors', () => window.location.href = '/detectors');
            this.commands.set('go to detection', () => window.location.href = '/detection');
            this.commands.set('go to monitoring', () => window.location.href = '/monitoring');
            this.commands.set('go to experiments', () => window.location.href = '/experiments');
            
            // Search commands
            this.commands.set('search', () => {
                const searchInput = document.getElementById('global-search');
                if (searchInput) {
                    searchInput.focus();
                    announceToScreenReader('Search field focused. Please enter your search term.');
                }
            });
            
            // Accessibility commands
            this.commands.set('toggle high contrast', () => {
                const button = document.getElementById('high-contrast-toggle');
                if (button) button.click();
            });
            
            this.commands.set('increase font size', () => {
                const button = document.getElementById('increase-font-size');
                if (button) button.click();
            });
            
            this.commands.set('decrease font size', () => {
                const button = document.getElementById('decrease-font-size');
                if (button) button.click();
            });
            
            this.commands.set('toggle dark mode', () => {
                const button = document.getElementById('theme-toggle');
                if (button) button.click();
            });
            
            // Form commands
            this.commands.set('submit form', () => {
                const form = document.querySelector('form');
                if (form) {
                    form.dispatchEvent(new Event('submit'));
                }
            });
            
            this.commands.set('cancel', () => {
                const cancelButton = document.querySelector('button[type="button"]:contains("Cancel")');
                if (cancelButton) {
                    cancelButton.click();
                } else {
                    window.history.back();
                }
            });
            
            // Help command
            this.commands.set('help', () => {
                this.showVoiceHelp();
            });
            
            this.commands.set('what can i say', () => {
                this.showVoiceHelp();
            });
        }

        processCommand(command) {
            announceToScreenReader(`Voice command received: ${command}`);
            
            // Try exact match first
            if (this.commands.has(command)) {
                this.commands.get(command)();
                announceToScreenReader('Voice command executed successfully.');
                return;
            }
            
            // Try partial matches
            for (const [registeredCommand, action] of this.commands) {
                if (registeredCommand.includes(command) || command.includes(registeredCommand)) {
                    action();
                    announceToScreenReader(`Voice command executed: ${registeredCommand}`);
                    return;
                }
            }
            
            announceToScreenReader('Voice command not recognized. Say "help" to hear available commands.');
        }

        startListening() {
            if (this.recognition && !this.isListening) {
                this.isListening = true;
                this.recognition.start();
                this.updateVoiceButton();
                announceToScreenReader('Voice commands activated. Listening...');
            }
        }

        stopListening() {
            if (this.recognition && this.isListening) {
                this.recognition.stop();
                this.isListening = false;
                this.updateVoiceButton();
                announceToScreenReader('Voice commands deactivated.');
            }
        }

        updateVoiceButton() {
            const button = document.getElementById('voice-toggle');
            if (button) {
                button.setAttribute('aria-pressed', this.isListening);
                button.title = this.isListening ? 'Stop voice commands' : 'Start voice commands';
                
                const icon = button.querySelector('.voice-icon');
                if (icon) {
                    icon.textContent = this.isListening ? '🎤' : '🔇';
                }
            }
        }

        showVoiceHelp() {
            const commands = Array.from(this.commands.keys()).join(', ');
            announceToScreenReader(`Available voice commands: ${commands}`);
            
            // Create and show help modal
            this.createVoiceHelpModal(commands);
        }

        createVoiceHelpModal(commands) {
            const modal = document.createElement('div');
            modal.className = 'modal show';
            modal.setAttribute('role', 'dialog');
            modal.setAttribute('aria-labelledby', 'voice-help-title');
            modal.setAttribute('aria-describedby', 'voice-help-content');
            
            modal.innerHTML = `
                <div class="modal-content" tabindex="-1">
                    <div class="modal-header">
                        <h2 id="voice-help-title">Voice Commands Help</h2>
                        <button type="button" class="modal-close" aria-label="Close help">
                            <span aria-hidden="true">&times;</span>
                        </button>
                    </div>
                    <div id="voice-help-content" class="modal-body">
                        <p>You can use voice commands to navigate and control the application. Here are the available commands:</p>
                        <ul class="voice-commands-list">
                            ${Array.from(this.commands.keys()).map(cmd => `<li>"${cmd}"</li>`).join('')}
                        </ul>
                        <p>To use voice commands:</p>
                        <ol>
                            <li>Click the microphone button or press Ctrl+Shift+V</li>
                            <li>Wait for the "listening" announcement</li>
                            <li>Speak your command clearly</li>
                            <li>The command will be executed automatically</li>
                        </ol>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-primary modal-close">Got it</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            // Focus management
            const closeButtons = modal.querySelectorAll('.modal-close');
            closeButtons.forEach(button => {
                button.addEventListener('click', () => {
                    document.body.removeChild(modal);
                });
            });
            
            // Trap focus in modal
            trapFocus(modal);
            modal.querySelector('.modal-content').focus();
        }
    }

    // === Enhanced Keyboard Navigation ===
    class KeyboardNavigationManager {
        constructor() {
            this.currentFocusIndex = -1;
            this.focusableElements = [];
            this.setupKeyboardShortcuts();
            this.setupFocusableElementsTracking();
        }

        setupKeyboardShortcuts() {
            document.addEventListener('keydown', (e) => {
                // Ctrl+Shift+V: Toggle voice commands
                if (e.ctrlKey && e.shiftKey && e.key === 'V') {
                    e.preventDefault();
                    const voiceButton = document.getElementById('voice-toggle');
                    if (voiceButton) voiceButton.click();
                }
                
                // Ctrl+Shift+S: Focus search
                if (e.ctrlKey && e.shiftKey && e.key === 'S') {
                    e.preventDefault();
                    const searchInput = document.getElementById('global-search');
                    if (searchInput) {
                        searchInput.focus();
                        announceToScreenReader('Search field focused');
                    }
                }
                
                // Ctrl+Shift+H: Go home/dashboard
                if (e.ctrlKey && e.shiftKey && e.key === 'H') {
                    e.preventDefault();
                    window.location.href = '/';
                }
                
                // Ctrl+Shift+M: Open main menu (mobile)
                if (e.ctrlKey && e.shiftKey && e.key === 'M') {
                    e.preventDefault();
                    const menuButton = document.querySelector('[aria-controls="mobile-menu"]');
                    if (menuButton) menuButton.click();
                }
                
                // Ctrl+Shift+?: Show keyboard shortcuts help
                if (e.ctrlKey && e.shiftKey && e.key === '?') {
                    e.preventDefault();
                    this.showKeyboardShortcuts();
                }
                
                // Arrow keys for grid navigation
                if (e.target.matches('.metric-card, .action-card, .nav-link')) {
                    this.handleArrowNavigation(e);
                }
                
                // Escape: Close modals, dropdowns, etc.
                if (e.key === 'Escape') {
                    this.handleEscape();
                }
            });
        }

        setupFocusableElementsTracking() {
            // Update focusable elements list when DOM changes
            const observer = new MutationObserver(() => {
                this.updateFocusableElements();
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['disabled', 'tabindex', 'hidden']
            });
            
            this.updateFocusableElements();
        }

        updateFocusableElements() {
            const selector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
            this.focusableElements = Array.from(document.querySelectorAll(selector))
                .filter(el => !el.hidden && getComputedStyle(el).display !== 'none');
        }

        handleArrowNavigation(e) {
            const container = e.target.closest('.grid, .flex, .space-x-4, .space-y-1');
            if (!container) return;
            
            const items = Array.from(container.children).filter(child => 
                child.matches('.metric-card, .action-card, .nav-link')
            );
            
            const currentIndex = items.indexOf(e.target);
            let newIndex = currentIndex;
            
            switch (e.key) {
                case 'ArrowRight':
                    newIndex = Math.min(currentIndex + 1, items.length - 1);
                    break;
                case 'ArrowLeft':
                    newIndex = Math.max(currentIndex - 1, 0);
                    break;
                case 'ArrowDown':
                    // For grid layouts, move to next row
                    const columns = Math.floor(container.offsetWidth / items[0].offsetWidth);
                    newIndex = Math.min(currentIndex + columns, items.length - 1);
                    break;
                case 'ArrowUp':
                    // For grid layouts, move to previous row
                    const cols = Math.floor(container.offsetWidth / items[0].offsetWidth);
                    newIndex = Math.max(currentIndex - cols, 0);
                    break;
                default:
                    return;
            }
            
            if (newIndex !== currentIndex && items[newIndex]) {
                e.preventDefault();
                items[newIndex].focus();
                announceToScreenReader(`Focused ${items[newIndex].textContent || items[newIndex].getAttribute('aria-label')}`);
            }
        }

        handleEscape() {
            // Close open modals
            const openModal = document.querySelector('.modal.show');
            if (openModal) {
                const closeButton = openModal.querySelector('.modal-close');
                if (closeButton) closeButton.click();
                return;
            }
            
            // Close open dropdowns
            const openDropdown = document.querySelector('.dropdown-menu:not(.hidden)');
            if (openDropdown) {
                openDropdown.classList.add('hidden');
                const trigger = document.querySelector(`[aria-controls="${openDropdown.id}"]`);
                if (trigger) {
                    trigger.setAttribute('aria-expanded', 'false');
                    trigger.focus();
                }
                return;
            }
            
            // Close mobile menu
            const mobileMenu = document.querySelector('#mobile-menu');
            if (mobileMenu && !mobileMenu.classList.contains('hidden')) {
                const menuButton = document.querySelector('[aria-controls="mobile-menu"]');
                if (menuButton) menuButton.click();
                return;
            }
        }

        showKeyboardShortcuts() {
            const shortcuts = [
                { keys: 'Ctrl+Shift+S', description: 'Focus search field' },
                { keys: 'Ctrl+Shift+H', description: 'Go to dashboard' },
                { keys: 'Ctrl+Shift+V', description: 'Toggle voice commands' },
                { keys: 'Ctrl+Shift+M', description: 'Open main menu' },
                { keys: 'Ctrl+Shift+?', description: 'Show keyboard shortcuts' },
                { keys: 'Escape', description: 'Close modals/dropdowns' },
                { keys: 'Tab', description: 'Navigate to next element' },
                { keys: 'Shift+Tab', description: 'Navigate to previous element' },
                { keys: 'Enter/Space', description: 'Activate buttons and links' },
                { keys: 'Arrow keys', description: 'Navigate grid items' }
            ];
            
            const modal = document.createElement('div');
            modal.className = 'modal show';
            modal.setAttribute('role', 'dialog');
            modal.setAttribute('aria-labelledby', 'shortcuts-title');
            
            modal.innerHTML = `
                <div class="modal-content" tabindex="-1">
                    <div class="modal-header">
                        <h2 id="shortcuts-title">Keyboard Shortcuts</h2>
                        <button type="button" class="modal-close" aria-label="Close shortcuts help">
                            <span aria-hidden="true">&times;</span>
                        </button>
                    </div>
                    <div class="modal-body">
                        <p>Use these keyboard shortcuts to navigate more efficiently:</p>
                        <dl class="shortcuts-list">
                            ${shortcuts.map(shortcut => `
                                <div class="shortcut-item">
                                    <dt class="shortcut-keys">${shortcut.keys}</dt>
                                    <dd class="shortcut-description">${shortcut.description}</dd>
                                </div>
                            `).join('')}
                        </dl>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-primary modal-close">Close</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            const closeButtons = modal.querySelectorAll('.modal-close');
            closeButtons.forEach(button => {
                button.addEventListener('click', () => {
                    document.body.removeChild(modal);
                });
            });
            
            trapFocus(modal);
            modal.querySelector('.modal-content').focus();
            
            announceToScreenReader('Keyboard shortcuts help opened');
        }
    }

    // === Motor Accessibility Features ===
    class MotorAccessibilityManager {
        constructor() {
            this.stickyHover = false;
            this.interactionTimeout = 5000; // Default 5 seconds
            this.setupMotorAccessibilityFeatures();
        }

        setupMotorAccessibilityFeatures() {
            this.addAccessibilityControls();
            this.setupStickyHover();
            this.setupAdjustableTiming();
            this.setupSingleClickAlternatives();
        }

        addAccessibilityControls() {
            const accessibilityControls = document.querySelector('.flex.items-center.space-x-2');
            if (!accessibilityControls) return;
            
            // Sticky hover toggle
            const stickyHoverButton = document.createElement('button');
            stickyHoverButton.id = 'sticky-hover-toggle';
            stickyHoverButton.className = 'p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary';
            stickyHoverButton.setAttribute('aria-label', 'Toggle sticky hover');
            stickyHoverButton.setAttribute('title', 'Toggle sticky hover mode');
            stickyHoverButton.innerHTML = `
                <span class="sr-only">Toggle sticky hover mode</span>
                <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                </svg>
            `;
            
            stickyHoverButton.addEventListener('click', () => {
                this.toggleStickyHover();
            });
            
            accessibilityControls.appendChild(stickyHoverButton);
            
            // Interaction timing control
            const timingButton = document.createElement('button');
            timingButton.id = 'timing-toggle';
            timingButton.className = 'p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary';
            timingButton.setAttribute('aria-label', 'Adjust interaction timing');
            timingButton.setAttribute('title', 'Adjust interaction timing');
            timingButton.innerHTML = `
                <span class="sr-only">Adjust interaction timing</span>
                <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            `;
            
            timingButton.addEventListener('click', () => {
                this.showTimingControls();
            });
            
            accessibilityControls.appendChild(timingButton);
        }

        toggleStickyHover() {
            this.stickyHover = !this.stickyHover;
            localStorage.setItem('sticky-hover', this.stickyHover);
            
            const button = document.getElementById('sticky-hover-toggle');
            if (button) {
                button.setAttribute('aria-pressed', this.stickyHover);
            }
            
            if (this.stickyHover) {
                document.body.classList.add('sticky-hover-mode');
                announceToScreenReader('Sticky hover mode enabled. Click elements to toggle hover state.');
            } else {
                document.body.classList.remove('sticky-hover-mode');
                announceToScreenReader('Sticky hover mode disabled.');
            }
        }

        setupStickyHover() {
            // Apply saved preference
            const savedStickyHover = localStorage.getItem('sticky-hover');
            if (savedStickyHover === 'true') {
                this.toggleStickyHover();
            }
            
            // Handle sticky hover interactions
            document.addEventListener('click', (e) => {
                if (!this.stickyHover) return;
                
                const hoverableElement = e.target.closest('.hover\\:bg-gray-100, .hover\\:text-gray-700, .hover\\:border-gray-300, .metric-card, .action-card');
                if (hoverableElement) {
                    hoverableElement.classList.toggle('force-hover');
                }
            });
        }

        setupAdjustableTiming() {
            // Apply saved timing preference
            const savedTiming = localStorage.getItem('interaction-timeout');
            if (savedTiming) {
                this.interactionTimeout = parseInt(savedTiming);
            }
            
            // Override default timeouts for interactive elements
            this.setupCustomTimeouts();
        }

        setupCustomTimeouts() {
            // Custom timeout for dropdowns
            const dropdownTriggers = document.querySelectorAll('[aria-haspopup="true"]');
            dropdownTriggers.forEach(trigger => {
                let timeoutId;
                
                trigger.addEventListener('mouseenter', () => {
                    clearTimeout(timeoutId);
                    timeoutId = setTimeout(() => {
                        const menu = document.getElementById(trigger.getAttribute('aria-controls'));
                        if (menu) {
                            menu.classList.remove('hidden');
                            trigger.setAttribute('aria-expanded', 'true');
                        }
                    }, this.interactionTimeout / 10); // Faster for mouse
                });
                
                trigger.addEventListener('mouseleave', () => {
                    clearTimeout(timeoutId);
                    timeoutId = setTimeout(() => {
                        const menu = document.getElementById(trigger.getAttribute('aria-controls'));
                        if (menu) {
                            menu.classList.add('hidden');
                            trigger.setAttribute('aria-expanded', 'false');
                        }
                    }, this.interactionTimeout);
                });
            });
        }

        setupSingleClickAlternatives() {
            // Convert double-click actions to single-click with confirmation
            document.addEventListener('dblclick', (e) => {
                e.preventDefault();
                
                // Convert to single click with confirmation
                const confirmAction = confirm('Execute action? (Double-click converted to single-click for accessibility)');
                if (confirmAction) {
                    e.target.click();
                }
            });
            
            // Add keyboard alternatives for mouse-only actions
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    const target = e.target;
                    
                    // Handle card interactions
                    if (target.matches('.metric-card, .action-card') && !target.matches('button, a')) {
                        e.preventDefault();
                        
                        // Find clickable element within card
                        const clickable = target.querySelector('a, button');
                        if (clickable) {
                            clickable.click();
                        }
                    }
                }
            });
        }

        showTimingControls() {
            const modal = document.createElement('div');
            modal.className = 'modal show';
            modal.setAttribute('role', 'dialog');
            modal.setAttribute('aria-labelledby', 'timing-title');
            
            modal.innerHTML = `
                <div class="modal-content" tabindex="-1">
                    <div class="modal-header">
                        <h2 id="timing-title">Interaction Timing Settings</h2>
                        <button type="button" class="modal-close" aria-label="Close timing settings">
                            <span aria-hidden="true">&times;</span>
                        </button>
                    </div>
                    <div class="modal-body">
                        <p>Adjust timing for interactive elements to match your needs:</p>
                        <div class="form-group">
                            <label for="timeout-slider" class="form-label">Interaction Timeout (seconds)</label>
                            <input type="range" id="timeout-slider" min="1" max="30" value="${this.interactionTimeout / 1000}" 
                                   class="form-input" aria-describedby="timeout-help">
                            <div id="timeout-help" class="form-help">
                                Current setting: <span id="timeout-value">${this.interactionTimeout / 1000}</span> seconds
                            </div>
                        </div>
                        <div class="form-group">
                            <p>This affects:</p>
                            <ul>
                                <li>How long dropdown menus stay open</li>
                                <li>Timeout for completing interactions</li>
                                <li>Auto-hide delays for notifications</li>
                            </ul>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-outline modal-close">Cancel</button>
                        <button type="button" class="btn btn-primary" id="save-timing">Save Settings</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            const slider = modal.querySelector('#timeout-slider');
            const valueDisplay = modal.querySelector('#timeout-value');
            
            slider.addEventListener('input', () => {
                valueDisplay.textContent = slider.value;
            });
            
            const saveButton = modal.querySelector('#save-timing');
            saveButton.addEventListener('click', () => {
                this.interactionTimeout = parseInt(slider.value) * 1000;
                localStorage.setItem('interaction-timeout', this.interactionTimeout);
                announceToScreenReader(`Interaction timeout set to ${slider.value} seconds`);
                
                document.body.removeChild(modal);
                this.setupCustomTimeouts();
            });
            
            const closeButtons = modal.querySelectorAll('.modal-close');
            closeButtons.forEach(button => {
                button.addEventListener('click', () => {
                    document.body.removeChild(modal);
                });
            });
            
            trapFocus(modal);
            slider.focus();
        }
    }

    // === Utility Functions ===
    function announceToScreenReader(message) {
        const liveRegion = document.getElementById('live-announcements');
        if (liveRegion) {
            liveRegion.textContent = message;
            setTimeout(() => {
                liveRegion.textContent = '';
            }, 1000);
        }
    }

    function trapFocus(container) {
        const focusableElements = container.querySelectorAll(
            'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        
        container.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                if (e.shiftKey) {
                    if (document.activeElement === firstElement) {
                        e.preventDefault();
                        lastElement.focus();
                    }
                } else {
                    if (document.activeElement === lastElement) {
                        e.preventDefault();
                        firstElement.focus();
                    }
                }
            }
        });
    }

    // === Add Voice Toggle Button ===
    function addVoiceToggleButton() {
        const accessibilityControls = document.querySelector('.flex.items-center.space-x-2');
        if (!accessibilityControls) return;
        
        const voiceToggleButton = document.createElement('button');
        voiceToggleButton.id = 'voice-toggle';
        voiceToggleButton.className = 'p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary';
        voiceToggleButton.setAttribute('aria-label', 'Toggle voice commands');
        voiceToggleButton.setAttribute('aria-pressed', 'false');
        voiceToggleButton.setAttribute('title', 'Toggle voice commands');
        voiceToggleButton.innerHTML = `
            <span class="sr-only">Toggle voice commands</span>
            <span class="voice-icon text-lg" aria-hidden="true">🔇</span>
        `;
        
        voiceToggleButton.addEventListener('click', () => {
            if (window.voiceHandler.isListening) {
                window.voiceHandler.stopListening();
            } else {
                window.voiceHandler.startListening();
            }
        });
        
        accessibilityControls.appendChild(voiceToggleButton);
    }

    // === CSS for Accessibility Features ===
    function addAccessibilityStyles() {
        const style = document.createElement('style');
        style.textContent = `
            /* Sticky hover mode styles */
            .sticky-hover-mode .force-hover {
                background-color: var(--hover-bg-color, #f3f4f6) !important;
                color: var(--hover-text-color, #374151) !important;
                border-color: var(--hover-border-color, #d1d5db) !important;
            }
            
            /* Motor accessibility styles */
            .large-touch-targets button,
            .large-touch-targets a,
            .large-touch-targets input {
                min-height: 48px !important;
                min-width: 48px !important;
                padding: 16px !important;
            }
            
            /* Keyboard shortcuts modal styles */
            .shortcuts-list {
                display: grid;
                gap: 12px;
                margin: 0;
            }
            
            .shortcut-item {
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 16px;
                align-items: center;
                padding: 8px;
                border-radius: 4px;
                background-color: #f9fafb;
            }
            
            .shortcut-keys {
                font-family: monospace;
                font-weight: bold;
                background-color: #e5e7eb;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                margin: 0;
            }
            
            .shortcut-description {
                margin: 0;
                font-size: 14px;
            }
            
            /* Voice commands list styles */
            .voice-commands-list {
                max-height: 300px;
                overflow-y: auto;
                padding-left: 20px;
            }
            
            .voice-commands-list li {
                margin-bottom: 4px;
                font-family: monospace;
                background-color: #f3f4f6;
                padding: 2px 6px;
                border-radius: 3px;
                display: inline-block;
                margin-right: 8px;
            }
            
            /* Enhanced focus indicators for keyboard navigation */
            .keyboard-nav-active *:focus {
                outline: 3px solid #0066cc !important;
                outline-offset: 2px !important;
                box-shadow: 0 0 0 1px #ffffff, 0 0 0 3px #0066cc !important;
            }
            
            /* Grid navigation visual feedback */
            .metric-card:focus,
            .action-card:focus {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }
            
            /* High contrast mode adjustments */
            .high-contrast .voice-icon {
                filter: contrast(2) brightness(1.5);
            }
            
            .high-contrast .shortcut-keys {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #ffffff;
            }
        `;
        
        document.head.appendChild(style);
    }

    // === Initialize Everything ===
    function initializeAccessibilityEnhancements() {
        // Check for required browser features
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Speech recognition not supported in this browser');
        }
        
        // Add CSS styles
        addAccessibilityStyles();
        
        // Initialize managers
        window.voiceHandler = new VoiceCommandHandler();
        window.keyboardManager = new KeyboardNavigationManager();
        window.motorAccessibility = new MotorAccessibilityManager();
        
        // Add UI controls
        addVoiceToggleButton();
        
        // Announce readiness
        setTimeout(() => {
            announceToScreenReader('Accessibility enhancements loaded. Press Ctrl+Shift+? for keyboard shortcuts.');
        }, 1000);
    }

    // === Start When DOM is Ready ===
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeAccessibilityEnhancements);
    } else {
        initializeAccessibilityEnhancements();
    }

})();