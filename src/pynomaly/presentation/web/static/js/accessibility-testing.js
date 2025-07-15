/**
 * Pynomaly Accessibility Testing and Validation Module
 * 
 * Provides automated accessibility testing and validation tools including:
 * - WCAG 2.1 compliance checking
 * - Screen reader compatibility testing
 * - Keyboard navigation validation
 * - Accessibility audit scoring
 */

(function() {
    'use strict';

    class AccessibilityTester {
        constructor() {
            this.testResults = {
                wcag: { passed: 0, failed: 0, warnings: 0, score: 0 },
                keyboard: { passed: 0, failed: 0, score: 0 },
                screenReader: { passed: 0, failed: 0, score: 0 },
                overall: { score: 0, grade: 'F' }
            };
            this.issues = [];
            this.init();
        }

        init() {
            this.addTestingInterface();
            this.setupAutomaticTesting();
        }

        // === WCAG 2.1 Compliance Testing ===
        async runWCAGTests() {
            console.log('Running WCAG 2.1 compliance tests...');
            this.testResults.wcag = { passed: 0, failed: 0, warnings: 0, score: 0 };
            this.issues = [];

            // Test color contrast
            await this.testColorContrast();
            
            // Test heading structure
            await this.testHeadingStructure();
            
            // Test form labels
            await this.testFormLabels();
            
            // Test image alt text
            await this.testImageAltText();
            
            // Test keyboard accessibility
            await this.testKeyboardAccessibility();
            
            // Test ARIA implementation
            await this.testARIAImplementation();
            
            // Test focus management
            await this.testFocusManagement();
            
            // Test live regions
            await this.testLiveRegions();
            
            // Test skip links
            await this.testSkipLinks();
            
            // Calculate WCAG score
            this.calculateWCAGScore();
            
            return this.testResults.wcag;
        }

        async testColorContrast() {
            const elements = document.querySelectorAll('*');
            let contrastIssues = 0;
            
            for (const element of elements) {
                const style = window.getComputedStyle(element);
                const textColor = style.color;
                const backgroundColor = style.backgroundColor;
                
                if (textColor && backgroundColor && textColor !== 'rgba(0, 0, 0, 0)' && backgroundColor !== 'rgba(0, 0, 0, 0)') {
                    const contrast = this.calculateContrast(textColor, backgroundColor);
                    const fontSize = parseFloat(style.fontSize);
                    const isLargeText = fontSize >= 18 || (fontSize >= 14 && style.fontWeight >= 700);
                    
                    const requiredRatio = isLargeText ? 3 : 4.5; // AA standard
                    
                    if (contrast < requiredRatio) {
                        contrastIssues++;
                        this.issues.push({
                            type: 'contrast',
                            severity: 'error',
                            element: element,
                            message: `Insufficient color contrast: ${contrast.toFixed(2)}:1 (required: ${requiredRatio}:1)`,
                            wcagCriteria: '1.4.3'
                        });
                    }
                }
            }
            
            if (contrastIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        async testHeadingStructure() {
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            let previousLevel = 0;
            let structureIssues = 0;
            
            // Check for h1
            const h1Count = document.querySelectorAll('h1').length;
            if (h1Count === 0) {
                structureIssues++;
                this.issues.push({
                    type: 'heading',
                    severity: 'error',
                    message: 'Page is missing an h1 heading',
                    wcagCriteria: '1.3.1'
                });
            } else if (h1Count > 1) {
                structureIssues++;
                this.issues.push({
                    type: 'heading',
                    severity: 'warning',
                    message: 'Multiple h1 headings found. Consider using only one per page.',
                    wcagCriteria: '1.3.1'
                });
            }
            
            // Check heading hierarchy
            headings.forEach((heading, index) => {
                const level = parseInt(heading.tagName.substring(1));
                
                if (index > 0 && level > previousLevel + 1) {
                    structureIssues++;
                    this.issues.push({
                        type: 'heading',
                        severity: 'error',
                        element: heading,
                        message: `Heading level skipped from h${previousLevel} to h${level}`,
                        wcagCriteria: '1.3.1'
                    });
                }
                
                // Check for empty headings
                if (!heading.textContent.trim()) {
                    structureIssues++;
                    this.issues.push({
                        type: 'heading',
                        severity: 'error',
                        element: heading,
                        message: 'Empty heading found',
                        wcagCriteria: '1.3.1'
                    });
                }
                
                previousLevel = level;
            });
            
            if (structureIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        async testFormLabels() {
            const inputs = document.querySelectorAll('input, select, textarea');
            let labelIssues = 0;
            
            inputs.forEach(input => {
                const hasLabel = input.labels && input.labels.length > 0;
                const hasAriaLabel = input.getAttribute('aria-label');
                const hasAriaLabelledBy = input.getAttribute('aria-labelledby');
                const hasTitle = input.getAttribute('title');
                
                if (!hasLabel && !hasAriaLabel && !hasAriaLabelledBy && !hasTitle) {
                    labelIssues++;
                    this.issues.push({
                        type: 'form',
                        severity: 'error',
                        element: input,
                        message: 'Form control missing accessible label',
                        wcagCriteria: '1.3.1'
                    });
                }
                
                // Check for placeholder-only labels
                if (!hasLabel && !hasAriaLabel && !hasAriaLabelledBy && input.placeholder) {
                    this.issues.push({
                        type: 'form',
                        severity: 'warning',
                        element: input,
                        message: 'Using placeholder as label. Consider adding proper label.',
                        wcagCriteria: '1.3.1'
                    });
                }
            });
            
            if (labelIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        async testImageAltText() {
            const images = document.querySelectorAll('img');
            let altTextIssues = 0;
            
            images.forEach(img => {
                const hasAlt = img.hasAttribute('alt');
                const altText = img.getAttribute('alt');
                
                if (!hasAlt) {
                    altTextIssues++;
                    this.issues.push({
                        type: 'image',
                        severity: 'error',
                        element: img,
                        message: 'Image missing alt attribute',
                        wcagCriteria: '1.1.1'
                    });
                } else if (altText && altText.length > 100) {
                    this.issues.push({
                        type: 'image',
                        severity: 'warning',
                        element: img,
                        message: 'Alt text is very long. Consider shorter description.',
                        wcagCriteria: '1.1.1'
                    });
                }
            });
            
            if (altTextIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        async testKeyboardAccessibility() {
            const interactive = document.querySelectorAll('a, button, input, select, textarea, [tabindex]');
            let keyboardIssues = 0;
            
            interactive.forEach(element => {
                const tabIndex = element.getAttribute('tabindex');
                
                // Check for positive tabindex (anti-pattern)
                if (tabIndex && parseInt(tabIndex) > 0) {
                    keyboardIssues++;
                    this.issues.push({
                        type: 'keyboard',
                        severity: 'warning',
                        element: element,
                        message: 'Positive tabindex found. Consider using tabindex="0" or removing.',
                        wcagCriteria: '2.4.3'
                    });
                }
                
                // Check for elements that look interactive but aren't keyboard accessible
                if (element.onclick && !element.tabIndex && element.tagName !== 'BUTTON' && element.tagName !== 'A') {
                    keyboardIssues++;
                    this.issues.push({
                        type: 'keyboard',
                        severity: 'error',
                        element: element,
                        message: 'Interactive element not keyboard accessible',
                        wcagCriteria: '2.1.1'
                    });
                }
            });
            
            if (keyboardIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        async testARIAImplementation() {
            let ariaIssues = 0;
            
            // Test invalid ARIA attributes
            const elementsWithAria = document.querySelectorAll('[aria-expanded], [aria-haspopup], [aria-controls], [aria-describedby], [aria-labelledby]');
            
            elementsWithAria.forEach(element => {
                const ariaExpanded = element.getAttribute('aria-expanded');
                const ariaControls = element.getAttribute('aria-controls');
                
                // Check if aria-controls points to existing element
                if (ariaControls && !document.getElementById(ariaControls)) {
                    ariaIssues++;
                    this.issues.push({
                        type: 'aria',
                        severity: 'error',
                        element: element,
                        message: `aria-controls references non-existent element: ${ariaControls}`,
                        wcagCriteria: '4.1.2'
                    });
                }
                
                // Check aria-expanded usage
                if (ariaExpanded && !['true', 'false'].includes(ariaExpanded)) {
                    ariaIssues++;
                    this.issues.push({
                        type: 'aria',
                        severity: 'error',
                        element: element,
                        message: 'aria-expanded must be "true" or "false"',
                        wcagCriteria: '4.1.2'
                    });
                }
            });
            
            // Test for missing ARIA on custom interactive elements
            const customInteractive = document.querySelectorAll('[onclick]:not(button):not(a):not([role])');
            customInteractive.forEach(element => {
                ariaIssues++;
                this.issues.push({
                    type: 'aria',
                    severity: 'error',
                    element: element,
                    message: 'Interactive element missing appropriate role',
                    wcagCriteria: '4.1.2'
                });
            });
            
            if (ariaIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        async testFocusManagement() {
            let focusIssues = 0;
            
            // Test that all interactive elements are focusable
            const interactive = document.querySelectorAll('button, a, input, select, textarea');
            interactive.forEach(element => {
                if (element.disabled || element.style.display === 'none' || element.hidden) return;
                
                const tabIndex = element.tabIndex;
                if (tabIndex === -1 && element.tagName !== 'INPUT') {
                    focusIssues++;
                    this.issues.push({
                        type: 'focus',
                        severity: 'warning',
                        element: element,
                        message: 'Interactive element has tabindex="-1" making it unfocusable',
                        wcagCriteria: '2.4.3'
                    });
                }
            });
            
            // Test for focus traps in modals
            const modals = document.querySelectorAll('.modal, [role="dialog"]');
            modals.forEach(modal => {
                if (!modal.hidden && !modal.classList.contains('hidden')) {
                    const focusableInModal = modal.querySelectorAll('button, a, input, select, textarea, [tabindex]:not([tabindex="-1"])');
                    if (focusableInModal.length === 0) {
                        focusIssues++;
                        this.issues.push({
                            type: 'focus',
                            severity: 'error',
                            element: modal,
                            message: 'Modal has no focusable elements',
                            wcagCriteria: '2.4.3'
                        });
                    }
                }
            });
            
            if (focusIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        async testLiveRegions() {
            const liveRegions = document.querySelectorAll('[aria-live], [role="status"], [role="alert"]');
            let liveRegionIssues = 0;
            
            // Check for proper live region setup
            if (liveRegions.length === 0) {
                liveRegionIssues++;
                this.issues.push({
                    type: 'live-region',
                    severity: 'warning',
                    message: 'No ARIA live regions found. Consider adding for dynamic content.',
                    wcagCriteria: '4.1.3'
                });
            }
            
            liveRegions.forEach(region => {
                const ariaLive = region.getAttribute('aria-live');
                if (ariaLive && !['polite', 'assertive', 'off'].includes(ariaLive)) {
                    liveRegionIssues++;
                    this.issues.push({
                        type: 'live-region',
                        severity: 'error',
                        element: region,
                        message: 'Invalid aria-live value. Must be "polite", "assertive", or "off"',
                        wcagCriteria: '4.1.3'
                    });
                }
            });
            
            if (liveRegionIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        async testSkipLinks() {
            const skipLinks = document.querySelectorAll('.skip-link, a[href^="#"]:first-child');
            let skipLinkIssues = 0;
            
            if (skipLinks.length === 0) {
                skipLinkIssues++;
                this.issues.push({
                    type: 'skip-link',
                    severity: 'error',
                    message: 'No skip links found. Add skip links for keyboard navigation.',
                    wcagCriteria: '2.4.1'
                });
            } else {
                skipLinks.forEach(link => {
                    const href = link.getAttribute('href');
                    if (href && href.startsWith('#')) {
                        const target = document.getElementById(href.substring(1));
                        if (!target) {
                            skipLinkIssues++;
                            this.issues.push({
                                type: 'skip-link',
                                severity: 'error',
                                element: link,
                                message: `Skip link target not found: ${href}`,
                                wcagCriteria: '2.4.1'
                            });
                        }
                    }
                });
            }
            
            if (skipLinkIssues === 0) {
                this.testResults.wcag.passed++;
            } else {
                this.testResults.wcag.failed++;
            }
        }

        calculateWCAGScore() {
            const total = this.testResults.wcag.passed + this.testResults.wcag.failed;
            this.testResults.wcag.score = total > 0 ? Math.round((this.testResults.wcag.passed / total) * 100) : 0;
        }

        // === Keyboard Navigation Testing ===
        async testKeyboardNavigation() {
            console.log('Testing keyboard navigation...');
            this.testResults.keyboard = { passed: 0, failed: 0, score: 0 };
            
            // Test tab order
            await this.testTabOrder();
            
            // Test keyboard shortcuts
            await this.testKeyboardShortcuts();
            
            // Test escape key functionality
            await this.testEscapeKey();
            
            // Calculate keyboard score
            const total = this.testResults.keyboard.passed + this.testResults.keyboard.failed;
            this.testResults.keyboard.score = total > 0 ? Math.round((this.testResults.keyboard.passed / total) * 100) : 0;
            
            return this.testResults.keyboard;
        }

        async testTabOrder() {
            const focusableElements = document.querySelectorAll(
                'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
            );
            
            let tabOrderIssues = 0;
            let previousTabIndex = -1;
            
            focusableElements.forEach(element => {
                const tabIndex = parseInt(element.getAttribute('tabindex') || '0');
                
                if (tabIndex > 0 && tabIndex < previousTabIndex) {
                    tabOrderIssues++;
                    this.issues.push({
                        type: 'tab-order',
                        severity: 'warning',
                        element: element,
                        message: 'Tab order may be confusing due to positive tabindex values',
                        wcagCriteria: '2.4.3'
                    });
                }
                
                if (tabIndex > 0) {
                    previousTabIndex = tabIndex;
                }
            });
            
            if (tabOrderIssues === 0) {
                this.testResults.keyboard.passed++;
            } else {
                this.testResults.keyboard.failed++;
            }
        }

        async testKeyboardShortcuts() {
            const shortcuts = [
                { keys: 'Escape', description: 'Close modals/dropdowns' },
                { keys: 'Tab', description: 'Navigate forward' },
                { keys: 'Shift+Tab', description: 'Navigate backward' }
            ];
            
            // This would need to be implemented with actual keyboard event simulation
            // For now, we'll just check if the handlers exist
            let shortcutIssues = 0;
            
            // Check for escape key handlers
            const modals = document.querySelectorAll('.modal, [role="dialog"]');
            modals.forEach(modal => {
                // Check if modal has escape key handler (simplified check)
                if (!modal.hasAttribute('data-escape-handler')) {
                    shortcutIssues++;
                    this.issues.push({
                        type: 'keyboard-shortcut',
                        severity: 'warning',
                        element: modal,
                        message: 'Modal may not respond to Escape key',
                        wcagCriteria: '2.1.1'
                    });
                }
            });
            
            if (shortcutIssues === 0) {
                this.testResults.keyboard.passed++;
            } else {
                this.testResults.keyboard.failed++;
            }
        }

        async testEscapeKey() {
            // Check that escape key works to close things
            const openDropdowns = document.querySelectorAll('[aria-expanded="true"]');
            const openModals = document.querySelectorAll('.modal:not(.hidden), .modal.show');
            
            let escapeIssues = 0;
            
            if (openDropdowns.length > 0 || openModals.length > 0) {
                // Simulate escape key
                const escapeEvent = new KeyboardEvent('keydown', { key: 'Escape' });
                document.dispatchEvent(escapeEvent);
                
                // Check if things closed (simplified test)
                setTimeout(() => {
                    const stillOpenDropdowns = document.querySelectorAll('[aria-expanded="true"]');
                    const stillOpenModals = document.querySelectorAll('.modal:not(.hidden), .modal.show');
                    
                    if (stillOpenDropdowns.length > 0 || stillOpenModals.length > 0) {
                        escapeIssues++;
                        this.issues.push({
                            type: 'escape-key',
                            severity: 'error',
                            message: 'Some elements do not respond to Escape key',
                            wcagCriteria: '2.1.1'
                        });
                    }
                    
                    if (escapeIssues === 0) {
                        this.testResults.keyboard.passed++;
                    } else {
                        this.testResults.keyboard.failed++;
                    }
                }, 100);
            } else {
                this.testResults.keyboard.passed++;
            }
        }

        // === Screen Reader Testing ===
        async testScreenReaderCompatibility() {
            console.log('Testing screen reader compatibility...');
            this.testResults.screenReader = { passed: 0, failed: 0, score: 0 };
            
            // Test semantic markup
            await this.testSemanticMarkup();
            
            // Test ARIA labels
            await this.testARIALabels();
            
            // Test reading order
            await this.testReadingOrder();
            
            // Calculate screen reader score
            const total = this.testResults.screenReader.passed + this.testResults.screenReader.failed;
            this.testResults.screenReader.score = total > 0 ? Math.round((this.testResults.screenReader.passed / total) * 100) : 0;
            
            return this.testResults.screenReader;
        }

        async testSemanticMarkup() {
            let semanticIssues = 0;
            
            // Check for proper use of semantic elements
            const hasMain = document.querySelector('main') !== null;
            const hasNav = document.querySelector('nav') !== null;
            const hasHeader = document.querySelector('header') !== null;
            const hasFooter = document.querySelector('footer') !== null;
            
            if (!hasMain) {
                semanticIssues++;
                this.issues.push({
                    type: 'semantic',
                    severity: 'error',
                    message: 'Page missing <main> landmark',
                    wcagCriteria: '1.3.1'
                });
            }
            
            if (!hasNav) {
                semanticIssues++;
                this.issues.push({
                    type: 'semantic',
                    severity: 'warning',
                    message: 'Page missing <nav> landmark',
                    wcagCriteria: '1.3.1'
                });
            }
            
            // Check for divs that should be buttons
            const clickableDivs = document.querySelectorAll('div[onclick], span[onclick]');
            clickableDivs.forEach(div => {
                if (!div.getAttribute('role')) {
                    semanticIssues++;
                    this.issues.push({
                        type: 'semantic',
                        severity: 'error',
                        element: div,
                        message: 'Use <button> instead of clickable div/span',
                        wcagCriteria: '4.1.2'
                    });
                }
            });
            
            if (semanticIssues === 0) {
                this.testResults.screenReader.passed++;
            } else {
                this.testResults.screenReader.failed++;
            }
        }

        async testARIALabels() {
            let ariaLabelIssues = 0;
            
            // Check for buttons with proper labels
            const buttons = document.querySelectorAll('button');
            buttons.forEach(button => {
                const hasText = button.textContent.trim().length > 0;
                const hasAriaLabel = button.getAttribute('aria-label');
                const hasAriaLabelledBy = button.getAttribute('aria-labelledby');
                
                if (!hasText && !hasAriaLabel && !hasAriaLabelledBy) {
                    ariaLabelIssues++;
                    this.issues.push({
                        type: 'aria-label',
                        severity: 'error',
                        element: button,
                        message: 'Button has no accessible name',
                        wcagCriteria: '4.1.2'
                    });
                }
            });
            
            // Check for links with proper labels
            const links = document.querySelectorAll('a[href]');
            links.forEach(link => {
                const hasText = link.textContent.trim().length > 0;
                const hasAriaLabel = link.getAttribute('aria-label');
                
                if (!hasText && !hasAriaLabel) {
                    ariaLabelIssues++;
                    this.issues.push({
                        type: 'aria-label',
                        severity: 'error',
                        element: link,
                        message: 'Link has no accessible name',
                        wcagCriteria: '4.1.2'
                    });
                }
            });
            
            if (ariaLabelIssues === 0) {
                this.testResults.screenReader.passed++;
            } else {
                this.testResults.screenReader.failed++;
            }
        }

        async testReadingOrder() {
            // Test that reading order makes sense
            const focusableElements = Array.from(document.querySelectorAll(
                'h1, h2, h3, h4, h5, h6, p, a, button, input, select, textarea'
            ));
            
            let readingOrderIssues = 0;
            
            // Check if elements are in logical reading order
            let previousY = -1;
            focusableElements.forEach(element => {
                const rect = element.getBoundingClientRect();
                const currentY = rect.top;
                
                // If element is significantly above previous element, it might be out of order
                if (currentY < previousY - 50) {
                    readingOrderIssues++;
                    this.issues.push({
                        type: 'reading-order',
                        severity: 'warning',
                        element: element,
                        message: 'Element may be out of logical reading order',
                        wcagCriteria: '1.3.2'
                    });
                }
                
                previousY = currentY;
            });
            
            if (readingOrderIssues === 0) {
                this.testResults.screenReader.passed++;
            } else {
                this.testResults.screenReader.failed++;
            }
        }

        // === Overall Score Calculation ===
        calculateOverallScore() {
            const wcagWeight = 0.5;
            const keyboardWeight = 0.3;
            const screenReaderWeight = 0.2;
            
            this.testResults.overall.score = Math.round(
                (this.testResults.wcag.score * wcagWeight) +
                (this.testResults.keyboard.score * keyboardWeight) +
                (this.testResults.screenReader.score * screenReaderWeight)
            );
            
            // Assign grade
            if (this.testResults.overall.score >= 95) {
                this.testResults.overall.grade = 'A+';
            } else if (this.testResults.overall.score >= 90) {
                this.testResults.overall.grade = 'A';
            } else if (this.testResults.overall.score >= 80) {
                this.testResults.overall.grade = 'B';
            } else if (this.testResults.overall.score >= 70) {
                this.testResults.overall.grade = 'C';
            } else if (this.testResults.overall.score >= 60) {
                this.testResults.overall.grade = 'D';
            } else {
                this.testResults.overall.grade = 'F';
            }
        }

        // === Run All Tests ===
        async runAllTests() {
            console.log('Running comprehensive accessibility audit...');
            
            await this.runWCAGTests();
            await this.testKeyboardNavigation();
            await this.testScreenReaderCompatibility();
            
            this.calculateOverallScore();
            
            return {
                results: this.testResults,
                issues: this.issues
            };
        }

        // === UI Components ===
        addTestingInterface() {
            const accessibilityControls = document.querySelector('.flex.items-center.space-x-2');
            if (!accessibilityControls) return;
            
            const testButton = document.createElement('button');
            testButton.id = 'accessibility-test';
            testButton.className = 'p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary';
            testButton.setAttribute('aria-label', 'Run accessibility tests');
            testButton.setAttribute('title', 'Run accessibility audit');
            testButton.innerHTML = `
                <span class="sr-only">Run accessibility audit</span>
                <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            `;
            
            testButton.addEventListener('click', async () => {
                this.showTestingModal();
            });
            
            accessibilityControls.appendChild(testButton);
        }

        async showTestingModal() {
            const modal = document.createElement('div');
            modal.className = 'modal show';
            modal.setAttribute('role', 'dialog');
            modal.setAttribute('aria-labelledby', 'test-title');
            
            modal.innerHTML = `
                <div class="modal-content" tabindex="-1">
                    <div class="modal-header">
                        <h2 id="test-title">Accessibility Audit</h2>
                        <button type="button" class="modal-close" aria-label="Close audit">
                            <span aria-hidden="true">&times;</span>
                        </button>
                    </div>
                    <div class="modal-body">
                        <div id="test-progress" class="mb-4">
                            <p>Running accessibility tests...</p>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: 0%"></div>
                            </div>
                        </div>
                        <div id="test-results" class="hidden">
                            <!-- Results will be populated here -->
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-outline modal-close">Close</button>
                        <button type="button" class="btn btn-primary" id="export-results" disabled>Export Report</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            // Set up close handlers
            const closeButtons = modal.querySelectorAll('.modal-close');
            closeButtons.forEach(button => {
                button.addEventListener('click', () => {
                    document.body.removeChild(modal);
                });
            });
            
            // Run tests and show results
            const progressBar = modal.querySelector('.progress-fill');
            const testProgress = modal.querySelector('#test-progress');
            const testResults = modal.querySelector('#test-results');
            
            try {
                // Simulate progress
                progressBar.style.width = '20%';
                await new Promise(resolve => setTimeout(resolve, 500));
                
                progressBar.style.width = '60%';
                const auditResults = await this.runAllTests();
                
                progressBar.style.width = '100%';
                await new Promise(resolve => setTimeout(resolve, 300));
                
                // Hide progress, show results
                testProgress.classList.add('hidden');
                testResults.classList.remove('hidden');
                
                this.populateTestResults(testResults, auditResults);
                
                // Enable export button
                const exportButton = modal.querySelector('#export-results');
                exportButton.disabled = false;
                exportButton.addEventListener('click', () => {
                    this.exportTestResults(auditResults);
                });
                
            } catch (error) {
                testProgress.innerHTML = `<p class="text-red-600">Error running tests: ${error.message}</p>`;
            }
            
            // Focus management
            modal.querySelector('.modal-content').focus();
        }

        populateTestResults(container, auditResults) {
            const { results, issues } = auditResults;
            
            container.innerHTML = `
                <div class="test-summary">
                    <h3>Audit Summary</h3>
                    <div class="score-card">
                        <div class="overall-score">
                            <span class="score-number">${results.overall.score}</span>
                            <span class="score-grade">${results.overall.grade}</span>
                        </div>
                        <p class="score-description">
                            ${results.overall.score >= 95 ? 'Excellent accessibility!' : 
                              results.overall.score >= 80 ? 'Good accessibility with room for improvement' :
                              'Accessibility needs significant improvement'}
                        </p>
                    </div>
                    
                    <div class="test-breakdown">
                        <div class="test-category">
                            <h4>WCAG 2.1 Compliance</h4>
                            <div class="score">${results.wcag.score}%</div>
                            <div class="details">${results.wcag.passed} passed, ${results.wcag.failed} failed</div>
                        </div>
                        <div class="test-category">
                            <h4>Keyboard Navigation</h4>
                            <div class="score">${results.keyboard.score}%</div>
                            <div class="details">${results.keyboard.passed} passed, ${results.keyboard.failed} failed</div>
                        </div>
                        <div class="test-category">
                            <h4>Screen Reader Support</h4>
                            <div class="score">${results.screenReader.score}%</div>
                            <div class="details">${results.screenReader.passed} passed, ${results.screenReader.failed} failed</div>
                        </div>
                    </div>
                </div>
                
                <div class="issues-list">
                    <h3>Issues Found (${issues.length})</h3>
                    ${issues.length === 0 ? '<p>No issues found! 🎉</p>' : ''}
                    ${issues.map(issue => `
                        <div class="issue-item ${issue.severity}">
                            <div class="issue-header">
                                <span class="issue-type">${issue.type}</span>
                                <span class="issue-severity ${issue.severity}">${issue.severity}</span>
                                <span class="wcag-criteria">WCAG ${issue.wcagCriteria || 'N/A'}</span>
                            </div>
                            <div class="issue-message">${issue.message}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        exportTestResults(auditResults) {
            const report = {
                timestamp: new Date().toISOString(),
                url: window.location.href,
                userAgent: navigator.userAgent,
                results: auditResults.results,
                issues: auditResults.issues,
                summary: {
                    totalIssues: auditResults.issues.length,
                    errorCount: auditResults.issues.filter(i => i.severity === 'error').length,
                    warningCount: auditResults.issues.filter(i => i.severity === 'warning').length,
                    overallScore: auditResults.results.overall.score,
                    grade: auditResults.results.overall.grade
                }
            };
            
            const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `accessibility-audit-${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        setupAutomaticTesting() {
            // Run automatic tests on page load
            if (localStorage.getItem('accessibility-auto-test') === 'true') {
                setTimeout(() => {
                    this.runAllTests().then(results => {
                        console.log('Automatic accessibility audit completed:', results);
                    });
                }, 2000);
            }
        }

        // === Helper Functions ===
        calculateContrast(color1, color2) {
            // Simplified contrast calculation
            // In a real implementation, this would properly parse RGB values
            // and calculate luminance according to WCAG guidelines
            
            const getRGB = (color) => {
                const match = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
                return match ? [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])] : [0, 0, 0];
            };
            
            const rgb1 = getRGB(color1);
            const rgb2 = getRGB(color2);
            
            const luminance = (rgb) => {
                const [r, g, b] = rgb.map(c => {
                    c = c / 255;
                    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
                });
                return 0.2126 * r + 0.7152 * g + 0.0722 * b;
            };
            
            const lum1 = luminance(rgb1);
            const lum2 = luminance(rgb2);
            
            const lighter = Math.max(lum1, lum2);
            const darker = Math.min(lum1, lum2);
            
            return (lighter + 0.05) / (darker + 0.05);
        }
    }

    // === CSS for Testing Interface ===
    function addTestingStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .progress-bar {
                width: 100%;
                height: 8px;
                background-color: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background-color: #3b82f6;
                transition: width 0.3s ease;
            }
            
            .test-summary {
                margin-bottom: 24px;
            }
            
            .score-card {
                display: flex;
                align-items: center;
                gap: 16px;
                padding: 16px;
                background-color: #f9fafb;
                border-radius: 8px;
                margin: 16px 0;
            }
            
            .overall-score {
                display: flex;
                flex-direction: column;
                align-items: center;
                min-width: 80px;
            }
            
            .score-number {
                font-size: 32px;
                font-weight: bold;
                color: #1f2937;
            }
            
            .score-grade {
                font-size: 18px;
                font-weight: bold;
                color: #6b7280;
            }
            
            .test-breakdown {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-top: 16px;
            }
            
            .test-category {
                padding: 16px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                text-align: center;
            }
            
            .test-category h4 {
                margin: 0 0 8px 0;
                font-size: 14px;
                color: #6b7280;
            }
            
            .test-category .score {
                font-size: 24px;
                font-weight: bold;
                color: #1f2937;
            }
            
            .test-category .details {
                font-size: 12px;
                color: #9ca3af;
                margin-top: 4px;
            }
            
            .issues-list {
                margin-top: 24px;
            }
            
            .issue-item {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
            }
            
            .issue-item.error {
                border-left: 4px solid #ef4444;
                background-color: #fef2f2;
            }
            
            .issue-item.warning {
                border-left: 4px solid #f59e0b;
                background-color: #fffbeb;
            }
            
            .issue-header {
                display: flex;
                gap: 12px;
                align-items: center;
                margin-bottom: 8px;
            }
            
            .issue-type {
                font-weight: bold;
                text-transform: capitalize;
                color: #1f2937;
            }
            
            .issue-severity {
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }
            
            .issue-severity.error {
                background-color: #ef4444;
                color: white;
            }
            
            .issue-severity.warning {
                background-color: #f59e0b;
                color: white;
            }
            
            .wcag-criteria {
                font-size: 12px;
                color: #6b7280;
                background-color: #f3f4f6;
                padding: 2px 6px;
                border-radius: 4px;
            }
            
            .issue-message {
                color: #374151;
                font-size: 14px;
            }
        `;
        
        document.head.appendChild(style);
    }

    // === Initialize Testing Framework ===
    function initializeAccessibilityTesting() {
        addTestingStyles();
        window.accessibilityTester = new AccessibilityTester();
        
        console.log('Accessibility testing framework initialized');
    }

    // === Start When DOM is Ready ===
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeAccessibilityTesting);
    } else {
        initializeAccessibilityTesting();
    }

})();