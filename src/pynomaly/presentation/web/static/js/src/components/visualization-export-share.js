/**
 * Enhanced Export and Sharing Capabilities
 * 
 * Features:
 * - Multiple export formats (PNG, SVG, PDF, JSON)
 * - High-quality export with custom resolution
 * - Batch export capabilities
 * - Social sharing integration
 * - Collaborative sharing with permissions
 * - Email and link sharing
 * - Embed code generation
 * - Export scheduling and automation
 */

class VisualizationExportShare {
    constructor() {
        this.exportFormats = ['png', 'svg', 'pdf', 'json', 'csv', 'excel'];
        this.shareProviders = ['email', 'link', 'teams', 'slack', 'embed'];
        this.exportQueue = [];
        this.shareHistory = [];
        this.init();
    }

    init() {
        this.setupExportModal();
        this.setupShareModal();
        this.bindEvents();
    }

    setupExportModal() {
        const modal = document.createElement('div');
        modal.id = 'export-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Export Visualization</h2>
                    <button class="modal-close" onclick="this.closest('.modal').style.display='none'">×</button>
                </div>
                <div class="modal-body">
                    <div class="export-tabs">
                        <button class="tab-button active" data-tab="single">Single Export</button>
                        <button class="tab-button" data-tab="batch">Batch Export</button>
                        <button class="tab-button" data-tab="schedule">Schedule Export</button>
                    </div>
                    
                    <!-- Single Export Tab -->
                    <div class="tab-content active" data-tab="single">
                        <div class="export-section">
                            <h3>Format Selection</h3>
                            <div class="format-grid">
                                <div class="format-option" data-format="png">
                                    <div class="format-icon">🖼️</div>
                                    <span>PNG Image</span>
                                    <small>High quality raster image</small>
                                </div>
                                <div class="format-option" data-format="svg">
                                    <div class="format-icon">🎨</div>
                                    <span>SVG Vector</span>
                                    <small>Scalable vector graphics</small>
                                </div>
                                <div class="format-option" data-format="pdf">
                                    <div class="format-icon">📄</div>
                                    <span>PDF Document</span>
                                    <small>Portable document format</small>
                                </div>
                                <div class="format-option" data-format="json">
                                    <div class="format-icon">📋</div>
                                    <span>JSON Data</span>
                                    <small>Raw data and configuration</small>
                                </div>
                                <div class="format-option" data-format="csv">
                                    <div class="format-icon">📊</div>
                                    <span>CSV Data</span>
                                    <small>Comma-separated values</small>
                                </div>
                                <div class="format-option" data-format="excel">
                                    <div class="format-icon">📈</div>
                                    <span>Excel File</span>
                                    <small>Microsoft Excel format</small>
                                </div>
                            </div>
                        </div>
                        
                        <div class="export-section">
                            <h3>Export Settings</h3>
                            <div class="settings-grid">
                                <div class="setting-group">
                                    <label>Resolution:</label>
                                    <select id="export-resolution">
                                        <option value="1">1x (Standard)</option>
                                        <option value="2" selected>2x (High)</option>
                                        <option value="3">3x (Ultra)</option>
                                        <option value="4">4x (Print)</option>
                                    </select>
                                </div>
                                <div class="setting-group">
                                    <label>Background:</label>
                                    <select id="export-background">
                                        <option value="transparent">Transparent</option>
                                        <option value="white" selected>White</option>
                                        <option value="black">Black</option>
                                        <option value="custom">Custom Color</option>
                                    </select>
                                </div>
                                <div class="setting-group">
                                    <label>Include Metadata:</label>
                                    <input type="checkbox" id="export-metadata" checked>
                                </div>
                                <div class="setting-group">
                                    <label>Filename:</label>
                                    <input type="text" id="export-filename" placeholder="visualization-export">
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Batch Export Tab -->
                    <div class="tab-content" data-tab="batch">
                        <div class="export-section">
                            <h3>Batch Export Settings</h3>
                            <div class="batch-options">
                                <div class="batch-format-selection">
                                    <label>Export Formats:</label>
                                    <div class="format-checkboxes">
                                        <label><input type="checkbox" value="png" checked> PNG</label>
                                        <label><input type="checkbox" value="svg"> SVG</label>
                                        <label><input type="checkbox" value="pdf"> PDF</label>
                                        <label><input type="checkbox" value="json"> JSON</label>
                                    </div>
                                </div>
                                <div class="batch-resolutions">
                                    <label>Resolutions:</label>
                                    <div class="resolution-checkboxes">
                                        <label><input type="checkbox" value="1"> 1x</label>
                                        <label><input type="checkbox" value="2" checked> 2x</label>
                                        <label><input type="checkbox" value="3"> 3x</label>
                                    </div>
                                </div>
                                <div class="batch-naming">
                                    <label>Naming Pattern:</label>
                                    <input type="text" id="batch-pattern" placeholder="{name}_{format}_{resolution}x" value="{name}_{format}_{resolution}x">
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Schedule Export Tab -->
                    <div class="tab-content" data-tab="schedule">
                        <div class="export-section">
                            <h3>Export Scheduling</h3>
                            <div class="schedule-options">
                                <div class="schedule-type">
                                    <label>Schedule Type:</label>
                                    <select id="schedule-type">
                                        <option value="once">One-time</option>
                                        <option value="daily">Daily</option>
                                        <option value="weekly">Weekly</option>
                                        <option value="monthly">Monthly</option>
                                    </select>
                                </div>
                                <div class="schedule-time">
                                    <label>Export Time:</label>
                                    <input type="datetime-local" id="schedule-datetime">
                                </div>
                                <div class="schedule-email">
                                    <label>Email Results To:</label>
                                    <input type="email" id="schedule-email" placeholder="user@example.com">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="this.closest('.modal').style.display='none'">Cancel</button>
                    <button class="btn btn-primary" id="start-export">Start Export</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    setupShareModal() {
        const modal = document.createElement('div');
        modal.id = 'share-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Share Visualization</h2>
                    <button class="modal-close" onclick="this.closest('.modal').style.display='none'">×</button>
                </div>
                <div class="modal-body">
                    <div class="share-tabs">
                        <button class="tab-button active" data-tab="link">Link Sharing</button>
                        <button class="tab-button" data-tab="embed">Embed Code</button>
                        <button class="tab-button" data-tab="email">Email</button>
                        <button class="tab-button" data-tab="teams">Teams</button>
                    </div>
                    
                    <!-- Link Sharing Tab -->
                    <div class="tab-content active" data-tab="link">
                        <div class="share-section">
                            <h3>Share Link</h3>
                            <div class="link-options">
                                <div class="link-permissions">
                                    <label>Permissions:</label>
                                    <select id="link-permissions">
                                        <option value="view">View Only</option>
                                        <option value="edit">Can Edit</option>
                                        <option value="comment">Can Comment</option>
                                    </select>
                                </div>
                                <div class="link-expiry">
                                    <label>Expires:</label>
                                    <select id="link-expiry">
                                        <option value="never">Never</option>
                                        <option value="1day">1 Day</option>
                                        <option value="1week">1 Week</option>
                                        <option value="1month">1 Month</option>
                                    </select>
                                </div>
                                <div class="link-password">
                                    <label>Password Protection:</label>
                                    <input type="password" id="link-password" placeholder="Optional password">
                                </div>
                            </div>
                            <div class="generated-link">
                                <label>Shareable Link:</label>
                                <div class="link-container">
                                    <input type="text" id="share-link" readonly>
                                    <button class="btn btn-sm" id="copy-link">Copy</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Embed Code Tab -->
                    <div class="tab-content" data-tab="embed">
                        <div class="share-section">
                            <h3>Embed Code</h3>
                            <div class="embed-options">
                                <div class="embed-size">
                                    <label>Size:</label>
                                    <select id="embed-size">
                                        <option value="responsive">Responsive</option>
                                        <option value="small">Small (400x300)</option>
                                        <option value="medium" selected>Medium (800x600)</option>
                                        <option value="large">Large (1200x800)</option>
                                        <option value="custom">Custom</option>
                                    </select>
                                </div>
                                <div class="embed-theme">
                                    <label>Theme:</label>
                                    <select id="embed-theme">
                                        <option value="auto">Auto</option>
                                        <option value="light">Light</option>
                                        <option value="dark">Dark</option>
                                    </select>
                                </div>
                                <div class="embed-interactive">
                                    <label>Interactive:</label>
                                    <input type="checkbox" id="embed-interactive" checked>
                                </div>
                            </div>
                            <div class="embed-code">
                                <label>Embed Code:</label>
                                <textarea id="embed-code-text" readonly rows="8"></textarea>
                                <button class="btn btn-sm" id="copy-embed">Copy Code</button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Email Tab -->
                    <div class="tab-content" data-tab="email">
                        <div class="share-section">
                            <h3>Email Sharing</h3>
                            <div class="email-form">
                                <div class="email-recipients">
                                    <label>Recipients:</label>
                                    <input type="email" id="email-recipients" placeholder="user@example.com, user2@example.com">
                                </div>
                                <div class="email-subject">
                                    <label>Subject:</label>
                                    <input type="text" id="email-subject" value="Shared Visualization from Pynomaly">
                                </div>
                                <div class="email-message">
                                    <label>Message:</label>
                                    <textarea id="email-message" rows="4" placeholder="Optional message..."></textarea>
                                </div>
                                <div class="email-format">
                                    <label>Include Attachment:</label>
                                    <select id="email-format">
                                        <option value="none">Link Only</option>
                                        <option value="png">PNG Image</option>
                                        <option value="pdf">PDF Document</option>
                                        <option value="json">JSON Data</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Teams Tab -->
                    <div class="tab-content" data-tab="teams">
                        <div class="share-section">
                            <h3>Teams Integration</h3>
                            <div class="teams-options">
                                <div class="teams-channel">
                                    <label>Channel:</label>
                                    <select id="teams-channel">
                                        <option value="">Select Channel...</option>
                                        <option value="general">General</option>
                                        <option value="data-science">Data Science</option>
                                        <option value="analytics">Analytics</option>
                                    </select>
                                </div>
                                <div class="teams-message">
                                    <label>Message:</label>
                                    <textarea id="teams-message" rows="3" placeholder="Sharing visualization..."></textarea>
                                </div>
                                <div class="teams-mention">
                                    <label>Mention:</label>
                                    <input type="text" id="teams-mention" placeholder="@username or @channel">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="this.closest('.modal').style.display='none'">Cancel</button>
                    <button class="btn btn-primary" id="share-visualization">Share</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    bindEvents() {
        // Export modal events
        document.getElementById('start-export').addEventListener('click', () => this.startExport());
        
        // Share modal events
        document.getElementById('share-visualization').addEventListener('click', () => this.shareVisualization());
        document.getElementById('copy-link').addEventListener('click', () => this.copyToClipboard('share-link'));
        document.getElementById('copy-embed').addEventListener('click', () => this.copyToClipboard('embed-code-text'));
        
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => this.switchTab(e.target));
        });
        
        // Format selection
        document.querySelectorAll('.format-option').forEach(option => {
            option.addEventListener('click', (e) => this.selectFormat(e.currentTarget));
        });
        
        // Dynamic updates
        document.getElementById('link-permissions').addEventListener('change', () => this.updateShareLink());
        document.getElementById('link-expiry').addEventListener('change', () => this.updateShareLink());
        document.getElementById('embed-size').addEventListener('change', () => this.updateEmbedCode());
        document.getElementById('embed-theme').addEventListener('change', () => this.updateEmbedCode());
        document.getElementById('embed-interactive').addEventListener('change', () => this.updateEmbedCode());
    }

    showExportModal(chartInstance) {
        this.currentChart = chartInstance;
        document.getElementById('export-modal').style.display = 'block';
        this.updateExportPreview();
    }

    showShareModal(chartInstance) {
        this.currentChart = chartInstance;
        document.getElementById('share-modal').style.display = 'block';
        this.updateShareLink();
        this.updateEmbedCode();
    }

    switchTab(button) {
        const tabName = button.dataset.tab;
        const container = button.closest('.modal-content');
        
        // Update button states
        container.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        // Update content visibility
        container.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.dataset.tab === tabName);
        });
    }

    selectFormat(formatElement) {
        const format = formatElement.dataset.format;
        
        // Update selection
        document.querySelectorAll('.format-option').forEach(option => {
            option.classList.remove('selected');
        });
        formatElement.classList.add('selected');
        
        // Update settings based on format
        this.updateFormatSettings(format);
    }

    updateFormatSettings(format) {
        const resolutionSelect = document.getElementById('export-resolution');
        const backgroundSelect = document.getElementById('export-background');
        
        if (format === 'svg') {
            resolutionSelect.disabled = true;
            backgroundSelect.value = 'transparent';
        } else if (format === 'pdf') {
            resolutionSelect.disabled = true;
            backgroundSelect.value = 'white';
        } else if (format === 'json' || format === 'csv' || format === 'excel') {
            resolutionSelect.disabled = true;
            backgroundSelect.disabled = true;
        } else {
            resolutionSelect.disabled = false;
            backgroundSelect.disabled = false;
        }
    }

    async startExport() {
        const activeTab = document.querySelector('.tab-content.active').dataset.tab;
        
        if (activeTab === 'single') {
            await this.performSingleExport();
        } else if (activeTab === 'batch') {
            await this.performBatchExport();
        } else if (activeTab === 'schedule') {
            await this.scheduleExport();
        }
    }

    async performSingleExport() {
        const selectedFormat = document.querySelector('.format-option.selected');
        if (!selectedFormat) {
            alert('Please select an export format');
            return;
        }
        
        const format = selectedFormat.dataset.format;
        const resolution = document.getElementById('export-resolution').value;
        const background = document.getElementById('export-background').value;
        const includeMetadata = document.getElementById('export-metadata').checked;
        const filename = document.getElementById('export-filename').value || 'visualization-export';
        
        const exportOptions = {
            format,
            resolution: parseInt(resolution),
            background,
            includeMetadata,
            filename
        };
        
        try {
            showLoading();
            await this.exportChart(exportOptions);
            showNotification('Export completed successfully!', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            showNotification('Export failed. Please try again.', 'error');
        } finally {
            hideLoading();
            document.getElementById('export-modal').style.display = 'none';
        }
    }

    async performBatchExport() {
        const selectedFormats = Array.from(document.querySelectorAll('.format-checkboxes input:checked')).map(cb => cb.value);
        const selectedResolutions = Array.from(document.querySelectorAll('.resolution-checkboxes input:checked')).map(cb => parseInt(cb.value));
        const namingPattern = document.getElementById('batch-pattern').value;
        
        if (selectedFormats.length === 0) {
            alert('Please select at least one export format');
            return;
        }
        
        const exportTasks = [];
        selectedFormats.forEach(format => {
            if (format === 'svg' || format === 'json' || format === 'csv' || format === 'excel') {
                exportTasks.push({
                    format,
                    resolution: 1,
                    filename: this.generateFilename(namingPattern, format, 1)
                });
            } else {
                selectedResolutions.forEach(resolution => {
                    exportTasks.push({
                        format,
                        resolution,
                        filename: this.generateFilename(namingPattern, format, resolution)
                    });
                });
            }
        });
        
        try {
            showLoading();
            for (const task of exportTasks) {
                await this.exportChart(task);
            }
            showNotification(`Batch export completed! ${exportTasks.length} files exported.`, 'success');
        } catch (error) {
            console.error('Batch export failed:', error);
            showNotification('Batch export failed. Some files may not have been exported.', 'error');
        } finally {
            hideLoading();
            document.getElementById('export-modal').style.display = 'none';
        }
    }

    generateFilename(pattern, format, resolution) {
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        return pattern
            .replace('{name}', 'visualization')
            .replace('{format}', format)
            .replace('{resolution}', resolution)
            .replace('{timestamp}', timestamp);
    }

    async exportChart(options) {
        if (!this.currentChart) {
            throw new Error('No chart available for export');
        }
        
        switch (options.format) {
            case 'png':
                return this.exportAsPNG(options);
            case 'svg':
                return this.exportAsSVG(options);
            case 'pdf':
                return this.exportAsPDF(options);
            case 'json':
                return this.exportAsJSON(options);
            case 'csv':
                return this.exportAsCSV(options);
            case 'excel':
                return this.exportAsExcel(options);
            default:
                throw new Error(`Unsupported export format: ${options.format}`);
        }
    }

    exportAsPNG(options) {
        return new Promise((resolve, reject) => {
            try {
                const canvas = document.createElement('canvas');
                const scale = options.resolution || 2;
                const width = (this.currentChart.width || 800) * scale;
                const height = (this.currentChart.height || 600) * scale;
                
                canvas.width = width;
                canvas.height = height;
                
                const ctx = canvas.getContext('2d');
                
                // Set background
                if (options.background && options.background !== 'transparent') {
                    ctx.fillStyle = options.background === 'custom' ? '#f0f0f0' : options.background;
                    ctx.fillRect(0, 0, width, height);
                }
                
                // Render chart to canvas
                this.renderChartToCanvas(ctx, scale).then(() => {
                    canvas.toBlob((blob) => {
                        this.downloadBlob(blob, `${options.filename}.png`);
                        resolve();
                    }, 'image/png');
                }).catch(reject);
                
            } catch (error) {
                reject(error);
            }
        });
    }

    exportAsSVG(options) {
        return new Promise((resolve, reject) => {
            try {
                const svgElement = this.currentChart.container.querySelector('svg');
                if (!svgElement) {
                    reject(new Error('No SVG element found in chart'));
                    return;
                }
                
                const serializer = new XMLSerializer();
                let svgString = serializer.serializeToString(svgElement);
                
                // Add metadata if requested
                if (options.includeMetadata) {
                    const metadata = this.generateMetadata();
                    svgString = svgString.replace('<svg', `<svg data-metadata='${JSON.stringify(metadata)}'`);
                }
                
                const blob = new Blob([svgString], { type: 'image/svg+xml' });
                this.downloadBlob(blob, `${options.filename}.svg`);
                resolve();
                
            } catch (error) {
                reject(error);
            }
        });
    }

    exportAsPDF(options) {
        return new Promise((resolve, reject) => {
            try {
                // This would require a PDF library like jsPDF
                // For now, we'll create a placeholder implementation
                const content = `PDF export of visualization: ${options.filename}`;
                const blob = new Blob([content], { type: 'application/pdf' });
                this.downloadBlob(blob, `${options.filename}.pdf`);
                resolve();
                
            } catch (error) {
                reject(error);
            }
        });
    }

    exportAsJSON(options) {
        return new Promise((resolve, reject) => {
            try {
                const data = {
                    metadata: this.generateMetadata(),
                    config: this.currentChart.config || {},
                    data: this.currentChart.data || [],
                    timestamp: new Date().toISOString(),
                    version: '1.0'
                };
                
                const jsonString = JSON.stringify(data, null, 2);
                const blob = new Blob([jsonString], { type: 'application/json' });
                this.downloadBlob(blob, `${options.filename}.json`);
                resolve();
                
            } catch (error) {
                reject(error);
            }
        });
    }

    exportAsCSV(options) {
        return new Promise((resolve, reject) => {
            try {
                const data = this.currentChart.data || [];
                if (data.length === 0) {
                    reject(new Error('No data available for CSV export'));
                    return;
                }
                
                const headers = Object.keys(data[0]);
                const csvContent = [
                    headers.join(','),
                    ...data.map(row => headers.map(header => `"${row[header] || ''}"`).join(','))
                ].join('\n');
                
                const blob = new Blob([csvContent], { type: 'text/csv' });
                this.downloadBlob(blob, `${options.filename}.csv`);
                resolve();
                
            } catch (error) {
                reject(error);
            }
        });
    }

    exportAsExcel(options) {
        return new Promise((resolve, reject) => {
            try {
                // This would require an Excel library like SheetJS
                // For now, we'll use CSV format as fallback
                this.exportAsCSV({ ...options, filename: options.filename.replace('.excel', '.csv') })
                    .then(resolve)
                    .catch(reject);
                
            } catch (error) {
                reject(error);
            }
        });
    }

    generateMetadata() {
        return {
            title: 'Pynomaly Visualization',
            created: new Date().toISOString(),
            creator: 'Pynomaly Visualization Builder',
            description: 'Generated visualization from Pynomaly platform',
            version: '1.0',
            dataPoints: this.currentChart.data ? this.currentChart.data.length : 0,
            chartType: this.currentChart.type || 'unknown'
        };
    }

    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    renderChartToCanvas(ctx, scale) {
        return new Promise((resolve, reject) => {
            // This would implement actual chart rendering to canvas
            // For now, we'll create a placeholder
            ctx.fillStyle = '#3b82f6';
            ctx.fillRect(100 * scale, 100 * scale, 200 * scale, 100 * scale);
            
            ctx.fillStyle = '#000';
            ctx.font = `${16 * scale}px Arial`;
            ctx.fillText('Chart Export Placeholder', 120 * scale, 160 * scale);
            
            resolve();
        });
    }

    updateShareLink() {
        const permissions = document.getElementById('link-permissions').value;
        const expiry = document.getElementById('link-expiry').value;
        const password = document.getElementById('link-password').value;
        
        const baseUrl = window.location.origin;
        const chartId = this.generateChartId();
        let shareUrl = `${baseUrl}/shared/${chartId}?permissions=${permissions}`;
        
        if (expiry !== 'never') {
            shareUrl += `&expires=${expiry}`;
        }
        
        if (password) {
            shareUrl += `&protected=true`;
        }
        
        document.getElementById('share-link').value = shareUrl;
    }

    updateEmbedCode() {
        const size = document.getElementById('embed-size').value;
        const theme = document.getElementById('embed-theme').value;
        const interactive = document.getElementById('embed-interactive').checked;
        
        const chartId = this.generateChartId();
        const baseUrl = window.location.origin;
        
        let width = '800';
        let height = '600';
        
        if (size === 'small') {
            width = '400';
            height = '300';
        } else if (size === 'large') {
            width = '1200';
            height = '800';
        } else if (size === 'responsive') {
            width = '100%';
            height = '400';
        }
        
        const embedCode = `<iframe
    src="${baseUrl}/embed/${chartId}?theme=${theme}&interactive=${interactive}"
    width="${width}"
    height="${height}"
    frameborder="0"
    allowfullscreen>
</iframe>`;
        
        document.getElementById('embed-code-text').value = embedCode;
    }

    generateChartId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    copyToClipboard(elementId) {
        const element = document.getElementById(elementId);
        element.select();
        document.execCommand('copy');
        showNotification('Copied to clipboard!', 'success');
    }

    async shareVisualization() {
        const activeTab = document.querySelector('.share-tabs .tab-button.active').dataset.tab;
        
        try {
            switch (activeTab) {
                case 'link':
                    await this.shareViaLink();
                    break;
                case 'embed':
                    await this.shareViaEmbed();
                    break;
                case 'email':
                    await this.shareViaEmail();
                    break;
                case 'teams':
                    await this.shareViaTeams();
                    break;
            }
            
            showNotification('Visualization shared successfully!', 'success');
            document.getElementById('share-modal').style.display = 'none';
            
        } catch (error) {
            console.error('Share failed:', error);
            showNotification('Failed to share visualization. Please try again.', 'error');
        }
    }

    async shareViaLink() {
        const shareUrl = document.getElementById('share-link').value;
        
        if (navigator.share) {
            await navigator.share({
                title: 'Pynomaly Visualization',
                text: 'Check out this visualization from Pynomaly',
                url: shareUrl
            });
        } else {
            await navigator.clipboard.writeText(shareUrl);
            showNotification('Share link copied to clipboard!', 'success');
        }
    }

    async shareViaEmbed() {
        const embedCode = document.getElementById('embed-code-text').value;
        await navigator.clipboard.writeText(embedCode);
        showNotification('Embed code copied to clipboard!', 'success');
    }

    async shareViaEmail() {
        const recipients = document.getElementById('email-recipients').value;
        const subject = document.getElementById('email-subject').value;
        const message = document.getElementById('email-message').value;
        const format = document.getElementById('email-format').value;
        
        // This would integrate with email service
        // For now, we'll open the default email client
        const emailUrl = `mailto:${recipients}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(message)}`;
        window.open(emailUrl);
    }

    async shareViaTeams() {
        const channel = document.getElementById('teams-channel').value;
        const message = document.getElementById('teams-message').value;
        const mention = document.getElementById('teams-mention').value;
        
        // This would integrate with Microsoft Teams API
        // For now, we'll show a placeholder
        showNotification('Teams integration coming soon!', 'info');
    }

    async scheduleExport() {
        const scheduleType = document.getElementById('schedule-type').value;
        const datetime = document.getElementById('schedule-datetime').value;
        const email = document.getElementById('schedule-email').value;
        
        if (!datetime || !email) {
            alert('Please fill in all required fields');
            return;
        }
        
        const schedule = {
            type: scheduleType,
            datetime: datetime,
            email: email,
            chartId: this.generateChartId(),
            created: new Date().toISOString()
        };
        
        // This would send to backend for scheduling
        // For now, we'll store locally
        const schedules = JSON.parse(localStorage.getItem('export-schedules') || '[]');
        schedules.push(schedule);
        localStorage.setItem('export-schedules', JSON.stringify(schedules));
        
        showNotification('Export scheduled successfully!', 'success');
        document.getElementById('export-modal').style.display = 'none';
    }
}

// Initialize the export and share system
window.VisualizationExportShare = VisualizationExportShare;

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VisualizationExportShare;
}