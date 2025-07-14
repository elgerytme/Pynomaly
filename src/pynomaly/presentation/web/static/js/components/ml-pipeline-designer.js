/**
 * ML Pipeline Designer
 * Visual workflow creation for machine learning pipelines
 */

// Main ML Pipeline Designer State Management
function mlPipelineDesigner() {
    return {
        // Core State
        loading: false,
        sidebarOpen: window.innerWidth >= 1024,
        darkMode: localStorage.getItem('darkMode') === 'true',
        
        // Pipeline State
        pipelineComponents: [],
        connections: [],
        selectedComponents: [],
        pipelineModified: false,
        currentPipeline: null,
        
        // Canvas State
        canvasZoom: 1,
        canvasPan: { x: 0, y: 0 },
        isDragging: false,
        dragOffset: { x: 0, y: 0 },
        
        // Component Library
        componentSearch: '',
        componentCategories: [
            {
                name: 'Data Sources',
                expanded: true,
                components: [
                    {
                        id: 'csv-source',
                        name: 'CSV Source',
                        icon: '📄',
                        description: 'Load data from CSV files',
                        type: 'source',
                        inputs: [],
                        outputs: ['data'],
                        parameters: [
                            { name: 'file_path', label: 'File Path', type: 'text', value: '' },
                            { name: 'delimiter', label: 'Delimiter', type: 'select', value: ',', options: [
                                { value: ',', label: 'Comma (,)' },
                                { value: ';', label: 'Semicolon (;)' },
                                { value: '\t', label: 'Tab' }
                            ]},
                            { name: 'header', label: 'Has Header', type: 'boolean', value: true }
                        ]
                    },
                    {
                        id: 'json-source',
                        name: 'JSON Source',
                        icon: '📋',
                        description: 'Load data from JSON files',
                        type: 'source',
                        inputs: [],
                        outputs: ['data'],
                        parameters: [
                            { name: 'file_path', label: 'File Path', type: 'text', value: '' },
                            { name: 'normalize', label: 'Normalize Structure', type: 'boolean', value: false }
                        ]
                    },
                    {
                        id: 'database-source',
                        name: 'Database Source',
                        icon: '🗄️',
                        description: 'Load data from database',
                        type: 'source',
                        inputs: [],
                        outputs: ['data'],
                        parameters: [
                            { name: 'connection_string', label: 'Connection String', type: 'text', value: '' },
                            { name: 'query', label: 'SQL Query', type: 'text', value: 'SELECT * FROM table' }
                        ]
                    }
                ]
            },
            {
                name: 'Data Preprocessing',
                expanded: true,
                components: [
                    {
                        id: 'data-cleaner',
                        name: 'Data Cleaner',
                        icon: '🧹',
                        description: 'Clean and prepare data',
                        type: 'transformer',
                        inputs: ['data'],
                        outputs: ['cleaned_data'],
                        parameters: [
                            { name: 'remove_nulls', label: 'Remove Null Values', type: 'boolean', value: true },
                            { name: 'remove_duplicates', label: 'Remove Duplicates', type: 'boolean', value: true },
                            { name: 'fill_strategy', label: 'Fill Strategy', type: 'select', value: 'mean', options: [
                                { value: 'mean', label: 'Mean' },
                                { value: 'median', label: 'Median' },
                                { value: 'mode', label: 'Mode' },
                                { value: 'forward_fill', label: 'Forward Fill' }
                            ]}
                        ]
                    },
                    {
                        id: 'feature-scaler',
                        name: 'Feature Scaler',
                        icon: '📏',
                        description: 'Scale numerical features',
                        type: 'transformer',
                        inputs: ['data'],
                        outputs: ['scaled_data'],
                        parameters: [
                            { name: 'method', label: 'Scaling Method', type: 'select', value: 'standard', options: [
                                { value: 'standard', label: 'Standard Scaler' },
                                { value: 'minmax', label: 'Min-Max Scaler' },
                                { value: 'robust', label: 'Robust Scaler' },
                                { value: 'normalizer', label: 'Normalizer' }
                            ]},
                            { name: 'features', label: 'Features to Scale', type: 'text', value: '' }
                        ]
                    },
                    {
                        id: 'feature-encoder',
                        name: 'Feature Encoder',
                        icon: '🔤',
                        description: 'Encode categorical features',
                        type: 'transformer',
                        inputs: ['data'],
                        outputs: ['encoded_data'],
                        parameters: [
                            { name: 'method', label: 'Encoding Method', type: 'select', value: 'onehot', options: [
                                { value: 'onehot', label: 'One-Hot Encoding' },
                                { value: 'label', label: 'Label Encoding' },
                                { value: 'target', label: 'Target Encoding' },
                                { value: 'binary', label: 'Binary Encoding' }
                            ]},
                            { name: 'categorical_features', label: 'Categorical Features', type: 'text', value: '' }
                        ]
                    }
                ]
            },
            {
                name: 'Anomaly Detection',
                expanded: true,
                components: [
                    {
                        id: 'isolation-forest',
                        name: 'Isolation Forest',
                        icon: '🌲',
                        description: 'Isolation Forest anomaly detector',
                        type: 'detector',
                        inputs: ['data'],
                        outputs: ['predictions', 'scores'],
                        parameters: [
                            { name: 'n_estimators', label: 'Number of Estimators', type: 'number', value: 100, min: 10, max: 1000 },
                            { name: 'contamination', label: 'Contamination Rate', type: 'number', value: 0.1, min: 0.01, max: 0.5, step: 0.01 },
                            { name: 'random_state', label: 'Random State', type: 'number', value: 42 }
                        ]
                    },
                    {
                        id: 'one-class-svm',
                        name: 'One-Class SVM',
                        icon: '🎯',
                        description: 'One-Class SVM anomaly detector',
                        type: 'detector',
                        inputs: ['data'],
                        outputs: ['predictions', 'scores'],
                        parameters: [
                            { name: 'kernel', label: 'Kernel', type: 'select', value: 'rbf', options: [
                                { value: 'rbf', label: 'RBF' },
                                { value: 'linear', label: 'Linear' },
                                { value: 'poly', label: 'Polynomial' },
                                { value: 'sigmoid', label: 'Sigmoid' }
                            ]},
                            { name: 'nu', label: 'Nu Parameter', type: 'number', value: 0.05, min: 0.01, max: 1.0, step: 0.01 },
                            { name: 'gamma', label: 'Gamma', type: 'select', value: 'scale', options: [
                                { value: 'scale', label: 'Scale' },
                                { value: 'auto', label: 'Auto' }
                            ]}
                        ]
                    },
                    {
                        id: 'local-outlier-factor',
                        name: 'Local Outlier Factor',
                        icon: '📍',
                        description: 'LOF anomaly detector',
                        type: 'detector',
                        inputs: ['data'],
                        outputs: ['predictions', 'scores'],
                        parameters: [
                            { name: 'n_neighbors', label: 'Number of Neighbors', type: 'number', value: 20, min: 1, max: 100 },
                            { name: 'contamination', label: 'Contamination Rate', type: 'number', value: 0.1, min: 0.01, max: 0.5, step: 0.01 },
                            { name: 'metric', label: 'Distance Metric', type: 'select', value: 'minkowski', options: [
                                { value: 'minkowski', label: 'Minkowski' },
                                { value: 'euclidean', label: 'Euclidean' },
                                { value: 'manhattan', label: 'Manhattan' },
                                { value: 'cosine', label: 'Cosine' }
                            ]}
                        ]
                    }
                ]
            },
            {
                name: 'Evaluation',
                expanded: false,
                components: [
                    {
                        id: 'metrics-calculator',
                        name: 'Metrics Calculator',
                        icon: '📊',
                        description: 'Calculate performance metrics',
                        type: 'evaluator',
                        inputs: ['predictions', 'ground_truth'],
                        outputs: ['metrics'],
                        parameters: [
                            { name: 'metrics', label: 'Metrics to Calculate', type: 'text', value: 'precision,recall,f1,auc' }
                        ]
                    },
                    {
                        id: 'confusion-matrix',
                        name: 'Confusion Matrix',
                        icon: '🔢',
                        description: 'Generate confusion matrix',
                        type: 'evaluator',
                        inputs: ['predictions', 'ground_truth'],
                        outputs: ['matrix'],
                        parameters: []
                    }
                ]
            },
            {
                name: 'Visualization',
                expanded: false,
                components: [
                    {
                        id: 'scatter-plot',
                        name: 'Scatter Plot',
                        icon: '📈',
                        description: 'Create scatter plot visualization',
                        type: 'visualizer',
                        inputs: ['data'],
                        outputs: ['plot'],
                        parameters: [
                            { name: 'x_column', label: 'X Column', type: 'text', value: '' },
                            { name: 'y_column', label: 'Y Column', type: 'text', value: '' },
                            { name: 'color_column', label: 'Color Column', type: 'text', value: '' }
                        ]
                    },
                    {
                        id: 'histogram',
                        name: 'Histogram',
                        icon: '📊',
                        description: 'Create histogram visualization',
                        type: 'visualizer',
                        inputs: ['data'],
                        outputs: ['plot'],
                        parameters: [
                            { name: 'column', label: 'Column', type: 'text', value: '' },
                            { name: 'bins', label: 'Number of Bins', type: 'number', value: 30, min: 5, max: 100 }
                        ]
                    }
                ]
            }
        ],
        
        // Connection State
        tempConnection: null,
        tempConnectionPath: '',
        dragConnection: null,
        
        // Computed Properties
        get filteredComponentCategories() {
            if (!this.componentSearch) return this.componentCategories;
            
            return this.componentCategories.map(category => ({
                ...category,
                components: category.components.filter(component =>
                    component.name.toLowerCase().includes(this.componentSearch.toLowerCase()) ||
                    component.description.toLowerCase().includes(this.componentSearch.toLowerCase())
                )
            })).filter(category => category.components.length > 0);
        },
        
        get selectedComponent() {
            if (this.selectedComponents.length === 1) {
                return this.pipelineComponents.find(c => c.id === this.selectedComponents[0]);
            }
            return null;
        },

        // Initialization
        init() {
            this.initializeTheme();
            this.setupEventListeners();
            this.loadSamplePipeline();
        },

        // Theme Management
        initializeTheme() {
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            }
        },

        toggleTheme() {
            this.darkMode = !this.darkMode;
            localStorage.setItem('darkMode', this.darkMode);
            
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }
        },

        // Pipeline Management
        newPipeline() {
            if (this.pipelineModified) {
                if (!confirm('You have unsaved changes. Create new pipeline?')) {
                    return;
                }
            }
            
            this.pipelineComponents = [];
            this.connections = [];
            this.selectedComponents = [];
            this.currentPipeline = null;
            this.pipelineModified = false;
            
            this.showNotification('New pipeline created', 'success');
        },

        async savePipeline() {
            this.loading = true;
            
            try {
                const pipelineData = {
                    name: this.currentPipeline?.name || 'Untitled Pipeline',
                    components: this.pipelineComponents,
                    connections: this.connections,
                    metadata: {
                        created: new Date().toISOString(),
                        version: '1.0'
                    }
                };
                
                // Simulate API call
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                this.pipelineModified = false;
                this.showNotification('Pipeline saved successfully', 'success');
                
            } catch (error) {
                console.error('Error saving pipeline:', error);
                this.showNotification('Error saving pipeline', 'error');
            } finally {
                this.loading = false;
            }
        },

        async runPipeline() {
            if (!this.isValidPipeline()) {
                this.showNotification('Pipeline is not valid. Check connections and parameters.', 'error');
                return;
            }
            
            this.loading = true;
            
            try {
                // Update component statuses
                this.pipelineComponents.forEach(component => {
                    component.status = 'running';
                });
                
                // Simulate pipeline execution
                for (let i = 0; i < this.pipelineComponents.length; i++) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    this.pipelineComponents[i].status = 'completed';
                }
                
                this.showNotification('Pipeline executed successfully', 'success');
                
            } catch (error) {
                console.error('Error running pipeline:', error);
                this.pipelineComponents.forEach(component => {
                    if (component.status === 'running') {
                        component.status = 'error';
                    }
                });
                this.showNotification('Error running pipeline', 'error');
            } finally {
                this.loading = false;
            }
        },

        isValidPipeline() {
            // Check if pipeline has at least one component
            if (this.pipelineComponents.length === 0) return false;
            
            // Check if all components have valid connections
            for (const component of this.pipelineComponents) {
                if (component.inputs.length > 0) {
                    const hasInputConnections = this.connections.some(conn => 
                        conn.target.componentId === component.id
                    );
                    if (!hasInputConnections && component.type !== 'source') {
                        return false;
                    }
                }
            }
            
            return true;
        },

        // Component Management
        startDrag(event, componentTemplate) {
            event.dataTransfer.setData('application/json', JSON.stringify(componentTemplate));
            event.dataTransfer.effectAllowed = 'copy';
        },

        handleDrop(event) {
            event.preventDefault();
            
            try {
                const componentTemplate = JSON.parse(event.dataTransfer.getData('application/json'));
                const rect = event.currentTarget.getBoundingClientRect();
                
                const component = {
                    ...componentTemplate,
                    id: `${componentTemplate.id}-${Date.now()}`,
                    x: (event.clientX - rect.left - this.canvasPan.x) / this.canvasZoom,
                    y: (event.clientY - rect.top - this.canvasPan.y) / this.canvasZoom,
                    status: 'pending',
                    validation: [],
                    parameters: componentTemplate.parameters ? [...componentTemplate.parameters] : []
                };
                
                this.pipelineComponents.push(component);
                this.pipelineModified = true;
                this.validatePipeline();
                
            } catch (error) {
                console.error('Error adding component:', error);
            }
        },

        selectComponent(componentId, event) {
            if (event.ctrlKey || event.metaKey) {
                // Multi-select
                if (this.selectedComponents.includes(componentId)) {
                    this.selectedComponents = this.selectedComponents.filter(id => id !== componentId);
                } else {
                    this.selectedComponents.push(componentId);
                }
            } else {
                // Single select
                this.selectedComponents = [componentId];
            }
        },

        deselectAll() {
            this.selectedComponents = [];
        },

        removeComponent(componentId) {
            // Remove component
            this.pipelineComponents = this.pipelineComponents.filter(c => c.id !== componentId);
            
            // Remove related connections
            this.connections = this.connections.filter(conn => 
                conn.source.componentId !== componentId && conn.target.componentId !== componentId
            );
            
            // Remove from selection
            this.selectedComponents = this.selectedComponents.filter(id => id !== componentId);
            
            this.pipelineModified = true;
            this.validatePipeline();
        },

        // Canvas Management
        zoomIn() {
            this.canvasZoom = Math.min(this.canvasZoom * 1.2, 3.0);
        },

        zoomOut() {
            this.canvasZoom = Math.max(this.canvasZoom / 1.2, 0.2);
        },

        resetZoom() {
            this.canvasZoom = 1.0;
            this.canvasPan = { x: 0, y: 0 };
        },

        autoLayout() {
            // Simple auto-layout algorithm
            const layers = this.calculateComponentLayers();
            let currentY = 50;
            
            layers.forEach((layer, index) => {
                let currentX = 50;
                layer.forEach(component => {
                    component.x = currentX;
                    component.y = currentY;
                    currentX += 200;
                });
                currentY += 150;
            });
            
            this.pipelineModified = true;
        },

        calculateComponentLayers() {
            const layers = [];
            const visited = new Set();
            const sourceComponents = this.pipelineComponents.filter(c => c.type === 'source');
            
            // Start with source components
            if (sourceComponents.length > 0) {
                layers.push(sourceComponents);
                sourceComponents.forEach(c => visited.add(c.id));
            }
            
            // Build subsequent layers
            while (visited.size < this.pipelineComponents.length) {
                const nextLayer = [];
                
                for (const component of this.pipelineComponents) {
                    if (visited.has(component.id)) continue;
                    
                    // Check if all inputs are satisfied
                    const inputConnections = this.connections.filter(conn => 
                        conn.target.componentId === component.id
                    );
                    
                    const allInputsSatisfied = inputConnections.every(conn => 
                        visited.has(conn.source.componentId)
                    );
                    
                    if (allInputsSatisfied || inputConnections.length === 0) {
                        nextLayer.push(component);
                        visited.add(component.id);
                    }
                }
                
                if (nextLayer.length > 0) {
                    layers.push(nextLayer);
                } else {
                    // Add remaining components to avoid infinite loop
                    const remaining = this.pipelineComponents.filter(c => !visited.has(c.id));
                    if (remaining.length > 0) {
                        layers.push(remaining);
                    }
                    break;
                }
            }
            
            return layers;
        },

        clearCanvas() {
            if (this.pipelineComponents.length > 0) {
                if (confirm('Are you sure you want to clear the entire pipeline?')) {
                    this.pipelineComponents = [];
                    this.connections = [];
                    this.selectedComponents = [];
                    this.pipelineModified = true;
                }
            }
        },

        // Connection Management
        startConnection(event, componentId, portType, portIndex) {
            event.preventDefault();
            event.stopPropagation();
            
            const component = this.pipelineComponents.find(c => c.id === componentId);
            if (!component) return;
            
            this.dragConnection = {
                source: { componentId, portType, portIndex },
                startX: event.clientX,
                startY: event.clientY
            };
            
            this.tempConnection = true;
            
            // Add mouse move and up listeners
            document.addEventListener('mousemove', this.handleConnectionDrag);
            document.addEventListener('mouseup', this.endConnection);
        },

        handleConnectionDrag(event) {
            if (!this.dragConnection) return;
            
            const rect = document.getElementById('pipeline-canvas').getBoundingClientRect();
            const endX = (event.clientX - rect.left - this.canvasPan.x) / this.canvasZoom;
            const endY = (event.clientY - rect.top - this.canvasPan.y) / this.canvasZoom;
            
            const sourceComponent = this.pipelineComponents.find(c => c.id === this.dragConnection.source.componentId);
            const startX = sourceComponent.x + 150; // Approximate output point position
            const startY = sourceComponent.y + 20 + this.dragConnection.source.portIndex * 15;
            
            this.tempConnectionPath = `M ${startX} ${startY} Q ${(startX + endX) / 2} ${startY} ${endX} ${endY}`;
        },

        endConnection(event) {
            this.tempConnection = false;
            this.tempConnectionPath = '';
            
            // Remove event listeners
            document.removeEventListener('mousemove', this.handleConnectionDrag);
            document.removeEventListener('mouseup', this.endConnection);
            
            this.dragConnection = null;
        },

        connectComponents(event, targetComponentId, portType, portIndex) {
            if (!this.dragConnection) return;
            
            const sourceComponentId = this.dragConnection.source.componentId;
            const sourcePortIndex = this.dragConnection.source.portIndex;
            
            // Prevent self-connection
            if (sourceComponentId === targetComponentId) {
                this.showNotification('Cannot connect component to itself', 'error');
                return;
            }
            
            // Check if connection already exists
            const existingConnection = this.connections.find(conn =>
                conn.source.componentId === sourceComponentId &&
                conn.source.portIndex === sourcePortIndex &&
                conn.target.componentId === targetComponentId &&
                conn.target.portIndex === portIndex
            );
            
            if (existingConnection) {
                this.showNotification('Connection already exists', 'warning');
                return;
            }
            
            // Create new connection
            const connection = {
                id: `conn-${Date.now()}`,
                source: {
                    componentId: sourceComponentId,
                    portType: 'output',
                    portIndex: sourcePortIndex
                },
                target: {
                    componentId: targetComponentId,
                    portType: 'input',
                    portIndex: portIndex
                }
            };
            
            this.connections.push(connection);
            this.pipelineModified = true;
            this.validatePipeline();
            
            this.showNotification('Components connected', 'success');
        },

        getConnectionPath(connection) {
            const sourceComponent = this.pipelineComponents.find(c => c.id === connection.source.componentId);
            const targetComponent = this.pipelineComponents.find(c => c.id === connection.target.componentId);
            
            if (!sourceComponent || !targetComponent) return '';
            
            const startX = sourceComponent.x + 150; // Output point position
            const startY = sourceComponent.y + 20 + connection.source.portIndex * 15;
            const endX = targetComponent.x; // Input point position
            const endY = targetComponent.y + 20 + connection.target.portIndex * 15;
            
            // Create curved path
            const controlPointX = (startX + endX) / 2;
            
            return `M ${startX} ${startY} Q ${controlPointX} ${startY} ${endX} ${endY}`;
        },

        // Pipeline Validation
        validatePipeline() {
            this.pipelineComponents.forEach(component => {
                component.validation = [];
                
                // Check required parameters
                component.parameters?.forEach(param => {
                    if (param.required && (!param.value || param.value === '')) {
                        component.validation.push({
                            id: `${param.name}-required`,
                            type: 'error',
                            message: `${param.label || param.name} is required`
                        });
                    }
                });
                
                // Check input connections for non-source components
                if (component.type !== 'source' && component.inputs.length > 0) {
                    const inputConnections = this.connections.filter(conn => 
                        conn.target.componentId === component.id
                    );
                    
                    if (inputConnections.length === 0) {
                        component.validation.push({
                            id: 'no-input',
                            type: 'warning',
                            message: 'Component has no input connections'
                        });
                    }
                }
            });
        },

        // Event Handlers
        startDragComponent(event, component) {
            if (!this.selectedComponents.includes(component.id)) {
                this.selectedComponents = [component.id];
            }
            
            this.isDragging = true;
            this.dragOffset = {
                x: event.clientX - component.x * this.canvasZoom,
                y: event.clientY - component.y * this.canvasZoom
            };
            
            document.addEventListener('mousemove', this.handleComponentDrag);
            document.addEventListener('mouseup', this.endComponentDrag);
        },

        handleComponentDrag(event) {
            if (!this.isDragging) return;
            
            const rect = document.getElementById('pipeline-canvas').getBoundingClientRect();
            
            this.selectedComponents.forEach(componentId => {
                const component = this.pipelineComponents.find(c => c.id === componentId);
                if (component) {
                    component.x = (event.clientX - this.dragOffset.x) / this.canvasZoom;
                    component.y = (event.clientY - this.dragOffset.y) / this.canvasZoom;
                }
            });
            
            this.pipelineModified = true;
        },

        endComponentDrag() {
            this.isDragging = false;
            document.removeEventListener('mousemove', this.handleComponentDrag);
            document.removeEventListener('mouseup', this.endComponentDrag);
        },

        setupEventListeners() {
            // Keyboard shortcuts
            document.addEventListener('keydown', (event) => {
                if (event.ctrlKey || event.metaKey) {
                    switch (event.key) {
                        case 's':
                            event.preventDefault();
                            this.savePipeline();
                            break;
                        case 'n':
                            event.preventDefault();
                            this.newPipeline();
                            break;
                        case 'r':
                            event.preventDefault();
                            this.runPipeline();
                            break;
                        case 'Delete':
                        case 'Backspace':
                            event.preventDefault();
                            this.deleteSelectedComponents();
                            break;
                    }
                }
            });
        },

        deleteSelectedComponents() {
            if (this.selectedComponents.length === 0) return;
            
            this.selectedComponents.forEach(componentId => {
                this.removeComponent(componentId);
            });
            
            this.selectedComponents = [];
        },

        // Sample Data
        loadSamplePipeline() {
            // Load a sample pipeline for demonstration
            this.pipelineComponents = [
                {
                    id: 'csv-source-demo',
                    name: 'Sample Data',
                    icon: '📄',
                    type: 'source',
                    x: 50,
                    y: 50,
                    status: 'pending',
                    validation: [],
                    inputs: [],
                    outputs: ['data'],
                    parameters: [
                        { name: 'file_path', label: 'File Path', type: 'text', value: '/data/sample.csv' },
                        { name: 'delimiter', label: 'Delimiter', type: 'select', value: ',', options: [] },
                        { name: 'header', label: 'Has Header', type: 'boolean', value: true }
                    ]
                },
                {
                    id: 'data-cleaner-demo',
                    name: 'Clean Data',
                    icon: '🧹',
                    type: 'transformer',
                    x: 300,
                    y: 50,
                    status: 'pending',
                    validation: [],
                    inputs: ['data'],
                    outputs: ['cleaned_data'],
                    parameters: [
                        { name: 'remove_nulls', label: 'Remove Null Values', type: 'boolean', value: true },
                        { name: 'remove_duplicates', label: 'Remove Duplicates', type: 'boolean', value: true }
                    ]
                }
            ];
            
            this.connections = [
                {
                    id: 'conn-demo',
                    source: { componentId: 'csv-source-demo', portType: 'output', portIndex: 0 },
                    target: { componentId: 'data-cleaner-demo', portType: 'input', portIndex: 0 }
                }
            ];
        },

        // Utility Functions
        showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.textContent = message;
            
            const container = document.getElementById('notifications');
            container.appendChild(notification);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 5000);
        }
    };
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Register Alpine.js component
    if (window.Alpine) {
        window.Alpine.data('mlPipelineDesigner', mlPipelineDesigner);
    }
});

export { mlPipelineDesigner };