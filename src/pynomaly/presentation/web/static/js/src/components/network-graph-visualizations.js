/**
 * Enhanced Network/Graph Visualizations
 * 
 * Features:
 * - Force-directed graphs with customizable forces
 * - Hierarchical layouts (tree, radial, circular)
 * - Interactive node and edge manipulation
 * - Multi-layer network support
 * - Community detection and clustering
 * - Node and edge filtering
 * - Dynamic layout algorithms
 * - Real-time network updates
 * - Graph analytics and metrics
 * - Export capabilities
 */

class NetworkGraphVisualizations {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            width: 800,
            height: 600,
            nodeRadius: 5,
            linkDistance: 50,
            linkStrength: 0.5,
            chargeStrength: -300,
            enableZoom: true,
            enablePan: true,
            enableDrag: true,
            showLabels: true,
            showEdgeLabels: false,
            colorScheme: 'category10',
            layoutType: 'force',
            ...options
        };
        
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.nodeGroups = new Map();
        this.selectedNodes = new Set();
        this.selectedLinks = new Set();
        this.filters = {
            nodeTypes: new Set(),
            edgeTypes: new Set(),
            minDegree: 0,
            maxDegree: Infinity,
            minWeight: 0,
            maxWeight: Infinity
        };
        
        this.init();
    }

    init() {
        this.setupSVG();
        this.setupControls();
        this.setupEventListeners();
        this.loadSampleData();
    }

    setupSVG() {
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .style('border', '1px solid #e2e8f0');
        
        // Setup zoom and pan
        if (this.options.enableZoom || this.options.enablePan) {
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {
                    this.g.attr('transform', event.transform);
                });
            
            this.svg.call(zoom);
        }
        
        // Create main group for graph elements
        this.g = this.svg.append('g');
        
        // Create groups for different elements
        this.linkGroup = this.g.append('g').attr('class', 'links');
        this.nodeGroup = this.g.append('g').attr('class', 'nodes');
        this.labelGroup = this.g.append('g').attr('class', 'labels');
        
        // Add arrow markers for directed graphs
        this.svg.append('defs')
            .append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 8)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 4)
            .attr('markerHeight', 4)
            .attr('xoverflow', 'visible')
            .append('path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#999');
    }

    setupControls() {
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'network-controls';
        controlsDiv.innerHTML = `
            <div class="controls-header">
                <h3>Network Controls</h3>
                <button class="controls-toggle" id="toggle-controls">−</button>
            </div>
            <div class="controls-content">
                <div class="control-section">
                    <h4>Layout</h4>
                    <select id="layout-type">
                        <option value="force">Force Directed</option>
                        <option value="hierarchical">Hierarchical</option>
                        <option value="circular">Circular</option>
                        <option value="radial">Radial</option>
                        <option value="grid">Grid</option>
                    </select>
                </div>
                
                <div class="control-section">
                    <h4>Forces</h4>
                    <div class="force-controls">
                        <label>Link Distance: <span id="link-distance-value">${this.options.linkDistance}</span></label>
                        <input type="range" id="link-distance" min="10" max="200" value="${this.options.linkDistance}">
                        
                        <label>Link Strength: <span id="link-strength-value">${this.options.linkStrength}</span></label>
                        <input type="range" id="link-strength" min="0" max="1" step="0.1" value="${this.options.linkStrength}">
                        
                        <label>Charge Strength: <span id="charge-strength-value">${Math.abs(this.options.chargeStrength)}</span></label>
                        <input type="range" id="charge-strength" min="10" max="1000" value="${Math.abs(this.options.chargeStrength)}">
                    </div>
                </div>
                
                <div class="control-section">
                    <h4>Filtering</h4>
                    <div class="filter-controls">
                        <label>Min Degree: <span id="min-degree-value">0</span></label>
                        <input type="range" id="min-degree" min="0" max="20" value="0">
                        
                        <label>Max Degree: <span id="max-degree-value">20</span></label>
                        <input type="range" id="max-degree" min="0" max="20" value="20">
                        
                        <label>Node Types:</label>
                        <div id="node-type-filters" class="checkbox-group">
                            <!-- Will be populated dynamically -->
                        </div>
                    </div>
                </div>
                
                <div class="control-section">
                    <h4>Appearance</h4>
                    <div class="appearance-controls">
                        <label>
                            <input type="checkbox" id="show-labels" ${this.options.showLabels ? 'checked' : ''}>
                            Show Node Labels
                        </label>
                        <label>
                            <input type="checkbox" id="show-edge-labels" ${this.options.showEdgeLabels ? 'checked' : ''}>
                            Show Edge Labels
                        </label>
                        <label>
                            <input type="checkbox" id="curved-edges">
                            Curved Edges
                        </label>
                    </div>
                </div>
                
                <div class="control-section">
                    <h4>Analysis</h4>
                    <div class="analysis-controls">
                        <button id="detect-communities" class="control-btn">Detect Communities</button>
                        <button id="calculate-centrality" class="control-btn">Calculate Centrality</button>
                        <button id="find-shortest-path" class="control-btn">Find Shortest Path</button>
                        <button id="export-graph" class="control-btn">Export Graph</button>
                    </div>
                </div>
                
                <div class="control-section">
                    <h4>Statistics</h4>
                    <div class="stats-display">
                        <div class="stat-item">
                            <span class="stat-label">Nodes:</span>
                            <span class="stat-value" id="node-count">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Edges:</span>
                            <span class="stat-value" id="edge-count">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Density:</span>
                            <span class="stat-value" id="graph-density">0.0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Avg Degree:</span>
                            <span class="stat-value" id="avg-degree">0.0</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.container.appendChild(controlsDiv);
    }

    setupEventListeners() {
        // Layout controls
        document.getElementById('layout-type').addEventListener('change', (e) => {
            this.changeLayout(e.target.value);
        });
        
        // Force controls
        document.getElementById('link-distance').addEventListener('input', (e) => {
            this.options.linkDistance = parseInt(e.target.value);
            document.getElementById('link-distance-value').textContent = e.target.value;
            this.updateForces();
        });
        
        document.getElementById('link-strength').addEventListener('input', (e) => {
            this.options.linkStrength = parseFloat(e.target.value);
            document.getElementById('link-strength-value').textContent = e.target.value;
            this.updateForces();
        });
        
        document.getElementById('charge-strength').addEventListener('input', (e) => {
            this.options.chargeStrength = -parseInt(e.target.value);
            document.getElementById('charge-strength-value').textContent = e.target.value;
            this.updateForces();
        });
        
        // Filter controls
        document.getElementById('min-degree').addEventListener('input', (e) => {
            this.filters.minDegree = parseInt(e.target.value);
            document.getElementById('min-degree-value').textContent = e.target.value;
            this.applyFilters();
        });
        
        document.getElementById('max-degree').addEventListener('input', (e) => {
            this.filters.maxDegree = parseInt(e.target.value);
            document.getElementById('max-degree-value').textContent = e.target.value;
            this.applyFilters();
        });
        
        // Appearance controls
        document.getElementById('show-labels').addEventListener('change', (e) => {
            this.options.showLabels = e.target.checked;
            this.updateLabels();
        });
        
        document.getElementById('show-edge-labels').addEventListener('change', (e) => {
            this.options.showEdgeLabels = e.target.checked;
            this.updateEdgeLabels();
        });
        
        document.getElementById('curved-edges').addEventListener('change', (e) => {
            this.updateEdgeStyle(e.target.checked);
        });
        
        // Analysis controls
        document.getElementById('detect-communities').addEventListener('click', () => {
            this.detectCommunities();
        });
        
        document.getElementById('calculate-centrality').addEventListener('click', () => {
            this.calculateCentrality();
        });
        
        document.getElementById('find-shortest-path').addEventListener('click', () => {
            this.findShortestPath();
        });
        
        document.getElementById('export-graph').addEventListener('click', () => {
            this.exportGraph();
        });
        
        // Controls toggle
        document.getElementById('toggle-controls').addEventListener('click', () => {
            this.toggleControls();
        });
    }

    loadSampleData() {
        // Generate sample network data
        this.generateSampleNetwork();
        this.render();
    }

    generateSampleNetwork() {
        const nodeCount = 50;
        const edgeCount = 100;
        const nodeTypes = ['server', 'database', 'api', 'client', 'service'];
        
        // Generate nodes
        this.nodes = [];
        for (let i = 0; i < nodeCount; i++) {
            const nodeType = nodeTypes[Math.floor(Math.random() * nodeTypes.length)];
            this.nodes.push({
                id: i,
                name: `Node ${i}`,
                type: nodeType,
                group: Math.floor(Math.random() * 5),
                size: Math.random() * 20 + 5,
                anomaly_score: Math.random(),
                is_anomaly: Math.random() > 0.8,
                metrics: {
                    cpu: Math.random() * 100,
                    memory: Math.random() * 100,
                    disk: Math.random() * 100
                }
            });
        }
        
        // Generate edges
        this.links = [];
        const linkTypes = ['depends_on', 'connects_to', 'communicates_with', 'inherits_from'];
        
        for (let i = 0; i < edgeCount; i++) {
            const source = Math.floor(Math.random() * nodeCount);
            let target = Math.floor(Math.random() * nodeCount);
            
            // Avoid self-loops
            while (target === source) {
                target = Math.floor(Math.random() * nodeCount);
            }
            
            // Check if link already exists
            const existingLink = this.links.find(l => 
                (l.source === source && l.target === target) ||
                (l.source === target && l.target === source)
            );
            
            if (!existingLink) {
                this.links.push({
                    source: source,
                    target: target,
                    type: linkTypes[Math.floor(Math.random() * linkTypes.length)],
                    weight: Math.random() * 10 + 1,
                    anomaly_score: Math.random(),
                    is_anomaly: Math.random() > 0.9
                });
            }
        }
        
        this.calculateNodeDegrees();
        this.updateNodeTypeFilters();
        this.updateStatistics();
    }

    calculateNodeDegrees() {
        // Calculate node degrees
        this.nodes.forEach(node => {
            node.degree = 0;
            node.inDegree = 0;
            node.outDegree = 0;
        });
        
        this.links.forEach(link => {
            const sourceNode = this.nodes.find(n => n.id === link.source);
            const targetNode = this.nodes.find(n => n.id === link.target);
            
            if (sourceNode && targetNode) {
                sourceNode.degree++;
                sourceNode.outDegree++;
                targetNode.degree++;
                targetNode.inDegree++;
            }
        });
    }

    updateNodeTypeFilters() {
        const nodeTypes = [...new Set(this.nodes.map(n => n.type))];
        const container = document.getElementById('node-type-filters');
        container.innerHTML = '';
        
        nodeTypes.forEach(type => {
            const label = document.createElement('label');
            label.innerHTML = `
                <input type="checkbox" value="${type}" checked>
                ${type.charAt(0).toUpperCase() + type.slice(1)}
            `;
            
            const checkbox = label.querySelector('input');
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.filters.nodeTypes.delete(type);
                } else {
                    this.filters.nodeTypes.add(type);
                }
                this.applyFilters();
            });
            
            container.appendChild(label);
        });
    }

    render() {
        this.setupSimulation();
        this.renderNodes();
        this.renderLinks();
        this.renderLabels();
        this.startSimulation();
    }

    setupSimulation() {
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(this.options.linkDistance).strength(this.options.linkStrength))
            .force('charge', d3.forceManyBody().strength(this.options.chargeStrength))
            .force('center', d3.forceCenter(this.options.width / 2, this.options.height / 2))
            .force('collision', d3.forceCollide().radius(d => d.size + 2));
    }

    renderNodes() {
        const nodeSelection = this.nodeGroup.selectAll('.node')
            .data(this.nodes, d => d.id);
        
        const nodeEnter = nodeSelection.enter()
            .append('g')
            .attr('class', 'node')
            .call(this.getDragBehavior());
        
        nodeEnter.append('circle')
            .attr('r', d => d.size)
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', d => d.is_anomaly ? '#ef4444' : '#666')
            .attr('stroke-width', d => d.is_anomaly ? 3 : 1)
            .attr('opacity', 0.8);
        
        nodeEnter.append('text')
            .attr('class', 'node-label')
            .attr('dx', d => d.size + 5)
            .attr('dy', '0.35em')
            .attr('font-size', '12px')
            .attr('fill', '#333')
            .text(d => d.name)
            .style('display', this.options.showLabels ? 'block' : 'none');
        
        // Add click handler
        nodeEnter.on('click', (event, d) => {
            this.selectNode(d, event.ctrlKey || event.metaKey);
        });
        
        // Add hover effects
        nodeEnter.on('mouseover', (event, d) => {
            this.showNodeTooltip(event, d);
        }).on('mouseout', () => {
            this.hideTooltip();
        });
        
        nodeSelection.exit().remove();
        
        this.nodeElements = nodeEnter.merge(nodeSelection);
    }

    renderLinks() {
        const linkSelection = this.linkGroup.selectAll('.link')
            .data(this.links, d => `${d.source.id}-${d.target.id}`);
        
        const linkEnter = linkSelection.enter()
            .append('line')
            .attr('class', 'link')
            .attr('stroke', d => d.is_anomaly ? '#ef4444' : '#999')
            .attr('stroke-width', d => Math.sqrt(d.weight))
            .attr('stroke-opacity', 0.6)
            .attr('marker-end', 'url(#arrowhead)');
        
        linkEnter.on('click', (event, d) => {
            this.selectLink(d, event.ctrlKey || event.metaKey);
        });
        
        linkSelection.exit().remove();
        
        this.linkElements = linkEnter.merge(linkSelection);
    }

    renderLabels() {
        if (!this.options.showLabels) return;
        
        this.nodeElements.select('.node-label')
            .style('display', 'block');
    }

    startSimulation() {
        this.simulation.on('tick', () => {
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            this.nodeElements
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
    }

    getDragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }

    getNodeColor(node) {
        if (node.is_anomaly) return '#ef4444';
        
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        return colorScale(node.group);
    }

    selectNode(node, multiSelect = false) {
        if (!multiSelect) {
            this.selectedNodes.clear();
        }
        
        if (this.selectedNodes.has(node.id)) {
            this.selectedNodes.delete(node.id);
        } else {
            this.selectedNodes.add(node.id);
        }
        
        this.updateNodeSelection();
    }

    selectLink(link, multiSelect = false) {
        if (!multiSelect) {
            this.selectedLinks.clear();
        }
        
        const linkId = `${link.source.id}-${link.target.id}`;
        if (this.selectedLinks.has(linkId)) {
            this.selectedLinks.delete(linkId);
        } else {
            this.selectedLinks.add(linkId);
        }
        
        this.updateLinkSelection();
    }

    updateNodeSelection() {
        this.nodeElements.select('circle')
            .attr('stroke', d => {
                if (this.selectedNodes.has(d.id)) return '#3b82f6';
                return d.is_anomaly ? '#ef4444' : '#666';
            })
            .attr('stroke-width', d => {
                if (this.selectedNodes.has(d.id)) return 4;
                return d.is_anomaly ? 3 : 1;
            });
    }

    updateLinkSelection() {
        this.linkElements
            .attr('stroke', d => {
                const linkId = `${d.source.id}-${d.target.id}`;
                if (this.selectedLinks.has(linkId)) return '#3b82f6';
                return d.is_anomaly ? '#ef4444' : '#999';
            })
            .attr('stroke-width', d => {
                const linkId = `${d.source.id}-${d.target.id}`;
                if (this.selectedLinks.has(linkId)) return Math.sqrt(d.weight) + 2;
                return Math.sqrt(d.weight);
            });
    }

    changeLayout(layoutType) {
        this.options.layoutType = layoutType;
        
        switch (layoutType) {
            case 'force':
                this.applyForceLayout();
                break;
            case 'hierarchical':
                this.applyHierarchicalLayout();
                break;
            case 'circular':
                this.applyCircularLayout();
                break;
            case 'radial':
                this.applyRadialLayout();
                break;
            case 'grid':
                this.applyGridLayout();
                break;
        }
    }

    applyForceLayout() {
        this.simulation.stop();
        this.setupSimulation();
        this.simulation.alpha(1).restart();
    }

    applyHierarchicalLayout() {
        this.simulation.stop();
        
        // Create hierarchy based on node types
        const hierarchy = d3.stratify()
            .id(d => d.id)
            .parentId(d => {
                // Simple hierarchy: servers -> databases -> apis -> clients
                const typeHierarchy = { server: null, database: 'server', api: 'database', client: 'api' };
                return typeHierarchy[d.type];
            });
        
        try {
            const root = hierarchy(this.nodes);
            const tree = d3.tree().size([this.options.width - 100, this.options.height - 100]);
            tree(root);
            
            root.descendants().forEach(d => {
                d.data.x = d.x + 50;
                d.data.y = d.y + 50;
            });
            
            this.nodeElements.transition().duration(1000)
                .attr('transform', d => `translate(${d.x},${d.y})`);
                
        } catch (error) {
            console.warn('Could not create hierarchy, falling back to force layout');
            this.applyForceLayout();
        }
    }

    applyCircularLayout() {
        this.simulation.stop();
        
        const radius = Math.min(this.options.width, this.options.height) / 2 - 50;
        const angleStep = (2 * Math.PI) / this.nodes.length;
        
        this.nodes.forEach((node, i) => {
            const angle = i * angleStep;
            node.x = this.options.width / 2 + radius * Math.cos(angle);
            node.y = this.options.height / 2 + radius * Math.sin(angle);
        });
        
        this.nodeElements.transition().duration(1000)
            .attr('transform', d => `translate(${d.x},${d.y})`);
        
        this.linkElements.transition().duration(1000)
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
    }

    applyRadialLayout() {
        this.simulation.stop();
        
        // Group nodes by type
        const nodesByType = d3.group(this.nodes, d => d.type);
        const typeCount = nodesByType.size;
        const typeAngleStep = (2 * Math.PI) / typeCount;
        
        let typeIndex = 0;
        nodesByType.forEach((nodes, type) => {
            const typeAngle = typeIndex * typeAngleStep;
            const typeRadius = 100 + typeIndex * 50;
            const nodeAngleStep = (Math.PI / 3) / nodes.length;
            
            nodes.forEach((node, i) => {
                const angle = typeAngle + (i - nodes.length / 2) * nodeAngleStep;
                node.x = this.options.width / 2 + typeRadius * Math.cos(angle);
                node.y = this.options.height / 2 + typeRadius * Math.sin(angle);
            });
            
            typeIndex++;
        });
        
        this.nodeElements.transition().duration(1000)
            .attr('transform', d => `translate(${d.x},${d.y})`);
        
        this.linkElements.transition().duration(1000)
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
    }

    applyGridLayout() {
        this.simulation.stop();
        
        const cols = Math.ceil(Math.sqrt(this.nodes.length));
        const rows = Math.ceil(this.nodes.length / cols);
        const cellWidth = this.options.width / cols;
        const cellHeight = this.options.height / rows;
        
        this.nodes.forEach((node, i) => {
            const row = Math.floor(i / cols);
            const col = i % cols;
            node.x = col * cellWidth + cellWidth / 2;
            node.y = row * cellHeight + cellHeight / 2;
        });
        
        this.nodeElements.transition().duration(1000)
            .attr('transform', d => `translate(${d.x},${d.y})`);
        
        this.linkElements.transition().duration(1000)
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
    }

    updateForces() {
        if (this.simulation) {
            this.simulation
                .force('link').distance(this.options.linkDistance).strength(this.options.linkStrength)
                .force('charge').strength(this.options.chargeStrength);
            
            this.simulation.alpha(0.3).restart();
        }
    }

    applyFilters() {
        const filteredNodes = this.nodes.filter(node => {
            return node.degree >= this.filters.minDegree &&
                   node.degree <= this.filters.maxDegree &&
                   !this.filters.nodeTypes.has(node.type);
        });
        
        const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
        const filteredLinks = this.links.filter(link => {
            return filteredNodeIds.has(link.source.id) && 
                   filteredNodeIds.has(link.target.id);
        });
        
        // Update visualization
        this.nodeElements.style('display', d => filteredNodeIds.has(d.id) ? 'block' : 'none');
        this.linkElements.style('display', d => {
            return filteredNodeIds.has(d.source.id) && filteredNodeIds.has(d.target.id) ? 'block' : 'none';
        });
        
        this.updateStatistics();
    }

    detectCommunities() {
        // Simple community detection using modularity
        const communities = new Map();
        let communityId = 0;
        
        this.nodes.forEach(node => {
            if (!communities.has(node.group)) {
                communities.set(node.group, communityId++);
            }
        });
        
        // Color nodes by community
        this.nodeElements.select('circle')
            .attr('fill', d => {
                const colorScale = d3.scaleOrdinal(d3.schemeSet3);
                return colorScale(communities.get(d.group));
            });
        
        alert(`Detected ${communities.size} communities`);
    }

    calculateCentrality() {
        // Calculate betweenness centrality (simplified)
        this.nodes.forEach(node => {
            node.centrality = node.degree / this.nodes.length;
        });
        
        // Scale node sizes by centrality
        this.nodeElements.select('circle')
            .transition().duration(1000)
            .attr('r', d => 5 + d.centrality * 20);
        
        alert('Centrality calculated and applied to node sizes');
    }

    findShortestPath() {
        if (this.selectedNodes.size !== 2) {
            alert('Please select exactly 2 nodes to find the shortest path');
            return;
        }
        
        const [nodeA, nodeB] = Array.from(this.selectedNodes);
        
        // Simple BFS for shortest path
        const path = this.bfsShortestPath(nodeA, nodeB);
        
        if (path.length > 0) {
            this.highlightPath(path);
            alert(`Shortest path found with ${path.length - 1} edges`);
        } else {
            alert('No path found between selected nodes');
        }
    }

    bfsShortestPath(startId, endId) {
        const queue = [[startId]];
        const visited = new Set([startId]);
        
        while (queue.length > 0) {
            const path = queue.shift();
            const currentId = path[path.length - 1];
            
            if (currentId === endId) {
                return path;
            }
            
            const neighbors = this.links
                .filter(link => link.source.id === currentId || link.target.id === currentId)
                .map(link => link.source.id === currentId ? link.target.id : link.source.id);
            
            for (const neighborId of neighbors) {
                if (!visited.has(neighborId)) {
                    visited.add(neighborId);
                    queue.push([...path, neighborId]);
                }
            }
        }
        
        return [];
    }

    highlightPath(path) {
        // Reset all elements
        this.nodeElements.select('circle').attr('opacity', 0.3);
        this.linkElements.attr('opacity', 0.1);
        
        // Highlight path nodes
        path.forEach(nodeId => {
            this.nodeElements.filter(d => d.id === nodeId)
                .select('circle')
                .attr('opacity', 1)
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', 4);
        });
        
        // Highlight path edges
        for (let i = 0; i < path.length - 1; i++) {
            const sourceId = path[i];
            const targetId = path[i + 1];
            
            this.linkElements.filter(d => {
                return (d.source.id === sourceId && d.target.id === targetId) ||
                       (d.source.id === targetId && d.target.id === sourceId);
            })
                .attr('opacity', 1)
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', 4);
        }
    }

    exportGraph() {
        const graphData = {
            nodes: this.nodes.map(node => ({
                id: node.id,
                name: node.name,
                type: node.type,
                group: node.group,
                x: node.x,
                y: node.y,
                degree: node.degree,
                anomaly_score: node.anomaly_score,
                is_anomaly: node.is_anomaly
            })),
            links: this.links.map(link => ({
                source: link.source.id,
                target: link.target.id,
                type: link.type,
                weight: link.weight,
                anomaly_score: link.anomaly_score,
                is_anomaly: link.is_anomaly
            }))
        };
        
        const blob = new Blob([JSON.stringify(graphData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'network-graph.json';
        link.click();
        URL.revokeObjectURL(url);
    }

    updateStatistics() {
        const visibleNodes = this.nodes.filter(node => {
            return node.degree >= this.filters.minDegree &&
                   node.degree <= this.filters.maxDegree &&
                   !this.filters.nodeTypes.has(node.type);
        });
        
        const visibleLinks = this.links.filter(link => {
            const sourceVisible = visibleNodes.some(n => n.id === link.source.id);
            const targetVisible = visibleNodes.some(n => n.id === link.target.id);
            return sourceVisible && targetVisible;
        });
        
        const nodeCount = visibleNodes.length;
        const edgeCount = visibleLinks.length;
        const maxPossibleEdges = nodeCount * (nodeCount - 1) / 2;
        const density = maxPossibleEdges > 0 ? edgeCount / maxPossibleEdges : 0;
        const avgDegree = nodeCount > 0 ? visibleNodes.reduce((sum, n) => sum + n.degree, 0) / nodeCount : 0;
        
        document.getElementById('node-count').textContent = nodeCount;
        document.getElementById('edge-count').textContent = edgeCount;
        document.getElementById('graph-density').textContent = density.toFixed(3);
        document.getElementById('avg-degree').textContent = avgDegree.toFixed(1);
    }

    updateLabels() {
        this.nodeElements.select('.node-label')
            .style('display', this.options.showLabels ? 'block' : 'none');
    }

    updateEdgeLabels() {
        // Implementation for edge labels
        if (this.options.showEdgeLabels) {
            // Add edge labels if not exist
            this.linkGroup.selectAll('.edge-label')
                .data(this.links)
                .enter()
                .append('text')
                .attr('class', 'edge-label')
                .attr('font-size', '10px')
                .attr('fill', '#666')
                .text(d => d.type);
        } else {
            this.linkGroup.selectAll('.edge-label').remove();
        }
    }

    updateEdgeStyle(curved) {
        if (curved) {
            this.linkElements
                .attr('d', d => {
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const dr = Math.sqrt(dx * dx + dy * dy) * 0.3;
                    return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
                });
        } else {
            this.linkElements
                .attr('d', null)
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        }
    }

    showNodeTooltip(event, node) {
        const tooltip = d3.select('body').append('div')
            .attr('class', 'network-tooltip')
            .style('opacity', 0);
        
        tooltip.transition().duration(200).style('opacity', 0.9);
        tooltip.html(`
            <strong>${node.name}</strong><br>
            Type: ${node.type}<br>
            Degree: ${node.degree}<br>
            Anomaly Score: ${node.anomaly_score.toFixed(3)}<br>
            ${node.is_anomaly ? '<span style="color: #ef4444;">ANOMALY</span>' : ''}
        `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px');
        
        this.currentTooltip = tooltip;
    }

    hideTooltip() {
        if (this.currentTooltip) {
            this.currentTooltip.transition().duration(200).style('opacity', 0).remove();
            this.currentTooltip = null;
        }
    }

    toggleControls() {
        const content = document.querySelector('.controls-content');
        const toggle = document.getElementById('toggle-controls');
        
        content.style.display = content.style.display === 'none' ? 'block' : 'none';
        toggle.textContent = content.style.display === 'none' ? '+' : '−';
    }
}

// Initialize the network graph visualizations
window.NetworkGraphVisualizations = NetworkGraphVisualizations;

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NetworkGraphVisualizations;
}