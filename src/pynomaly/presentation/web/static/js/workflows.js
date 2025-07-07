/**
 * Workflow Management JavaScript
 * Handles workflow creation, execution, and monitoring
 */

// Global workflow manager
window.WorkflowManager = {
  workflows: new Map(),
  templates: new Map(),
  activeWorkflow: null,
  dragState: {
    isDragging: false,
    draggedElement: null,
    offset: { x: 0, y: 0 },
  },

  init() {
    this.initializeTemplates();
    this.setupEventListeners();
    this.loadWorkflows();
    this.initWorkflowCanvas();
  },

  initializeTemplates() {
    // Pre-defined workflow templates
    const templates = [
      {
        id: "basic-anomaly-detection",
        name: "Basic Anomaly Detection",
        description: "Load data, train detector, detect anomalies",
        steps: [
          {
            id: "load-data",
            type: "data_input",
            name: "Load Dataset",
            x: 100,
            y: 100,
          },
          {
            id: "train",
            type: "training",
            name: "Train Detector",
            x: 300,
            y: 100,
          },
          {
            id: "detect",
            type: "detection",
            name: "Detect Anomalies",
            x: 500,
            y: 100,
          },
          {
            id: "visualize",
            type: "visualization",
            name: "Visualize Results",
            x: 700,
            y: 100,
          },
        ],
        connections: [
          { from: "load-data", to: "train" },
          { from: "train", to: "detect" },
          { from: "detect", to: "visualize" },
        ],
      },
      {
        id: "ensemble-workflow",
        name: "Ensemble Detection",
        description: "Multiple detectors with ensemble aggregation",
        steps: [
          {
            id: "load-data",
            type: "data_input",
            name: "Load Dataset",
            x: 100,
            y: 200,
          },
          {
            id: "train1",
            type: "training",
            name: "Train Detector 1",
            x: 300,
            y: 150,
          },
          {
            id: "train2",
            type: "training",
            name: "Train Detector 2",
            x: 300,
            y: 250,
          },
          {
            id: "ensemble",
            type: "ensemble",
            name: "Ensemble Aggregation",
            x: 500,
            y: 200,
          },
          {
            id: "evaluate",
            type: "evaluation",
            name: "Evaluate Results",
            x: 700,
            y: 200,
          },
        ],
        connections: [
          { from: "load-data", to: "train1" },
          { from: "load-data", to: "train2" },
          { from: "train1", to: "ensemble" },
          { from: "train2", to: "ensemble" },
          { from: "ensemble", to: "evaluate" },
        ],
      },
      {
        id: "automl-optimization",
        name: "AutoML Optimization",
        description: "Automated hyperparameter optimization workflow",
        steps: [
          {
            id: "load-data",
            type: "data_input",
            name: "Load Dataset",
            x: 100,
            y: 100,
          },
          {
            id: "preprocess",
            type: "preprocessing",
            name: "Preprocess Data",
            x: 300,
            y: 100,
          },
          {
            id: "automl",
            type: "automl",
            name: "AutoML Optimization",
            x: 500,
            y: 100,
          },
          {
            id: "validate",
            type: "validation",
            name: "Cross Validation",
            x: 700,
            y: 100,
          },
          {
            id: "deploy",
            type: "deployment",
            name: "Deploy Model",
            x: 900,
            y: 100,
          },
        ],
        connections: [
          { from: "load-data", to: "preprocess" },
          { from: "preprocess", to: "automl" },
          { from: "automl", to: "validate" },
          { from: "validate", to: "deploy" },
        ],
      },
    ];

    templates.forEach((template) => {
      this.templates.set(template.id, template);
    });
  },

  setupEventListeners() {
    // Workflow creation
    document.addEventListener("click", (e) => {
      if (e.target.matches('[data-action="create-workflow"]')) {
        this.createNewWorkflow();
      }

      if (e.target.matches('[data-action="load-template"]')) {
        const templateId = e.target.dataset.templateId;
        this.loadTemplate(templateId);
      }

      if (e.target.matches('[data-action="save-workflow"]')) {
        this.saveCurrentWorkflow();
      }

      if (e.target.matches('[data-action="execute-workflow"]')) {
        this.executeWorkflow();
      }

      if (e.target.matches('[data-action="export-workflow"]')) {
        this.exportWorkflow();
      }
    });

    // Workflow step interactions
    document.addEventListener("mousedown", (e) => {
      if (
        e.target.matches(".workflow-step") ||
        e.target.closest(".workflow-step")
      ) {
        this.startDrag(e);
      }
    });

    document.addEventListener("mousemove", (e) => {
      if (this.dragState.isDragging) {
        this.updateDrag(e);
      }
    });

    document.addEventListener("mouseup", (e) => {
      if (this.dragState.isDragging) {
        this.endDrag(e);
      }
    });

    // Canvas interactions
    const canvas = document.getElementById("workflow-canvas");
    if (canvas) {
      canvas.addEventListener("click", (e) => {
        if (e.target === canvas) {
          this.clearSelection();
        }
      });

      canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        this.showContextMenu(e);
      });
    }
  },

  initWorkflowCanvas() {
    const canvas = document.getElementById("workflow-canvas");
    if (!canvas) return;

    // Set up SVG for connections
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.id = "workflow-connections";
    svg.style.position = "absolute";
    svg.style.top = "0";
    svg.style.left = "0";
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.style.pointerEvents = "none";
    svg.style.zIndex = "1";

    canvas.appendChild(svg);

    // Add defs for arrow markers
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const marker = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "marker",
    );
    marker.id = "arrowhead";
    marker.setAttribute("markerWidth", "10");
    marker.setAttribute("markerHeight", "7");
    marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "3.5");
    marker.setAttribute("orient", "auto");

    const polygon = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "polygon",
    );
    polygon.setAttribute("points", "0 0, 10 3.5, 0 7");
    polygon.setAttribute("fill", "#3B82F6");

    marker.appendChild(polygon);
    defs.appendChild(marker);
    svg.appendChild(defs);
  },

  createNewWorkflow() {
    const workflow = {
      id: this.generateId(),
      name: "New Workflow",
      description: "",
      steps: [],
      connections: [],
      status: "draft",
      created: new Date(),
      modified: new Date(),
    };

    this.workflows.set(workflow.id, workflow);
    this.setActiveWorkflow(workflow.id);
    this.renderWorkflow();
    this.updateWorkflowList();
  },

  loadTemplate(templateId) {
    const template = this.templates.get(templateId);
    if (!template) return;

    const workflow = {
      id: this.generateId(),
      name: template.name,
      description: template.description,
      steps: [...template.steps],
      connections: [...template.connections],
      status: "draft",
      created: new Date(),
      modified: new Date(),
    };

    this.workflows.set(workflow.id, workflow);
    this.setActiveWorkflow(workflow.id);
    this.renderWorkflow();
    this.updateWorkflowList();
  },

  setActiveWorkflow(workflowId) {
    this.activeWorkflow = workflowId;
    const workflow = this.workflows.get(workflowId);

    // Update UI
    const nameInput = document.getElementById("workflow-name");
    const descInput = document.getElementById("workflow-description");

    if (nameInput) nameInput.value = workflow.name;
    if (descInput) descInput.value = workflow.description;

    this.renderWorkflow();
  },

  renderWorkflow() {
    if (!this.activeWorkflow) return;

    const workflow = this.workflows.get(this.activeWorkflow);
    const canvas = document.getElementById("workflow-canvas");
    const svg = document.getElementById("workflow-connections");

    if (!canvas || !workflow) return;

    // Clear existing steps
    canvas.querySelectorAll(".workflow-step").forEach((step) => step.remove());

    // Clear existing connections
    if (svg) {
      svg.querySelectorAll("path").forEach((path) => path.remove());
    }

    // Render steps
    workflow.steps.forEach((step) => {
      this.renderStep(step);
    });

    // Render connections
    workflow.connections.forEach((connection) => {
      this.renderConnection(connection);
    });
  },

  renderStep(step) {
    const canvas = document.getElementById("workflow-canvas");
    if (!canvas) return;

    const stepElement = document.createElement("div");
    stepElement.className = `workflow-step workflow-step-${step.type}`;
    stepElement.dataset.stepId = step.id;
    stepElement.style.position = "absolute";
    stepElement.style.left = step.x + "px";
    stepElement.style.top = step.y + "px";
    stepElement.style.zIndex = "2";

    const icon = this.getStepIcon(step.type);
    const statusClass = this.getStepStatusClass(step.status || "pending");

    stepElement.innerHTML = `
            <div class="workflow-node bg-white border-2 rounded-lg p-4 shadow-lg cursor-move ${statusClass}">
                <div class="flex items-center space-x-3">
                    <div class="flex-shrink-0">
                        ${icon}
                    </div>
                    <div class="flex-1">
                        <h4 class="text-sm font-medium text-gray-900">${step.name}</h4>
                        <p class="text-xs text-gray-500">${step.type.replace("_", " ")}</p>
                    </div>
                </div>
                <div class="mt-2 flex justify-between items-center">
                    <span class="text-xs text-gray-400">${step.id}</span>
                    <div class="flex space-x-1">
                        <button class="step-config text-xs bg-blue-100 text-blue-600 px-2 py-1 rounded">Config</button>
                        <button class="step-delete text-xs bg-red-100 text-red-600 px-2 py-1 rounded">Delete</button>
                    </div>
                </div>
                <!-- Connection points -->
                <div class="connection-point input absolute -left-2 top-1/2 w-4 h-4 bg-green-500 rounded-full border-2 border-white"></div>
                <div class="connection-point output absolute -right-2 top-1/2 w-4 h-4 bg-blue-500 rounded-full border-2 border-white"></div>
            </div>
        `;

    // Add event listeners
    stepElement.querySelector(".step-config").addEventListener("click", (e) => {
      e.stopPropagation();
      this.configureStep(step.id);
    });

    stepElement.querySelector(".step-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      this.deleteStep(step.id);
    });

    canvas.appendChild(stepElement);
  },

  renderConnection(connection) {
    const svg = document.getElementById("workflow-connections");
    if (!svg) return;

    const fromElement = document.querySelector(
      `[data-step-id="${connection.from}"]`,
    );
    const toElement = document.querySelector(
      `[data-step-id="${connection.to}"]`,
    );

    if (!fromElement || !toElement) return;

    const fromRect = fromElement.getBoundingClientRect();
    const toRect = toElement.getBoundingClientRect();
    const canvasRect = svg.getBoundingClientRect();

    const fromX = fromRect.right - canvasRect.left;
    const fromY = fromRect.top + fromRect.height / 2 - canvasRect.top;
    const toX = toRect.left - canvasRect.left;
    const toY = toRect.top + toRect.height / 2 - canvasRect.top;

    // Create curved path
    const midX = (fromX + toX) / 2;
    const path = `M ${fromX} ${fromY} Q ${midX} ${fromY} ${midX} ${(fromY + toY) / 2} Q ${midX} ${toY} ${toX} ${toY}`;

    const pathElement = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    pathElement.setAttribute("d", path);
    pathElement.setAttribute("stroke", "#3B82F6");
    pathElement.setAttribute("stroke-width", "2");
    pathElement.setAttribute("fill", "none");
    pathElement.setAttribute("marker-end", "url(#arrowhead)");
    pathElement.dataset.connection = `${connection.from}-${connection.to}`;

    svg.appendChild(pathElement);
  },

  getStepIcon(type) {
    const icons = {
      data_input:
        '<svg class="w-6 h-6 text-blue-500" fill="currentColor" viewBox="0 0 20 20"><path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/></svg>',
      training:
        '<svg class="w-6 h-6 text-green-500" fill="currentColor" viewBox="0 0 20 20"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
      detection:
        '<svg class="w-6 h-6 text-yellow-500" fill="currentColor" viewBox="0 0 20 20"><path d="M12.432 0c1.34 0 2.01.912 2.01 1.957 0 1.305-1.164 2.512-2.679 2.512-1.269 0-2.009-.75-1.974-1.99C9.789 1.436 10.67 0 12.432 0zM8.309 20c-1.058 0-1.833-.652-1.093-3.524l1.214-5.092c.211-.814.246-1.141 0-1.141-.317 0-1.689.562-2.502 1.117l-.528-.88c2.572-2.186 5.531-3.467 6.801-3.467 1.057 0 1.233 1.273.705 3.23l-1.391 5.352c-.246.945-.141 1.271.106 1.271.317 0 1.357-.392 2.379-1.207l.6.814C12.098 19.02 9.365 20 8.309 20z"/></svg>',
      visualization:
        '<svg class="w-6 h-6 text-purple-500" fill="currentColor" viewBox="0 0 20 20"><path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/></svg>',
      ensemble:
        '<svg class="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/></svg>',
      evaluation:
        '<svg class="w-6 h-6 text-indigo-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>',
      preprocessing:
        '<svg class="w-6 h-6 text-cyan-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd"/></svg>',
      automl:
        '<svg class="w-6 h-6 text-orange-500" fill="currentColor" viewBox="0 0 20 20"><path d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z"/></svg>',
      validation:
        '<svg class="w-6 h-6 text-pink-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>',
      deployment:
        '<svg class="w-6 h-6 text-gray-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M2 5a2 2 0 012-2h8a2 2 0 012 2v10a2 2 0 002 2H4a2 2 0 01-2-2V5zm3 1h6v4H5V6zm6 6H5v2h6v-2z" clip-rule="evenodd"/><path d="M15 7h1a2 2 0 012 2v5.5a1.5 1.5 0 01-3 0V9a1 1 0 00-1-1h-1v-1z"/></svg>',
    };

    return icons[type] || icons.data_input;
  },

  getStepStatusClass(status) {
    const classes = {
      pending: "workflow-step-pending",
      running: "workflow-step-active",
      completed: "workflow-step-complete",
      failed: "border-red-500 bg-red-50",
    };

    return classes[status] || classes.pending;
  },

  startDrag(e) {
    const stepElement = e.target.closest(".workflow-step");
    if (!stepElement) return;

    this.dragState.isDragging = true;
    this.dragState.draggedElement = stepElement;

    const rect = stepElement.getBoundingClientRect();
    this.dragState.offset = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };

    stepElement.style.zIndex = "1000";
    document.body.style.cursor = "grabbing";
  },

  updateDrag(e) {
    if (!this.dragState.isDragging || !this.dragState.draggedElement) return;

    const canvas = document.getElementById("workflow-canvas");
    const canvasRect = canvas.getBoundingClientRect();

    const x = e.clientX - canvasRect.left - this.dragState.offset.x;
    const y = e.clientY - canvasRect.top - this.dragState.offset.y;

    this.dragState.draggedElement.style.left = x + "px";
    this.dragState.draggedElement.style.top = y + "px";

    // Update connections
    this.updateConnections();
  },

  endDrag(e) {
    if (!this.dragState.isDragging) return;

    if (this.dragState.draggedElement) {
      this.dragState.draggedElement.style.zIndex = "2";

      // Update step position in workflow
      const stepId = this.dragState.draggedElement.dataset.stepId;
      const workflow = this.workflows.get(this.activeWorkflow);
      const step = workflow.steps.find((s) => s.id === stepId);

      if (step) {
        step.x = parseInt(this.dragState.draggedElement.style.left);
        step.y = parseInt(this.dragState.draggedElement.style.top);
        workflow.modified = new Date();
      }
    }

    this.dragState.isDragging = false;
    this.dragState.draggedElement = null;
    document.body.style.cursor = "";
  },

  updateConnections() {
    const workflow = this.workflows.get(this.activeWorkflow);
    if (!workflow) return;

    workflow.connections.forEach((connection) => {
      this.renderConnection(connection);
    });
  },

  configureStep(stepId) {
    const workflow = this.workflows.get(this.activeWorkflow);
    const step = workflow.steps.find((s) => s.id === stepId);

    if (!step) return;

    // Show configuration modal
    this.showStepConfigModal(step);
  },

  showStepConfigModal(step) {
    const modal = document.getElementById("step-config-modal");
    if (!modal) return;

    const content = modal.querySelector(".modal-content");
    content.innerHTML = `
            <h3 class="text-lg font-bold mb-4">Configure Step: ${step.name}</h3>
            <form id="step-config-form" class="space-y-4">
                <div>
                    <label class="block text-sm font-medium mb-2">Step Name</label>
                    <input type="text" name="name" value="${step.name}" 
                           class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2">Type</label>
                    <select name="type" class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                        <option value="data_input" ${step.type === "data_input" ? "selected" : ""}>Data Input</option>
                        <option value="preprocessing" ${step.type === "preprocessing" ? "selected" : ""}>Preprocessing</option>
                        <option value="training" ${step.type === "training" ? "selected" : ""}>Training</option>
                        <option value="detection" ${step.type === "detection" ? "selected" : ""}>Detection</option>
                        <option value="ensemble" ${step.type === "ensemble" ? "selected" : ""}>Ensemble</option>
                        <option value="evaluation" ${step.type === "evaluation" ? "selected" : ""}>Evaluation</option>
                        <option value="visualization" ${step.type === "visualization" ? "selected" : ""}>Visualization</option>
                        <option value="automl" ${step.type === "automl" ? "selected" : ""}>AutoML</option>
                        <option value="validation" ${step.type === "validation" ? "selected" : ""}>Validation</option>
                        <option value="deployment" ${step.type === "deployment" ? "selected" : ""}>Deployment</option>
                    </select>
                </div>
                ${this.getStepSpecificConfig(step)}
                <div class="flex justify-end space-x-2 pt-4">
                    <button type="button" onclick="this.closest('.modal').style.display='none'" 
                            class="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50">
                        Cancel
                    </button>
                    <button type="submit" 
                            class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                        Save
                    </button>
                </div>
            </form>
        `;

    // Handle form submission
    const form = content.querySelector("#step-config-form");
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.saveStepConfig(step, new FormData(form));
      modal.style.display = "none";
    });

    modal.style.display = "block";
  },

  getStepSpecificConfig(step) {
    // Return specific configuration fields based on step type
    switch (step.type) {
      case "data_input":
        return `
                    <div>
                        <label class="block text-sm font-medium mb-2">Dataset Path</label>
                        <input type="text" name="dataset_path" value="${step.config?.dataset_path || ""}" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-2">File Format</label>
                        <select name="format" class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                            <option value="csv" ${step.config?.format === "csv" ? "selected" : ""}>CSV</option>
                            <option value="parquet" ${step.config?.format === "parquet" ? "selected" : ""}>Parquet</option>
                            <option value="json" ${step.config?.format === "json" ? "selected" : ""}>JSON</option>
                        </select>
                    </div>
                `;
      case "training":
        return `
                    <div>
                        <label class="block text-sm font-medium mb-2">Algorithm</label>
                        <select name="algorithm" class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                            <option value="IsolationForest" ${step.config?.algorithm === "IsolationForest" ? "selected" : ""}>Isolation Forest</option>
                            <option value="LocalOutlierFactor" ${step.config?.algorithm === "LocalOutlierFactor" ? "selected" : ""}>LOF</option>
                            <option value="OneClassSVM" ${step.config?.algorithm === "OneClassSVM" ? "selected" : ""}>One-Class SVM</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-2">Contamination</label>
                        <input type="number" name="contamination" value="${step.config?.contamination || 0.1}" 
                               step="0.01" min="0" max="1" class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                    </div>
                `;
      default:
        return "";
    }
  },

  saveStepConfig(step, formData) {
    step.name = formData.get("name");
    step.type = formData.get("type");
    step.config = step.config || {};

    // Save specific configuration
    for (const [key, value] of formData.entries()) {
      if (key !== "name" && key !== "type") {
        step.config[key] = value;
      }
    }

    // Update workflow
    const workflow = this.workflows.get(this.activeWorkflow);
    workflow.modified = new Date();

    // Re-render
    this.renderWorkflow();
  },

  deleteStep(stepId) {
    if (!confirm("Are you sure you want to delete this step?")) return;

    const workflow = this.workflows.get(this.activeWorkflow);

    // Remove step
    workflow.steps = workflow.steps.filter((s) => s.id !== stepId);

    // Remove connections
    workflow.connections = workflow.connections.filter(
      (c) => c.from !== stepId && c.to !== stepId,
    );

    workflow.modified = new Date();
    this.renderWorkflow();
  },

  executeWorkflow() {
    if (!this.activeWorkflow) return;

    const workflow = this.workflows.get(this.activeWorkflow);
    workflow.status = "running";

    // Simulate workflow execution
    this.simulateExecution(workflow);
  },

  simulateExecution(workflow) {
    const steps = [...workflow.steps];
    let currentStep = 0;

    const executeStep = () => {
      if (currentStep >= steps.length) {
        workflow.status = "completed";
        this.updateWorkflowStatus(
          "Workflow completed successfully!",
          "success",
        );
        return;
      }

      const step = steps[currentStep];
      step.status = "running";
      this.renderWorkflow();
      this.updateWorkflowStatus(`Executing: ${step.name}`, "info");

      // Simulate step execution time
      setTimeout(
        () => {
          step.status = "completed";
          this.renderWorkflow();
          currentStep++;
          executeStep();
        },
        1000 + Math.random() * 2000,
      );
    };

    executeStep();
  },

  updateWorkflowStatus(message, type) {
    const statusElement = document.getElementById("workflow-status");
    if (statusElement) {
      statusElement.innerHTML = `
                <div class="alert alert-${type} rounded-lg p-3">
                    <span class="text-sm">${message}</span>
                </div>
            `;
    }
  },

  saveCurrentWorkflow() {
    if (!this.activeWorkflow) return;

    const workflow = this.workflows.get(this.activeWorkflow);

    // Update from form
    const nameInput = document.getElementById("workflow-name");
    const descInput = document.getElementById("workflow-description");

    if (nameInput) workflow.name = nameInput.value;
    if (descInput) workflow.description = descInput.value;

    workflow.modified = new Date();

    // Save to storage (simulate)
    localStorage.setItem(
      "pynomaly-workflows",
      JSON.stringify(Array.from(this.workflows.entries())),
    );

    this.updateWorkflowStatus("Workflow saved successfully!", "success");
    this.updateWorkflowList();
  },

  loadWorkflows() {
    // Load from storage (simulate)
    const saved = localStorage.getItem("pynomaly-workflows");
    if (saved) {
      const workflows = new Map(JSON.parse(saved));
      this.workflows = workflows;
    }

    this.updateWorkflowList();
  },

  updateWorkflowList() {
    const list = document.getElementById("workflow-list");
    if (!list) return;

    list.innerHTML = "";

    for (const [id, workflow] of this.workflows) {
      const item = document.createElement("div");
      item.className = `workflow-item p-3 border rounded-lg cursor-pointer hover:bg-gray-50 ${this.activeWorkflow === id ? "border-blue-500 bg-blue-50" : "border-gray-200"}`;
      item.innerHTML = `
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <h4 class="font-medium text-gray-900">${workflow.name}</h4>
                        <p class="text-sm text-gray-500">${workflow.description || "No description"}</p>
                        <div class="mt-2 flex items-center space-x-4 text-xs text-gray-400">
                            <span>${workflow.steps.length} steps</span>
                            <span>${workflow.status}</span>
                            <span>Modified: ${workflow.modified.toLocaleDateString()}</span>
                        </div>
                    </div>
                    <div class="flex space-x-1">
                        <button class="text-blue-600 hover:text-blue-800" onclick="WorkflowManager.setActiveWorkflow('${id}')">
                            Open
                        </button>
                        <button class="text-red-600 hover:text-red-800" onclick="WorkflowManager.deleteWorkflow('${id}')">
                            Delete
                        </button>
                    </div>
                </div>
            `;

      list.appendChild(item);
    }
  },

  deleteWorkflow(workflowId) {
    if (!confirm("Are you sure you want to delete this workflow?")) return;

    this.workflows.delete(workflowId);

    if (this.activeWorkflow === workflowId) {
      this.activeWorkflow = null;
      this.renderWorkflow();
    }

    this.updateWorkflowList();
    this.saveCurrentWorkflow();
  },

  exportWorkflow() {
    if (!this.activeWorkflow) return;

    const workflow = this.workflows.get(this.activeWorkflow);
    const exportData = {
      version: "1.0",
      workflow: workflow,
      exported: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `${workflow.name.replace(/\s+/g, "_")}_workflow.json`;
    link.click();

    URL.revokeObjectURL(url);
  },

  generateId() {
    return (
      "workflow_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9)
    );
  },
};

// Alpine.js data for workflow management
function workflowManager() {
  return {
    searchQuery: "",
    selectedWorkflow: null,
    showTemplates: false,

    get filteredWorkflows() {
      const workflows = Array.from(WorkflowManager.workflows.values());
      if (!this.searchQuery) return workflows;

      return workflows.filter(
        (w) =>
          w.name.toLowerCase().includes(this.searchQuery.toLowerCase()) ||
          w.description.toLowerCase().includes(this.searchQuery.toLowerCase()),
      );
    },

    selectWorkflow(workflow) {
      this.selectedWorkflow = workflow;
      WorkflowManager.setActiveWorkflow(workflow.id);
    },

    createNewWorkflow() {
      WorkflowManager.createNewWorkflow();
    },

    importWorkflow() {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".json";
      input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (e) => {
            try {
              const data = JSON.parse(e.target.result);
              if (data.workflow) {
                const workflow = data.workflow;
                workflow.id = WorkflowManager.generateId();
                workflow.created = new Date();
                workflow.modified = new Date();

                WorkflowManager.workflows.set(workflow.id, workflow);
                WorkflowManager.setActiveWorkflow(workflow.id);
                WorkflowManager.updateWorkflowList();
              }
            } catch (err) {
              alert("Invalid workflow file");
            }
          };
          reader.readAsText(file);
        }
      };
      input.click();
    },
  };
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", function () {
  if (typeof WorkflowManager !== "undefined") {
    WorkflowManager.init();
  }
});
