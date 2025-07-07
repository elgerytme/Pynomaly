/**
 * AutoML Interface Component
 * User interface for automated model training and hyperparameter optimization
 * Provides intuitive workflow for enterprise machine learning automation
 */

/**
 * AutoML Configuration Wizard
 * Guides users through AutoML setup with intelligent defaults
 */
class AutoMLConfigWizard {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      enableAdvancedOptions: true,
      showExpertMode: false,
      enableTemplates: true,
      ...options,
    };

    this.currentStep = 0;
    this.totalSteps = 5;
    this.config = {};
    this.templates = this.getConfigTemplates();

    this.eventListeners = new Map();
    this.init();
  }

  init() {
    this.createWizardStructure();
    this.setupEventListeners();
    this.showStep(0);
  }

  createWizardStructure() {
    this.container.innerHTML = `
      <div class="automl-wizard">
        <div class="wizard-header">
          <h2 class="wizard-title">AutoML Configuration Wizard</h2>
          <div class="wizard-progress">
            <div class="progress-bar">
              <div class="progress-fill" style="width: 0%"></div>
            </div>
            <span class="progress-text">Step 1 of ${this.totalSteps}</span>
          </div>
        </div>

        <div class="wizard-content">
          <!-- Step content will be dynamically inserted here -->
        </div>

        <div class="wizard-actions">
          <button class="btn btn-secondary wizard-prev" disabled>Previous</button>
          <button class="btn btn-primary wizard-next">Next</button>
          <button class="btn btn-success wizard-finish" style="display: none;">Start AutoML</button>
        </div>
      </div>
    `;

    this.wizardContent = this.container.querySelector(".wizard-content");
    this.prevButton = this.container.querySelector(".wizard-prev");
    this.nextButton = this.container.querySelector(".wizard-next");
    this.finishButton = this.container.querySelector(".wizard-finish");
    this.progressFill = this.container.querySelector(".progress-fill");
    this.progressText = this.container.querySelector(".progress-text");
  }

  setupEventListeners() {
    this.prevButton.addEventListener("click", () => this.previousStep());
    this.nextButton.addEventListener("click", () => this.nextStep());
    this.finishButton.addEventListener("click", () => this.finishWizard());
  }

  showStep(stepIndex) {
    this.currentStep = stepIndex;
    this.updateProgress();
    this.updateButtons();

    switch (stepIndex) {
      case 0:
        this.showDatasetStep();
        break;
      case 1:
        this.showTemplateStep();
        break;
      case 2:
        this.showAlgorithmStep();
        break;
      case 3:
        this.showOptimizationStep();
        break;
      case 4:
        this.showSummaryStep();
        break;
    }
  }

  showDatasetStep() {
    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="0">
        <h3>Dataset Configuration</h3>
        <p>Configure your dataset and target variable for anomaly detection.</p>
        
        <div class="form-group">
          <label for="dataset-source">Data Source</label>
          <select id="dataset-source" class="form-control">
            <option value="upload">Upload CSV File</option>
            <option value="database">Database Connection</option>
            <option value="api">API Endpoint</option>
            <option value="streaming">Real-time Stream</option>
          </select>
        </div>

        <div class="form-group" id="file-upload-group">
          <label for="dataset-file">Upload Dataset</label>
          <input type="file" id="dataset-file" class="form-control" accept=".csv,.json,.parquet">
          <small class="form-text text-muted">Supported formats: CSV, JSON, Parquet</small>
        </div>

        <div class="form-group">
          <label for="target-column">Target Column (Optional)</label>
          <select id="target-column" class="form-control">
            <option value="">Auto-detect anomalies (unsupervised)</option>
            <option value="is_anomaly">is_anomaly</option>
            <option value="label">label</option>
            <option value="target">target</option>
            <option value="outlier">outlier</option>
          </select>
          <small class="form-text text-muted">Leave empty for unsupervised anomaly detection</small>
        </div>

        <div class="form-group">
          <label for="data-preview">Data Preview</label>
          <div id="data-preview" class="data-preview-container">
            <div class="preview-placeholder">
              <i class="fas fa-upload"></i>
              <p>Upload a dataset to see preview</p>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupDatasetStepListeners();
  }

  setupDatasetStepListeners() {
    const fileInput = this.wizardContent.querySelector("#dataset-file");
    const dataPreview = this.wizardContent.querySelector("#data-preview");

    fileInput?.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) {
        this.loadDatasetPreview(file, dataPreview);
      }
    });
  }

  async loadDatasetPreview(file, container) {
    try {
      container.innerHTML =
        '<div class="loading-spinner">Loading preview...</div>';

      // Simulate file reading and preview generation
      await new Promise((resolve) => setTimeout(resolve, 1000));

      const mockPreview = this.generateMockDataPreview();
      container.innerHTML = `
        <div class="data-preview">
          <div class="preview-stats">
            <div class="stat">
              <span class="stat-value">${mockPreview.rows}</span>
              <span class="stat-label">Rows</span>
            </div>
            <div class="stat">
              <span class="stat-value">${mockPreview.columns}</span>
              <span class="stat-label">Columns</span>
            </div>
            <div class="stat">
              <span class="stat-value">${mockPreview.missing}%</span>
              <span class="stat-label">Missing</span>
            </div>
          </div>
          <div class="preview-table">
            <table class="table table-sm">
              <thead>
                <tr>
                  ${mockPreview.headers.map((h) => `<th>${h}</th>`).join("")}
                </tr>
              </thead>
              <tbody>
                ${mockPreview.rows_data
                  .map(
                    (row) =>
                      `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`,
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>
      `;

      // Store dataset info in config
      this.config.dataset = {
        filename: file.name,
        size: file.size,
        rows: mockPreview.rows,
        columns: mockPreview.columns,
        preview: mockPreview,
      };
    } catch (error) {
      container.innerHTML = `<div class="error-message">Error loading file: ${error.message}</div>`;
    }
  }

  showTemplateStep() {
    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="1">
        <h3>Configuration Template</h3>
        <p>Choose a pre-configured template or start with custom settings.</p>
        
        <div class="template-grid">
          ${this.templates
            .map(
              (template) => `
            <div class="template-card" data-template="${template.id}">
              <div class="template-header">
                <h4>${template.name}</h4>
                <span class="template-badge ${template.complexity}">${template.complexity}</span>
              </div>
              <p class="template-description">${template.description}</p>
              <div class="template-features">
                <h5>Features:</h5>
                <ul>
                  ${template.features.map((feature) => `<li>${feature}</li>`).join("")}
                </ul>
              </div>
              <div class="template-specs">
                <div class="spec">
                  <span class="spec-label">Training Time:</span>
                  <span class="spec-value">${template.estimatedTime}</span>
                </div>
                <div class="spec">
                  <span class="spec-label">Algorithms:</span>
                  <span class="spec-value">${template.algorithms.length}</span>
                </div>
              </div>
            </div>
          `,
            )
            .join("")}
        </div>

        <div class="custom-option">
          <div class="template-card custom-template" data-template="custom">
            <div class="template-header">
              <h4>Custom Configuration</h4>
              <span class="template-badge expert">Expert</span>
            </div>
            <p class="template-description">Create a custom configuration with full control over all parameters.</p>
            <div class="template-features">
              <h5>Features:</h5>
              <ul>
                <li>Full parameter control</li>
                <li>Advanced optimization</li>
                <li>Custom metrics</li>
                <li>Expert tuning</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupTemplateStepListeners();
  }

  setupTemplateStepListeners() {
    const templateCards = this.wizardContent.querySelectorAll(".template-card");

    templateCards.forEach((card) => {
      card.addEventListener("click", () => {
        // Remove previous selection
        templateCards.forEach((c) => c.classList.remove("selected"));

        // Select current template
        card.classList.add("selected");

        const templateId = card.dataset.template;
        if (templateId === "custom") {
          this.config.template = "custom";
        } else {
          const template = this.templates.find((t) => t.id === templateId);
          this.config.template = template;
          this.applyTemplate(template);
        }
      });
    });
  }

  showAlgorithmStep() {
    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="2">
        <h3>Algorithm Selection</h3>
        <p>Choose which anomaly detection algorithms to include in the AutoML search.</p>
        
        <div class="algorithm-categories">
          <div class="category">
            <h4>
              <input type="checkbox" id="cat-statistical" checked>
              <label for="cat-statistical">Statistical Methods</label>
            </h4>
            <div class="algorithm-list" data-category="statistical">
              <div class="algorithm-item">
                <input type="checkbox" id="alg-isolation-forest" checked>
                <label for="alg-isolation-forest">
                  <span class="algorithm-name">Isolation Forest</span>
                  <span class="algorithm-description">Tree-based ensemble for outlier detection</span>
                  <span class="algorithm-complexity">Fast</span>
                </label>
              </div>
              <div class="algorithm-item">
                <input type="checkbox" id="alg-lof" checked>
                <label for="alg-lof">
                  <span class="algorithm-name">Local Outlier Factor</span>
                  <span class="algorithm-description">Density-based local outlier detection</span>
                  <span class="algorithm-complexity">Medium</span>
                </label>
              </div>
              <div class="algorithm-item">
                <input type="checkbox" id="alg-elliptic-envelope">
                <label for="alg-elliptic-envelope">
                  <span class="algorithm-name">Elliptic Envelope</span>
                  <span class="algorithm-description">Gaussian distribution assumption</span>
                  <span class="algorithm-complexity">Fast</span>
                </label>
              </div>
            </div>
          </div>

          <div class="category">
            <h4>
              <input type="checkbox" id="cat-neural" checked>
              <label for="cat-neural">Neural Networks</label>
            </h4>
            <div class="algorithm-list" data-category="neural">
              <div class="algorithm-item">
                <input type="checkbox" id="alg-autoencoder" checked>
                <label for="alg-autoencoder">
                  <span class="algorithm-name">Autoencoder</span>
                  <span class="algorithm-description">Neural network reconstruction error</span>
                  <span class="algorithm-complexity">Slow</span>
                </label>
              </div>
              <div class="algorithm-item">
                <input type="checkbox" id="alg-deep-svdd">
                <label for="alg-deep-svdd">
                  <span class="algorithm-name">Deep SVDD</span>
                  <span class="algorithm-description">Deep one-class classification</span>
                  <span class="algorithm-complexity">Slow</span>
                </label>
              </div>
            </div>
          </div>

          <div class="category">
            <h4>
              <input type="checkbox" id="cat-ensemble" checked>
              <label for="cat-ensemble">Ensemble Methods</label>
            </h4>
            <div class="algorithm-list" data-category="ensemble">
              <div class="algorithm-item">
                <input type="checkbox" id="alg-feature-bagging" checked>
                <label for="alg-feature-bagging">
                  <span class="algorithm-name">Feature Bagging</span>
                  <span class="algorithm-description">Ensemble of base detectors</span>
                  <span class="algorithm-complexity">Medium</span>
                </label>
              </div>
              <div class="algorithm-item">
                <input type="checkbox" id="alg-copod">
                <label for="alg-copod">
                  <span class="algorithm-name">COPOD</span>
                  <span class="algorithm-description">Copula-based outlier detection</span>
                  <span class="algorithm-complexity">Fast</span>
                </label>
              </div>
            </div>
          </div>
        </div>

        <div class="algorithm-summary">
          <h4>Selection Summary</h4>
          <div class="summary-stats">
            <div class="stat">
              <span class="stat-value" id="selected-algorithms">0</span>
              <span class="stat-label">Selected Algorithms</span>
            </div>
            <div class="stat">
              <span class="stat-value" id="estimated-time">0</span>
              <span class="stat-label">Est. Training Time</span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupAlgorithmStepListeners();
  }

  setupAlgorithmStepListeners() {
    const algorithmCheckboxes = this.wizardContent.querySelectorAll(
      '.algorithm-item input[type="checkbox"]',
    );
    const categoryCheckboxes = this.wizardContent.querySelectorAll(
      '.category > h4 input[type="checkbox"]',
    );

    // Update selection summary
    const updateSummary = () => {
      const selectedCount = this.wizardContent.querySelectorAll(
        '.algorithm-item input[type="checkbox"]:checked',
      ).length;
      const estimatedTime = selectedCount * 5; // 5 minutes per algorithm estimate

      this.wizardContent.querySelector("#selected-algorithms").textContent =
        selectedCount;
      this.wizardContent.querySelector("#estimated-time").textContent =
        `${estimatedTime}min`;

      // Store selected algorithms in config
      const selectedAlgorithms = Array.from(algorithmCheckboxes)
        .filter((cb) => cb.checked)
        .map((cb) => cb.id.replace("alg-", "").replace("-", "_"));

      this.config.algorithms = selectedAlgorithms;
    };

    // Category checkbox handlers
    categoryCheckboxes.forEach((catCheckbox) => {
      catCheckbox.addEventListener("change", () => {
        const category = catCheckbox.id.replace("cat-", "");
        const categoryAlgorithms = this.wizardContent.querySelectorAll(
          `[data-category="${category}"] input[type="checkbox"]`,
        );

        categoryAlgorithms.forEach((algCheckbox) => {
          algCheckbox.checked = catCheckbox.checked;
        });

        updateSummary();
      });
    });

    // Algorithm checkbox handlers
    algorithmCheckboxes.forEach((algCheckbox) => {
      algCheckbox.addEventListener("change", updateSummary);
    });

    // Initial summary update
    updateSummary();
  }

  showOptimizationStep() {
    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="3">
        <h3>Optimization Settings</h3>
        <p>Configure hyperparameter optimization and resource limits.</p>
        
        <div class="optimization-grid">
          <div class="optimization-section">
            <h4>Hyperparameter Optimization</h4>
            
            <div class="form-group">
              <label for="optimization-algorithm">Optimization Algorithm</label>
              <select id="optimization-algorithm" class="form-control">
                <option value="bayesian" selected>Bayesian Optimization (Recommended)</option>
                <option value="random_search">Random Search</option>
                <option value="grid_search">Grid Search</option>
                <option value="evolutionary">Evolutionary Algorithm</option>
                <option value="optuna">Optuna TPE</option>
              </select>
            </div>

            <div class="form-group">
              <label for="max-evaluations">Maximum Evaluations</label>
              <input type="range" id="max-evaluations" class="form-control-range" 
                     min="10" max="500" value="100" step="10">
              <div class="range-labels">
                <span>10 (Fast)</span>
                <span id="eval-value">100</span>
                <span>500 (Thorough)</span>
              </div>
            </div>

            <div class="form-group">
              <label for="optimization-timeout">Timeout (minutes)</label>
              <input type="number" id="optimization-timeout" class="form-control" 
                     value="60" min="5" max="480">
            </div>
          </div>

          <div class="optimization-section">
            <h4>Cross-Validation</h4>
            
            <div class="form-group">
              <label for="cv-folds">Cross-Validation Folds</label>
              <select id="cv-folds" class="form-control">
                <option value="3">3-Fold (Fast)</option>
                <option value="5" selected>5-Fold (Recommended)</option>
                <option value="10">10-Fold (Thorough)</option>
              </select>
            </div>

            <div class="form-group">
              <label for="scoring-metric">Scoring Metric</label>
              <select id="scoring-metric" class="form-control">
                <option value="roc_auc" selected>ROC AUC</option>
                <option value="average_precision">Average Precision</option>
                <option value="f1_score">F1 Score</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
              </select>
            </div>
          </div>

          <div class="optimization-section">
            <h4>Resource Limits</h4>
            
            <div class="form-group">
              <label for="max-training-time">Max Training Time (minutes)</label>
              <input type="number" id="max-training-time" class="form-control" 
                     value="120" min="10" max="1440">
            </div>

            <div class="form-group">
              <label for="memory-limit">Memory Limit (GB)</label>
              <input type="number" id="memory-limit" class="form-control" 
                     value="8" min="1" max="64" step="1">
            </div>

            <div class="form-group">
              <div class="form-check">
                <input type="checkbox" id="gpu-enabled" class="form-check-input">
                <label for="gpu-enabled" class="form-check-label">
                  Enable GPU Acceleration (if available)
                </label>
              </div>
            </div>

            <div class="form-group">
              <label for="n-jobs">Parallel Jobs</label>
              <select id="n-jobs" class="form-control">
                <option value="1">1 (Single-threaded)</option>
                <option value="2">2 cores</option>
                <option value="4">4 cores</option>
                <option value="-1" selected>All available cores</option>
              </select>
            </div>
          </div>
        </div>

        <div class="estimation-panel">
          <h4>Training Estimation</h4>
          <div class="estimation-grid">
            <div class="estimation-item">
              <span class="estimation-label">Estimated Duration:</span>
              <span class="estimation-value" id="estimated-duration">~2-3 hours</span>
            </div>
            <div class="estimation-item">
              <span class="estimation-label">Memory Usage:</span>
              <span class="estimation-value" id="estimated-memory">~4-6 GB</span>
            </div>
            <div class="estimation-item">
              <span class="estimation-label">Total Trials:</span>
              <span class="estimation-value" id="estimated-trials">~500</span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupOptimizationStepListeners();
  }

  setupOptimizationStepListeners() {
    const maxEvaluations = this.wizardContent.querySelector("#max-evaluations");
    const evalValue = this.wizardContent.querySelector("#eval-value");

    maxEvaluations.addEventListener("input", () => {
      evalValue.textContent = maxEvaluations.value;
      this.updateEstimations();
    });

    // Store optimization settings in config
    const formElements = this.wizardContent.querySelectorAll("input, select");
    formElements.forEach((element) => {
      element.addEventListener("change", () => {
        this.updateOptimizationConfig();
        this.updateEstimations();
      });
    });

    this.updateOptimizationConfig();
    this.updateEstimations();
  }

  updateOptimizationConfig() {
    const getValue = (id) => {
      const element = this.wizardContent.querySelector(`#${id}`);
      if (element.type === "checkbox") return element.checked;
      if (element.type === "number" || element.type === "range")
        return parseInt(element.value);
      return element.value;
    };

    this.config.optimization = {
      algorithm: getValue("optimization-algorithm"),
      max_evaluations: getValue("max-evaluations"),
      timeout_minutes: getValue("optimization-timeout"),
      cv_folds: getValue("cv-folds"),
      scoring_metric: getValue("scoring-metric"),
      max_training_time: getValue("max-training-time"),
      memory_limit: getValue("memory-limit"),
      gpu_enabled: getValue("gpu-enabled"),
      n_jobs: getValue("n-jobs"),
    };
  }

  updateEstimations() {
    const algorithms = this.config.algorithms?.length || 5;
    const evaluations = this.config.optimization?.max_evaluations || 100;
    const parallelJobs =
      this.config.optimization?.n_jobs === "-1"
        ? 4
        : parseInt(this.config.optimization?.n_jobs || 1);

    // Rough estimation formulas
    const totalTrials = algorithms * evaluations;
    const estimatedMinutes = Math.ceil((totalTrials * 2) / parallelJobs); // 2 minutes per trial
    const estimatedHours = Math.floor(estimatedMinutes / 60);
    const remainingMinutes = estimatedMinutes % 60;

    const durationText =
      estimatedHours > 0
        ? `~${estimatedHours}h ${remainingMinutes}m`
        : `~${estimatedMinutes}m`;

    const memoryUsage = Math.min(
      this.config.optimization?.memory_limit || 8,
      algorithms * 1.5,
    );

    this.wizardContent.querySelector("#estimated-duration").textContent =
      durationText;
    this.wizardContent.querySelector("#estimated-memory").textContent =
      `~${memoryUsage.toFixed(1)} GB`;
    this.wizardContent.querySelector("#estimated-trials").textContent =
      `~${totalTrials}`;
  }

  showSummaryStep() {
    const configSummary = this.generateConfigSummary();

    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="4">
        <h3>Configuration Summary</h3>
        <p>Review your AutoML configuration before starting the training process.</p>
        
        <div class="summary-grid">
          <div class="summary-section">
            <h4>Dataset</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Source:</span>
                <span class="summary-value">${configSummary.dataset.source}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Samples:</span>
                <span class="summary-value">${configSummary.dataset.samples}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Features:</span>
                <span class="summary-value">${configSummary.dataset.features}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Target:</span>
                <span class="summary-value">${configSummary.dataset.target}</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Algorithms</h4>
            <div class="summary-content">
              <div class="algorithm-chips">
                ${configSummary.algorithms
                  .map((alg) => `<span class="algorithm-chip">${alg}</span>`)
                  .join("")}
              </div>
              <div class="summary-item">
                <span class="summary-label">Total:</span>
                <span class="summary-value">${configSummary.algorithms.length} algorithms</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Optimization</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Algorithm:</span>
                <span class="summary-value">${configSummary.optimization.algorithm}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Max Evaluations:</span>
                <span class="summary-value">${configSummary.optimization.evaluations}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">CV Folds:</span>
                <span class="summary-value">${configSummary.optimization.cv_folds}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Scoring:</span>
                <span class="summary-value">${configSummary.optimization.scoring}</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Resources</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Max Time:</span>
                <span class="summary-value">${configSummary.resources.max_time}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Memory Limit:</span>
                <span class="summary-value">${configSummary.resources.memory}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Parallel Jobs:</span>
                <span class="summary-value">${configSummary.resources.parallel_jobs}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">GPU:</span>
                <span class="summary-value">${configSummary.resources.gpu ? "Enabled" : "Disabled"}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="estimation-summary">
          <h4>Training Estimation</h4>
          <div class="estimation-highlight">
            <div class="estimation-main">
              <span class="estimation-duration">${configSummary.estimation.duration}</span>
              <span class="estimation-label">Estimated Training Time</span>
            </div>
            <div class="estimation-details">
              <span>~${configSummary.estimation.trials} total trials</span>
              <span>~${configSummary.estimation.memory} memory usage</span>
            </div>
          </div>
        </div>

        <div class="configuration-export">
          <h4>Configuration Export</h4>
          <div class="export-actions">
            <button class="btn btn-outline-secondary" id="export-config">
              <i class="fas fa-download"></i> Export Configuration
            </button>
            <button class="btn btn-outline-secondary" id="save-template">
              <i class="fas fa-save"></i> Save as Template
            </button>
          </div>
        </div>
      </div>
    `;

    this.setupSummaryStepListeners();
  }

  setupSummaryStepListeners() {
    const exportButton = this.wizardContent.querySelector("#export-config");
    const saveTemplateButton =
      this.wizardContent.querySelector("#save-template");

    exportButton?.addEventListener("click", () => {
      this.exportConfiguration();
    });

    saveTemplateButton?.addEventListener("click", () => {
      this.saveAsTemplate();
    });
  }

  generateConfigSummary() {
    return {
      dataset: {
        source: this.config.dataset?.filename || "Not specified",
        samples: this.config.dataset?.rows || "Unknown",
        features: this.config.dataset?.columns || "Unknown",
        target: "Auto-detect (unsupervised)",
      },
      algorithms: this.config.algorithms || [],
      optimization: {
        algorithm: this.config.optimization?.algorithm || "bayesian",
        evaluations: this.config.optimization?.max_evaluations || 100,
        cv_folds: this.config.optimization?.cv_folds || 5,
        scoring: this.config.optimization?.scoring_metric || "roc_auc",
      },
      resources: {
        max_time: `${this.config.optimization?.max_training_time || 120} minutes`,
        memory: `${this.config.optimization?.memory_limit || 8} GB`,
        parallel_jobs:
          this.config.optimization?.n_jobs === "-1"
            ? "All cores"
            : this.config.optimization?.n_jobs || 1,
        gpu: this.config.optimization?.gpu_enabled || false,
      },
      estimation: {
        duration: "~2-3 hours",
        trials: "500",
        memory: "4-6 GB",
      },
    };
  }

  updateProgress() {
    const progressPercent = ((this.currentStep + 1) / this.totalSteps) * 100;
    this.progressFill.style.width = `${progressPercent}%`;
    this.progressText.textContent = `Step ${this.currentStep + 1} of ${this.totalSteps}`;
  }

  updateButtons() {
    this.prevButton.disabled = this.currentStep === 0;
    this.nextButton.style.display =
      this.currentStep === this.totalSteps - 1 ? "none" : "inline-block";
    this.finishButton.style.display =
      this.currentStep === this.totalSteps - 1 ? "inline-block" : "none";
  }

  nextStep() {
    if (this.validateCurrentStep() && this.currentStep < this.totalSteps - 1) {
      this.showStep(this.currentStep + 1);
    }
  }

  previousStep() {
    if (this.currentStep > 0) {
      this.showStep(this.currentStep - 1);
    }
  }

  validateCurrentStep() {
    // Add validation logic for each step
    switch (this.currentStep) {
      case 0: // Dataset step
        return this.config.dataset || true; // Allow proceeding even without file for demo
      case 1: // Template step
        return this.config.template !== undefined;
      case 2: // Algorithm step
        return this.config.algorithms && this.config.algorithms.length > 0;
      case 3: // Optimization step
        return this.config.optimization !== undefined;
      default:
        return true;
    }
  }

  finishWizard() {
    const finalConfig = this.buildFinalConfig();
    this.emit("wizard-complete", { config: finalConfig });
  }

  buildFinalConfig() {
    // Build complete AutoML configuration
    return {
      dataset: this.config.dataset,
      model_search: {
        algorithms: this.config.algorithms,
        max_trials: 50,
        early_stopping: true,
      },
      hyperparameter_optimization: {
        algorithm: this.config.optimization?.algorithm || "bayesian",
        max_evaluations: this.config.optimization?.max_evaluations || 100,
        timeout_minutes: this.config.optimization?.timeout_minutes || 60,
        cv_folds: this.config.optimization?.cv_folds || 5,
        scoring_metric: this.config.optimization?.scoring_metric || "roc_auc",
      },
      performance: {
        max_training_time_minutes:
          this.config.optimization?.max_training_time || 120,
        memory_limit_gb: this.config.optimization?.memory_limit || 8,
        gpu_enabled: this.config.optimization?.gpu_enabled || false,
        n_jobs: this.config.optimization?.n_jobs || -1,
      },
      ensemble: {
        enable: true,
        strategy: "ensemble",
        max_models: 5,
      },
      validation: {
        test_size: 0.2,
        cross_validation: true,
        cv_folds: this.config.optimization?.cv_folds || 5,
      },
    };
  }

  getConfigTemplates() {
    return [
      {
        id: "quick",
        name: "Quick Start",
        complexity: "beginner",
        description:
          "Fast anomaly detection with basic algorithms and minimal tuning.",
        features: [
          "Fast training (~30 minutes)",
          "Basic algorithms",
          "Minimal resource usage",
          "Good baseline performance",
        ],
        algorithms: ["isolation_forest", "local_outlier_factor"],
        estimatedTime: "30 minutes",
      },
      {
        id: "balanced",
        name: "Balanced Performance",
        complexity: "intermediate",
        description:
          "Balance between training time and model performance with moderate tuning.",
        features: [
          "Moderate training (~2 hours)",
          "Multiple algorithms",
          "Hyperparameter optimization",
          "Ensemble methods",
        ],
        algorithms: [
          "isolation_forest",
          "local_outlier_factor",
          "one_class_svm",
          "autoencoder",
        ],
        estimatedTime: "2 hours",
      },
      {
        id: "comprehensive",
        name: "Comprehensive Search",
        complexity: "advanced",
        description:
          "Exhaustive search across all algorithms for maximum performance.",
        features: [
          "Extensive training (~6 hours)",
          "All available algorithms",
          "Advanced optimization",
          "Neural networks included",
        ],
        algorithms: [
          "isolation_forest",
          "local_outlier_factor",
          "one_class_svm",
          "autoencoder",
          "deep_svdd",
          "feature_bagging",
          "copod",
        ],
        estimatedTime: "6 hours",
      },
      {
        id: "neural",
        name: "Neural Network Focus",
        complexity: "advanced",
        description:
          "Focus on deep learning approaches for complex pattern detection.",
        features: [
          "GPU acceleration",
          "Deep learning algorithms",
          "Advanced feature learning",
          "Complex pattern detection",
        ],
        algorithms: ["autoencoder", "deep_svdd"],
        estimatedTime: "4 hours",
      },
    ];
  }

  generateMockDataPreview() {
    return {
      rows: 10000 + Math.floor(Math.random() * 50000),
      columns: 8 + Math.floor(Math.random() * 15),
      missing: Math.floor(Math.random() * 15),
      headers: [
        "timestamp",
        "sensor_1",
        "sensor_2",
        "temperature",
        "pressure",
        "flow_rate",
      ],
      rows_data: [
        ["2024-01-01 10:00:00", "0.823", "1.234", "25.4", "101.3", "15.2"],
        ["2024-01-01 10:01:00", "0.801", "1.189", "25.7", "101.2", "15.8"],
        ["2024-01-01 10:02:00", "0.856", "1.267", "25.1", "101.4", "14.9"],
        ["2024-01-01 10:03:00", "0.798", "1.145", "25.9", "101.1", "16.1"],
      ],
    };
  }

  applyTemplate(template) {
    this.config.algorithms = [...template.algorithms];
    // Apply other template-specific configurations
  }

  exportConfiguration() {
    const config = this.buildFinalConfig();
    const blob = new Blob([JSON.stringify(config, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `automl-config-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  saveAsTemplate() {
    // Implementation for saving custom template
    const templateName = prompt("Enter template name:");
    if (templateName) {
      // Save to local storage or send to server
      console.log("Saving template:", templateName, this.buildFinalConfig());
    }
  }

  // Event system
  on(event, listener) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event).add(listener);
    return () => this.off(event, listener);
  }

  off(event, listener) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).delete(listener);
    }
  }

  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          console.error("AutoML wizard event error:", error);
        }
      });
    }
  }
}

// Export class
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    AutoMLConfigWizard,
  };
} else {
  // Browser environment
  window.AutoMLConfigWizard = AutoMLConfigWizard;
}
