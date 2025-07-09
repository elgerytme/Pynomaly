(()=>{var d=(c,t)=>()=>(t||c((t={exports:{}}).exports,t),t.exports);var u=d((v,l)=>{var r=class{constructor(t,a={}){this.container=t,this.options={enableAdvancedOptions:!0,showExpertMode:!1,enableTemplates:!0,...a},this.currentStep=0,this.totalSteps=5,this.config={},this.templates=this.getConfigTemplates(),this.eventListeners=new Map,this.init()}init(){this.createWizardStructure(),this.setupEventListeners(),this.showStep(0)}createWizardStructure(){this.container.innerHTML=`
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
    `,this.wizardContent=this.container.querySelector(".wizard-content"),this.prevButton=this.container.querySelector(".wizard-prev"),this.nextButton=this.container.querySelector(".wizard-next"),this.finishButton=this.container.querySelector(".wizard-finish"),this.progressFill=this.container.querySelector(".progress-fill"),this.progressText=this.container.querySelector(".progress-text")}setupEventListeners(){this.prevButton.addEventListener("click",()=>this.previousStep()),this.nextButton.addEventListener("click",()=>this.nextStep()),this.finishButton.addEventListener("click",()=>this.finishWizard())}showStep(t){switch(this.currentStep=t,this.updateProgress(),this.updateButtons(),t){case 0:this.showDatasetStep();break;case 1:this.showTemplateStep();break;case 2:this.showAlgorithmStep();break;case 3:this.showOptimizationStep();break;case 4:this.showSummaryStep();break}}showDatasetStep(){this.wizardContent.innerHTML=`
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
    `,this.setupDatasetStepListeners()}setupDatasetStepListeners(){let t=this.wizardContent.querySelector("#dataset-file"),a=this.wizardContent.querySelector("#data-preview");t?.addEventListener("change",i=>{let e=i.target.files[0];e&&this.loadDatasetPreview(e,a)})}async loadDatasetPreview(t,a){try{a.innerHTML='<div class="loading-spinner">Loading preview...</div>',await new Promise(e=>setTimeout(e,1e3));let i=this.generateMockDataPreview();a.innerHTML=`
        <div class="data-preview">
          <div class="preview-stats">
            <div class="stat">
              <span class="stat-value">${i.rows}</span>
              <span class="stat-label">Rows</span>
            </div>
            <div class="stat">
              <span class="stat-value">${i.columns}</span>
              <span class="stat-label">Columns</span>
            </div>
            <div class="stat">
              <span class="stat-value">${i.missing}%</span>
              <span class="stat-label">Missing</span>
            </div>
          </div>
          <div class="preview-table">
            <table class="table table-sm">
              <thead>
                <tr>
                  ${i.headers.map(e=>`<th>${e}</th>`).join("")}
                </tr>
              </thead>
              <tbody>
                ${i.rows_data.map(e=>`<tr>${e.map(s=>`<td>${s}</td>`).join("")}</tr>`).join("")}
              </tbody>
            </table>
          </div>
        </div>
      `,this.config.dataset={filename:t.name,size:t.size,rows:i.rows,columns:i.columns,preview:i}}catch(i){a.innerHTML=`<div class="error-message">Error loading file: ${i.message}</div>`}}showTemplateStep(){this.wizardContent.innerHTML=`
      <div class="wizard-step" data-step="1">
        <h3>Configuration Template</h3>
        <p>Choose a pre-configured template or start with custom settings.</p>
        
        <div class="template-grid">
          ${this.templates.map(t=>`
            <div class="template-card" data-template="${t.id}">
              <div class="template-header">
                <h4>${t.name}</h4>
                <span class="template-badge ${t.complexity}">${t.complexity}</span>
              </div>
              <p class="template-description">${t.description}</p>
              <div class="template-features">
                <h5>Features:</h5>
                <ul>
                  ${t.features.map(a=>`<li>${a}</li>`).join("")}
                </ul>
              </div>
              <div class="template-specs">
                <div class="spec">
                  <span class="spec-label">Training Time:</span>
                  <span class="spec-value">${t.estimatedTime}</span>
                </div>
                <div class="spec">
                  <span class="spec-label">Algorithms:</span>
                  <span class="spec-value">${t.algorithms.length}</span>
                </div>
              </div>
            </div>
          `).join("")}
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
    `,this.setupTemplateStepListeners()}setupTemplateStepListeners(){let t=this.wizardContent.querySelectorAll(".template-card");t.forEach(a=>{a.addEventListener("click",()=>{t.forEach(e=>e.classList.remove("selected")),a.classList.add("selected");let i=a.dataset.template;if(i==="custom")this.config.template="custom";else{let e=this.templates.find(s=>s.id===i);this.config.template=e,this.applyTemplate(e)}})})}showAlgorithmStep(){this.wizardContent.innerHTML=`
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
    `,this.setupAlgorithmStepListeners()}setupAlgorithmStepListeners(){let t=this.wizardContent.querySelectorAll('.algorithm-item input[type="checkbox"]'),a=this.wizardContent.querySelectorAll('.category > h4 input[type="checkbox"]'),i=()=>{let e=this.wizardContent.querySelectorAll('.algorithm-item input[type="checkbox"]:checked').length,s=e*5;this.wizardContent.querySelector("#selected-algorithms").textContent=e,this.wizardContent.querySelector("#estimated-time").textContent=`${s}min`;let n=Array.from(t).filter(o=>o.checked).map(o=>o.id.replace("alg-","").replace("-","_"));this.config.algorithms=n};a.forEach(e=>{e.addEventListener("change",()=>{let s=e.id.replace("cat-","");this.wizardContent.querySelectorAll(`[data-category="${s}"] input[type="checkbox"]`).forEach(o=>{o.checked=e.checked}),i()})}),t.forEach(e=>{e.addEventListener("change",i)}),i()}showOptimizationStep(){this.wizardContent.innerHTML=`
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
    `,this.setupOptimizationStepListeners()}setupOptimizationStepListeners(){let t=this.wizardContent.querySelector("#max-evaluations"),a=this.wizardContent.querySelector("#eval-value");t.addEventListener("input",()=>{a.textContent=t.value,this.updateEstimations()}),this.wizardContent.querySelectorAll("input, select").forEach(e=>{e.addEventListener("change",()=>{this.updateOptimizationConfig(),this.updateEstimations()})}),this.updateOptimizationConfig(),this.updateEstimations()}updateOptimizationConfig(){let t=a=>{let i=this.wizardContent.querySelector(`#${a}`);return i.type==="checkbox"?i.checked:i.type==="number"||i.type==="range"?parseInt(i.value):i.value};this.config.optimization={algorithm:t("optimization-algorithm"),max_evaluations:t("max-evaluations"),timeout_minutes:t("optimization-timeout"),cv_folds:t("cv-folds"),scoring_metric:t("scoring-metric"),max_training_time:t("max-training-time"),memory_limit:t("memory-limit"),gpu_enabled:t("gpu-enabled"),n_jobs:t("n-jobs")}}updateEstimations(){let t=this.config.algorithms?.length||5,a=this.config.optimization?.max_evaluations||100,i=this.config.optimization?.n_jobs==="-1"?4:parseInt(this.config.optimization?.n_jobs||1),e=t*a,s=Math.ceil(e*2/i),n=Math.floor(s/60),o=s%60,m=n>0?`~${n}h ${o}m`:`~${s}m`,p=Math.min(this.config.optimization?.memory_limit||8,t*1.5);this.wizardContent.querySelector("#estimated-duration").textContent=m,this.wizardContent.querySelector("#estimated-memory").textContent=`~${p.toFixed(1)} GB`,this.wizardContent.querySelector("#estimated-trials").textContent=`~${e}`}showSummaryStep(){let t=this.generateConfigSummary();this.wizardContent.innerHTML=`
      <div class="wizard-step" data-step="4">
        <h3>Configuration Summary</h3>
        <p>Review your AutoML configuration before starting the training process.</p>
        
        <div class="summary-grid">
          <div class="summary-section">
            <h4>Dataset</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Source:</span>
                <span class="summary-value">${t.dataset.source}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Samples:</span>
                <span class="summary-value">${t.dataset.samples}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Features:</span>
                <span class="summary-value">${t.dataset.features}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Target:</span>
                <span class="summary-value">${t.dataset.target}</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Algorithms</h4>
            <div class="summary-content">
              <div class="algorithm-chips">
                ${t.algorithms.map(a=>`<span class="algorithm-chip">${a}</span>`).join("")}
              </div>
              <div class="summary-item">
                <span class="summary-label">Total:</span>
                <span class="summary-value">${t.algorithms.length} algorithms</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Optimization</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Algorithm:</span>
                <span class="summary-value">${t.optimization.algorithm}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Max Evaluations:</span>
                <span class="summary-value">${t.optimization.evaluations}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">CV Folds:</span>
                <span class="summary-value">${t.optimization.cv_folds}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Scoring:</span>
                <span class="summary-value">${t.optimization.scoring}</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Resources</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Max Time:</span>
                <span class="summary-value">${t.resources.max_time}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Memory Limit:</span>
                <span class="summary-value">${t.resources.memory}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Parallel Jobs:</span>
                <span class="summary-value">${t.resources.parallel_jobs}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">GPU:</span>
                <span class="summary-value">${t.resources.gpu?"Enabled":"Disabled"}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="estimation-summary">
          <h4>Training Estimation</h4>
          <div class="estimation-highlight">
            <div class="estimation-main">
              <span class="estimation-duration">${t.estimation.duration}</span>
              <span class="estimation-label">Estimated Training Time</span>
            </div>
            <div class="estimation-details">
              <span>~${t.estimation.trials} total trials</span>
              <span>~${t.estimation.memory} memory usage</span>
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
    `,this.setupSummaryStepListeners()}setupSummaryStepListeners(){let t=this.wizardContent.querySelector("#export-config"),a=this.wizardContent.querySelector("#save-template");t?.addEventListener("click",()=>{this.exportConfiguration()}),a?.addEventListener("click",()=>{this.saveAsTemplate()})}generateConfigSummary(){return{dataset:{source:this.config.dataset?.filename||"Not specified",samples:this.config.dataset?.rows||"Unknown",features:this.config.dataset?.columns||"Unknown",target:"Auto-detect (unsupervised)"},algorithms:this.config.algorithms||[],optimization:{algorithm:this.config.optimization?.algorithm||"bayesian",evaluations:this.config.optimization?.max_evaluations||100,cv_folds:this.config.optimization?.cv_folds||5,scoring:this.config.optimization?.scoring_metric||"roc_auc"},resources:{max_time:`${this.config.optimization?.max_training_time||120} minutes`,memory:`${this.config.optimization?.memory_limit||8} GB`,parallel_jobs:this.config.optimization?.n_jobs==="-1"?"All cores":this.config.optimization?.n_jobs||1,gpu:this.config.optimization?.gpu_enabled||!1},estimation:{duration:"~2-3 hours",trials:"500",memory:"4-6 GB"}}}updateProgress(){let t=(this.currentStep+1)/this.totalSteps*100;this.progressFill.style.width=`${t}%`,this.progressText.textContent=`Step ${this.currentStep+1} of ${this.totalSteps}`}updateButtons(){this.prevButton.disabled=this.currentStep===0,this.nextButton.style.display=this.currentStep===this.totalSteps-1?"none":"inline-block",this.finishButton.style.display=this.currentStep===this.totalSteps-1?"inline-block":"none"}nextStep(){this.validateCurrentStep()&&this.currentStep<this.totalSteps-1&&this.showStep(this.currentStep+1)}previousStep(){this.currentStep>0&&this.showStep(this.currentStep-1)}validateCurrentStep(){switch(this.currentStep){case 0:return this.config.dataset||!0;case 1:return this.config.template!==void 0;case 2:return this.config.algorithms&&this.config.algorithms.length>0;case 3:return this.config.optimization!==void 0;default:return!0}}finishWizard(){let t=this.buildFinalConfig();this.emit("wizard-complete",{config:t})}buildFinalConfig(){return{dataset:this.config.dataset,model_search:{algorithms:this.config.algorithms,max_trials:50,early_stopping:!0},hyperparameter_optimization:{algorithm:this.config.optimization?.algorithm||"bayesian",max_evaluations:this.config.optimization?.max_evaluations||100,timeout_minutes:this.config.optimization?.timeout_minutes||60,cv_folds:this.config.optimization?.cv_folds||5,scoring_metric:this.config.optimization?.scoring_metric||"roc_auc"},performance:{max_training_time_minutes:this.config.optimization?.max_training_time||120,memory_limit_gb:this.config.optimization?.memory_limit||8,gpu_enabled:this.config.optimization?.gpu_enabled||!1,n_jobs:this.config.optimization?.n_jobs||-1},ensemble:{enable:!0,strategy:"ensemble",max_models:5},validation:{test_size:.2,cross_validation:!0,cv_folds:this.config.optimization?.cv_folds||5}}}getConfigTemplates(){return[{id:"quick",name:"Quick Start",complexity:"beginner",description:"Fast anomaly detection with basic algorithms and minimal tuning.",features:["Fast training (~30 minutes)","Basic algorithms","Minimal resource usage","Good baseline performance"],algorithms:["isolation_forest","local_outlier_factor"],estimatedTime:"30 minutes"},{id:"balanced",name:"Balanced Performance",complexity:"intermediate",description:"Balance between training time and model performance with moderate tuning.",features:["Moderate training (~2 hours)","Multiple algorithms","Hyperparameter optimization","Ensemble methods"],algorithms:["isolation_forest","local_outlier_factor","one_class_svm","autoencoder"],estimatedTime:"2 hours"},{id:"comprehensive",name:"Comprehensive Search",complexity:"advanced",description:"Exhaustive search across all algorithms for maximum performance.",features:["Extensive training (~6 hours)","All available algorithms","Advanced optimization","Neural networks included"],algorithms:["isolation_forest","local_outlier_factor","one_class_svm","autoencoder","deep_svdd","feature_bagging","copod"],estimatedTime:"6 hours"},{id:"neural",name:"Neural Network Focus",complexity:"advanced",description:"Focus on deep learning approaches for complex pattern detection.",features:["GPU acceleration","Deep learning algorithms","Advanced feature learning","Complex pattern detection"],algorithms:["autoencoder","deep_svdd"],estimatedTime:"4 hours"}]}generateMockDataPreview(){return{rows:1e4+Math.floor(Math.random()*5e4),columns:8+Math.floor(Math.random()*15),missing:Math.floor(Math.random()*15),headers:["timestamp","sensor_1","sensor_2","temperature","pressure","flow_rate"],rows_data:[["2024-01-01 10:00:00","0.823","1.234","25.4","101.3","15.2"],["2024-01-01 10:01:00","0.801","1.189","25.7","101.2","15.8"],["2024-01-01 10:02:00","0.856","1.267","25.1","101.4","14.9"],["2024-01-01 10:03:00","0.798","1.145","25.9","101.1","16.1"]]}}applyTemplate(t){this.config.algorithms=[...t.algorithms]}exportConfiguration(){let t=this.buildFinalConfig(),a=new Blob([JSON.stringify(t,null,2)],{type:"application/json"}),i=URL.createObjectURL(a),e=document.createElement("a");e.href=i,e.download=`automl-config-${Date.now()}.json`,document.body.appendChild(e),e.click(),document.body.removeChild(e),URL.revokeObjectURL(i)}saveAsTemplate(){let t=prompt("Enter template name:");t&&console.log("Saving template:",t,this.buildFinalConfig())}on(t,a){return this.eventListeners.has(t)||this.eventListeners.set(t,new Set),this.eventListeners.get(t).add(a),()=>this.off(t,a)}off(t,a){this.eventListeners.has(t)&&this.eventListeners.get(t).delete(a)}emit(t,a){this.eventListeners.has(t)&&this.eventListeners.get(t).forEach(i=>{try{i(a)}catch(e){console.error("AutoML wizard event error:",e)}})}};typeof l<"u"&&l.exports?l.exports={AutoMLConfigWizard:r}:window.AutoMLConfigWizard=r});u();})();
