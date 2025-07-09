(()=>{var n=class{constructor(t,e={}){this.container=t,this.config={type:"scatter",width:800,height:400,margin:{top:20,right:20,bottom:40,left:40},...e},this.chart=null,this.data=e.data||[],this.init()}init(){this.setupContainer(),this.createChart()}setupContainer(){this.container.innerHTML="",this.container.style.width="100%",this.container.style.height=`${this.config.height}px`}createChart(){switch(this.config.type){case"scatter":this.createScatterPlot();break;case"timeline":this.createTimeline();break;case"distribution":this.createDistribution();break;default:console.warn(`Unknown chart type: ${this.config.type}`)}}createScatterPlot(){this.container.innerHTML=`
      <div class="chart-placeholder">
        <div class="placeholder-icon">\u{1F4CA}</div>
        <div class="placeholder-text">Scatter Plot Visualization</div>
        <div class="placeholder-description">D3.js scatter plot will be rendered here</div>
      </div>
    `}createTimeline(){this.container.innerHTML=`
      <div class="chart-placeholder">
        <div class="placeholder-icon">\u{1F4C8}</div>
        <div class="placeholder-text">Timeline Visualization</div>
        <div class="placeholder-description">Time series chart will be rendered here</div>
      </div>
    `}createDistribution(){this.container.innerHTML=`
      <div class="chart-placeholder">
        <div class="placeholder-icon">\u{1F4CA}</div>
        <div class="placeholder-text">Distribution Visualization</div>
        <div class="placeholder-description">Distribution chart will be rendered here</div>
      </div>
    `}updateData(t){this.data=t,this.createChart()}destroy(){this.chart&&this.chart.dispose&&this.chart.dispose(),this.container.innerHTML=""}};var r=class{constructor(t){this.element=t,this.config=this.getConfig(),this.state={isProcessing:!1,currentDataset:null,selectedAlgorithm:null,results:null,realTimeMode:!1},this.charts=new Map,this.websocket=null,this.init()}init(){console.log("\u{1F50D} Initializing Anomaly Detector component"),this.createInterface(),this.bindEvents(),this.loadAlgorithms(),this.setupRealTimeCapabilities(),this.element.dispatchEvent(new CustomEvent("component:ready",{detail:{component:"anomaly-detector",element:this.element}}))}getConfig(){let t=this.element;return{apiEndpoint:t.dataset.apiEndpoint||"/api/anomaly-detection",websocketUrl:t.dataset.websocketUrl||"/ws/anomaly-detection",autoDetect:t.dataset.autoDetect==="true",realTime:t.dataset.realTime==="true",maxFileSize:parseInt(t.dataset.maxFileSize)||10*1024*1024,allowedFormats:(t.dataset.allowedFormats||"csv,json,parquet").split(","),algorithms:t.dataset.algorithms?JSON.parse(t.dataset.algorithms):null}}createInterface(){this.element.innerHTML=`
      <div class="anomaly-detector-container">
        <!-- Header -->
        <div class="detector-header card-header">
          <h3 class="text-lg font-semibold text-neutral-900">Anomaly Detection</h3>
          <div class="detector-controls flex items-center space-x-2">
            <button class="btn btn-sm btn-outline" data-action="toggle-realtime">
              <span class="realtime-icon">\u{1F4E1}</span>
              <span class="realtime-text">Real-time</span>
            </button>
            <button class="btn btn-sm btn-ghost" data-action="toggle-settings">
              <span>\u2699\uFE0F</span>
            </button>
          </div>
        </div>

        <!-- Main Content -->
        <div class="detector-content">
          <!-- Step 1: Data Input -->
          <div class="detection-step" data-step="data-input">
            <div class="step-header">
              <h4 class="text-md font-medium">1. Select Data Source</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>
            
            <div class="data-input-section">
              <div class="input-tabs">
                <button class="tab-button active" data-tab="upload">Upload File</button>
                <button class="tab-button" data-tab="dataset">Existing Dataset</button>
                <button class="tab-button" data-tab="stream">Real-time Stream</button>
              </div>
              
              <!-- File Upload Tab -->
              <div class="tab-content active" data-tab-content="upload">
                <div class="file-upload-area" data-drop-zone>
                  <div class="upload-icon">\u{1F4C1}</div>
                  <div class="upload-text">
                    <p class="text-sm font-medium">Drop your data file here or click to browse</p>
                    <p class="text-xs text-neutral-500">Supported formats: CSV, JSON, Parquet (max ${this.config.maxFileSize/1024/1024}MB)</p>
                  </div>
                  <input type="file" class="file-input" accept=".csv,.json,.parquet" hidden>
                </div>
                <div class="file-info hidden" data-file-info></div>
              </div>
              
              <!-- Existing Dataset Tab -->
              <div class="tab-content" data-tab-content="dataset">
                <div class="dataset-selector">
                  <select class="form-select" data-dataset-select>
                    <option value="">Select a dataset...</option>
                  </select>
                  <button class="btn btn-sm btn-outline" data-action="refresh-datasets">
                    <span>\u{1F504}</span> Refresh
                  </button>
                </div>
                <div class="dataset-info hidden" data-dataset-info></div>
              </div>
              
              <!-- Real-time Stream Tab -->
              <div class="tab-content" data-tab-content="stream">
                <div class="stream-config">
                  <div class="form-group">
                    <label class="form-label">Stream Source</label>
                    <select class="form-select" data-stream-source>
                      <option value="">Select stream source...</option>
                      <option value="kafka">Kafka Topic</option>
                      <option value="mqtt">MQTT</option>
                      <option value="websocket">WebSocket</option>
                      <option value="api">REST API Polling</option>
                    </select>
                  </div>
                  <div class="stream-connection-config hidden" data-stream-config></div>
                  <button class="btn btn-primary" data-action="connect-stream">Connect Stream</button>
                </div>
              </div>
            </div>
          </div>

          <!-- Step 2: Algorithm Selection -->
          <div class="detection-step" data-step="algorithm-selection">
            <div class="step-header">
              <h4 class="text-md font-medium">2. Choose Detection Algorithm</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>
            
            <div class="algorithm-selection">
              <div class="algorithm-tabs">
                <button class="tab-button active" data-algo-tab="recommended">Recommended</button>
                <button class="tab-button" data-algo-tab="all">All Algorithms</button>
                <button class="tab-button" data-algo-tab="ensemble">Ensemble</button>
                <button class="tab-button" data-algo-tab="custom">Custom</button>
              </div>
              
              <div class="algorithm-grid" data-algorithm-grid></div>
              
              <div class="algorithm-params hidden" data-algorithm-params>
                <h5 class="text-sm font-medium mb-2">Algorithm Parameters</h5>
                <div class="params-form" data-params-form></div>
              </div>
            </div>
          </div>

          <!-- Step 3: Detection Execution -->
          <div class="detection-step" data-step="execution">
            <div class="step-header">
              <h4 class="text-md font-medium">3. Run Detection</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>
            
            <div class="execution-controls">
              <button class="btn btn-primary btn-lg" data-action="start-detection" disabled>
                <span class="btn-icon">\u{1F680}</span>
                <span class="btn-text">Start Detection</span>
              </button>
              
              <div class="execution-options">
                <label class="checkbox-label">
                  <input type="checkbox" data-option="explain-results">
                  <span class="checkmark"></span>
                  Generate explanations
                </label>
                <label class="checkbox-label">
                  <input type="checkbox" data-option="save-model">
                  <span class="checkmark"></span>
                  Save trained model
                </label>
                <label class="checkbox-label">
                  <input type="checkbox" data-option="auto-threshold">
                  <span class="checkmark"></span>
                  Auto-optimize threshold
                </label>
              </div>
            </div>
            
            <div class="execution-progress hidden" data-execution-progress>
              <div class="progress-bar">
                <div class="progress-fill" data-progress-fill></div>
              </div>
              <div class="progress-text" data-progress-text>Initializing...</div>
              <button class="btn btn-sm btn-accent" data-action="cancel-detection">Cancel</button>
            </div>
          </div>

          <!-- Step 4: Results Visualization -->
          <div class="detection-step" data-step="results">
            <div class="step-header">
              <h4 class="text-md font-medium">4. Detection Results</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>
            
            <div class="results-container hidden" data-results-container>
              <!-- Results Summary -->
              <div class="results-summary">
                <div class="metric-cards">
                  <div class="metric-card">
                    <div class="metric-value" data-metric="total-samples">-</div>
                    <div class="metric-label">Total Samples</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value text-accent-600" data-metric="anomalies-detected">-</div>
                    <div class="metric-label">Anomalies Detected</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" data-metric="anomaly-rate">-</div>
                    <div class="metric-label">Anomaly Rate</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" data-metric="confidence-score">-</div>
                    <div class="metric-label">Confidence Score</div>
                  </div>
                </div>
              </div>
              
              <!-- Results Visualization -->
              <div class="results-visualization">
                <div class="chart-tabs">
                  <button class="tab-button active" data-chart-tab="scatter">Scatter Plot</button>
                  <button class="tab-button" data-chart-tab="timeline">Timeline</button>
                  <button class="tab-button" data-chart-tab="distribution">Distribution</button>
                  <button class="tab-button" data-chart-tab="heatmap">Feature Heatmap</button>
                </div>
                
                <div class="chart-container" data-chart-container>
                  <div class="chart-loading skeleton h-64"></div>
                </div>
                
                <div class="chart-controls">
                  <div class="threshold-control">
                    <label class="form-label">Anomaly Threshold</label>
                    <input type="range" class="threshold-slider" data-threshold-slider min="0" max="1" step="0.01" value="0.5">
                    <span class="threshold-value" data-threshold-value>0.5</span>
                  </div>
                  
                  <div class="filter-controls">
                    <button class="btn btn-sm btn-outline" data-filter="show-all">Show All</button>
                    <button class="btn btn-sm btn-outline" data-filter="show-anomalies">Anomalies Only</button>
                    <button class="btn btn-sm btn-outline" data-filter="show-normal">Normal Only</button>
                  </div>
                </div>
              </div>
              
              <!-- Results Actions -->
              <div class="results-actions">
                <button class="btn btn-primary" data-action="export-results">
                  <span>\u{1F4CA}</span> Export Results
                </button>
                <button class="btn btn-secondary" data-action="save-model">
                  <span>\u{1F4BE}</span> Save Model
                </button>
                <button class="btn btn-outline" data-action="generate-report">
                  <span>\u{1F4C4}</span> Generate Report
                </button>
                <button class="btn btn-ghost" data-action="explain-results">
                  <span>\u{1F50D}</span> Explain Results
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Settings Panel -->
        <div class="settings-panel hidden" data-settings-panel>
          <div class="settings-header">
            <h4 class="text-md font-medium">Detection Settings</h4>
            <button class="btn btn-sm btn-ghost" data-action="close-settings">\u2715</button>
          </div>
          
          <div class="settings-content">
            <div class="setting-group">
              <label class="form-label">Default Algorithm</label>
              <select class="form-select" data-setting="default-algorithm">
                <option value="auto">Auto-select</option>
                <option value="isolation-forest">Isolation Forest</option>
                <option value="one-class-svm">One-Class SVM</option>
                <option value="lof">Local Outlier Factor</option>
              </select>
            </div>
            
            <div class="setting-group">
              <label class="form-label">Contamination Rate</label>
              <input type="number" class="form-input" data-setting="contamination-rate" min="0" max="1" step="0.01" value="0.1">
            </div>
            
            <div class="setting-group">
              <label class="checkbox-label">
                <input type="checkbox" data-setting="auto-preprocess">
                <span class="checkmark"></span>
                Auto-preprocess data
              </label>
            </div>
            
            <div class="setting-group">
              <label class="checkbox-label">
                <input type="checkbox" data-setting="enable-notifications">
                <span class="checkmark"></span>
                Enable notifications
              </label>
            </div>
          </div>
        </div>
      </div>
    `}bindEvents(){let t=this.element;t.addEventListener("click",s=>{s.target.matches(".tab-button")&&this.switchTab(s.target)});let e=t.querySelector(".file-input"),a=t.querySelector("[data-drop-zone]");e?.addEventListener("change",s=>{this.handleFileUpload(s.target.files[0])}),a?.addEventListener("click",()=>e?.click()),a?.addEventListener("dragover",s=>{s.preventDefault(),a.classList.add("drag-over")}),a?.addEventListener("dragleave",()=>{a.classList.remove("drag-over")}),a?.addEventListener("drop",s=>{s.preventDefault(),a.classList.remove("drag-over"),this.handleFileUpload(s.dataTransfer.files[0])}),t.addEventListener("click",s=>{let o=s.target.closest("[data-action]")?.dataset.action;o&&this.handleAction(o,s.target)}),t.addEventListener("click",s=>{s.target.matches(".algorithm-card")&&this.selectAlgorithm(s.target)}),t.querySelector("[data-threshold-slider]")?.addEventListener("input",s=>{this.updateThreshold(parseFloat(s.target.value))}),t.addEventListener("keydown",s=>{if(s.ctrlKey||s.metaKey)switch(s.key){case"Enter":s.preventDefault(),this.startDetection();break;case"r":s.preventDefault(),this.resetDetector();break}})}async loadAlgorithms(){try{let e=await(await fetch("/api/algorithms")).json();this.renderAlgorithmGrid(e),this.updateAlgorithmRecommendations(e)}catch(t){console.error("Failed to load algorithms:",t),this.showError("Failed to load detection algorithms")}}renderAlgorithmGrid(t){let e=this.element.querySelector("[data-algorithm-grid]");e&&(e.innerHTML=t.map(a=>`
      <div class="algorithm-card" data-algorithm="${a.id}">
        <div class="algorithm-header">
          <h5 class="algorithm-name">${a.name}</h5>
          <div class="algorithm-type">${a.type}</div>
        </div>
        <div class="algorithm-description">${a.description}</div>
        <div class="algorithm-metrics">
          <div class="metric">
            <span class="metric-label">Accuracy</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: ${a.accuracy*100}%"></div>
            </div>
          </div>
          <div class="metric">
            <span class="metric-label">Speed</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: ${a.speed*100}%"></div>
            </div>
          </div>
        </div>
        <div class="algorithm-tags">
          ${a.tags.map(i=>`<span class="tag">${i}</span>`).join("")}
        </div>
      </div>
    `).join(""))}handleFileUpload(t){if(!t)return;let e=t.name.split(".").pop().toLowerCase();if(!this.config.allowedFormats.includes(e)){this.showError(`Unsupported file format: ${e}`);return}if(t.size>this.config.maxFileSize){this.showError(`File too large: ${(t.size/1024/1024).toFixed(1)}MB (max: ${this.config.maxFileSize/1024/1024}MB)`);return}this.showFileInfo(t),this.uploadFile(t)}async uploadFile(t){let e=new FormData;e.append("file",t);try{this.updateStepStatus("data-input","processing");let a=await fetch("/api/datasets/upload",{method:"POST",body:e});if(!a.ok)throw new Error(`Upload failed: ${a.statusText}`);let i=await a.json();this.state.currentDataset=i.dataset,this.updateStepStatus("data-input","completed"),this.enableStep("algorithm-selection"),this.announceToScreenReader(`File uploaded successfully: ${t.name}`)}catch(a){console.error("Upload failed:",a),this.updateStepStatus("data-input","error"),this.showError(`Upload failed: ${a.message}`)}}async startDetection(){if(!this.state.currentDataset||!this.state.selectedAlgorithm){this.showError("Please select data and algorithm first");return}this.state.isProcessing=!0,this.updateStepStatus("execution","processing");let t=this.element.querySelector("[data-execution-progress]");t?.classList.remove("hidden");try{let e=this.getAlgorithmParams(),a=this.getDetectionOptions(),i=await fetch("/api/anomaly-detection/detect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({dataset_id:this.state.currentDataset.id,algorithm:this.state.selectedAlgorithm,parameters:e,options:a})});if(!i.ok)throw new Error(`Detection failed: ${i.statusText}`);let s=await i.json();this.state.results=s,this.updateStepStatus("execution","completed"),this.updateStepStatus("results","completed"),this.showResults(s),this.announceToScreenReader(`Detection completed. Found ${s.anomaly_count} anomalies.`)}catch(e){console.error("Detection failed:",e),this.updateStepStatus("execution","error"),this.showError(`Detection failed: ${e.message}`)}finally{this.state.isProcessing=!1,t?.classList.add("hidden")}}showResults(t){this.element.querySelector("[data-results-container]")?.classList.remove("hidden"),this.updateMetric("total-samples",t.total_samples.toLocaleString()),this.updateMetric("anomalies-detected",t.anomaly_count.toLocaleString()),this.updateMetric("anomaly-rate",`${(t.anomaly_rate*100).toFixed(2)}%`),this.updateMetric("confidence-score",t.confidence_score.toFixed(3)),this.createResultsCharts(t)}createResultsCharts(t){let e=this.element.querySelector("[data-chart-container]");if(!e)return;let a=new n(e,{type:"scatter",data:t.visualization_data.scatter,options:{title:"Anomaly Detection Results",xAxis:{title:"Feature 1"},yAxis:{title:"Feature 2"},colorScale:{normal:"#22c55e",anomaly:"#ef4444"}}});this.charts.set("scatter",a)}switchTab(t){let e=t.closest(".input-tabs, .algorithm-tabs, .chart-tabs"),a=t.dataset.tab||t.dataset.algoTab||t.dataset.chartTab;e.querySelectorAll(".tab-button").forEach(o=>{o.classList.remove("active")}),t.classList.add("active");let i=e.nextElementSibling;i.querySelectorAll(".tab-content").forEach(o=>{o.classList.remove("active")}),i.querySelector(`[data-tab-content="${a}"]`)?.classList.add("active")}updateStepStatus(t,e){let i=this.element.querySelector(`[data-step="${t}"]`)?.querySelector("[data-status]");i&&(i.className=`step-status status-${e}`,i.textContent=e.charAt(0).toUpperCase()+e.slice(1),i.dataset.status=e)}enableStep(t){this.element.querySelector(`[data-step="${t}"]`)?.classList.remove("disabled")}updateMetric(t,e){let a=this.element.querySelector(`[data-metric="${t}"]`);a&&(a.textContent=e)}showError(t){this.element.dispatchEvent(new CustomEvent("component:error",{detail:{component:"anomaly-detector",message:t}}))}announceToScreenReader(t){window.PynomalyApp&&window.PynomalyApp.announceToScreenReader(t)}setupRealTimeCapabilities(){this.config.realTime&&this.initWebSocket()}initWebSocket(){if(this.config.websocketUrl)try{this.websocket=new WebSocket(this.config.websocketUrl),this.websocket.onmessage=t=>{let e=JSON.parse(t.data);this.handleRealTimeUpdate(e)},this.websocket.onerror=t=>{console.error("WebSocket error:",t)}}catch(t){console.error("Failed to initialize WebSocket:",t)}}handleRealTimeUpdate(t){t.type==="anomaly_detected"?this.showAnomalyAlert(t.anomaly):t.type==="model_updated"&&this.refreshResults()}handleAction(t,e){switch(t){case"start-detection":this.startDetection();break;case"toggle-realtime":this.toggleRealTimeMode();break;case"export-results":this.exportResults();break;case"generate-report":this.generateReport();break;default:console.warn(`Unknown action: ${t}`)}}getAlgorithmParams(){let t=this.element.querySelector("[data-params-form]");if(!t)return{};let e=new FormData(t),a={};for(let[i,s]of e.entries())a[i]=s;return a}getDetectionOptions(){return{explain_results:this.element.querySelector('[data-option="explain-results"]')?.checked||!1,save_model:this.element.querySelector('[data-option="save-model"]')?.checked||!1,auto_threshold:this.element.querySelector('[data-option="auto-threshold"]')?.checked||!1}}destroy(){this.websocket&&this.websocket.close(),this.charts.forEach(t=>{t.destroy&&t.destroy()}),this.charts.clear()}};document.addEventListener("DOMContentLoaded",()=>{document.querySelectorAll('[data-component="anomaly-detector"]').forEach(l=>{new r(l)})});})();
