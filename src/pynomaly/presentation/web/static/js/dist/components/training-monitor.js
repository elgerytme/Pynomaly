(()=>{var o=class{constructor(t={}){this.options={url:t.url||this.getWebSocketUrl(),protocols:t.protocols||["anomaly-detection-v1"],maxReconnectAttempts:t.maxReconnectAttempts||10,reconnectInterval:t.reconnectInterval||3e3,maxReconnectDelay:t.maxReconnectDelay||3e4,heartbeatInterval:t.heartbeatInterval||3e4,messageQueueSize:t.messageQueueSize||1e3,enableMessageQueue:t.enableMessageQueue!==!1,enableCompression:t.enableCompression!==!1,enableLogging:t.enableLogging||!1,autoConnect:t.autoConnect!==!1,authentication:t.authentication||null,...t},this.ws=null,this.isConnected=!1,this.reconnectAttempts=0,this.listeners=new Map,this.subscriptions=new Set,this.heartbeatTimer=null,this.reconnectTimer=null,this.messageQueue=[],this.connectionId=null,this.bindMethods(),this.options.autoConnect&&this.connect()}bindMethods(){this.handleOpen=this.handleOpen.bind(this),this.handleMessage=this.handleMessage.bind(this),this.handleError=this.handleError.bind(this),this.handleClose=this.handleClose.bind(this),this.sendHeartbeat=this.sendHeartbeat.bind(this)}getWebSocketUrl(){let t=window.location.protocol==="https:"?"wss:":"ws:",e=window.location.host;return`${t}//${e}/ws/anomaly-detection`}connect(){return this.isConnected||this.ws&&this.ws.readyState===WebSocket.CONNECTING?Promise.resolve():new Promise((t,e)=>{try{this.log("Connecting to WebSocket...",this.options.url),this.ws=new WebSocket(this.options.url,this.options.protocols),this.options.enableCompression&&this.ws.extensions&&(this.ws.extensions="permessage-deflate");let s=setTimeout(()=>{this.ws.readyState===WebSocket.CONNECTING&&(this.ws.close(),e(new Error("Connection timeout")))},1e4);this.ws.addEventListener("open",i=>{clearTimeout(s),this.handleOpen(i),t()}),this.ws.addEventListener("message",this.handleMessage),this.ws.addEventListener("error",i=>{clearTimeout(s),this.handleError(i),e(i)}),this.ws.addEventListener("close",this.handleClose)}catch(s){this.log("Connection error:",s),e(s)}})}handleOpen(t){this.log("WebSocket connected"),this.isConnected=!0,this.reconnectAttempts=0,this.startHeartbeat(),this.processMessageQueue(),this.resubscribe(),this.emit("connected",{event:t,connectionId:this.connectionId})}handleMessage(t){try{let e=JSON.parse(t.data);if(this.log("Received message:",e),e.type==="system"){this.handleSystemMessage(e);return}if(e.type==="pong"){this.log("Heartbeat acknowledged");return}this.emit("message",e),e.type&&this.emit(e.type,e.data||e),e.subscription&&this.emit(`subscription:${e.subscription}`,e.data||e)}catch(e){this.log("Error parsing message:",e,t.data),this.emit("error",{type:"parse_error",error:e,rawData:t.data})}}handleSystemMessage(t){switch(t.action){case"connection_established":this.connectionId=t.connectionId,this.log("Connection ID received:",this.connectionId);break;case"subscription_confirmed":this.log("Subscription confirmed:",t.subscription),this.emit("subscription_confirmed",t);break;case"subscription_error":this.log("Subscription error:",t.error),this.emit("subscription_error",t);break;case"rate_limit_exceeded":this.log("Rate limit exceeded"),this.emit("rate_limit_exceeded",t);break;case"server_shutdown":this.log("Server shutdown notification"),this.emit("server_shutdown",t);break;default:this.log("Unknown system message:",t)}}handleError(t){this.log("WebSocket error:",t),this.emit("error",{type:"connection_error",error:t})}handleClose(t){this.log("WebSocket closed:",t.code,t.reason),this.isConnected=!1,this.stopHeartbeat(),this.emit("disconnected",{code:t.code,reason:t.reason,wasClean:t.wasClean}),!t.wasClean&&this.reconnectAttempts<this.options.maxReconnectAttempts&&this.scheduleReconnect()}scheduleReconnect(){this.reconnectTimer&&clearTimeout(this.reconnectTimer),this.reconnectAttempts++;let t=Math.min(this.options.reconnectInterval*Math.pow(2,this.reconnectAttempts-1),3e4);this.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${t}ms`),this.reconnectTimer=setTimeout(()=>{this.log(`Reconnect attempt ${this.reconnectAttempts}`),this.connect().catch(e=>{this.log("Reconnect failed:",e),this.reconnectAttempts<this.options.maxReconnectAttempts?this.scheduleReconnect():this.emit("max_reconnect_attempts_reached")})},t)}startHeartbeat(){this.heartbeatTimer&&clearInterval(this.heartbeatTimer),this.heartbeatTimer=setInterval(this.sendHeartbeat,this.options.heartbeatInterval)}stopHeartbeat(){this.heartbeatTimer&&(clearInterval(this.heartbeatTimer),this.heartbeatTimer=null)}sendHeartbeat(){this.isConnected&&this.send({type:"ping",timestamp:Date.now()})}send(t){if(!this.isConnected)return this.log("Queueing message (not connected):",t),this.messageQueue.push(t),!1;try{let e=typeof t=="string"?t:JSON.stringify(t);return this.ws.send(e),this.log("Sent message:",t),!0}catch(e){return this.log("Error sending message:",e),this.emit("error",{type:"send_error",error:e,data:t}),!1}}processMessageQueue(){for(;this.messageQueue.length>0;){let t=this.messageQueue.shift();this.send(t)}}subscribe(t,e={}){let s={type:"subscribe",subscription:t,params:e,timestamp:Date.now()};return this.subscriptions.add(t),this.send(s),this.log("Subscribed to:",t,e),()=>this.unsubscribe(t)}unsubscribe(t){this.subscriptions.delete(t),this.send({type:"unsubscribe",subscription:t,timestamp:Date.now()}),this.log("Unsubscribed from:",t)}resubscribe(){this.subscriptions.forEach(t=>{this.send({type:"subscribe",subscription:t,timestamp:Date.now()})})}on(t,e){return this.listeners.has(t)||this.listeners.set(t,new Set),this.listeners.get(t).add(e),()=>this.off(t,e)}off(t,e){let s=this.listeners.get(t);s&&(s.delete(e),s.size===0&&this.listeners.delete(t))}emit(t,e){let s=this.listeners.get(t);s&&s.forEach(i=>{try{i(e)}catch(a){this.log("Error in event listener:",a)}})}isConnected(){return this.isConnected&&this.ws&&this.ws.readyState===WebSocket.OPEN}getConnectionState(){if(!this.ws)return"disconnected";switch(this.ws.readyState){case WebSocket.CONNECTING:return"connecting";case WebSocket.OPEN:return"connected";case WebSocket.CLOSING:return"closing";case WebSocket.CLOSED:return"disconnected";default:return"unknown"}}getConnectionInfo(){return{connectionId:this.connectionId,state:this.getConnectionState(),reconnectAttempts:this.reconnectAttempts,subscriptions:Array.from(this.subscriptions),queuedMessages:this.messageQueue.length,url:this.options.url}}disconnect(){this.log("Disconnecting WebSocket"),this.stopHeartbeat(),this.reconnectTimer&&(clearTimeout(this.reconnectTimer),this.reconnectTimer=null),this.ws&&this.ws.close(1e3,"Client disconnect"),this.isConnected=!1,this.subscriptions.clear(),this.messageQueue=[]}log(...t){this.options.enableLogging&&console.log("[WebSocketService]",...t)}destroy(){this.disconnect(),this.listeners.clear(),this.resizeObserver&&this.resizeObserver.disconnect()}};var c=class{constructor(t,e={}){this.container=typeof t=="string"?document.querySelector(t):t,this.options={enableWebSocket:!0,enableAutoRefresh:!0,refreshInterval:5e3,showResourceMonitoring:!0,showTrainingHistory:!0,maxHistoryItems:50,...e},this.activeTrainings=new Map,this.trainingHistory=[],this.websocketService=null,this.components={toolbar:null,activeTrainings:null,trainingDetails:null,metricsChart:null,resourceChart:null,historyTable:null},this.listeners=new Map,this.init()}init(){this.createLayout(),this.setupWebSocket(),this.loadInitialData(),this.bindEvents(),this.options.enableAutoRefresh&&this.startAutoRefresh()}createLayout(){this.container.innerHTML=`
            <div class="training-monitor">
                <!-- Header and Controls -->
                <div class="training-monitor__header">
                    <div class="training-monitor__title">
                        <h2>Automated Training Monitor</h2>
                        <div class="training-monitor__status">
                            <span class="status-indicator" data-status="disconnected">
                                <span class="status-dot"></span>
                                <span class="status-text">Connecting...</span>
                            </span>
                        </div>
                    </div>
                    
                    <div class="training-monitor__toolbar">
                        <button class="btn btn--primary" data-action="start-training">
                            <i class="icon-play"></i> Start Training
                        </button>
                        <button class="btn btn--secondary" data-action="refresh">
                            <i class="icon-refresh"></i> Refresh
                        </button>
                        <button class="btn btn--secondary" data-action="settings">
                            <i class="icon-settings"></i> Settings
                        </button>
                    </div>
                </div>
                
                <!-- Active Trainings Grid -->
                <div class="training-monitor__content">
                    <div class="training-monitor__grid">
                        <!-- Active Trainings Panel -->
                        <div class="training-panel">
                            <div class="panel-header">
                                <h3>Active Trainings</h3>
                                <span class="badge badge--info" data-count="active-count">0</span>
                            </div>
                            <div class="panel-content">
                                <div class="training-list" data-component="active-trainings">
                                    <div class="empty-state">
                                        <i class="icon-training"></i>
                                        <p>No active trainings</p>
                                        <button class="btn btn--primary btn--sm" data-action="start-training">
                                            Start New Training
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Training Details Panel -->
                        <div class="training-panel">
                            <div class="panel-header">
                                <h3>Training Details</h3>
                                <div class="panel-actions">
                                    <button class="btn btn--icon" data-action="expand-details" title="Expand">
                                        <i class="icon-expand"></i>
                                    </button>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="training-details" data-component="training-details">
                                    <div class="empty-state">
                                        <p>Select a training to view details</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Metrics Visualization Panel -->
                        <div class="training-panel training-panel--wide">
                            <div class="panel-header">
                                <h3>Training Metrics</h3>
                                <div class="panel-controls">
                                    <select class="form-select form-select--sm" data-control="metric-type">
                                        <option value="score">Score Progress</option>
                                        <option value="loss">Loss Curve</option>
                                        <option value="trials">Trial Results</option>
                                        <option value="resource">Resource Usage</option>
                                    </select>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="metrics-chart" data-component="metrics-chart">
                                    <svg class="chart-svg"></svg>
                                    <div class="chart-legend"></div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Training History Panel -->
                        <div class="training-panel training-panel--full">
                            <div class="panel-header">
                                <h3>Training History</h3>
                                <div class="panel-controls">
                                    <input type="search" class="form-input form-input--sm" 
                                           placeholder="Search trainings..." data-control="history-search">
                                    <select class="form-select form-select--sm" data-control="history-filter">
                                        <option value="">All Statuses</option>
                                        <option value="completed">Completed</option>
                                        <option value="failed">Failed</option>
                                        <option value="cancelled">Cancelled</option>
                                    </select>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="training-history" data-component="training-history">
                                    <div class="table-container">
                                        <table class="data-table">
                                            <thead>
                                                <tr>
                                                    <th>Training ID</th>
                                                    <th>Detector</th>
                                                    <th>Algorithm</th>
                                                    <th>Score</th>
                                                    <th>Duration</th>
                                                    <th>Status</th>
                                                    <th>Started</th>
                                                    <th>Actions</th>
                                                </tr>
                                            </thead>
                                            <tbody data-target="history-rows">
                                                <!-- Dynamic content -->
                                            </tbody>
                                        </table>
                                    </div>
                                    <div class="table-pagination">
                                        <button class="btn btn--sm" data-action="prev-page" disabled>Previous</button>
                                        <span class="pagination-info">Page 1 of 1</span>
                                        <button class="btn btn--sm" data-action="next-page" disabled>Next</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Start Training Modal -->
                <div class="modal" data-modal="start-training">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>Start New Training</h3>
                            <button class="btn btn--icon" data-action="close-modal">
                                <i class="icon-close"></i>
                            </button>
                        </div>
                        <div class="modal-body">
                            <form class="training-form" data-form="start-training">
                                <div class="form-grid">
                                    <div class="form-group">
                                        <label>Detector</label>
                                        <select class="form-select" name="detector_id" required>
                                            <option value="">Select detector...</option>
                                        </select>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Dataset</label>
                                        <select class="form-select" name="dataset_id" required>
                                            <option value="">Select dataset...</option>
                                        </select>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Experiment Name</label>
                                        <input type="text" class="form-input" name="experiment_name" 
                                               placeholder="Optional experiment name">
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Optimization Objective</label>
                                        <select class="form-select" name="optimization_objective">
                                            <option value="auc">AUC</option>
                                            <option value="precision">Precision</option>
                                            <option value="recall">Recall</option>
                                            <option value="f1_score">F1 Score</option>
                                        </select>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Max Algorithms</label>
                                        <input type="number" class="form-input" name="max_algorithms" 
                                               value="3" min="1" max="10">
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Max Optimization Time (minutes)</label>
                                        <input type="number" class="form-input" name="max_optimization_time" 
                                               value="60" min="1" max="1440">
                                    </div>
                                </div>
                                
                                <div class="form-group">
                                    <div class="form-checkboxes">
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_automl" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable AutoML optimization
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_ensemble" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable ensemble creation
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_early_stopping" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable early stopping
                                        </label>
                                    </div>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn--secondary" data-action="close-modal">Cancel</button>
                            <button class="btn btn--primary" data-action="submit-training">Start Training</button>
                        </div>
                    </div>
                </div>
            </div>
        `,this.components.toolbar=this.container.querySelector(".training-monitor__toolbar"),this.components.activeTrainings=this.container.querySelector('[data-component="active-trainings"]'),this.components.trainingDetails=this.container.querySelector('[data-component="training-details"]'),this.components.metricsChart=this.container.querySelector('[data-component="metrics-chart"]'),this.components.historyTable=this.container.querySelector('[data-component="training-history"]'),this.setupCharts()}setupCharts(){this.initializeMetricsChart(),this.initializeResourceChart()}initializeMetricsChart(){let t=this.components.metricsChart.querySelector(".chart-svg"),e=t.clientWidth||600,s=300,i={top:20,right:30,bottom:40,left:50},a=d3.select(t).attr("width",e).attr("height",s);a.selectAll("*").remove();let n=a.append("g").attr("transform",`translate(${i.left},${i.top})`);this.chartScales={x:d3.scaleLinear().range([0,e-i.left-i.right]),y:d3.scaleLinear().range([s-i.top-i.bottom,0])},this.chartAxes={x:d3.axisBottom(this.chartScales.x),y:d3.axisLeft(this.chartScales.y)},n.append("g").attr("class","axis axis--x").attr("transform",`translate(0,${s-i.top-i.bottom})`),n.append("g").attr("class","axis axis--y"),n.append("text").attr("class","chart-title").attr("x",(e-i.left-i.right)/2).attr("y",-5).attr("text-anchor","middle").text("Training Progress"),this.chartGroup=n}initializeResourceChart(){}setupWebSocket(){this.options.enableWebSocket&&(this.websocketService=new o({url:this.getWebSocketUrl(),enableLogging:!0}),this.websocketService.on("connected",()=>{this.updateConnectionStatus("connected"),this.subscribeToTrainingUpdates()}),this.websocketService.on("disconnected",()=>{this.updateConnectionStatus("disconnected")}),this.websocketService.on("error",t=>{console.error("Training WebSocket error:",t),this.updateConnectionStatus("error")}),this.websocketService.on("training_update",t=>{this.handleTrainingUpdate(t)}),this.websocketService.on("training_progress",t=>{this.handleTrainingProgress(t.data)}))}getWebSocketUrl(){let t=window.location.protocol==="https:"?"wss:":"ws:",e=window.location.host;return`${t}//${e}/ws/training`}subscribeToTrainingUpdates(){this.websocketService&&this.websocketService.send({type:"subscribe_training_updates"})}updateConnectionStatus(t){let e=this.container.querySelector(".status-indicator"),s=e.querySelector(".status-text");switch(e.setAttribute("data-status",t),t){case"connected":s.textContent="Connected";break;case"disconnected":s.textContent="Disconnected";break;case"error":s.textContent="Connection Error";break;default:s.textContent="Connecting..."}}handleTrainingUpdate(t){console.log("Training update received:",t),t.training_id?this.updateTrainingItem(t.training_id,t):this.refreshActiveTrainings()}handleTrainingProgress(t){console.log("Training progress:",t),this.updateActiveTraining(t),this.getSelectedTrainingId()===t.training_id&&(this.updateTrainingDetails(t),this.updateMetricsChart(t))}updateActiveTraining(t){this.activeTrainings.set(t.training_id,t),this.renderActiveTrainings()}renderActiveTrainings(){let t=this.components.activeTrainings,e=Array.from(this.activeTrainings.values());if(e.length===0){t.innerHTML=`
                <div class="empty-state">
                    <i class="icon-training"></i>
                    <p>No active trainings</p>
                    <button class="btn btn--primary btn--sm" data-action="start-training">
                        Start New Training
                    </button>
                </div>
            `,this.updateActiveCount(0);return}let s=e.map(i=>this.renderTrainingItem(i)).join("");t.innerHTML=s,this.updateActiveCount(e.length)}renderTrainingItem(t){let e=this.getStatusClass(t.status),s=Math.round(t.progress_percentage);return`
            <div class="training-item" data-training-id="${t.training_id}">
                <div class="training-item__header">
                    <div class="training-item__title">
                        <strong>${t.training_id.substring(0,8)}...</strong>
                        <span class="status-badge status-badge--${e}">${t.status}</span>
                    </div>
                    <div class="training-item__actions">
                        <button class="btn btn--icon btn--sm" data-action="view-training" 
                                data-training-id="${t.training_id}" title="View Details">
                            <i class="icon-eye"></i>
                        </button>
                        ${t.status==="running"?`
                            <button class="btn btn--icon btn--sm btn--danger" data-action="cancel-training" 
                                    data-training-id="${t.training_id}" title="Cancel">
                                <i class="icon-stop"></i>
                            </button>
                        `:""}
                    </div>
                </div>
                
                <div class="training-item__progress">
                    <div class="progress-bar">
                        <div class="progress-bar__fill" style="width: ${s}%"></div>
                    </div>
                    <div class="progress-text">
                        <span>${t.current_step}</span>
                        <span>${s}%</span>
                    </div>
                </div>
                
                <div class="training-item__details">
                    ${t.current_algorithm?`
                        <div class="detail-item">
                            <span class="detail-label">Algorithm:</span>
                            <span class="detail-value">${t.current_algorithm}</span>
                        </div>
                    `:""}
                    ${t.best_score?`
                        <div class="detail-item">
                            <span class="detail-label">Best Score:</span>
                            <span class="detail-value">${t.best_score.toFixed(4)}</span>
                        </div>
                    `:""}
                    ${t.current_message?`
                        <div class="detail-item detail-item--full">
                            <span class="detail-message">${t.current_message}</span>
                        </div>
                    `:""}
                </div>
                
                ${t.warnings&&t.warnings.length>0?`
                    <div class="training-item__warnings">
                        ${t.warnings.map(i=>`
                            <div class="warning-item">
                                <i class="icon-warning"></i>
                                <span>${i}</span>
                            </div>
                        `).join("")}
                    </div>
                `:""}
            </div>
        `}getStatusClass(t){return{idle:"secondary",scheduled:"info",running:"primary",optimizing:"warning",evaluating:"info",completed:"success",failed:"danger",cancelled:"secondary"}[t]||"secondary"}updateActiveCount(t){let e=this.container.querySelector('[data-count="active-count"]');e&&(e.textContent=t)}updateTrainingDetails(t){let e=this.components.trainingDetails;if(!t){e.innerHTML=`
                <div class="empty-state">
                    <p>Select a training to view details</p>
                </div>
            `;return}e.innerHTML=`
            <div class="training-details-content">
                <div class="details-header">
                    <h4>Training ${t.training_id.substring(0,8)}...</h4>
                    <span class="status-badge status-badge--${this.getStatusClass(t.status)}">
                        ${t.status}
                    </span>
                </div>
                
                <div class="details-grid">
                    <div class="detail-group">
                        <h5>Progress</h5>
                        <div class="detail-item">
                            <span class="detail-label">Current Step:</span>
                            <span class="detail-value">${t.current_step}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Progress:</span>
                            <span class="detail-value">${Math.round(t.progress_percentage)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Started:</span>
                            <span class="detail-value">${new Date(t.start_time).toLocaleString()}</span>
                        </div>
                        ${t.estimated_completion?`
                            <div class="detail-item">
                                <span class="detail-label">ETA:</span>
                                <span class="detail-value">${new Date(t.estimated_completion).toLocaleString()}</span>
                            </div>
                        `:""}
                    </div>
                    
                    ${t.current_algorithm?`
                        <div class="detail-group">
                            <h5>Algorithm</h5>
                            <div class="detail-item">
                                <span class="detail-label">Current:</span>
                                <span class="detail-value">${t.current_algorithm}</span>
                            </div>
                            ${t.current_trial&&t.total_trials?`
                                <div class="detail-item">
                                    <span class="detail-label">Trial:</span>
                                    <span class="detail-value">${t.current_trial} / ${t.total_trials}</span>
                                </div>
                            `:""}
                            ${t.best_score?`
                                <div class="detail-item">
                                    <span class="detail-label">Best Score:</span>
                                    <span class="detail-value">${t.best_score.toFixed(4)}</span>
                                </div>
                            `:""}
                            ${t.current_score?`
                                <div class="detail-item">
                                    <span class="detail-label">Current Score:</span>
                                    <span class="detail-value">${t.current_score.toFixed(4)}</span>
                                </div>
                            `:""}
                        </div>
                    `:""}
                    
                    ${this.options.showResourceMonitoring?`
                        <div class="detail-group">
                            <h5>Resources</h5>
                            ${t.memory_usage_mb?`
                                <div class="detail-item">
                                    <span class="detail-label">Memory:</span>
                                    <span class="detail-value">${Math.round(t.memory_usage_mb)} MB</span>
                                </div>
                            `:""}
                            ${t.cpu_usage_percent?`
                                <div class="detail-item">
                                    <span class="detail-label">CPU:</span>
                                    <span class="detail-value">${Math.round(t.cpu_usage_percent)}%</span>
                                </div>
                            `:""}
                        </div>
                    `:""}
                </div>
                
                ${t.current_message?`
                    <div class="detail-message">
                        <h5>Status Message</h5>
                        <p>${t.current_message}</p>
                    </div>
                `:""}
            </div>
        `}updateMetricsChart(t){if(!this.chartGroup||!t.best_score)return;let e=[{trial:t.current_trial||1,score:t.current_score||t.best_score}];this.chartScales.x.domain([0,t.total_trials||100]),this.chartScales.y.domain([0,1]),this.chartGroup.select(".axis--x").call(this.chartAxes.x),this.chartGroup.select(".axis--y").call(this.chartAxes.y);let s=this.chartGroup.selectAll(".data-point").data(e);s.enter().append("circle").attr("class","data-point").attr("r",4).attr("fill","#3b82f6").merge(s).attr("cx",i=>this.chartScales.x(i.trial)).attr("cy",i=>this.chartScales.y(i.score)),s.exit().remove()}bindEvents(){this.container.addEventListener("click",t=>{switch(t.target.closest("[data-action]")?.dataset.action){case"start-training":this.showStartTrainingModal();break;case"refresh":this.refreshAll();break;case"settings":this.showSettings();break;case"view-training":let s=t.target.closest("[data-training-id]")?.dataset.trainingId;this.selectTraining(s);break;case"cancel-training":let i=t.target.closest("[data-training-id]")?.dataset.trainingId;this.cancelTraining(i);break;case"close-modal":this.closeModal();break;case"submit-training":this.submitTraining();break}}),this.container.addEventListener("submit",t=>{t.preventDefault(),t.target.matches('[data-form="start-training"]')&&this.submitTraining()})}async loadInitialData(){try{await this.refreshActiveTrainings(),await this.refreshTrainingHistory(),await this.loadFormOptions()}catch(t){console.error("Failed to load initial data:",t),this.showError("Failed to load training data")}}async refreshActiveTrainings(){try{let e=await(await fetch("/api/training/active")).json();this.activeTrainings.clear(),e.forEach(s=>{this.activeTrainings.set(s.training_id,s)}),this.renderActiveTrainings()}catch(t){console.error("Failed to refresh active trainings:",t)}}async refreshTrainingHistory(){try{let e=await(await fetch("/api/training/history")).json();this.trainingHistory=e.trainings||[],this.renderTrainingHistory()}catch(t){console.error("Failed to refresh training history:",t)}}renderTrainingHistory(){let t=this.container.querySelector('[data-target="history-rows"]');if(this.trainingHistory.length===0){t.innerHTML=`
                <tr>
                    <td colspan="8" class="empty-cell">No training history available</td>
                </tr>
            `;return}t.innerHTML=this.trainingHistory.map(e=>`
            <tr>
                <td><code>${e.training_id.substring(0,8)}...</code></td>
                <td>${e.detector_id.substring(0,8)}...</td>
                <td>${e.best_algorithm||"N/A"}</td>
                <td>${e.best_score?e.best_score.toFixed(4):"N/A"}</td>
                <td>${e.training_time_seconds?this.formatDuration(e.training_time_seconds):"N/A"}</td>
                <td><span class="status-badge status-badge--${this.getStatusClass(e.status)}">${e.status}</span></td>
                <td>${e.start_time?new Date(e.start_time).toLocaleDateString():"N/A"}</td>
                <td>
                    <button class="btn btn--icon btn--sm" data-action="view-result" 
                            data-training-id="${e.training_id}" title="View Details">
                        <i class="icon-eye"></i>
                    </button>
                </td>
            </tr>
        `).join("")}formatDuration(t){return t<60?`${Math.round(t)}s`:t<3600?`${Math.round(t/60)}m`:`${Math.round(t/3600)}h`}async loadFormOptions(){try{let e=await(await fetch("/api/detectors")).json(),s=this.container.querySelector('select[name="detector_id"]');s.innerHTML='<option value="">Select detector...</option>'+e.map(r=>`<option value="${r.id}">${r.name}</option>`).join("");let a=await(await fetch("/api/datasets")).json(),n=this.container.querySelector('select[name="dataset_id"]');n.innerHTML='<option value="">Select dataset...</option>'+a.map(r=>`<option value="${r.id}">${r.name}</option>`).join("")}catch(t){console.error("Failed to load form options:",t)}}showStartTrainingModal(){this.container.querySelector('[data-modal="start-training"]').classList.add("modal--active")}closeModal(){this.container.querySelectorAll(".modal").forEach(e=>e.classList.remove("modal--active"))}async submitTraining(){let t=this.container.querySelector('[data-form="start-training"]'),e=new FormData(t),s={detector_id:e.get("detector_id"),dataset_id:e.get("dataset_id"),experiment_name:e.get("experiment_name")||null,optimization_objective:e.get("optimization_objective"),max_algorithms:parseInt(e.get("max_algorithms")),max_optimization_time:parseInt(e.get("max_optimization_time"))*60,enable_automl:e.has("enable_automl"),enable_ensemble:e.has("enable_ensemble"),enable_early_stopping:e.has("enable_early_stopping")};try{let i=await fetch("/api/training/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(s)});if(!i.ok)throw new Error(`HTTP ${i.status}: ${i.statusText}`);let a=await i.json();this.closeModal(),this.showSuccess(`Training started: ${a.training_id}`),await this.refreshActiveTrainings()}catch(i){console.error("Failed to start training:",i),this.showError(`Failed to start training: ${i.message}`)}}async cancelTraining(t){if(confirm("Are you sure you want to cancel this training?"))try{let e=await fetch(`/api/training/cancel/${t}`,{method:"POST"});if(!e.ok)throw new Error(`HTTP ${e.status}: ${e.statusText}`);this.showSuccess("Training cancelled successfully"),await this.refreshActiveTrainings()}catch(e){console.error("Failed to cancel training:",e),this.showError(`Failed to cancel training: ${e.message}`)}}selectTraining(t){let e=this.activeTrainings.get(t);e&&(this.updateTrainingDetails(e),this.updateMetricsChart(e),this.container.querySelectorAll(".training-item").forEach(s=>{s.classList.toggle("training-item--selected",s.dataset.trainingId===t)}))}getSelectedTrainingId(){return this.container.querySelector(".training-item--selected")?.dataset.trainingId||null}startAutoRefresh(){setInterval(()=>{document.visibilityState==="visible"&&this.refreshActiveTrainings()},this.options.refreshInterval)}async refreshAll(){await Promise.all([this.refreshActiveTrainings(),this.refreshTrainingHistory()])}showSuccess(t){console.log("Success:",t)}showError(t){console.error("Error:",t)}showSettings(){console.log("Show settings")}destroy(){this.websocketService&&this.websocketService.disconnect(),this.listeners.clear(),this.activeTrainings.clear()}},u=c;})();
