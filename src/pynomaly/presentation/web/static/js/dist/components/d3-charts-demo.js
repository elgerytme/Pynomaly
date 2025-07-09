(()=>{var b=(v,e)=>()=>(e||v((e={exports:{}}).exports,e),e.exports);var f=b((D,h)=>{var m=class{constructor(){this.charts=new Map,this.demoData=this.generateDemoData(),this.realTimeInterval=null,this.isRealTimeEnabled=!1,this.init()}init(){this.setupDemoControls(),this.createAllDemos(),this.setupEventListeners()}generateDemoData(){let e=new Date,t=[],a=[],n=[];for(let o=0;o<100;o++){let l=new Date(e.getTime()-(100-o)*6e4),r=50+20*Math.sin(o*.1)+Math.random()*10,c=Math.random()<.05,u=c?r+(Math.random()-.5)*80:r,d=c?.7+Math.random()*.3:.1+Math.random()*.3;t.push({timestamp:l,value:u,isAnomaly:c,confidence:d})}for(let o=0;o<200;o++){let l=Math.random()*100,r=Math.random()*100,c=Math.sqrt((l-25)**2+(r-25)**2)<15,u=Math.sqrt((l-75)**2+(r-75)**2)<15,d=!c&&!u&&Math.random()<.1,p=d?.7+Math.random()*.3:Math.random()*.3,g=d?.8+Math.random()*.2:.3+Math.random()*.4;a.push({x:l,y:r,anomalyScore:p,confidence:g,isAnomaly:d})}let i=["CPU","Memory","Disk","Network","Response Time"],s=["00:00","04:00","08:00","12:00","16:00","20:00"];for(let o of i)for(let l of s)n.push({x:l,y:o,value:Math.random()});return{timeSeries:t,scatter:a,heatmap:n}}setupDemoControls(){let e=document.getElementById("demo-controls");e&&(e.innerHTML=`
      <div class="demo-controls-grid">
        <div class="control-group">
          <h3>Real-Time Simulation</h3>
          <button id="realtime-toggle" class="btn btn-primary">Start Real-Time</button>
          <label>
            Update Interval:
            <select id="update-interval">
              <option value="1000">1 second</option>
              <option value="2000" selected>2 seconds</option>
              <option value="5000">5 seconds</option>
            </select>
          </label>
        </div>
        
        <div class="control-group">
          <h3>Theme</h3>
          <button id="theme-toggle" class="btn btn-secondary">Switch to Dark</button>
        </div>
        
        <div class="control-group">
          <h3>Chart Controls</h3>
          <button id="refresh-data" class="btn btn-secondary">Refresh Data</button>
          <button id="export-charts" class="btn btn-secondary">Export Charts</button>
        </div>
        
        <div class="control-group">
          <h3>Accessibility</h3>
          <button id="announce-data" class="btn btn-secondary">Announce Data</button>
          <label>
            <input type="checkbox" id="high-contrast" /> High Contrast
          </label>
        </div>
      </div>
      
      <div id="chart-announcer" aria-live="polite" class="sr-only"></div>
    `)}createAllDemos(){this.createTimeSeriesDemo(),this.createScatterPlotDemo(),this.createHeatmapDemo(),this.createInteractiveDemo()}createTimeSeriesDemo(){let e=document.getElementById("timeseries-demo");if(!e)return;e.innerHTML=`
      <div class="demo-section">
        <h2>Time Series Chart - Anomaly Detection Timeline</h2>
        <p>Interactive time series visualization showing anomaly detection results over time with real-time updates.</p>
        
        <div class="chart-controls">
          <label>
            <input type="checkbox" id="show-confidence" checked /> Show Confidence Bands
          </label>
          <label>
            <input type="checkbox" id="show-anomalies" checked /> Show Anomaly Markers
          </label>
          <button id="zoom-reset-ts" class="btn btn-sm">Reset Zoom</button>
        </div>
        
        <div id="timeseries-chart" class="chart-container"></div>
        
        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Real-time data updates</li>
            <li>Interactive tooltips</li>
            <li>Anomaly highlighting</li>
            <li>Confidence bands</li>
            <li>Keyboard navigation</li>
            <li>Screen reader support</li>
          </ul>
        </div>
      </div>
    `;let t=new TimeSeriesChart("#timeseries-chart",{title:"Anomaly Detection Timeline",description:"Time series chart showing detected anomalies with confidence intervals",showConfidenceBands:!0,animated:!0,responsive:!0});t.setData(this.demoData.timeSeries),this.charts.set("timeseries",t),document.getElementById("show-confidence")?.addEventListener("change",a=>{t.options.showConfidenceBands=a.target.checked,t.render()}),document.getElementById("show-anomalies")?.addEventListener("change",a=>{t.options.showAnomalies=a.target.checked,t.render()}),e.addEventListener("anomaly-selected",a=>{let{data:n}=a.detail;this.showAnomalyDetails(n,"Time Series")})}createScatterPlotDemo(){let e=document.getElementById("scatter-demo");if(!e)return;e.innerHTML=`
      <div class="demo-section">
        <h2>Scatter Plot - 2D Anomaly Detection</h2>
        <p>Interactive scatter plot for detecting anomalies in two-dimensional data space with brushing and zoom.</p>
        
        <div class="chart-controls">
          <label>
            <input type="checkbox" id="enable-brushing" checked /> Enable Brushing
          </label>
          <label>
            <input type="checkbox" id="enable-zoom" checked /> Enable Zoom
          </label>
          <button id="clear-selection" class="btn btn-sm">Clear Selection</button>
        </div>
        
        <div id="scatter-chart" class="chart-container"></div>
        
        <div class="selection-info">
          <h4>Selection Info:</h4>
          <div id="selection-details">No points selected</div>
        </div>
        
        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Brush selection</li>
            <li>Zoom and pan</li>
            <li>Color-coded anomaly scores</li>
            <li>Size-coded confidence</li>
            <li>Interactive legends</li>
          </ul>
        </div>
      </div>
    `;let t=new ScatterPlotChart("#scatter-chart",{title:"2D Anomaly Detection",description:"Scatter plot showing anomalies in two-dimensional feature space",xLabel:"Feature 1",yLabel:"Feature 2",enableBrushing:!0,enableZoom:!0,animated:!0});t.setData(this.demoData.scatter),this.charts.set("scatter",t),document.getElementById("enable-brushing")?.addEventListener("change",a=>{t.options.enableBrushing=a.target.checked,t.render()}),document.getElementById("enable-zoom")?.addEventListener("change",a=>{t.options.enableZoom=a.target.checked,t.render()}),e.addEventListener("points-selected",a=>{let{data:n}=a.detail;this.updateSelectionInfo(n)}),e.addEventListener("point-selected",a=>{let{data:n}=a.detail;this.showAnomalyDetails(n,"Scatter Plot")})}createHeatmapDemo(){let e=document.getElementById("heatmap-demo");if(!e)return;e.innerHTML=`
      <div class="demo-section">
        <h2>Heatmap - Feature Correlation Matrix</h2>
        <p>Interactive heatmap showing correlations between features and anomaly densities across time periods.</p>
        
        <div class="chart-controls">
          <label>
            <input type="checkbox" id="show-labels" checked /> Show Value Labels
          </label>
          <select id="color-scheme">
            <option value="interpolateViridis">Viridis</option>
            <option value="interpolatePlasma">Plasma</option>
            <option value="interpolateInferno">Inferno</option>
            <option value="interpolateTurbo">Turbo</option>
          </select>
        </div>
        
        <div id="heatmap-chart" class="chart-container"></div>
        
        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Interactive cells</li>
            <li>Color legends</li>
            <li>Value labels</li>
            <li>Multiple color schemes</li>
            <li>Responsive design</li>
          </ul>
        </div>
      </div>
    `;let t=new HeatmapChart("#heatmap-chart",{title:"Feature Correlation Heatmap",description:"Heatmap showing correlation values between different system features",showLabels:!0,animated:!0});t.setData(this.demoData.heatmap),this.charts.set("heatmap",t),document.getElementById("show-labels")?.addEventListener("change",a=>{t.options.showLabels=a.target.checked,t.render()}),document.getElementById("color-scheme")?.addEventListener("change",a=>{t.options.colorScheme=d3[a.target.value],t.render()}),e.addEventListener("cell-selected",a=>{let{data:n}=a.detail;this.showCellDetails(n)})}createInteractiveDemo(){let e=document.getElementById("interactive-demo");if(!e)return;e.innerHTML=`
      <div class="demo-section">
        <h2>Interactive Dashboard Demo</h2>
        <p>Combined visualization dashboard showing real-time anomaly detection across multiple chart types.</p>
        
        <div class="dashboard-grid">
          <div class="dashboard-item">
            <h3>Real-Time Stream</h3>
            <div id="realtime-chart" class="mini-chart"></div>
          </div>
          
          <div class="dashboard-item">
            <h3>Anomaly Distribution</h3>
            <div id="distribution-chart" class="mini-chart"></div>
          </div>
          
          <div class="dashboard-item">
            <h3>Feature Correlation</h3>
            <div id="correlation-chart" class="mini-chart"></div>
          </div>
          
          <div class="dashboard-item stats-panel">
            <h3>Statistics</h3>
            <div id="stats-content">
              <div class="stat-item">
                <span class="stat-label">Total Anomalies:</span>
                <span class="stat-value" id="total-anomalies">-</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">Avg Confidence:</span>
                <span class="stat-value" id="avg-confidence">-</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">Last Updated:</span>
                <span class="stat-value" id="last-updated">-</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;let t=new TimeSeriesChart("#realtime-chart",{title:"Real-time Anomaly Stream",height:200,margin:{top:10,right:20,bottom:30,left:40},showConfidenceBands:!1,animated:!1}),a=new ScatterPlotChart("#distribution-chart",{title:"Anomaly Distribution",height:200,margin:{top:10,right:20,bottom:30,left:40},enableBrushing:!1,enableZoom:!1,animated:!1}),n=new HeatmapChart("#correlation-chart",{title:"Feature Correlation",height:200,margin:{top:10,right:20,bottom:30,left:40},showLabels:!1,animated:!1});t.setData(this.demoData.timeSeries.slice(-20)),a.setData(this.demoData.scatter.slice(0,50)),n.setData(this.demoData.heatmap),this.charts.set("realtime",t),this.charts.set("distribution",a),this.charts.set("correlation",n),this.updateStatistics()}setupEventListeners(){document.getElementById("realtime-toggle")?.addEventListener("click",e=>{this.toggleRealTime(),e.target.textContent=this.isRealTimeEnabled?"Stop Real-Time":"Start Real-Time"}),document.getElementById("theme-toggle")?.addEventListener("click",e=>{let t=chartLibrary.currentTheme==="light"?"dark":"light";this.switchTheme(t),e.target.textContent=`Switch to ${t==="light"?"Dark":"Light"}`}),document.getElementById("refresh-data")?.addEventListener("click",()=>{this.refreshAllData()}),document.getElementById("export-charts")?.addEventListener("click",()=>{this.exportCharts()}),document.getElementById("announce-data")?.addEventListener("click",()=>{this.announceDataSummary()}),document.getElementById("high-contrast")?.addEventListener("change",e=>{document.body.classList.toggle("high-contrast",e.target.checked)})}toggleRealTime(){if(this.isRealTimeEnabled)clearInterval(this.realTimeInterval),this.isRealTimeEnabled=!1;else{let e=parseInt(document.getElementById("update-interval")?.value||"2000");this.realTimeInterval=setInterval(()=>{this.updateRealTimeData()},e),this.isRealTimeEnabled=!0}}updateRealTimeData(){let e=this.demoData.timeSeries[this.demoData.timeSeries.length-1],t=new Date,a=50+20*Math.sin(Date.now()*1e-4)+Math.random()*10,n=Math.random()<.05,i=n?a+(Math.random()-.5)*80:a,s={timestamp:t,value:i,isAnomaly:n,confidence:n?.7+Math.random()*.3:.1+Math.random()*.3},o=this.charts.get("timeseries");o&&o.addDataPoint(s);let l=this.charts.get("realtime");l&&l.addDataPoint(s,20),this.updateStatistics(),n&&s.confidence>.8&&this.announceToScreenReader(`High confidence anomaly detected: value ${i.toFixed(2)}, confidence ${(s.confidence*100).toFixed(1)}%`)}switchTheme(e){document.documentElement.setAttribute("data-theme",e),document.dispatchEvent(new CustomEvent("theme-changed",{detail:{theme:e}}))}refreshAllData(){this.demoData=this.generateDemoData(),this.charts.forEach((e,t)=>{t==="timeseries"||t==="realtime"?e.setData(this.demoData.timeSeries):t==="scatter"||t==="distribution"?e.setData(this.demoData.scatter):(t==="heatmap"||t==="correlation")&&e.setData(this.demoData.heatmap)}),this.updateStatistics(),this.announceToScreenReader("All chart data refreshed")}exportCharts(){let e={timestamp:new Date().toISOString(),charts:{}};this.charts.forEach((i,s)=>{i.data&&(e.charts[s]={type:i.constructor.name,data:i.data,options:i.options})});let t=new Blob([JSON.stringify(e,null,2)],{type:"application/json"}),a=URL.createObjectURL(t),n=document.createElement("a");n.href=a,n.download=`pynomaly-charts-export-${Date.now()}.json`,document.body.appendChild(n),n.click(),document.body.removeChild(n),URL.revokeObjectURL(a),this.announceToScreenReader("Chart data exported successfully")}updateStatistics(){let e=this.demoData.timeSeries.filter(n=>n.isAnomaly),t=e.length,a=t>0?e.reduce((n,i)=>n+i.confidence,0)/t:0;document.getElementById("total-anomalies").textContent=t,document.getElementById("avg-confidence").textContent=(a*100).toFixed(1)+"%",document.getElementById("last-updated").textContent=new Date().toLocaleTimeString()}updateSelectionInfo(e){let t=document.getElementById("selection-details");if(!t)return;if(e.length===0){t.textContent="No points selected";return}let a=e.filter(i=>i.isAnomaly).length,n=e.reduce((i,s)=>i+s.anomalyScore,0)/e.length;t.innerHTML=`
      <strong>${e.length} points selected</strong><br/>
      Anomalies: ${a}<br/>
      Average Score: ${n.toFixed(3)}
    `}showAnomalyDetails(e,t){let a=document.createElement("div");a.className="anomaly-modal",a.innerHTML=`
      <div class="modal-content">
        <div class="modal-header">
          <h3>Anomaly Details - ${t}</h3>
          <button class="modal-close" aria-label="Close">&times;</button>
        </div>
        <div class="modal-body">
          <div class="detail-grid">
            ${e.timestamp?`<div><strong>Timestamp:</strong> ${e.timestamp.toLocaleString()}</div>`:""}
            ${e.value!==void 0?`<div><strong>Value:</strong> ${e.value.toFixed(3)}</div>`:""}
            ${e.x!==void 0?`<div><strong>X:</strong> ${e.x.toFixed(3)}</div>`:""}
            ${e.y!==void 0?`<div><strong>Y:</strong> ${e.y.toFixed(3)}</div>`:""}
            ${e.confidence!==void 0?`<div><strong>Confidence:</strong> ${(e.confidence*100).toFixed(1)}%</div>`:""}
            ${e.anomalyScore!==void 0?`<div><strong>Anomaly Score:</strong> ${e.anomalyScore.toFixed(3)}</div>`:""}
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary modal-close">Close</button>
        </div>
      </div>
    `,document.body.appendChild(a),a.querySelectorAll(".modal-close").forEach(i=>{i.addEventListener("click",()=>{document.body.removeChild(a)})}),a.addEventListener("click",i=>{i.target===a&&document.body.removeChild(a)}),a.querySelector("button")?.focus()}showCellDetails(e){this.announceToScreenReader(`Heatmap cell selected: ${e.x}, ${e.y}, value ${e.value.toFixed(3)}`)}announceDataSummary(){let e=this.demoData.timeSeries.length,t=this.demoData.timeSeries.filter(i=>i.isAnomaly).length,a=(t/e*100).toFixed(1),n=`Data summary: ${e} total data points, ${t} anomalies detected, ${a}% anomaly rate`;this.announceToScreenReader(n)}announceToScreenReader(e){let t=document.getElementById("chart-announcer");t&&(t.textContent=e)}destroy(){this.realTimeInterval&&clearInterval(this.realTimeInterval),this.charts.forEach(e=>{e.destroy()}),this.charts.clear()}};document.addEventListener("DOMContentLoaded",()=>{document.querySelector(".d3-charts-demo")&&(window.d3ChartsDemo=new m)});typeof h<"u"&&h.exports?h.exports=m:window.D3ChartsDemo=m});f();})();
