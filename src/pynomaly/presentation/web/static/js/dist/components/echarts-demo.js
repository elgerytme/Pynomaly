(()=>{var l=(c,t)=>()=>(t||c((t={exports:{}}).exports,t),t.exports);var m=l((p,d)=>{var r=class{constructor(){this.charts=new Map,this.demoData=this.generateDemoData(),this.realTimeInterval=null,this.isRealTimeEnabled=!1,this.init()}init(){this.setupDemoControls(),this.createAllDemos(),this.setupEventListeners()}generateDemoData(){let t=new Date,e=[];for(let o=0;o<50;o++){let s=new Date(t.getTime()-(50-o)*6e4);e.push({timestamp:s.toISOString(),cpu:Math.random()*80+10,memory:Math.random()*70+20,network:Math.random()*50+5,disk:Math.random()*60+10})}let a=["Statistical Outlier","Temporal Anomaly","Pattern Deviation","Threshold Violation","Trend Anomaly"],i=[];for(let o=0;o<100;o++){let s=Math.random();i.push({id:o,type:a[Math.floor(Math.random()*a.length)],confidence:s,score:s,timestamp:new Date(t.getTime()-Math.random()*864e5*7).toISOString(),severity:s>.8?"high":s>.5?"medium":"low"})}let n=[];for(let o=0;o<200;o++){let s=Math.random();n.push({timestamp:new Date(t.getTime()-Math.random()*864e5).toISOString(),confidence:s,score:s,type:a[Math.floor(Math.random()*a.length)]})}return{performance:e,anomalies:i,timeline:n}}setupDemoControls(){let t=document.getElementById("echarts-demo-controls");t&&(t.innerHTML=`
      <div class="demo-controls-grid">
        <div class="control-group">
          <h3>Real-Time Simulation</h3>
          <button id="echarts-realtime-toggle" class="btn btn-primary">Start Real-Time</button>
          <label>
            Update Interval:
            <select id="echarts-update-interval">
              <option value="1000">1 second</option>
              <option value="2000" selected>2 seconds</option>
              <option value="5000">5 seconds</option>
            </select>
          </label>
        </div>
        
        <div class="control-group">
          <h3>Theme</h3>
          <button id="echarts-theme-toggle" class="btn btn-secondary">Switch to Dark</button>
          <label>
            <input type="checkbox" id="echarts-high-contrast" /> High Contrast
          </label>
        </div>
        
        <div class="control-group">
          <h3>Chart Controls</h3>
          <button id="echarts-refresh-data" class="btn btn-secondary">Refresh Data</button>
          <button id="echarts-export-charts" class="btn btn-secondary">Export Charts</button>
        </div>
        
        <div class="control-group">
          <h3>Performance</h3>
          <label>
            Metrics:
            <select id="performance-metrics" multiple>
              <option value="cpu" selected>CPU</option>
              <option value="memory" selected>Memory</option>
              <option value="network" selected>Network</option>
              <option value="disk">Disk I/O</option>
            </select>
          </label>
        </div>
        
        <div class="control-group">
          <h3>Anomaly Charts</h3>
          <label>
            Distribution Type:
            <select id="distribution-type">
              <option value="pie" selected>Pie Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="histogram">Histogram</option>
            </select>
          </label>
        </div>
        
        <div class="control-group">
          <h3>Timeline</h3>
          <label>
            Time Range:
            <select id="timeline-range">
              <option value="1h">1 Hour</option>
              <option value="6h">6 Hours</option>
              <option value="24h" selected>24 Hours</option>
              <option value="7d">7 Days</option>
              <option value="30d">30 Days</option>
            </select>
          </label>
        </div>
      </div>
      
      <div id="echarts-announcer" aria-live="polite" class="sr-only"></div>
    `)}createAllDemos(){this.createPerformanceDemo(),this.createAnomalyDistributionDemo(),this.createTimelineDemo(),this.createDashboardDemo()}createPerformanceDemo(){let t=document.getElementById("performance-demo");if(!t)return;t.innerHTML=`
      <div class="demo-section">
        <h2>Performance Metrics Chart</h2>
        <p>Real-time system performance monitoring with CPU, memory, network, and disk utilization metrics.</p>
        
        <div class="chart-controls">
          <label>
            <input type="checkbox" id="performance-animation" checked /> Enable Animations
          </label>
          <label>
            Max Data Points:
            <select id="performance-max-points">
              <option value="50">50 points</option>
              <option value="100" selected>100 points</option>
              <option value="200">200 points</option>
            </select>
          </label>
          <button id="performance-reset" class="btn btn-sm">Reset Data</button>
        </div>
        
        <div id="performance-chart" class="chart-container" style="height: 400px;"></div>
        
        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Real-time data streaming</li>
            <li>Multiple metric tracking</li>
            <li>Smooth line animations</li>
            <li>Interactive tooltips</li>
            <li>Responsive design</li>
            <li>Alert notifications</li>
          </ul>
        </div>
      </div>
    `;let e=new PerformanceMetricsChart("#performance-chart",{title:"System Performance Metrics",description:"Real-time monitoring of system CPU, memory, and network utilization",metrics:["cpu","memory","network"],updateInterval:0,maxDataPoints:100,animation:!0});e.setData(this.demoData.performance),this.charts.set("performance",e),this.setupPerformanceControls()}setupPerformanceControls(){document.getElementById("performance-animation")?.addEventListener("change",t=>{let e=this.charts.get("performance");e&&(e.options.animation=t.target.checked,e.updateChart())}),document.getElementById("performance-max-points")?.addEventListener("change",t=>{let e=this.charts.get("performance");e&&(e.options.maxDataPoints=parseInt(t.target.value),e.setData(this.demoData.performance.slice(-e.options.maxDataPoints)))}),document.getElementById("performance-reset")?.addEventListener("click",()=>{this.demoData.performance=this.generateDemoData().performance;let t=this.charts.get("performance");t&&t.setData(this.demoData.performance)}),document.getElementById("performance-metrics")?.addEventListener("change",t=>{let e=Array.from(t.target.selectedOptions).map(i=>i.value),a=this.charts.get("performance");a&&(a.options.metrics=e,a.updateChart())})}createAnomalyDistributionDemo(){let t=document.getElementById("anomaly-distribution-demo");if(!t)return;t.innerHTML=`
      <div class="demo-section">
        <h2>Anomaly Distribution Analysis</h2>
        <p>Statistical visualization of detected anomalies by type, confidence level, and distribution patterns.</p>
        
        <div class="chart-controls">
          <label>
            Chart Type:
            <select id="distribution-chart-type">
              <option value="pie" selected>Pie Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="histogram">Histogram</option>
            </select>
          </label>
          <label>
            <input type="checkbox" id="distribution-animation" checked /> Enable Animations
          </label>
          <button id="distribution-randomize" class="btn btn-sm">Randomize Data</button>
        </div>
        
        <div id="anomaly-distribution-chart" class="chart-container" style="height: 400px;"></div>
        
        <div class="stats-panel">
          <h4>Statistics:</h4>
          <div id="distribution-stats">
            <div class="stat-item">
              <span class="stat-label">Total Anomalies:</span>
              <span class="stat-value" id="total-anomalies-count">-</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Average Confidence:</span>
              <span class="stat-value" id="avg-confidence-value">-</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">High Confidence:</span>
              <span class="stat-value" id="high-confidence-count">-</span>
            </div>
          </div>
        </div>
        
        <div class="chart-info">
          <h4>Chart Types:</h4>
          <ul>
            <li><strong>Pie Chart</strong> - Distribution by anomaly type</li>
            <li><strong>Bar Chart</strong> - Confidence level ranges</li>
            <li><strong>Histogram</strong> - Score distribution density</li>
          </ul>
        </div>
      </div>
    `;let e=new AnomalyDistributionChart("#anomaly-distribution-chart",{title:"Anomaly Distribution by Type",description:"Pie chart showing distribution of anomaly types detected in the system",chartType:"pie",animation:!0});e.setData(this.demoData.anomalies),this.charts.set("distribution",e),this.updateDistributionStats(),this.setupDistributionControls()}setupDistributionControls(){document.getElementById("distribution-chart-type")?.addEventListener("change",t=>{let e=this.charts.get("distribution");if(e){e.options.chartType=t.target.value;let a={pie:"Anomaly Distribution by Type",bar:"Anomalies by Confidence Level",histogram:"Anomaly Score Distribution"};e.options.title=a[t.target.value],e.updateChart()}}),document.getElementById("distribution-animation")?.addEventListener("change",t=>{let e=this.charts.get("distribution");e&&(e.options.animation=t.target.checked,e.updateChart())}),document.getElementById("distribution-randomize")?.addEventListener("click",()=>{this.demoData.anomalies=this.generateDemoData().anomalies;let t=this.charts.get("distribution");t&&(t.setData(this.demoData.anomalies),this.updateDistributionStats())}),document.getElementById("distribution-type")?.addEventListener("change",t=>{let e=this.charts.get("distribution");e&&(e.options.chartType=t.target.value,e.updateChart())})}updateDistributionStats(){let t=this.demoData.anomalies,e=t.length,a=t.reduce((n,o)=>n+o.confidence,0)/e,i=t.filter(n=>n.confidence>.8).length;document.getElementById("total-anomalies-count").textContent=e,document.getElementById("avg-confidence-value").textContent=(a*100).toFixed(1)+"%",document.getElementById("high-confidence-count").textContent=i}createTimelineDemo(){let t=document.getElementById("timeline-demo");if(!t)return;t.innerHTML=`
      <div class="demo-section">
        <h2>Detection Timeline</h2>
        <p>Chronological visualization of anomaly detection events with severity indicators and time-based analysis.</p>
        
        <div class="chart-controls">
          <label>
            Time Range:
            <select id="timeline-time-range">
              <option value="1h">1 Hour</option>
              <option value="6h">6 Hours</option>
              <option value="24h" selected>24 Hours</option>
              <option value="7d">7 Days</option>
              <option value="30d">30 Days</option>
            </select>
          </label>
          <label>
            <input type="checkbox" id="timeline-severity" checked /> Show Severity Levels
          </label>
          <button id="timeline-refresh" class="btn btn-sm">Refresh Timeline</button>
        </div>
        
        <div id="detection-timeline-chart" class="chart-container" style="height: 400px;"></div>
        
        <div class="timeline-summary">
          <h4>Timeline Summary:</h4>
          <div id="timeline-summary-content">
            <div class="summary-item">
              <span class="severity-indicator critical"></span>
              <span>Critical: <strong id="critical-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator high"></span>
              <span>High: <strong id="high-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator medium"></span>
              <span>Medium: <strong id="medium-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator low"></span>
              <span>Low: <strong id="low-count">0</strong></span>
            </div>
          </div>
        </div>
        
        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Multiple time range options</li>
            <li>Severity level stacking</li>
            <li>Interactive timeline navigation</li>
            <li>Real-time updates</li>
            <li>Trend analysis</li>
          </ul>
        </div>
      </div>
    `;let e=new DetectionTimelineChart("#detection-timeline-chart",{title:"Anomaly Detection Timeline - 24 Hours",description:"Stacked bar chart showing anomaly detection events over time by severity level",timeRange:"24h",showSeverityLevels:!0,animation:!0});e.setData(this.demoData.timeline),this.charts.set("timeline",e),this.updateTimelineSummary(),this.setupTimelineControls()}setupTimelineControls(){document.getElementById("timeline-time-range")?.addEventListener("change",t=>{let e=this.charts.get("timeline");e&&(e.options.timeRange=t.target.value,e.options.title=`Anomaly Detection Timeline - ${t.target.value.toUpperCase()}`,e.updateChart())}),document.getElementById("timeline-severity")?.addEventListener("change",t=>{let e=this.charts.get("timeline");e&&(e.options.showSeverityLevels=t.target.checked,e.updateChart())}),document.getElementById("timeline-refresh")?.addEventListener("click",()=>{this.demoData.timeline=this.generateDemoData().timeline;let t=this.charts.get("timeline");t&&(t.setData(this.demoData.timeline),this.updateTimelineSummary())}),document.getElementById("timeline-range")?.addEventListener("change",t=>{let e=this.charts.get("timeline");e&&(e.options.timeRange=t.target.value,e.updateChart())})}updateTimelineSummary(){let t=this.demoData.timeline,e={critical:0,high:0,medium:0,low:0};t.forEach(a=>{let i=a.confidence||0;i>=.95?e.critical++:i>=.8?e.high++:i>=.5?e.medium++:e.low++}),document.getElementById("critical-count").textContent=e.critical,document.getElementById("high-count").textContent=e.high,document.getElementById("medium-count").textContent=e.medium,document.getElementById("low-count").textContent=e.low}createDashboardDemo(){let t=document.getElementById("echarts-dashboard-demo");t&&(t.innerHTML=`
      <div class="demo-section">
        <h2>Integrated Dashboard</h2>
        <p>Comprehensive anomaly detection dashboard combining multiple chart types for complete system overview.</p>
        
        <div class="dashboard-grid">
          <div class="dashboard-widget">
            <h3>Performance Overview</h3>
            <div id="dashboard-performance" class="mini-chart" style="height: 200px;"></div>
          </div>
          
          <div class="dashboard-widget">
            <h3>Anomaly Types</h3>
            <div id="dashboard-distribution" class="mini-chart" style="height: 200px;"></div>
          </div>
          
          <div class="dashboard-widget">
            <h3>Detection Events</h3>
            <div id="dashboard-timeline" class="mini-chart" style="height: 200px;"></div>
          </div>
          
          <div class="dashboard-widget stats-summary">
            <h3>Key Metrics</h3>
            <div class="metrics-grid">
              <div class="metric">
                <div class="metric-value" id="dashboard-total-anomalies">0</div>
                <div class="metric-label">Total Anomalies</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-avg-confidence">0%</div>
                <div class="metric-label">Avg Confidence</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-critical-alerts">0</div>
                <div class="metric-label">Critical Alerts</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-system-health">Good</div>
                <div class="metric-label">System Health</div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="dashboard-controls">
          <button id="dashboard-refresh" class="btn btn-primary">Refresh Dashboard</button>
          <button id="dashboard-export" class="btn btn-secondary">Export Data</button>
          <label>
            <input type="checkbox" id="dashboard-realtime" /> Real-time Updates
          </label>
        </div>
      </div>
    `,this.createDashboardCharts(),this.updateDashboardMetrics())}createDashboardCharts(){let t=new PerformanceMetricsChart("#dashboard-performance",{title:"",metrics:["cpu","memory"],updateInterval:0,maxDataPoints:20,animation:!1});t.setData(this.demoData.performance.slice(-20)),this.charts.set("dashboard-performance",t);let e=new AnomalyDistributionChart("#dashboard-distribution",{title:"",chartType:"pie",animation:!1});e.setData(this.demoData.anomalies.slice(0,50)),this.charts.set("dashboard-distribution",e);let a=new DetectionTimelineChart("#dashboard-timeline",{title:"",timeRange:"6h",animation:!1});a.setData(this.demoData.timeline.slice(-50)),this.charts.set("dashboard-timeline",a),this.setupDashboardControls()}setupDashboardControls(){document.getElementById("dashboard-refresh")?.addEventListener("click",()=>{this.refreshDashboard()}),document.getElementById("dashboard-export")?.addEventListener("click",()=>{this.exportDashboardData()}),document.getElementById("dashboard-realtime")?.addEventListener("change",t=>{t.target.checked?this.startDashboardRealTime():this.stopDashboardRealTime()})}updateDashboardMetrics(){let t=this.demoData.anomalies.length,e=this.demoData.anomalies.reduce((n,o)=>n+o.confidence,0)/t,a=this.demoData.anomalies.filter(n=>n.confidence>.95).length,i=a>5?"Critical":a>2?"Warning":"Good";document.getElementById("dashboard-total-anomalies").textContent=t,document.getElementById("dashboard-avg-confidence").textContent=(e*100).toFixed(1)+"%",document.getElementById("dashboard-critical-alerts").textContent=a,document.getElementById("dashboard-system-health").textContent=i}refreshDashboard(){this.demoData=this.generateDemoData();let t=this.charts.get("dashboard-performance");t&&t.setData(this.demoData.performance.slice(-20));let e=this.charts.get("dashboard-distribution");e&&e.setData(this.demoData.anomalies.slice(0,50));let a=this.charts.get("dashboard-timeline");a&&a.setData(this.demoData.timeline.slice(-50)),this.updateDashboardMetrics(),this.announceToScreenReader("Dashboard refreshed with new data")}startDashboardRealTime(){this.dashboardTimer||(this.dashboardTimer=setInterval(()=>{let t=new Date,e={timestamp:t.toISOString(),cpu:Math.random()*80+10,memory:Math.random()*70+20,network:Math.random()*50+5};this.demoData.performance.push(e),this.demoData.performance=this.demoData.performance.slice(-100);let a=this.charts.get("dashboard-performance");if(a&&a.setData(this.demoData.performance.slice(-20)),Math.random()<.1){let i={id:Date.now(),type:["Statistical Outlier","Temporal Anomaly","Pattern Deviation"][Math.floor(Math.random()*3)],confidence:Math.random(),timestamp:t.toISOString()};this.demoData.anomalies.push(i),this.demoData.timeline.push(i),this.updateDashboardMetrics()}},3e3))}stopDashboardRealTime(){this.dashboardTimer&&(clearInterval(this.dashboardTimer),this.dashboardTimer=null)}exportDashboardData(){let t={timestamp:new Date().toISOString(),performance:this.demoData.performance,anomalies:this.demoData.anomalies,timeline:this.demoData.timeline,metrics:{totalAnomalies:this.demoData.anomalies.length,avgConfidence:this.demoData.anomalies.reduce((n,o)=>n+o.confidence,0)/this.demoData.anomalies.length,criticalAlerts:this.demoData.anomalies.filter(n=>n.confidence>.95).length}},e=new Blob([JSON.stringify(t,null,2)],{type:"application/json"}),a=URL.createObjectURL(e),i=document.createElement("a");i.href=a,i.download=`echarts-dashboard-export-${Date.now()}.json`,document.body.appendChild(i),i.click(),document.body.removeChild(i),URL.revokeObjectURL(a),this.announceToScreenReader("Dashboard data exported successfully")}setupEventListeners(){document.getElementById("echarts-realtime-toggle")?.addEventListener("click",t=>{this.toggleRealTime(),t.target.textContent=this.isRealTimeEnabled?"Stop Real-Time":"Start Real-Time"}),document.getElementById("echarts-theme-toggle")?.addEventListener("click",t=>{let e=echartsManager.currentTheme==="light"?"dark":"light";this.switchTheme(e),t.target.textContent=`Switch to ${e==="light"?"Dark":"Light"}`}),document.getElementById("echarts-refresh-data")?.addEventListener("click",()=>{this.refreshAllData()}),document.getElementById("echarts-export-charts")?.addEventListener("click",()=>{this.exportCharts()})}toggleRealTime(){if(this.isRealTimeEnabled)clearInterval(this.realTimeInterval),this.isRealTimeEnabled=!1;else{let t=parseInt(document.getElementById("echarts-update-interval")?.value||"2000");this.realTimeInterval=setInterval(()=>{this.updateRealTimeData()},t),this.isRealTimeEnabled=!0}}updateRealTimeData(){let t=this.charts.get("performance");if(t&&t.addRandomDataPoint(),Math.random()<.05){let e=new Date,a={id:Date.now(),type:["Statistical Outlier","Temporal Anomaly","Pattern Deviation"][Math.floor(Math.random()*3)],confidence:Math.random(),timestamp:e.toISOString()};this.demoData.anomalies.push(a),this.demoData.timeline.push(a);let i=this.charts.get("distribution");i&&(i.setData(this.demoData.anomalies),this.updateDistributionStats());let n=this.charts.get("timeline");n&&(n.setData(this.demoData.timeline),this.updateTimelineSummary()),a.confidence>.9&&this.announceToScreenReader(`High confidence anomaly detected: ${a.type}, confidence ${(a.confidence*100).toFixed(1)}%`)}}switchTheme(t){document.documentElement.setAttribute("data-theme",t),document.dispatchEvent(new CustomEvent("theme-changed",{detail:{theme:t}}))}refreshAllData(){this.demoData=this.generateDemoData(),this.charts.forEach((t,e)=>{e.includes("performance")?t.setData(this.demoData.performance):e.includes("distribution")?(t.setData(this.demoData.anomalies),this.updateDistributionStats()):e.includes("timeline")&&(t.setData(this.demoData.timeline),this.updateTimelineSummary())}),this.updateDashboardMetrics(),this.announceToScreenReader("All chart data refreshed")}exportCharts(){let t={timestamp:new Date().toISOString(),chartTypes:["performance","distribution","timeline"],data:this.demoData,chartConfigurations:{}};this.charts.forEach((n,o)=>{t.chartConfigurations[o]={type:n.constructor.name,options:n.options}});let e=new Blob([JSON.stringify(t,null,2)],{type:"application/json"}),a=URL.createObjectURL(e),i=document.createElement("a");i.href=a,i.download=`echarts-demo-export-${Date.now()}.json`,document.body.appendChild(i),i.click(),document.body.removeChild(i),URL.revokeObjectURL(a),this.announceToScreenReader("ECharts data exported successfully")}announceToScreenReader(t){let e=document.getElementById("echarts-announcer");e&&(e.textContent=t)}destroy(){this.realTimeInterval&&clearInterval(this.realTimeInterval),this.dashboardTimer&&clearInterval(this.dashboardTimer),this.charts.forEach(t=>{t.dispose()}),this.charts.clear()}};document.addEventListener("DOMContentLoaded",()=>{document.querySelector(".echarts-demo")&&(window.echartsDemo=new r)});typeof d<"u"&&d.exports?d.exports=r:window.EChartsDemo=r});m();})();
