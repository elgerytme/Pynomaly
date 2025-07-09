(()=>{var y=class{constructor(){this.charts=new Map,this.datasets=new Map,this.results=new Map,this.currentDataset=null,this.currentResult=null,this.isInitialized=!1,this.init()}async init(){await this.loadCachedData(),this.setupEventListeners(),this.isInitialized=!0}async loadCachedData(){try{if("serviceWorker"in navigator){let t=await navigator.serviceWorker.getRegistration();if(t?.active)return t.active.postMessage({type:"GET_OFFLINE_DATASETS"}),t.active.postMessage({type:"GET_OFFLINE_RESULTS"}),new Promise(a=>{let e=!1,r=!1;navigator.serviceWorker.addEventListener("message",s=>{s.data.type==="OFFLINE_DATASETS"?(s.data.datasets.forEach(i=>{this.datasets.set(i.id,i)}),e=!0):s.data.type==="OFFLINE_RESULTS"&&(s.data.results.forEach(i=>{this.results.set(i.id,i)}),r=!0),e&&r&&a()})})}}catch(t){console.error("[OfflineVisualizer] Failed to load cached data:",t)}}setupEventListeners(){document.addEventListener("change",t=>{t.target.matches(".dataset-selector")&&this.selectDataset(t.target.value)}),document.addEventListener("change",t=>{t.target.matches(".result-selector")&&this.selectResult(t.target.value)}),document.addEventListener("change",t=>{t.target.matches(".viz-type-selector")&&this.changeVisualizationType(t.target.value)}),document.addEventListener("click",t=>{t.target.matches(".export-viz")&&this.exportVisualization(t.target.dataset.format)})}async selectDataset(t){let a=this.datasets.get(t);a&&(this.currentDataset=a,await this.renderDatasetVisualization(),this.updateResultSelector())}async selectResult(t){let a=this.results.get(t);a&&(this.currentResult=a,await this.renderResultVisualization())}async renderDatasetVisualization(){if(!this.currentDataset)return;let t=document.getElementById("dataset-visualization");if(!t)return;let a=this.currentDataset.data,e=this.extractFeatures(a);this.clearCharts(),t.innerHTML=`
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Data Distribution</h3>
            <select class="form-select form-select-sm viz-type-selector" data-target="distribution">
              <option value="histogram">Histogram</option>
              <option value="boxplot">Box Plot</option>
              <option value="violin">Violin Plot</option>
            </select>
          </div>
          <div class="card-body">
            <div id="distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Feature Correlation</h3>
            <select class="form-select form-select-sm viz-type-selector" data-target="correlation">
              <option value="heatmap">Heatmap</option>
              <option value="scatter">Scatter Matrix</option>
            </select>
          </div>
          <div class="card-body">
            <div id="correlation-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Feature Statistics</h3>
          </div>
          <div class="card-body">
            <div id="statistics-table"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Data Quality</h3>
          </div>
          <div class="card-body">
            <div id="quality-chart" style="height: 300px;"></div>
          </div>
        </div>
      </div>
    `,await Promise.all([this.renderDistributionChart(e),this.renderCorrelationChart(e),this.renderStatisticsTable(e),this.renderQualityChart(e)])}async renderResultVisualization(){if(!this.currentResult)return;let t=document.getElementById("result-visualization");t&&(t.innerHTML=`
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Anomaly Distribution</h3>
          </div>
          <div class="card-body">
            <div id="anomaly-distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Score Distribution</h3>
          </div>
          <div class="card-body">
            <div id="score-distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Anomaly Scatter Plot</h3>
          </div>
          <div class="card-body">
            <div id="anomaly-scatter-chart" style="height: 400px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Detection Summary</h3>
          </div>
          <div class="card-body">
            <div id="detection-summary"></div>
          </div>
        </div>
      </div>
    `,await Promise.all([this.renderAnomalyDistributionChart(),this.renderScoreDistributionChart(),this.renderAnomalyScatterPlot(),this.renderDetectionSummary()]))}async renderDistributionChart(t){let a=document.getElementById("distribution-chart");if(!a||!t.length)return;let e=echarts.init(a),r=t.find(c=>c.type==="numeric");if(!r)return;let s=r.values,i=this.calculateHistogramBins(s,20),n={title:{text:`Distribution: ${r.name}`,textStyle:{fontSize:14}},tooltip:{trigger:"axis",axisPointer:{type:"shadow"}},xAxis:{type:"category",data:i.map(c=>c.range)},yAxis:{type:"value",name:"Frequency"},series:[{name:"Frequency",type:"bar",data:i.map(c=>c.count),itemStyle:{color:"#3b82f6"}}]};e.setOption(n),this.charts.set("distribution-chart",e)}async renderCorrelationChart(t){let a=document.getElementById("correlation-chart");if(!a)return;let e=t.filter(n=>n.type==="numeric");if(e.length<2)return;let r=this.calculateCorrelationMatrix(e),s=echarts.init(a),i={title:{text:"Feature Correlation Matrix",textStyle:{fontSize:14}},tooltip:{position:"top",formatter:n=>`${n.name}<br/>Correlation: ${n.value[2].toFixed(3)}`},grid:{height:"50%",top:"10%"},xAxis:{type:"category",data:e.map(n=>n.name),splitArea:{show:!0}},yAxis:{type:"category",data:e.map(n=>n.name),splitArea:{show:!0}},visualMap:{min:-1,max:1,calculable:!0,orient:"horizontal",left:"center",bottom:"15%",inRange:{color:["#313695","#4575b4","#74add1","#abd9e9","#e0f3f8","#ffffbf","#fee090","#fdae61","#f46d43","#d73027","#a50026"]}},series:[{name:"Correlation",type:"heatmap",data:r,label:{show:!0,formatter:n=>n.value[2].toFixed(2)},emphasis:{itemStyle:{shadowBlur:10,shadowColor:"rgba(0, 0, 0, 0.5)"}}}]};s.setOption(i),this.charts.set("correlation-chart",s)}renderStatisticsTable(t){let a=document.getElementById("statistics-table");if(!a)return;let r=t.filter(s=>s.type==="numeric").map(s=>{let i=this.calculateBasicStats(s.values);return`
        <tr>
          <td class="font-medium">${s.name}</td>
          <td>${i.mean.toFixed(3)}</td>
          <td>${i.std.toFixed(3)}</td>
          <td>${i.min.toFixed(3)}</td>
          <td>${i.max.toFixed(3)}</td>
          <td>${i.median.toFixed(3)}</td>
        </tr>
      `}).join("");a.innerHTML=`
      <div class="overflow-x-auto">
        <table class="table table-striped">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Mean</th>
              <th>Std Dev</th>
              <th>Min</th>
              <th>Max</th>
              <th>Median</th>
            </tr>
          </thead>
          <tbody>
            ${r}
          </tbody>
        </table>
      </div>
    `}async renderQualityChart(t){let a=document.getElementById("quality-chart");if(!a)return;let e=t.map(i=>{let n=i.values.length,c=i.values.filter(d=>d==null||d==="").length,u=(n-c)/n*100;return{name:i.name,completeness:u,uniqueness:this.calculateUniqueness(i.values),validity:this.calculateValidity(i.values,i.type)}}),r=echarts.init(a),s={title:{text:"Data Quality Metrics",textStyle:{fontSize:14}},tooltip:{trigger:"axis",axisPointer:{type:"shadow"}},legend:{data:["Completeness","Uniqueness","Validity"],bottom:0},xAxis:{type:"category",data:e.map(i=>i.name),axisLabel:{rotate:45}},yAxis:{type:"value",name:"Percentage",max:100},series:[{name:"Completeness",type:"bar",data:e.map(i=>i.completeness),itemStyle:{color:"#10b981"}},{name:"Uniqueness",type:"bar",data:e.map(i=>i.uniqueness),itemStyle:{color:"#3b82f6"}},{name:"Validity",type:"bar",data:e.map(i=>i.validity),itemStyle:{color:"#f59e0b"}}]};r.setOption(s),this.charts.set("quality-chart",r)}async renderAnomalyDistributionChart(){let t=document.getElementById("anomaly-distribution-chart");if(!t||!this.currentResult)return;let a=this.currentResult,e=a.statistics?.totalSamples||0,r=a.statistics?.totalAnomalies||0,s=e-r,i=echarts.init(t),n={title:{text:"Normal vs Anomalous Data",textStyle:{fontSize:14}},tooltip:{trigger:"item",formatter:"{a} <br/>{b}: {c} ({d}%)"},series:[{name:"Data Distribution",type:"pie",radius:["40%","70%"],data:[{value:s,name:"Normal",itemStyle:{color:"#10b981"}},{value:r,name:"Anomalous",itemStyle:{color:"#ef4444"}}],emphasis:{itemStyle:{shadowBlur:10,shadowOffsetX:0,shadowColor:"rgba(0, 0, 0, 0.5)"}}}]};i.setOption(n),this.charts.set("anomaly-distribution-chart",i)}async renderScoreDistributionChart(){let t=document.getElementById("score-distribution-chart");if(!t||!this.currentResult)return;let a=this.currentResult.scores||[];if(!a.length)return;let e=this.calculateHistogramBins(a,30),r=echarts.init(t),s={title:{text:"Anomaly Score Distribution",textStyle:{fontSize:14}},tooltip:{trigger:"axis",axisPointer:{type:"shadow"}},xAxis:{type:"category",data:e.map(i=>i.range),name:"Anomaly Score"},yAxis:{type:"value",name:"Frequency"},series:[{name:"Frequency",type:"bar",data:e.map(i=>i.count),itemStyle:{color:"#8b5cf6"}}]};r.setOption(s),this.charts.set("score-distribution-chart",r)}async renderAnomalyScatterPlot(){let t=document.getElementById("anomaly-scatter-chart");if(!t||!this.currentResult||!this.currentDataset)return;let a=this.currentDataset.data,e=this.currentResult.anomalies||[],r=this.currentResult.scores||[],i=this.extractFeatures(a).filter(o=>o.type==="numeric").slice(0,2);if(i.length<2)return;let n=[],c=[],u=new Set(e.map(o=>o.index));a.forEach((o,h)=>{let m=[o[i[0].name]||0,o[i[1].name]||0,r[h]||0];u.has(h)?c.push(m):n.push(m)});let d=echarts.init(t),l={title:{text:`${i[0].name} vs ${i[1].name}`,textStyle:{fontSize:14}},tooltip:{trigger:"item",formatter:o=>{let[h,m,p]=o.data;return`${o.seriesName}<br/>
                  ${i[0].name}: ${h.toFixed(3)}<br/>
                  ${i[1].name}: ${m.toFixed(3)}<br/>
                  Score: ${p.toFixed(3)}`}},legend:{data:["Normal","Anomaly"],bottom:0},xAxis:{type:"value",name:i[0].name,scale:!0},yAxis:{type:"value",name:i[1].name,scale:!0},series:[{name:"Normal",type:"scatter",data:n,itemStyle:{color:"#10b981",opacity:.7},symbolSize:6},{name:"Anomaly",type:"scatter",data:c,itemStyle:{color:"#ef4444",opacity:.9},symbolSize:10}]};d.setOption(l),this.charts.set("anomaly-scatter-chart",d)}renderDetectionSummary(){let t=document.getElementById("detection-summary");if(!t||!this.currentResult)return;let a=this.currentResult,e=a.statistics||{};t.innerHTML=`
      <div class="space-y-4">
        <div class="grid grid-cols-2 gap-4">
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Algorithm</div>
            <div class="font-semibold">${a.algorithmId||"Unknown"}</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Processing Time</div>
            <div class="font-semibold">${a.processingTimeMs||0}ms</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Total Samples</div>
            <div class="font-semibold">${e.totalSamples||0}</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Anomaly Rate</div>
            <div class="font-semibold">${((e.anomalyRate||0)*100).toFixed(2)}%</div>
          </div>
        </div>

        <div class="bg-blue-50 p-4 rounded border border-blue-200">
          <h4 class="font-medium text-blue-900 mb-2">Detection Parameters</h4>
          <div class="text-sm text-blue-800">
            ${Object.entries(a.parameters||{}).map(([r,s])=>`<div><strong>${r}:</strong> ${s}</div>`).join("")}
          </div>
        </div>

        <div class="flex gap-2">
          <button class="btn-base btn-sm btn-primary export-viz" data-format="png">
            Export as PNG
          </button>
          <button class="btn-base btn-sm btn-secondary export-viz" data-format="pdf">
            Export as PDF
          </button>
        </div>
      </div>
    `}extractFeatures(t){if(!t||!t.length)return[];let a=[],e=t[0];return Object.keys(e).forEach(r=>{let s=t.map(n=>n[r]),i=this.inferDataType(s);a.push({name:r,type:i,values:s.filter(n=>n!=null)})}),a}inferDataType(t){let a=t.filter(s=>s!=null&&s!=="");return a.length?a.filter(s=>!isNaN(parseFloat(s))).length/a.length>.8?"numeric":"categorical":"unknown"}calculateHistogramBins(t,a){let e=Math.min(...t),s=(Math.max(...t)-e)/a,i=[];for(let n=0;n<a;n++){let c=e+n*s,u=e+(n+1)*s,d=t.filter(l=>l>=c&&(n===a-1?l<=u:l<u)).length;i.push({range:`${c.toFixed(2)}-${u.toFixed(2)}`,count:d})}return i}calculateCorrelationMatrix(t){let a=[];for(let e=0;e<t.length;e++)for(let r=0;r<t.length;r++){let s=this.calculateCorrelation(t[e].values,t[r].values);a.push([e,r,s])}return a}calculateCorrelation(t,a){let e=Math.min(t.length,a.length);if(e<2)return 0;let r=t.slice(0,e).reduce((l,o)=>l+o,0),s=a.slice(0,e).reduce((l,o)=>l+o,0),i=t.slice(0,e).reduce((l,o,h)=>l+o*a[h],0),n=t.slice(0,e).reduce((l,o)=>l+o*o,0),c=a.slice(0,e).reduce((l,o)=>l+o*o,0),u=e*i-r*s,d=Math.sqrt((e*n-r*r)*(e*c-s*s));return d===0?0:u/d}calculateBasicStats(t){let a=[...t].sort((s,i)=>s-i),e=t.reduce((s,i)=>s+i,0)/t.length,r=t.reduce((s,i)=>s+Math.pow(i-e,2),0)/t.length;return{mean:e,std:Math.sqrt(r),min:Math.min(...t),max:Math.max(...t),median:a[Math.floor(a.length/2)]}}calculateUniqueness(t){return new Set(t).size/t.length*100}calculateValidity(t,a){return a==="numeric"?t.filter(s=>!isNaN(parseFloat(s))).length/t.length*100:t.filter(r=>r!=null&&r!=="").length/t.length*100}updateResultSelector(){let t=document.querySelector(".result-selector");if(!t||!this.currentDataset)return;let a=Array.from(this.results.values()).filter(e=>e.datasetId===this.currentDataset.id);t.innerHTML=`
      <option value="">Select a result...</option>
      ${a.map(e=>`
        <option value="${e.id}">
          ${e.algorithmId} - ${new Date(e.timestamp).toLocaleDateString()}
        </option>
      `).join("")}
    `}changeVisualizationType(t){console.log("Changing visualization type to:",t)}exportVisualization(t){console.log("Exporting visualization as:",t)}clearCharts(){this.charts.forEach(t=>{t.dispose()}),this.charts.clear()}getAvailableDatasets(){return Array.from(this.datasets.values())}getAvailableResults(){return Array.from(this.results.values())}};typeof window<"u"&&(window.OfflineVisualizer=new y);})();
