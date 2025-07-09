(()=>{var h=class{constructor(){this.charts=new Map,this.cachedData={datasets:[],results:[],stats:{},algorithms:[]},this.isInitialized=!1,this.init()}async init(){await this.loadCachedData(),this.setupEventListeners(),this.renderDashboard(),this.isInitialized=!0}async loadCachedData(){try{if("serviceWorker"in navigator){let t=await navigator.serviceWorker.getRegistration();if(t?.active)return t.active.postMessage({type:"GET_OFFLINE_DASHBOARD_DATA"}),new Promise(e=>{navigator.serviceWorker.addEventListener("message",function a(i){i.data.type==="OFFLINE_DASHBOARD_DATA"&&(navigator.serviceWorker.removeEventListener("message",a),this.cachedData={...this.cachedData,...i.data.data},e(i.data.data))}.bind(this))})}}catch(t){console.error("[OfflineDashboard] Failed to load cached data:",t)}}setupEventListeners(){document.addEventListener("change",t=>{t.target.matches(".dataset-selector")&&this.onDatasetChange(t.target.value)}),document.addEventListener("change",t=>{t.target.matches(".algorithm-selector")&&this.onAlgorithmChange(t.target.value)}),document.addEventListener("click",t=>{t.target.matches(".refresh-dashboard")&&this.refreshDashboard()}),document.addEventListener("click",t=>{t.target.matches(".export-chart")&&this.exportChart(t.target.dataset.chartId)})}renderDashboard(){this.renderOverviewCards(),this.renderDatasetChart(),this.renderAlgorithmPerformanceChart(),this.renderAnomalyTimelineChart(),this.renderRecentActivity()}renderOverviewCards(){let t=document.getElementById("overview-cards");if(!t)return;let e=this.calculateStats();t.innerHTML=`
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Total Datasets</p>
                <p class="text-2xl font-bold">${e.totalDatasets}</p>
              </div>
              <div class="text-3xl">\u{1F4CA}</div>
            </div>
            <div class="mt-2 text-sm text-green-600">
              ${e.datasetsLastWeek} added this week
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Detections Run</p>
                <p class="text-2xl font-bold">${e.totalDetections}</p>
              </div>
              <div class="text-3xl">\u{1F50D}</div>
            </div>
            <div class="mt-2 text-sm text-blue-600">
              ${e.detectionsToday} today
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Anomalies Found</p>
                <p class="text-2xl font-bold">${e.totalAnomalies}</p>
              </div>
              <div class="text-3xl">\u26A0\uFE0F</div>
            </div>
            <div class="mt-2 text-sm ${e.anomalyRate>.1?"text-red-600":"text-gray-600"}">
              ${(e.anomalyRate*100).toFixed(1)}% anomaly rate
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Cached Data</p>
                <p class="text-2xl font-bold">${this.formatBytes(e.cacheSize)}</p>
              </div>
              <div class="text-3xl">\u{1F4BE}</div>
            </div>
            <div class="mt-2 text-sm text-purple-600">
              Available offline
            </div>
          </div>
        </div>
      </div>
    `}renderDatasetChart(){let t=document.getElementById("dataset-chart");if(!t)return;let a=(this.cachedData.datasets||[]).reduce((n,c)=>{let s=c.type||"unknown";return n[s]=(n[s]||0)+1,n},{}),i=Object.entries(a).map(([n,c])=>({name:n.charAt(0).toUpperCase()+n.slice(1),value:c})),o=echarts.init(t),l={title:{text:"Dataset Distribution",left:"center"},tooltip:{trigger:"item",formatter:"{a} <br/>{b}: {c} ({d}%)"},legend:{orient:"vertical",left:"left"},series:[{name:"Datasets",type:"pie",radius:"50%",data:i,emphasis:{itemStyle:{shadowBlur:10,shadowOffsetX:0,shadowColor:"rgba(0, 0, 0, 0.5)"}}}]};o.setOption(l),this.charts.set("dataset-chart",o),window.addEventListener("resize",()=>o.resize())}renderAlgorithmPerformanceChart(){let t=document.getElementById("algorithm-performance-chart");if(!t)return;let a=(this.cachedData.results||[]).reduce((s,r)=>{let d=r.algorithm||"unknown";return s[d]||(s[d]={count:0,totalTime:0,totalAnomalies:0}),s[d].count++,s[d].totalTime+=r.processingTime||0,s[d].totalAnomalies+=r.anomalies?.length||0,s},{}),i=Object.keys(a),o=i.map(s=>a[s].totalTime/a[s].count),l=i.map(s=>a[s].totalAnomalies),n=echarts.init(t),c={title:{text:"Algorithm Performance",left:"center"},tooltip:{trigger:"axis",axisPointer:{type:"cross"}},legend:{data:["Average Processing Time (ms)","Total Anomalies Found"],bottom:0},xAxis:{type:"category",data:i,axisPointer:{type:"shadow"}},yAxis:[{type:"value",name:"Time (ms)",position:"left"},{type:"value",name:"Anomalies",position:"right"}],series:[{name:"Average Processing Time (ms)",type:"bar",yAxisIndex:0,data:o,itemStyle:{color:"#3b82f6"}},{name:"Total Anomalies Found",type:"line",yAxisIndex:1,data:l,itemStyle:{color:"#ef4444"}}]};n.setOption(c),this.charts.set("algorithm-performance-chart",n),window.addEventListener("resize",()=>n.resize())}renderAnomalyTimelineChart(){let t=document.getElementById("anomaly-timeline-chart");if(!t)return;let a=(this.cachedData.results||[]).reduce((s,r)=>{let d=new Date(r.timestamp).toISOString().split("T")[0];return s[d]||(s[d]={detections:0,anomalies:0}),s[d].detections++,s[d].anomalies+=r.anomalies?.length||0,s},{}),i=Object.keys(a).sort(),o=i.map(s=>a[s].detections),l=i.map(s=>a[s].anomalies),n=echarts.init(t),c={title:{text:"Detection Activity Timeline",left:"center"},tooltip:{trigger:"axis",axisPointer:{type:"cross"}},legend:{data:["Detections Run","Anomalies Found"],bottom:0},grid:{left:"3%",right:"4%",bottom:"15%",containLabel:!0},xAxis:{type:"category",boundaryGap:!1,data:i},yAxis:{type:"value"},series:[{name:"Detections Run",type:"line",stack:"Total",areaStyle:{},data:o,itemStyle:{color:"#10b981"}},{name:"Anomalies Found",type:"line",stack:"Total",areaStyle:{},data:l,itemStyle:{color:"#f59e0b"}}]};n.setOption(c),this.charts.set("anomaly-timeline-chart",n),window.addEventListener("resize",()=>n.resize())}renderRecentActivity(){let t=document.getElementById("recent-activity");if(!t)return;let i=(this.cachedData.results||[]).sort((o,l)=>new Date(l.timestamp)-new Date(o.timestamp)).slice(0,10).map(o=>{let l=this.timeAgo(new Date(o.timestamp)),n=o.anomalies?.length||0,c=n>0?"text-orange-600":"text-green-600";return`
        <div class="flex items-start gap-3 p-3 border-b border-border last:border-b-0">
          <div class="text-xl">${n>0?"\u26A0\uFE0F":"\u2705"}</div>
          <div class="flex-grow">
            <div class="flex items-center justify-between">
              <h4 class="font-medium">${o.dataset||"Unknown Dataset"}</h4>
              <span class="text-sm text-text-secondary">${l}</span>
            </div>
            <p class="text-sm text-text-secondary">
              Algorithm: ${o.algorithm||"Unknown"}
            </p>
            <p class="text-sm ${c}">
              ${n} anomalies detected
            </p>
          </div>
        </div>
      `}).join("");t.innerHTML=`
      <div class="card">
        <div class="card-header">
          <h3 class="card-title">Recent Activity</h3>
          <button class="btn-base btn-sm refresh-dashboard">
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"></path>
            </svg>
            Refresh
          </button>
        </div>
        <div class="card-body p-0">
          ${i||'<div class="p-4 text-center text-text-secondary">No recent activity</div>'}
        </div>
      </div>
    `}onDatasetChange(t){let e=this.cachedData.datasets.find(a=>a.id===t);e&&this.renderDatasetDetails(e)}onAlgorithmChange(t){let e=this.cachedData.algorithms.find(a=>a.id===t);e&&this.renderAlgorithmDetails(e)}async refreshDashboard(){let t=document.querySelector(".refresh-dashboard");t&&(t.disabled=!0,t.innerHTML="Refreshing...");try{await this.loadCachedData(),this.renderDashboard()}catch(e){console.error("[OfflineDashboard] Failed to refresh:",e)}finally{t&&(t.disabled=!1,t.innerHTML=`
          <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"></path>
          </svg>
          Refresh
        `)}}exportChart(t){let e=this.charts.get(t);if(e){let a=e.getDataURL({type:"png",pixelRatio:2,backgroundColor:"#fff"}),i=document.createElement("a");i.download=`${t}-${Date.now()}.png`,i.href=a,i.click()}}calculateStats(){let t=this.cachedData.datasets||[],e=this.cachedData.results||[],a=new Date,i=new Date(a.getTime()-7*24*60*60*1e3),o=new Date(a.getTime()-24*60*60*1e3),l=t.filter(r=>new Date(r.timestamp)>i).length,n=e.filter(r=>new Date(r.timestamp)>o).length,c=e.reduce((r,d)=>r+(d.anomalies?.length||0),0),s=e.reduce((r,d)=>r+(d.totalSamples||0),0);return{totalDatasets:t.length,datasetsLastWeek:l,totalDetections:e.length,detectionsToday:n,totalAnomalies:c,anomalyRate:s>0?c/s:0,cacheSize:this.estimateCacheSize()}}estimateCacheSize(){return JSON.stringify(this.cachedData).length*2}formatBytes(t){if(t===0)return"0 B";let e=1024,a=["B","KB","MB","GB"],i=Math.floor(Math.log(t)/Math.log(e));return parseFloat((t/Math.pow(e,i)).toFixed(2))+" "+a[i]}timeAgo(t){let a=Math.floor((new Date-t)/1e3);return a<60?"Just now":a<3600?`${Math.floor(a/60)}m ago`:a<86400?`${Math.floor(a/3600)}h ago`:`${Math.floor(a/86400)}d ago`}};typeof window<"u"&&(window.OfflineDashboard=new h);})();
