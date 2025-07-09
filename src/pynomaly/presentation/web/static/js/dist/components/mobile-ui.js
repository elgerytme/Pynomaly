(()=>{var p=(d,e)=>()=>(e||d((e={exports:{}}).exports,e),e.exports);var m=p((v,h)=>{var l=class{constructor(e,t={}){this.element=e,this.options={swipeThreshold:50,tapTimeout:300,doubleTapTimeout:300,pinchThreshold:10,enablePinch:!0,enableSwipe:!0,enableTap:!0,enablePan:!0,preventDefault:!0,...t},this.touches=new Map,this.lastTap=null,this.isGesturing=!1,this.gestureStartDistance=0,this.gestureStartScale=1,this.listeners=new Map,this.init()}init(){this.bindEvents()}bindEvents(){this.element.addEventListener("touchstart",this.handleTouchStart.bind(this),{passive:!1}),this.element.addEventListener("touchmove",this.handleTouchMove.bind(this),{passive:!1}),this.element.addEventListener("touchend",this.handleTouchEnd.bind(this),{passive:!1}),this.element.addEventListener("touchcancel",this.handleTouchCancel.bind(this),{passive:!1}),this.element.addEventListener("mousedown",this.handleMouseDown.bind(this)),this.element.addEventListener("mousemove",this.handleMouseMove.bind(this)),this.element.addEventListener("mouseup",this.handleMouseUp.bind(this)),this.element.addEventListener("contextmenu",e=>{this.options.preventDefault&&e.preventDefault()})}handleTouchStart(e){this.options.preventDefault&&e.preventDefault(),Array.from(e.changedTouches).forEach(s=>{this.touches.set(s.identifier,{id:s.identifier,startX:s.clientX,startY:s.clientY,currentX:s.clientX,currentY:s.clientY,startTime:Date.now(),element:s.target})}),e.touches.length===2&&this.options.enablePinch&&this.startPinchGesture(e.touches),this.emit("touchstart",{touches:Array.from(this.touches.values()),originalEvent:e})}handleTouchMove(e){this.options.preventDefault&&e.preventDefault(),Array.from(e.changedTouches).forEach(s=>{let i=this.touches.get(s.identifier);i&&(i.currentX=s.clientX,i.currentY=s.clientY)}),e.touches.length===2&&this.options.enablePinch?this.handlePinchGesture(e.touches):e.touches.length===1&&this.options.enablePan&&this.handlePanGesture(Array.from(this.touches.values())[0]),this.emit("touchmove",{touches:Array.from(this.touches.values()),originalEvent:e})}handleTouchEnd(e){Array.from(e.changedTouches).forEach(s=>{let i=this.touches.get(s.identifier);i&&(this.processTouchEnd(i),this.touches.delete(s.identifier))}),e.touches.length===0&&(this.isGesturing=!1),this.emit("touchend",{touches:Array.from(this.touches.values()),originalEvent:e})}handleTouchCancel(e){e.changedTouches.forEach(t=>{this.touches.delete(t.identifier)}),this.isGesturing=!1}processTouchEnd(e){let t=Date.now()-e.startTime,s=e.currentX-e.startX,i=e.currentY-e.startY,a=Math.sqrt(s*s+i*i);this.options.enableTap&&t<this.options.tapTimeout&&a<10&&this.handleTap(e),this.options.enableSwipe&&a>this.options.swipeThreshold&&this.handleSwipe(e,s,i,a)}handleTap(e){let t=Date.now(),s={x:e.currentX,y:e.currentY,element:e.element,timestamp:t};this.lastTap&&t-this.lastTap.timestamp<this.options.doubleTapTimeout&&Math.abs(this.lastTap.x-s.x)<25&&Math.abs(this.lastTap.y-s.y)<25?(this.emit("doubletap",s),this.lastTap=null):(this.emit("tap",s),this.lastTap=s,setTimeout(()=>{this.lastTap===s&&(this.lastTap=null)},this.options.doubleTapTimeout))}handleSwipe(e,t,s,i){let a=this.getSwipeDirection(t,s);this.emit("swipe",{direction:a,deltaX:t,deltaY:s,distance:i,velocity:i/(Date.now()-e.startTime),startX:e.startX,startY:e.startY,endX:e.currentX,endY:e.currentY})}getSwipeDirection(e,t){let s=Math.abs(e),i=Math.abs(t);return s>i?e>0?"right":"left":t>0?"down":"up"}startPinchGesture(e){let t=e[0],s=e[1];this.gestureStartDistance=this.calculateDistance(t.clientX,t.clientY,s.clientX,s.clientY),this.isGesturing=!0}handlePinchGesture(e){if(!this.isGesturing)return;let t=e[0],s=e[1],i=this.calculateDistance(t.clientX,t.clientY,s.clientX,s.clientY),a=i/this.gestureStartDistance,n=(t.clientX+s.clientX)/2,r=(t.clientY+s.clientY)/2;this.emit("pinch",{scale:a,centerX:n,centerY:r,distance:i,startDistance:this.gestureStartDistance})}handlePanGesture(e){let t=e.currentX-e.startX,s=e.currentY-e.startY;this.emit("pan",{deltaX:t,deltaY:s,currentX:e.currentX,currentY:e.currentY,startX:e.startX,startY:e.startY})}calculateDistance(e,t,s,i){return Math.sqrt(Math.pow(s-e,2)+Math.pow(i-t,2))}handleMouseDown(e){this.touches.set("mouse",{id:"mouse",startX:e.clientX,startY:e.clientY,currentX:e.clientX,currentY:e.clientY,startTime:Date.now(),element:e.target})}handleMouseMove(e){let t=this.touches.get("mouse");t&&(t.currentX=e.clientX,t.currentY=e.clientY,this.handlePanGesture(t))}handleMouseUp(e){let t=this.touches.get("mouse");t&&(this.processTouchEnd(t),this.touches.delete("mouse"))}on(e,t){return this.listeners.has(e)||this.listeners.set(e,new Set),this.listeners.get(e).add(t),()=>this.off(e,t)}off(e,t){this.listeners.has(e)&&this.listeners.get(e).delete(t)}emit(e,t){this.listeners.has(e)&&this.listeners.get(e).forEach(s=>{try{s(t)}catch(i){console.error("Touch gesture callback error:",i)}})}destroy(){this.touches.clear(),this.listeners.clear(),this.lastTap=null}},c=class{constructor(e,t={}){this.container=e,this.options={enableSwipeNavigation:!0,enablePullToRefresh:!0,enableCollapsiblePanels:!0,tabBarHeight:60,headerHeight:56,minPanelHeight:200,maxColumns:{mobile:1,tablet:2,desktop:3},breakpoints:{mobile:768,tablet:1024,desktop:1200},...t},this.currentLayout="mobile",this.widgets=new Map,this.panels=new Map,this.activeTab=0,this.isRefreshing=!1,this.header=null,this.tabBar=null,this.contentArea=null,this.pullToRefreshIndicator=null,this.init()}init(){this.detectLayout(),this.createMobileStructure(),this.setupGestureHandling(),this.setupResizeListener(),this.setupPullToRefresh()}detectLayout(){let e=window.innerWidth;e<=this.options.breakpoints.mobile?this.currentLayout="mobile":e<=this.options.breakpoints.tablet?this.currentLayout="tablet":this.currentLayout="desktop"}createMobileStructure(){this.container.className=`mobile-dashboard ${this.currentLayout}`,this.container.innerHTML=`
      <header class="mobile-header">
        <div class="header-content">
          <button class="menu-button" aria-label="Menu">
            <span class="hamburger"></span>
          </button>
          <h1 class="header-title">Pynomaly</h1>
          <div class="header-actions">
            <button class="refresh-button" aria-label="Refresh">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
              </svg>
            </button>
            <button class="settings-button" aria-label="Settings">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
              </svg>
            </button>
          </div>
        </div>
      </header>

      <div class="pull-to-refresh-indicator">
        <div class="refresh-spinner"></div>
        <span class="refresh-text">Pull to refresh</span>
      </div>

      <main class="content-area">
        <div class="dashboard-tabs" role="tablist">
          <!-- Tabs will be dynamically generated -->
        </div>
        
        <div class="tab-panels">
          <!-- Panel content will be dynamically generated -->
        </div>
      </main>

      <nav class="tab-bar" role="tablist">
        <button class="tab-button active" data-tab="0" role="tab" aria-selected="true">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
          </svg>
          <span>Dashboard</span>
        </button>
        <button class="tab-button" data-tab="1" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/>
          </svg>
          <span>Analytics</span>
        </button>
        <button class="tab-button" data-tab="2" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
          <span>Alerts</span>
        </button>
        <button class="tab-button" data-tab="3" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
          <span>Models</span>
        </button>
      </nav>
    `,this.header=this.container.querySelector(".mobile-header"),this.tabBar=this.container.querySelector(".tab-bar"),this.contentArea=this.container.querySelector(".content-area"),this.pullToRefreshIndicator=this.container.querySelector(".pull-to-refresh-indicator"),this.setupTabNavigation()}setupTabNavigation(){this.tabBar.querySelectorAll(".tab-button").forEach((t,s)=>{t.addEventListener("click",()=>{this.switchTab(s)})}),this.switchTab(0)}switchTab(e){let t=this.tabBar.querySelectorAll(".tab-button"),s=this.contentArea.querySelectorAll(".tab-panel");t.forEach((i,a)=>{let n=a===e;i.classList.toggle("active",n),i.setAttribute("aria-selected",n)}),s.forEach((i,a)=>{i.classList.toggle("active",a===e)}),this.activeTab=e,this.emit("tab-changed",{activeTab:e})}setupGestureHandling(){let e=new l(this.contentArea,{enableSwipe:this.options.enableSwipeNavigation,enablePinch:!1});this.options.enableSwipeNavigation&&e.on("swipe",t=>{Math.abs(t.deltaY)<50&&(t.direction==="left"&&this.activeTab<3?this.switchTab(this.activeTab+1):t.direction==="right"&&this.activeTab>0&&this.switchTab(this.activeTab-1))})}setupPullToRefresh(){if(!this.options.enablePullToRefresh)return;let e=0,t=0,s=0,i=!1,a=new l(this.contentArea);a.on("touchstart",n=>{this.contentArea.scrollTop===0&&(e=n.touches[0].currentY,i=!0)}),a.on("touchmove",n=>{if(i&&(t=n.touches[0].currentY,s=t-e,s>0&&this.contentArea.scrollTop===0)){let r=Math.min(s,100),o=Math.min(r/60,1);this.pullToRefreshIndicator.style.transform=`translateY(${r}px)`,this.pullToRefreshIndicator.style.opacity=o,r>60?(this.pullToRefreshIndicator.classList.add("ready"),this.pullToRefreshIndicator.querySelector(".refresh-text").textContent="Release to refresh"):(this.pullToRefreshIndicator.classList.remove("ready"),this.pullToRefreshIndicator.querySelector(".refresh-text").textContent="Pull to refresh")}}),a.on("touchend",()=>{i&&(i=!1,s>60&&!this.isRefreshing?this.triggerRefresh():this.resetPullToRefresh())})}triggerRefresh(){this.isRefreshing=!0,this.pullToRefreshIndicator.classList.add("refreshing"),this.pullToRefreshIndicator.querySelector(".refresh-text").textContent="Refreshing...",this.emit("refresh-requested"),setTimeout(()=>{this.isRefreshing&&this.resetPullToRefresh()},3e3)}resetPullToRefresh(){this.isRefreshing=!1,this.pullToRefreshIndicator.classList.remove("ready","refreshing"),this.pullToRefreshIndicator.style.transform="translateY(-100%)",this.pullToRefreshIndicator.style.opacity="0",this.pullToRefreshIndicator.querySelector(".refresh-text").textContent="Pull to refresh"}setupResizeListener(){let e;window.addEventListener("resize",()=>{clearTimeout(e),e=setTimeout(()=>{let t=this.currentLayout;this.detectLayout(),t!==this.currentLayout&&(this.container.className=`mobile-dashboard ${this.currentLayout}`,this.updateLayoutForScreen())},250)})}updateLayoutForScreen(){let e=this.options.maxColumns[this.currentLayout];this.panels.forEach(t=>{t.updateLayout(e)}),this.emit("layout-changed",{layout:this.currentLayout})}createPanel(e,t,s,i=0){let a=document.createElement("div");a.className=`tab-panel ${i===this.activeTab?"active":""}`,a.setAttribute("role","tabpanel"),a.innerHTML=`
      <div class="panel-header">
        <h2 class="panel-title">${t}</h2>
        <div class="panel-actions">
          <button class="panel-collapse-btn" aria-label="Collapse panel">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7.41 8.84L12 13.42l4.59-4.58L18 10.25l-6 6-6-6z"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="panel-content">
        ${s}
      </div>
    `;let n={id:e,element:a,title:t,tabIndex:i,isCollapsed:!1,updateLayout:u=>{a.style.gridColumn=`span ${Math.min(1,u)}`}},r=a.querySelector(".panel-collapse-btn"),o=a.querySelector(".panel-content");return r.addEventListener("click",()=>{n.isCollapsed=!n.isCollapsed,a.classList.toggle("collapsed",n.isCollapsed),n.isCollapsed?(o.style.height="0",r.style.transform="rotate(-90deg)"):(o.style.height="auto",r.style.transform="rotate(0deg)")}),this.panels.set(e,n),this.contentArea.querySelector(".tab-panels").appendChild(a),n}createWidget(e,t,s,i){let a={id:e,type:t,config:s,panelId:i,element:null,touchOptimized:!0};switch(t){case"chart":a.element=this.createChartWidget(s);break;case"metric":a.element=this.createMetricWidget(s);break;case"list":a.element=this.createListWidget(s);break;case"form":a.element=this.createFormWidget(s);break;default:a.element=this.createDefaultWidget(s)}this.optimizeWidgetForTouch(a),this.widgets.set(e,a);let n=this.panels.get(i);return n&&n.element.querySelector(".panel-content").appendChild(a.element),a}createChartWidget(e){let t=document.createElement("div");return t.className="widget chart-widget touch-optimized",t.innerHTML=`
      <div class="widget-header">
        <h3 class="widget-title">${e.title||"Chart"}</h3>
        <div class="widget-controls">
          <button class="zoom-out-btn" aria-label="Zoom out">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
              <path d="M7 9h5v1H7z"/>
            </svg>
          </button>
          <button class="fullscreen-btn" aria-label="Fullscreen">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="widget-content chart-container" style="height: ${e.height||"250px"};">
        <!-- Chart will be rendered here -->
      </div>
    `,t}createMetricWidget(e){let t=document.createElement("div");return t.className="widget metric-widget touch-optimized",t.innerHTML=`
      <div class="metric-display">
        <div class="metric-value ${e.trend||""}">${e.value||"0"}</div>
        <div class="metric-label">${e.label||"Metric"}</div>
        <div class="metric-change">${e.change||"+0%"}</div>
      </div>
    `,t}createListWidget(e){let t=document.createElement("div");return t.className="widget list-widget touch-optimized",t.innerHTML=`
      <div class="widget-header">
        <h3 class="widget-title">${e.title||"List"}</h3>
        <button class="refresh-widget-btn" aria-label="Refresh">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
          </svg>
        </button>
      </div>
      <div class="widget-content">
        <div class="list-container">
          <!-- List items will be dynamically added -->
        </div>
      </div>
    `,t}createFormWidget(e){let t=document.createElement("div");return t.className="widget form-widget touch-optimized",t.innerHTML=`
      <div class="widget-header">
        <h3 class="widget-title">${e.title||"Form"}</h3>
      </div>
      <div class="widget-content">
        <form class="mobile-form">
          <!-- Form fields will be dynamically added -->
        </form>
      </div>
    `,t}createDefaultWidget(e){let t=document.createElement("div");return t.className="widget default-widget touch-optimized",t.innerHTML=`
      <div class="widget-content">
        ${e.content||"Widget content"}
      </div>
    `,t}optimizeWidgetForTouch(e){let t=e.element;if(t.querySelectorAll("button").forEach(i=>{i.style.minHeight="44px",i.style.minWidth="44px",i.classList.add("touch-target")}),e.type==="chart"){let i=t.querySelector(".chart-container"),a=new l(i,{enablePinch:!0,enablePan:!0});a.on("pinch",n=>{this.emit("chart-zoom",{widgetId:e.id,scale:n.scale,centerX:n.centerX,centerY:n.centerY})}),a.on("pan",n=>{this.emit("chart-pan",{widgetId:e.id,deltaX:n.deltaX,deltaY:n.deltaY})}),a.on("doubletap",()=>{this.emit("chart-reset",{widgetId:e.id})})}t.addEventListener("touchstart",()=>{t.classList.add("touch-active")}),t.addEventListener("touchend",()=>{setTimeout(()=>{t.classList.remove("touch-active")},150)})}enableHapticFeedback(){return"vibrate"in navigator?{light:()=>navigator.vibrate(10),medium:()=>navigator.vibrate(20),heavy:()=>navigator.vibrate(50),success:()=>navigator.vibrate([50,50,50]),error:()=>navigator.vibrate([100,50,100])}:{light:()=>{},medium:()=>{},heavy:()=>{},success:()=>{},error:()=>{}}}showToast(e,t="info",s=3e3){let i=document.createElement("div");i.className=`mobile-toast ${t}`,i.innerHTML=`
      <div class="toast-content">
        <span class="toast-message">${e}</span>
        <button class="toast-close" aria-label="Close">\xD7</button>
      </div>
    `,document.body.appendChild(i),new l(i).on("swipe",n=>{(n.direction==="up"||n.direction==="right")&&this.dismissToast(i)}),i.querySelector(".toast-close").addEventListener("click",()=>{this.dismissToast(i)}),setTimeout(()=>{this.dismissToast(i)},s),requestAnimationFrame(()=>{i.classList.add("show")})}dismissToast(e){e.classList.add("dismiss"),setTimeout(()=>{e.parentNode&&e.parentNode.removeChild(e)},300)}on(e,t){return this.listeners||(this.listeners=new Map),this.listeners.has(e)||this.listeners.set(e,new Set),this.listeners.get(e).add(t),()=>this.off(e,t)}off(e,t){this.listeners&&this.listeners.has(e)&&this.listeners.get(e).delete(t)}emit(e,t){this.listeners&&this.listeners.has(e)&&this.listeners.get(e).forEach(s=>{try{s(t)}catch(i){console.error("Mobile dashboard event error:",i)}})}destroy(){this.listeners&&this.listeners.clear(),this.widgets.clear(),this.panels.clear()}};typeof h<"u"&&h.exports?h.exports={TouchGestureManager:l,MobileDashboardManager:c}:(window.TouchGestureManager=l,window.MobileDashboardManager=c)});m();})();
