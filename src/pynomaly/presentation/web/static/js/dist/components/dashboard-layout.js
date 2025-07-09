(()=>{var u=class{constructor(t,e={}){this.container=typeof t=="string"?document.querySelector(t):t,this.options={columns:12,rowHeight:60,margin:[10,10],containerPadding:[10,10],maxRows:1/0,isDraggable:!0,isResizable:!0,preventCollision:!1,autoSize:!0,compactType:"vertical",layouts:{},breakpoints:{lg:1200,md:996,sm:768,xs:480,xxs:0},responsiveLayouts:{lg:[],md:[],sm:[],xs:[],xxs:[]},...e},this.widgets=new Map,this.layout=[],this.currentBreakpoint="lg",this.isDragging=!1,this.isResizing=!1,this.draggedWidget=null,this.placeholder=null,this.init()}init(){this.setupContainer(),this.detectBreakpoint(),this.bindEvents(),this.render()}setupContainer(){this.container.classList.add("dashboard-layout"),this.container.innerHTML="",this.gridOverlay=document.createElement("div"),this.gridOverlay.className="grid-overlay",this.gridOverlay.style.display="none",this.container.appendChild(this.gridOverlay),Object.assign(this.container.style,{position:"relative",minHeight:"100vh",padding:`${this.options.containerPadding[1]}px ${this.options.containerPadding[0]}px`})}detectBreakpoint(){let t=this.container.clientWidth;for(let[e,i]of Object.entries(this.options.breakpoints))if(t>=i){this.currentBreakpoint=e;break}this.loadLayout(this.currentBreakpoint)}bindEvents(){window.ResizeObserver?(this.resizeObserver=new ResizeObserver(()=>{this.detectBreakpoint(),this.render()}),this.resizeObserver.observe(this.container)):window.addEventListener("resize",this.debounce(()=>{this.detectBreakpoint(),this.render()},250)),this.container.addEventListener("mousedown",this.onMouseDown.bind(this)),this.container.addEventListener("mousemove",this.onMouseMove.bind(this)),this.container.addEventListener("mouseup",this.onMouseUp.bind(this)),this.container.addEventListener("touchstart",this.onTouchStart.bind(this),{passive:!1}),this.container.addEventListener("touchmove",this.onTouchMove.bind(this),{passive:!1}),this.container.addEventListener("touchend",this.onTouchEnd.bind(this)),this.container.addEventListener("keydown",this.onKeyDown.bind(this))}addWidget(t){let e={id:t.id||this.generateId(),type:t.type||"default",title:t.title||"Widget",content:t.content||"",component:t.component||null,props:t.props||{},x:t.x||0,y:t.y||0,w:t.w||2,h:t.h||2,minW:t.minW||1,minH:t.minH||1,maxW:t.maxW||1/0,maxH:t.maxH||1/0,static:t.static||!1,isDraggable:t.isDraggable!==!1,isResizable:t.isResizable!==!1,moved:!1,resizeHandles:t.resizeHandles||["se"],...t};if(t.x===void 0||t.y===void 0){let i=this.findSuitablePosition(e.w,e.h);e.x=i.x,e.y=i.y}return this.widgets.set(e.id,e),this.layout.push(e),this.options.compactType&&this.compactLayout(),this.render(),this.saveLayout(),this.emitEvent("widgetAdded",{widget:e}),e}removeWidget(t){let e=this.widgets.get(t);if(!e)return!1;this.widgets.delete(t),this.layout=this.layout.filter(s=>s.id!==t);let i=this.container.querySelector(`[data-widget-id="${t}"]`);return i&&i.remove(),this.options.compactType&&this.compactLayout(),this.render(),this.saveLayout(),this.emitEvent("widgetRemoved",{widget:e}),!0}updateWidget(t,e){let i=this.widgets.get(t);if(!i)return!1;Object.assign(i,e);let s=this.layout.findIndex(n=>n.id===t);return s>=0&&(this.layout[s]=i),this.render(),this.saveLayout(),this.emitEvent("widgetUpdated",{widget:i,updates:e}),!0}findSuitablePosition(t,e){let i=this.options.columns;for(let n=0;n<this.options.maxRows;n++)for(let o=0;o<=i-t;o++)if(!this.hasCollision({x:o,y:n,w:t,h:e}))return{x:o,y:n};return{x:0,y:Math.max(0,...this.layout.map(n=>n.y+n.h))}}hasCollision(t,e=null){return this.layout.some(i=>i.id===e?!1:!(t.x>=i.x+i.w||t.x+t.w<=i.x||t.y>=i.y+i.h||t.y+t.h<=i.y))}compactLayout(){this.options.compactType==="vertical"?this.compactVertical():this.options.compactType==="horizontal"&&this.compactHorizontal()}compactVertical(){[...this.layout].sort((e,i)=>e.y===i.y?e.x-i.x:e.y-i.y).forEach(e=>{if(e.static)return;let i=0;for(let s=0;s<e.y;s++){let n={...e,y:s};if(!this.hasCollision(n,e.id))i=s;else break}e.y=i})}compactHorizontal(){[...this.layout].sort((e,i)=>e.x===i.x?e.y-i.y:e.x-i.x).forEach(e=>{if(e.static)return;let i=0;for(let s=0;s<e.x;s++){let n={...e,x:s};if(!this.hasCollision(n,e.id))i=s;else break}e.x=i})}render(){this.container.querySelectorAll(".dashboard-widget").forEach(s=>s.remove());let i=(this.container.clientWidth-this.options.containerPadding[0]*2-this.options.margin[0]*(this.options.columns-1))/this.options.columns;if(this.layout.forEach(s=>{let n=this.createWidgetElement(s,i);this.container.appendChild(n)}),this.options.autoSize){let n=Math.max(0,...this.layout.map(o=>o.y+o.h))*(this.options.rowHeight+this.options.margin[1])+this.options.containerPadding[1]*2;this.container.style.minHeight=`${n}px`}}createWidgetElement(t,e){let i=document.createElement("div");i.className="dashboard-widget",i.setAttribute("data-widget-id",t.id),i.setAttribute("tabindex","0"),i.setAttribute("role","article"),i.setAttribute("aria-label",`Widget: ${t.title}`);let s=t.x*(e+this.options.margin[0]),n=t.y*(this.options.rowHeight+this.options.margin[1]),o=t.w*e+(t.w-1)*this.options.margin[0],r=t.h*this.options.rowHeight+(t.h-1)*this.options.margin[1];Object.assign(i.style,{position:"absolute",left:`${s}px`,top:`${n}px`,width:`${o}px`,height:`${r}px`,backgroundColor:"white",border:"1px solid #e2e8f0",borderRadius:"8px",boxShadow:"0 1px 3px rgba(0, 0, 0, 0.1)",overflow:"hidden",transition:this.isDragging||this.isResizing?"none":"all 0.2s ease",cursor:t.isDraggable?"move":"default",zIndex:t.static?1:2});let h=document.createElement("div");h.className="widget-header",h.style.cssText=`
            padding: 12px 16px;
            border-bottom: 1px solid #e2e8f0;
            background: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
            min-height: 44px;
        `;let l=document.createElement("h3");l.className="widget-title",l.textContent=t.title,l.style.cssText=`
            margin: 0;
            font-size: 14px;
            font-weight: 600;
            color: #1f2937;
        `;let d=document.createElement("div");if(d.className="widget-actions",d.style.cssText=`
            display: flex;
            gap: 8px;
        `,!t.static){let p=this.createActionButton("\xD7",()=>this.removeWidget(t.id));p.setAttribute("aria-label","Remove widget"),d.appendChild(p)}h.appendChild(l),h.appendChild(d);let c=document.createElement("div");if(c.className="widget-content",c.style.cssText=`
            padding: 16px;
            height: calc(100% - 44px);
            overflow: auto;
        `,t.component&&typeof t.component=="function"){let p=t.component(t.props);c.appendChild(p)}else t.content?c.innerHTML=t.content:c.innerHTML='<p style="color: #6b7280;">No content available</p>';return i.appendChild(h),i.appendChild(c),t.isResizable&&!t.static&&this.addResizeHandles(i,t),i}createActionButton(t,e){let i=document.createElement("button");return i.textContent=t,i.style.cssText=`
            background: transparent;
            border: none;
            color: #6b7280;
            cursor: pointer;
            font-size: 16px;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            transition: all 0.2s ease;
        `,i.addEventListener("mouseenter",()=>{i.style.backgroundColor="#e5e7eb",i.style.color="#374151"}),i.addEventListener("mouseleave",()=>{i.style.backgroundColor="transparent",i.style.color="#6b7280"}),i.addEventListener("click",s=>{s.stopPropagation(),e()}),i}addResizeHandles(t,e){e.resizeHandles.forEach(i=>{let s=document.createElement("div");s.className=`resize-handle resize-handle-${i}`,s.style.cssText=this.getResizeHandleStyles(i),s.setAttribute("data-resize-direction",i),s.addEventListener("mousedown",n=>{n.stopPropagation(),this.startResize(n,e,i)}),t.appendChild(s)})}getResizeHandleStyles(t){let e=`
            position: absolute;
            background: #3b82f6;
            opacity: 0;
            transition: opacity 0.2s ease;
            cursor: ${this.getResizeCursor(t)};
        `;switch(t){case"se":return e+`
                    bottom: 0;
                    right: 0;
                    width: 12px;
                    height: 12px;
                    border-radius: 12px 0 0 0;
                `;case"s":return e+`
                    bottom: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 24px;
                    height: 4px;
                `;case"e":return e+`
                    right: 0;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 4px;
                    height: 24px;
                `;default:return e}}getResizeCursor(t){return{se:"nw-resize",s:"ns-resize",e:"ew-resize",ne:"ne-resize",nw:"nw-resize",sw:"sw-resize",n:"ns-resize",w:"ew-resize"}[t]||"default"}onMouseDown(t){let e=t.target.closest(".dashboard-widget");if(!e)return;let i=e.getAttribute("data-widget-id"),s=this.widgets.get(i);!s||s.static||!s.isDraggable||t.target.classList.contains("resize-handle")||this.startDrag(t,s)}startDrag(t,e){this.isDragging=!0,this.draggedWidget=e;let i=this.container.querySelector(`[data-widget-id="${e.id}"]`),s=i.getBoundingClientRect(),n=this.container.getBoundingClientRect();this.dragOffset={x:t.clientX-s.left,y:t.clientY-s.top},this.showGridOverlay(),i.classList.add("dragging"),this.createPlaceholder(e),this.emitEvent("dragStart",{widget:e})}startResize(t,e,i){this.isResizing=!0,this.draggedWidget=e,this.resizeDirection=i,this.container.querySelector(`[data-widget-id="${e.id}"]`).classList.add("resizing"),this.emitEvent("resizeStart",{widget:e,direction:i})}onMouseMove(t){this.isDragging?this.handleDrag(t):this.isResizing&&this.handleResize(t)}handleDrag(t){if(!this.draggedWidget)return;let e=this.container.getBoundingClientRect(),i=(this.container.clientWidth-this.options.containerPadding[0]*2-this.options.margin[0]*(this.options.columns-1))/this.options.columns,s=Math.round((t.clientX-e.left-this.dragOffset.x)/(i+this.options.margin[0])),n=Math.round((t.clientY-e.top-this.dragOffset.y)/(this.options.rowHeight+this.options.margin[1])),o=Math.max(0,Math.min(s,this.options.columns-this.draggedWidget.w)),r=Math.max(0,n);this.placeholder&&(this.placeholder.x=o,this.placeholder.y=r,this.updatePlaceholderPosition())}handleResize(t){if(!this.draggedWidget)return;let e=this.container.getBoundingClientRect(),i=(this.container.clientWidth-this.options.containerPadding[0]*2-this.options.margin[0]*(this.options.columns-1))/this.options.columns,s=this.draggedWidget,n=this.resizeDirection,o=s.w,r=s.h;if(n.includes("e")){let c=t.clientX-e.left-s.x*(i+this.options.margin[0]);o=Math.max(s.minW,Math.min(s.maxW,Math.round(c/(i+this.options.margin[0]))))}if(n.includes("s")){let c=t.clientY-e.top-s.y*(this.options.rowHeight+this.options.margin[1]);r=Math.max(s.minH,Math.min(s.maxH,Math.round(c/(this.options.rowHeight+this.options.margin[1]))))}let h=this.container.querySelector(`[data-widget-id="${s.id}"]`),l=o*i+(o-1)*this.options.margin[0],d=r*this.options.rowHeight+(r-1)*this.options.margin[1];h.style.width=`${l}px`,h.style.height=`${d}px`}onMouseUp(t){this.isDragging?this.endDrag():this.isResizing&&this.endResize()}endDrag(){if(!this.draggedWidget||!this.placeholder)return;let t=this.draggedWidget;t.x=this.placeholder.x,t.y=this.placeholder.y,t.moved=!0,this.container.querySelector(`[data-widget-id="${t.id}"]`).classList.remove("dragging"),this.removePlaceholder(),this.hideGridOverlay(),this.options.preventCollision||this.resolveCollisions(t),this.options.compactType&&this.compactLayout(),this.render(),this.saveLayout(),this.emitEvent("dragEnd",{widget:t}),this.isDragging=!1,this.draggedWidget=null}endResize(){if(!this.draggedWidget)return;let t=this.draggedWidget,e=this.container.querySelector(`[data-widget-id="${t.id}"]`),s=(this.container.clientWidth-this.options.containerPadding[0]*2-this.options.margin[0]*(this.options.columns-1))/this.options.columns,n=parseInt(e.style.width),o=parseInt(e.style.height);t.w=Math.round(n/(s+this.options.margin[0])),t.h=Math.round(o/(this.options.rowHeight+this.options.margin[1])),e.classList.remove("resizing"),this.options.preventCollision||this.resolveCollisions(t),this.options.compactType&&this.compactLayout(),this.render(),this.saveLayout(),this.emitEvent("resizeEnd",{widget:t}),this.isResizing=!1,this.draggedWidget=null,this.resizeDirection=null}resolveCollisions(t){this.layout.filter(i=>i.id!==t.id&&!i.static&&this.hasCollision(t,i.id)).forEach(i=>{i.y=t.y+t.h,i.moved=!0})}createPlaceholder(t){this.placeholder={x:t.x,y:t.y,w:t.w,h:t.h};let e=document.createElement("div");e.className="widget-placeholder",e.style.cssText=`
            position: absolute;
            background: rgba(59, 130, 246, 0.2);
            border: 2px dashed #3b82f6;
            border-radius: 8px;
            pointer-events: none;
            z-index: 1000;
        `,this.placeholderElement=e,this.container.appendChild(e),this.updatePlaceholderPosition()}updatePlaceholderPosition(){if(!this.placeholderElement||!this.placeholder)return;let e=(this.container.clientWidth-this.options.containerPadding[0]*2-this.options.margin[0]*(this.options.columns-1))/this.options.columns,i=this.placeholder.x*(e+this.options.margin[0]),s=this.placeholder.y*(this.options.rowHeight+this.options.margin[1]),n=this.placeholder.w*e+(this.placeholder.w-1)*this.options.margin[0],o=this.placeholder.h*this.options.rowHeight+(this.placeholder.h-1)*this.options.margin[1];Object.assign(this.placeholderElement.style,{left:`${i}px`,top:`${s}px`,width:`${n}px`,height:`${o}px`})}removePlaceholder(){this.placeholderElement&&(this.placeholderElement.remove(),this.placeholderElement=null),this.placeholder=null}showGridOverlay(){this.gridOverlay.style.display="block"}hideGridOverlay(){this.gridOverlay.style.display="none"}onTouchStart(t){if(t.touches.length===1){t.preventDefault();let e=t.touches[0];this.onMouseDown({...e,target:e.target,stopPropagation:()=>t.stopPropagation(),preventDefault:()=>t.preventDefault()})}}onTouchMove(t){if(t.touches.length===1&&(this.isDragging||this.isResizing)){t.preventDefault();let e=t.touches[0];this.onMouseMove(e)}}onTouchEnd(t){this.onMouseUp(t)}onKeyDown(t){let e=t.target.closest(".dashboard-widget");if(!e)return;let i=e.getAttribute("data-widget-id"),s=this.widgets.get(i);if(!s||s.static)return;let n=!1,o=t.shiftKey?5:1;switch(t.key){case"ArrowLeft":s.x=Math.max(0,s.x-o),n=!0;break;case"ArrowRight":s.x=Math.min(this.options.columns-s.w,s.x+o),n=!0;break;case"ArrowUp":s.y=Math.max(0,s.y-o),n=!0;break;case"ArrowDown":s.y=s.y+o,n=!0;break;case"Delete":case"Backspace":this.removeWidget(i),t.preventDefault();return}if(n){if(t.preventDefault(),this.hasCollision(s,s.id)){switch(t.key){case"ArrowLeft":s.x+=o;break;case"ArrowRight":s.x-=o;break;case"ArrowUp":s.y+=o;break;case"ArrowDown":s.y-=o;break}return}this.render(),this.saveLayout(),this.emitEvent("widgetMoved",{widget:s})}}saveLayout(){let t={[this.currentBreakpoint]:this.layout.map(e=>({i:e.id,x:e.x,y:e.y,w:e.w,h:e.h,static:e.static}))};try{localStorage.setItem("dashboard-layout",JSON.stringify(t)),this.emitEvent("layoutSaved",{layout:t})}catch(e){console.error("Failed to save layout:",e)}}loadLayout(t){try{let e=localStorage.getItem("dashboard-layout");if(e){let s=JSON.parse(e)[t];s&&(s.forEach(n=>{let o=this.widgets.get(n.i);o&&(o.x=n.x,o.y=n.y,o.w=n.w,o.h=n.h,o.static=n.static)}),this.emitEvent("layoutLoaded",{layout:s}))}}catch(e){console.error("Failed to load layout:",e)}}generateId(){return"widget_"+Math.random().toString(36).substr(2,9)}debounce(t,e){let i;return function(...n){let o=()=>{clearTimeout(i),t(...n)};clearTimeout(i),i=setTimeout(o,e)}}emitEvent(t,e){let i=new CustomEvent(`dashboard:${t}`,{detail:e,bubbles:!0,cancelable:!0});this.container.dispatchEvent(i)}getLayout(){return this.layout.map(t=>({...t}))}setLayout(t){this.layout=t.map(e=>({...e})),this.widgets.clear(),this.layout.forEach(e=>{this.widgets.set(e.id,e)}),this.render(),this.saveLayout()}exportLayout(){return{layout:this.getLayout(),breakpoint:this.currentBreakpoint,options:{...this.options}}}importLayout(t){t.layout&&this.setLayout(t.layout)}destroy(){this.resizeObserver&&this.resizeObserver.disconnect(),this.container.innerHTML="",this.widgets.clear(),this.layout=[]}},g=class{constructor(){this.widgets=new Map}register(t,e){this.widgets.set(t,e)}create(t,e={}){let i=this.widgets.get(t);if(!i)throw new Error(`Widget type '${t}' not found`);return{...i,props:{...i.defaultProps,...e}}}getTypes(){return Array.from(this.widgets.keys())}},m={metric:{type:"metric",title:"Metric Widget",w:2,h:2,component:a=>{let t=document.createElement("div");return t.className="metric-widget",t.innerHTML=`
                <div class="metric-value">${a.value||0}</div>
                <div class="metric-label">${a.label||"Metric"}</div>
                <div class="metric-change ${a.change>=0?"positive":"negative"}">
                    ${a.change>=0?"+":""}${a.change||0}%
                </div>
            `,t},defaultProps:{value:0,label:"Metric",change:0}},chart:{type:"chart",title:"Chart Widget",w:4,h:3,component:a=>{let t=document.createElement("div");return t.className="chart-widget",t.innerHTML=`
                <div class="chart-placeholder">
                    <div class="chart-icon">\u{1F4CA}</div>
                    <p>${a.chartType||"Chart"} visualization</p>
                </div>
            `,t},defaultProps:{chartType:"Line"}},table:{type:"table",title:"Data Table",w:6,h:4,component:a=>{let t=document.createElement("div");t.className="table-widget";let e=document.createElement("table");if(e.className="widget-table",a.columns){let s=document.createElement("thead"),n=document.createElement("tr");a.columns.forEach(o=>{let r=document.createElement("th");r.textContent=o,n.appendChild(r)}),s.appendChild(n),e.appendChild(s)}let i=document.createElement("tbody");if(a.data&&a.data.length>0)a.data.forEach(s=>{let n=document.createElement("tr");Object.values(s).forEach(o=>{let r=document.createElement("td");r.textContent=o,n.appendChild(r)}),i.appendChild(n)});else{let s=document.createElement("tr"),n=document.createElement("td");n.colSpan=a.columns?.length||1,n.textContent="No data available",n.style.textAlign="center",n.style.color="#6b7280",s.appendChild(n),i.appendChild(s)}return e.appendChild(i),t.appendChild(e),t},defaultProps:{columns:["Column 1","Column 2"],data:[]}}},y=`
.dashboard-layout {
    position: relative;
    background: #f8fafc;
}

.dashboard-widget {
    box-sizing: border-box;
    user-select: none;
}

.dashboard-widget:hover .resize-handle {
    opacity: 1;
}

.dashboard-widget.dragging {
    z-index: 1000;
    opacity: 0.8;
}

.dashboard-widget.resizing {
    z-index: 1000;
}

.widget-header {
    cursor: move;
}

.widget-content {
    position: relative;
}

.widget-placeholder {
    animation: pulse 1s ease-in-out infinite alternate;
}

@keyframes pulse {
    from { opacity: 0.3; }
    to { opacity: 0.7; }
}

.metric-widget {
    text-align: center;
    padding: 20px;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #1f2937;
}

.metric-label {
    font-size: 0.875rem;
    color: #6b7280;
    margin: 8px 0;
}

.metric-change {
    font-size: 0.875rem;
    font-weight: 500;
}

.metric-change.positive {
    color: #059669;
}

.metric-change.negative {
    color: #dc2626;
}

.chart-widget {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}

.chart-placeholder {
    text-align: center;
    color: #6b7280;
}

.chart-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.widget-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

.widget-table th,
.widget-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}

.widget-table th {
    background: #f9fafb;
    font-weight: 600;
    color: #374151;
}

.grid-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    z-index: 999;
    background-image: 
        linear-gradient(to right, rgba(0,0,0,0.1) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(0,0,0,0.1) 1px, transparent 1px);
    background-size: 60px 60px;
}
`;})();
