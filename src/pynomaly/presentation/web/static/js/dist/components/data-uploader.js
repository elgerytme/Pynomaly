(()=>{var s=class{constructor(e){this.element=e,this.config=this.getConfig(),this.uploadQueue=[],this.init()}getConfig(){let e=this.element;return{maxFileSize:parseInt(e.dataset.maxFileSize)||10*1024*1024,allowedFormats:(e.dataset.allowedFormats||"csv,json,parquet").split(","),multiple:e.dataset.multiple==="true",autoUpload:e.dataset.autoUpload==="true"}}init(){this.createInterface(),this.bindEvents()}createInterface(){this.element.innerHTML=`
      <div class="data-uploader">
        <div class="upload-zone" data-drop-zone>
          <div class="upload-icon">\u{1F4C1}</div>
          <div class="upload-text">
            <p>Drop files here or click to browse</p>
            <p class="text-sm text-neutral-500">Max size: ${this.config.maxFileSize/1024/1024}MB</p>
          </div>
          <input type="file" class="file-input" ${this.config.multiple?"multiple":""} hidden>
        </div>
        <div class="upload-queue" data-upload-queue></div>
      </div>
    `}bindEvents(){let e=this.element.querySelector(".file-input"),t=this.element.querySelector("[data-drop-zone]");t.addEventListener("click",()=>e.click()),e.addEventListener("change",i=>this.handleFiles(i.target.files)),t.addEventListener("dragover",i=>{i.preventDefault(),t.classList.add("drag-over")}),t.addEventListener("dragleave",()=>{t.classList.remove("drag-over")}),t.addEventListener("drop",i=>{i.preventDefault(),t.classList.remove("drag-over"),this.handleFiles(i.dataTransfer.files)})}handleFiles(e){Array.from(e).forEach(t=>{this.validateFile(t)&&this.addToQueue(t)})}validateFile(e){let t=e.name.split(".").pop().toLowerCase();return this.config.allowedFormats.includes(t)?e.size>this.config.maxFileSize?(this.showError(`File too large: ${e.name}`),!1):!0:(this.showError(`Unsupported format: ${t}`),!1)}addToQueue(e){this.uploadQueue.push(e),this.renderQueue(),this.config.autoUpload&&this.uploadFile(e)}renderQueue(){let e=this.element.querySelector("[data-upload-queue]");e.innerHTML=this.uploadQueue.map(t=>`
      <div class="upload-item" data-file="${t.name}">
        <div class="file-info">
          <div class="file-name">${t.name}</div>
          <div class="file-size">${(t.size/1024).toFixed(1)} KB</div>
        </div>
        <div class="upload-progress">
          <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
        </div>
      </div>
    `).join("")}async uploadFile(e){console.log("Uploading file:",e.name)}showError(e){console.error("Upload error:",e)}};})();
