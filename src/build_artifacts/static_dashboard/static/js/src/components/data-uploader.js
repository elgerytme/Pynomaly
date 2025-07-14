// Data Uploader Component - File upload with validation and preview
export class DataUploader {
  constructor(element) {
    this.element = element;
    this.config = this.getConfig();
    this.uploadQueue = [];

    this.init();
  }

  getConfig() {
    const element = this.element;
    return {
      maxFileSize: parseInt(element.dataset.maxFileSize) || 10 * 1024 * 1024,
      allowedFormats: (
        element.dataset.allowedFormats || "csv,json,parquet"
      ).split(","),
      multiple: element.dataset.multiple === "true",
      autoUpload: element.dataset.autoUpload === "true",
    };
  }

  init() {
    this.createInterface();
    this.bindEvents();
  }

  createInterface() {
    this.element.innerHTML = `
      <div class="data-uploader">
        <div class="upload-zone" data-drop-zone>
          <div class="upload-icon">📁</div>
          <div class="upload-text">
            <p>Drop files here or click to browse</p>
            <p class="text-sm text-neutral-500">Max size: ${this.config.maxFileSize / 1024 / 1024}MB</p>
          </div>
          <input type="file" class="file-input" ${this.config.multiple ? "multiple" : ""} hidden>
        </div>
        <div class="upload-queue" data-upload-queue></div>
      </div>
    `;
  }

  bindEvents() {
    const fileInput = this.element.querySelector(".file-input");
    const dropZone = this.element.querySelector("[data-drop-zone]");

    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) =>
      this.handleFiles(e.target.files),
    );

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      this.handleFiles(e.dataTransfer.files);
    });
  }

  handleFiles(files) {
    Array.from(files).forEach((file) => {
      if (this.validateFile(file)) {
        this.addToQueue(file);
      }
    });
  }

  validateFile(file) {
    const extension = file.name.split(".").pop().toLowerCase();

    if (!this.config.allowedFormats.includes(extension)) {
      this.showError(`Unsupported format: ${extension}`);
      return false;
    }

    if (file.size > this.config.maxFileSize) {
      this.showError(`File too large: ${file.name}`);
      return false;
    }

    return true;
  }

  addToQueue(file) {
    this.uploadQueue.push(file);
    this.renderQueue();

    if (this.config.autoUpload) {
      this.uploadFile(file);
    }
  }

  renderQueue() {
    const queueContainer = this.element.querySelector("[data-upload-queue]");
    queueContainer.innerHTML = this.uploadQueue
      .map(
        (file) => `
      <div class="upload-item" data-file="${file.name}">
        <div class="file-info">
          <div class="file-name">${file.name}</div>
          <div class="file-size">${(file.size / 1024).toFixed(1)} KB</div>
        </div>
        <div class="upload-progress">
          <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
        </div>
      </div>
    `,
      )
      .join("");
  }

  async uploadFile(file) {
    // Placeholder upload implementation
    console.log("Uploading file:", file.name);
  }

  showError(message) {
    console.error("Upload error:", message);
  }
}
