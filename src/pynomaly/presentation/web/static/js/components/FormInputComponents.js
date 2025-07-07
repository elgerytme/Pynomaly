/**
 * Advanced Form Input Components
 *
 * Supporting components for advanced form functionality:
 * - File Upload with drag-and-drop
 * - Date Range Picker
 * - Multi-Select with search
 * - Dynamic Fieldset
 */

// File Upload Component
class FileUploadComponent {
  constructor(container, options = {}, parent) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      accept: "*",
      maxFiles: 5,
      maxSize: 10 * 1024 * 1024, // 10MB
      allowDragDrop: true,
      showProgress: true,
      ...options,
    };
    this.parent = parent;
    this.files = [];
    this.uploading = false;

    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  render() {
    this.container.innerHTML = `
      <div class="file-upload ${this.options.disabled ? "disabled" : ""}" 
           tabindex="0" 
           role="button" 
           aria-label="Click to select files or drag and drop files here">
        <div class="file-upload-icon" aria-hidden="true">📁</div>
        <div class="file-upload-text">
          <strong>Click to upload</strong> or drag and drop
        </div>
        <div class="file-upload-hint">
          ${this.options.accept !== "*" ? `Accepts: ${this.options.accept}` : "All file types accepted"}
          ${this.options.maxSize ? ` • Max size: ${this.formatFileSize(this.options.maxSize)}` : ""}
        </div>
        <input type="file" 
               class="file-input sr-only" 
               ${this.options.maxFiles > 1 ? "multiple" : ""}
               ${this.options.accept !== "*" ? `accept="${this.options.accept}"` : ""}
               aria-describedby="file-upload-hint">
      </div>
      <div class="file-list" style="display: none;"></div>
    `;

    this.fileInput = this.container.querySelector(".file-input");
    this.fileList = this.container.querySelector(".file-list");
    this.uploadArea = this.container.querySelector(".file-upload");
  }

  bindEvents() {
    if (this.options.disabled) return;

    // Click to select files
    this.uploadArea.addEventListener("click", () => {
      this.fileInput.click();
    });

    // Keyboard accessibility
    this.uploadArea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        this.fileInput.click();
      }
    });

    // File selection
    this.fileInput.addEventListener("change", (e) => {
      this.handleFiles(Array.from(e.target.files));
    });

    // Drag and drop
    if (this.options.allowDragDrop) {
      this.uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        this.uploadArea.classList.add("dragover");
      });

      this.uploadArea.addEventListener("dragleave", (e) => {
        e.preventDefault();
        this.uploadArea.classList.remove("dragover");
      });

      this.uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        this.uploadArea.classList.remove("dragover");
        this.handleFiles(Array.from(e.dataTransfer.files));
      });
    }
  }

  handleFiles(newFiles) {
    const validFiles = [];
    const errors = [];

    for (const file of newFiles) {
      // Check file count
      if (this.files.length + validFiles.length >= this.options.maxFiles) {
        errors.push(`Maximum ${this.options.maxFiles} files allowed`);
        break;
      }

      // Check file size
      if (this.options.maxSize && file.size > this.options.maxSize) {
        errors.push(
          `${file.name} is too large (max ${this.formatFileSize(this.options.maxSize)})`,
        );
        continue;
      }

      // Check file type
      if (this.options.accept !== "*" && !this.isValidFileType(file)) {
        errors.push(`${file.name} is not an accepted file type`);
        continue;
      }

      validFiles.push(file);
    }

    if (errors.length > 0) {
      this.showErrors(errors);
    }

    if (validFiles.length > 0) {
      this.addFiles(validFiles);
    }
  }

  addFiles(files) {
    for (const file of files) {
      const fileInfo = {
        file,
        id: Math.random().toString(36).substr(2, 9),
        progress: 0,
        uploaded: false,
        error: null,
      };

      this.files.push(fileInfo);
    }

    this.updateDisplay();
    this.notifyChange();

    if (this.options.autoUpload) {
      this.uploadFiles();
    }
  }

  removeFile(fileId) {
    this.files = this.files.filter((f) => f.id !== fileId);
    this.updateDisplay();
    this.notifyChange();
  }

  updateDisplay() {
    if (this.files.length === 0) {
      this.fileList.style.display = "none";
      return;
    }

    this.fileList.style.display = "block";
    this.fileList.innerHTML = this.files
      .map((fileInfo) => this.renderFileItem(fileInfo))
      .join("");

    // Bind remove buttons
    this.fileList.querySelectorAll("[data-remove]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const fileId = e.target.dataset.remove;
        this.removeFile(fileId);
      });
    });
  }

  renderFileItem(fileInfo) {
    const { file, id, progress, uploaded, error } = fileInfo;

    return `
      <div class="file-item ${error ? "error" : uploaded ? "success" : ""}" data-file-id="${id}">
        <div class="file-info">
          <div class="file-name">${file.name}</div>
          <div class="file-size">${this.formatFileSize(file.size)}</div>
        </div>
        ${
          this.options.showProgress && !uploaded
            ? `
          <div class="file-progress">
            <div class="file-progress-bar" style="width: ${progress}%"></div>
          </div>
        `
            : ""
        }
        ${
          error
            ? `
          <div class="file-error">${error}</div>
        `
            : ""
        }
        <button type="button" 
                class="file-remove" 
                data-remove="${id}"
                aria-label="Remove ${file.name}">
          ×
        </button>
      </div>
    `;
  }

  async uploadFiles() {
    if (this.uploading || !this.options.uploadUrl) return;

    this.uploading = true;

    for (const fileInfo of this.files.filter((f) => !f.uploaded && !f.error)) {
      try {
        await this.uploadFile(fileInfo);
        fileInfo.uploaded = true;
        fileInfo.progress = 100;
      } catch (error) {
        fileInfo.error = error.message;
      }

      this.updateDisplay();
    }

    this.uploading = false;
    this.notifyChange();
  }

  uploadFile(fileInfo) {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append("file", fileInfo.file);

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          fileInfo.progress = (e.loaded / e.total) * 100;
          this.updateDisplay();
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status === 200) {
          resolve(xhr.response);
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`));
        }
      });

      xhr.addEventListener("error", () => {
        reject(new Error("Upload failed: Network error"));
      });

      xhr.open("POST", this.options.uploadUrl);
      xhr.send(formData);
    });
  }

  isValidFileType(file) {
    if (this.options.accept === "*") return true;

    const acceptedTypes = this.options.accept
      .split(",")
      .map((type) => type.trim());

    return acceptedTypes.some((type) => {
      if (type.startsWith(".")) {
        return file.name.toLowerCase().endsWith(type.toLowerCase());
      } else if (type.includes("/*")) {
        const mimePrefix = type.split("/")[0];
        return file.type.startsWith(mimePrefix);
      } else {
        return file.type === type;
      }
    });
  }

  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  showErrors(errors) {
    this.uploadArea.classList.add("error");

    setTimeout(() => {
      this.uploadArea.classList.remove("error");
    }, 3000);

    if (this.parent) {
      this.parent.announceToUser(`File upload errors: ${errors.join(", ")}`);
    }
  }

  notifyChange() {
    if (this.options.onFileChange) {
      this.options.onFileChange(this.files.map((f) => f.file));
    }
  }

  getValue() {
    return this.files.map((f) => f.file);
  }

  destroy() {
    this.container.innerHTML = "";
    this.files = [];
  }
}

// Date Range Picker Component
class DateRangePickerComponent {
  constructor(container, options = {}, parent) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      format: "YYYY-MM-DD",
      placeholder: { start: "Start date", end: "End date" },
      minDate: null,
      maxDate: null,
      ...options,
    };
    this.parent = parent;
    this.value = { start: null, end: null };

    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  render() {
    this.container.innerHTML = `
      <div class="date-range-picker">
        <div class="date-input-group">
          <label for="${this.options.name}-start" class="sr-only">Start date</label>
          <input type="date" 
                 id="${this.options.name}-start"
                 class="form-input date-start" 
                 placeholder="${this.options.placeholder.start}"
                 ${this.options.minDate ? `min="${this.options.minDate}"` : ""}
                 ${this.options.maxDate ? `max="${this.options.maxDate}"` : ""}
                 aria-label="Start date">
        </div>
        <div class="date-range-separator" aria-hidden="true">to</div>
        <div class="date-input-group">
          <label for="${this.options.name}-end" class="sr-only">End date</label>
          <input type="date" 
                 id="${this.options.name}-end"
                 class="form-input date-end" 
                 placeholder="${this.options.placeholder.end}"
                 ${this.options.minDate ? `min="${this.options.minDate}"` : ""}
                 ${this.options.maxDate ? `max="${this.options.maxDate}"` : ""}
                 aria-label="End date">
        </div>
      </div>
    `;

    this.startInput = this.container.querySelector(".date-start");
    this.endInput = this.container.querySelector(".date-end");
  }

  bindEvents() {
    this.startInput.addEventListener("change", () => {
      this.value.start = this.startInput.value
        ? new Date(this.startInput.value)
        : null;
      this.validateRange();
      this.notifyChange();
    });

    this.endInput.addEventListener("change", () => {
      this.value.end = this.endInput.value
        ? new Date(this.endInput.value)
        : null;
      this.validateRange();
      this.notifyChange();
    });
  }

  validateRange() {
    let isValid = true;

    // Clear previous errors
    this.startInput.classList.remove("error");
    this.endInput.classList.remove("error");

    if (this.value.start && this.value.end) {
      if (this.value.start > this.value.end) {
        this.endInput.classList.add("error");
        isValid = false;

        if (this.parent) {
          this.parent.announceToUser("End date must be after start date");
        }
      }
    }

    return isValid;
  }

  setValue(range) {
    this.value = { ...range };

    if (range.start) {
      this.startInput.value = this.formatDate(range.start);
    }

    if (range.end) {
      this.endInput.value = this.formatDate(range.end);
    }
  }

  getValue() {
    return this.value;
  }

  formatDate(date) {
    if (!(date instanceof Date)) return "";
    return date.toISOString().split("T")[0];
  }

  notifyChange() {
    if (this.options.onDateChange) {
      this.options.onDateChange(this.value);
    }
  }

  destroy() {
    this.container.innerHTML = "";
  }
}

// Multi-Select Component
class MultiSelectComponent {
  constructor(container, options = {}, parent) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      options: [],
      placeholder: "Select options...",
      searchable: true,
      clearable: true,
      maxSelections: null,
      ...options,
    };
    this.parent = parent;
    this.selected = [];
    this.isOpen = false;
    this.searchTerm = "";

    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  render() {
    this.container.innerHTML = `
      <div class="multi-select">
        <div class="multi-select-input form-input" 
             tabindex="0" 
             role="combobox" 
             aria-expanded="false"
             aria-haspopup="listbox"
             aria-label="${this.options.placeholder}">
          <span class="multi-select-placeholder">${this.options.placeholder}</span>
          <span class="multi-select-arrow" aria-hidden="true">▼</span>
        </div>
        <div class="multi-select-dropdown" style="display: none;" role="listbox" aria-multiselectable="true">
          ${
            this.options.searchable
              ? `
            <div class="multi-select-search">
              <input type="text" 
                     class="form-input" 
                     placeholder="Search options..." 
                     aria-label="Search options">
            </div>
          `
              : ""
          }
          <div class="multi-select-options">
            ${this.renderOptions()}
          </div>
        </div>
        <div class="selected-tags"></div>
      </div>
    `;

    this.input = this.container.querySelector(".multi-select-input");
    this.dropdown = this.container.querySelector(".multi-select-dropdown");
    this.searchInput = this.container.querySelector(
      ".multi-select-search input",
    );
    this.optionsContainer = this.container.querySelector(
      ".multi-select-options",
    );
    this.tagsContainer = this.container.querySelector(".selected-tags");
    this.placeholder = this.container.querySelector(
      ".multi-select-placeholder",
    );
  }

  renderOptions() {
    const filteredOptions = this.options.options.filter((option) =>
      option.label.toLowerCase().includes(this.searchTerm.toLowerCase()),
    );

    return filteredOptions
      .map(
        (option, index) => `
      <div class="multi-select-option ${this.selected.includes(option.value) ? "selected" : ""}" 
           data-value="${option.value}"
           role="option"
           aria-selected="${this.selected.includes(option.value)}"
           tabindex="-1">
        <input type="checkbox" 
               ${this.selected.includes(option.value) ? "checked" : ""}
               tabindex="-1"
               aria-hidden="true">
        <span>${option.label}</span>
      </div>
    `,
      )
      .join("");
  }

  bindEvents() {
    // Toggle dropdown
    this.input.addEventListener("click", () => {
      this.toggle();
    });

    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        this.toggle();
      } else if (e.key === "Escape") {
        this.close();
      }
    });

    // Search functionality
    if (this.searchInput) {
      this.searchInput.addEventListener("input", (e) => {
        this.searchTerm = e.target.value;
        this.updateOptions();
      });
    }

    // Option selection
    this.optionsContainer.addEventListener("click", (e) => {
      const option = e.target.closest(".multi-select-option");
      if (option) {
        this.toggleOption(option.dataset.value);
      }
    });

    // Close on outside click
    document.addEventListener("click", (e) => {
      if (!this.container.contains(e.target)) {
        this.close();
      }
    });

    // Keyboard navigation in dropdown
    this.dropdown.addEventListener("keydown", (e) => {
      this.handleKeyboardNavigation(e);
    });
  }

  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  open() {
    this.isOpen = true;
    this.dropdown.style.display = "block";
    this.input.setAttribute("aria-expanded", "true");

    if (this.searchInput) {
      this.searchInput.focus();
    }
  }

  close() {
    this.isOpen = false;
    this.dropdown.style.display = "none";
    this.input.setAttribute("aria-expanded", "false");
    this.input.focus();
  }

  toggleOption(value) {
    const index = this.selected.indexOf(value);

    if (index > -1) {
      this.selected.splice(index, 1);
    } else {
      if (
        this.options.maxSelections &&
        this.selected.length >= this.options.maxSelections
      ) {
        if (this.parent) {
          this.parent.announceToUser(
            `Maximum ${this.options.maxSelections} selections allowed`,
          );
        }
        return;
      }

      this.selected.push(value);
    }

    this.updateDisplay();
    this.notifyChange();
  }

  removeSelection(value) {
    const index = this.selected.indexOf(value);
    if (index > -1) {
      this.selected.splice(index, 1);
      this.updateDisplay();
      this.notifyChange();
    }
  }

  updateDisplay() {
    this.updateOptions();
    this.updateTags();
    this.updatePlaceholder();
  }

  updateOptions() {
    this.optionsContainer.innerHTML = this.renderOptions();
  }

  updateTags() {
    const selectedOptions = this.options.options.filter((option) =>
      this.selected.includes(option.value),
    );

    this.tagsContainer.innerHTML = selectedOptions
      .map(
        (option) => `
      <span class="tag" data-value="${option.value}">
        <span>${option.label}</span>
        <button type="button" 
                class="tag-remove" 
                data-remove="${option.value}"
                aria-label="Remove ${option.label}">
          ×
        </button>
      </span>
    `,
      )
      .join("");

    // Bind remove buttons
    this.tagsContainer.querySelectorAll("[data-remove]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.removeSelection(e.target.dataset.remove);
      });
    });

    this.tagsContainer.style.display =
      selectedOptions.length > 0 ? "flex" : "none";
  }

  updatePlaceholder() {
    if (this.selected.length === 0) {
      this.placeholder.textContent = this.options.placeholder;
      this.placeholder.style.display = "block";
    } else {
      this.placeholder.style.display = "none";
    }
  }

  handleKeyboardNavigation(e) {
    const options = this.optionsContainer.querySelectorAll(
      ".multi-select-option",
    );
    const currentFocus = this.optionsContainer.querySelector(
      ".multi-select-option:focus",
    );
    let focusIndex = Array.from(options).indexOf(currentFocus);

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        focusIndex = Math.min(focusIndex + 1, options.length - 1);
        options[focusIndex]?.focus();
        break;
      case "ArrowUp":
        e.preventDefault();
        focusIndex = Math.max(focusIndex - 1, 0);
        options[focusIndex]?.focus();
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (currentFocus) {
          this.toggleOption(currentFocus.dataset.value);
        }
        break;
      case "Escape":
        this.close();
        break;
    }
  }

  setValue(values) {
    this.selected = Array.isArray(values) ? [...values] : [];
    this.updateDisplay();
  }

  getValue() {
    return this.selected;
  }

  notifyChange() {
    if (this.options.onSelectionChange) {
      this.options.onSelectionChange(this.selected);
    }
  }

  destroy() {
    this.container.innerHTML = "";
  }
}

// Dynamic Fieldset Component
class DynamicFieldsetComponent {
  constructor(container, options = {}, parent) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      minItems: 1,
      maxItems: 10,
      template: {},
      addButtonText: "Add Item",
      removeButtonText: "Remove",
      ...options,
    };
    this.parent = parent;
    this.items = [];
    this.nextId = 1;

    this.init();
  }

  init() {
    this.render();
    this.bindEvents();

    // Add initial items
    for (let i = 0; i < this.options.minItems; i++) {
      this.addItem();
    }
  }

  render() {
    this.container.innerHTML = `
      <div class="dynamic-fieldset">
        <div class="fieldset-items"></div>
        <div class="fieldset-actions">
          <button type="button" class="btn btn--secondary add-item-btn">
            ${this.options.addButtonText}
          </button>
        </div>
      </div>
    `;

    this.itemsContainer = this.container.querySelector(".fieldset-items");
    this.addButton = this.container.querySelector(".add-item-btn");
  }

  bindEvents() {
    this.addButton.addEventListener("click", () => {
      this.addItem();
    });
  }

  addItem() {
    if (this.items.length >= this.options.maxItems) {
      if (this.parent) {
        this.parent.announceToUser(
          `Maximum ${this.options.maxItems} items allowed`,
        );
      }
      return;
    }

    const itemId = this.nextId++;
    const item = {
      id: itemId,
      data: {},
    };

    this.items.push(item);
    this.renderItem(item);
    this.updateAddButton();
    this.notifyChange();
  }

  removeItem(itemId) {
    if (this.items.length <= this.options.minItems) {
      if (this.parent) {
        this.parent.announceToUser(
          `Minimum ${this.options.minItems} items required`,
        );
      }
      return;
    }

    this.items = this.items.filter((item) => item.id !== itemId);
    this.container.querySelector(`[data-item-id="${itemId}"]`).remove();
    this.updateAddButton();
    this.notifyChange();
  }

  renderItem(item) {
    const itemElement = document.createElement("div");
    itemElement.className = "fieldset-item";
    itemElement.setAttribute("data-item-id", item.id);

    itemElement.innerHTML = `
      <div class="fieldset-item-header">
        <h4>Item ${this.items.length}</h4>
        ${
          this.items.length > this.options.minItems
            ? `
          <button type="button" 
                  class="btn btn--danger btn--sm remove-item-btn" 
                  data-remove="${item.id}"
                  aria-label="Remove item ${this.items.length}">
            ${this.options.removeButtonText}
          </button>
        `
            : ""
        }
      </div>
      <div class="fieldset-item-content">
        ${this.renderItemFields(item)}
      </div>
    `;

    this.itemsContainer.appendChild(itemElement);

    // Bind remove button
    const removeBtn = itemElement.querySelector(".remove-item-btn");
    if (removeBtn) {
      removeBtn.addEventListener("click", () => {
        this.removeItem(item.id);
      });
    }

    // Bind field events
    itemElement.querySelectorAll("input, select, textarea").forEach((field) => {
      field.addEventListener("input", () => {
        item.data[field.name] = field.value;
        this.notifyChange();
      });
    });
  }

  renderItemFields(item) {
    if (!this.options.template.fields) return "";

    return this.options.template.fields
      .map((field) => {
        const value = item.data[field.name] || field.defaultValue || "";
        const fieldId = `${field.name}_${item.id}`;

        return `
        <div class="form-field">
          <label for="${fieldId}" class="form-label ${field.required ? "required" : ""}">
            ${field.label}
          </label>
          ${this.renderTemplateField(field, fieldId, value)}
          ${field.helpText ? `<div class="form-help">${field.helpText}</div>` : ""}
        </div>
      `;
      })
      .join("");
  }

  renderTemplateField(field, fieldId, value) {
    const commonAttributes = `
      id="${fieldId}"
      name="${field.name}"
      class="form-input"
      ${field.required ? "required" : ""}
      ${field.disabled ? "disabled" : ""}
    `;

    switch (field.type) {
      case "text":
      case "email":
      case "number":
        return `
          <input type="${field.type}" 
                 ${commonAttributes}
                 value="${value}"
                 placeholder="${field.placeholder || ""}">
        `;
      case "textarea":
        return `
          <textarea ${commonAttributes} 
                    placeholder="${field.placeholder || ""}">${value}</textarea>
        `;
      case "select":
        return `
          <select ${commonAttributes}>
            ${field.placeholder ? `<option value="">${field.placeholder}</option>` : ""}
            ${field.options
              .map(
                (option) => `
              <option value="${option.value}" ${value === option.value ? "selected" : ""}>
                ${option.label}
              </option>
            `,
              )
              .join("")}
          </select>
        `;
      default:
        return `
          <input type="text" 
                 ${commonAttributes}
                 value="${value}"
                 placeholder="${field.placeholder || ""}">
        `;
    }
  }

  updateAddButton() {
    this.addButton.disabled = this.items.length >= this.options.maxItems;
  }

  getValue() {
    return this.items.map((item) => item.data);
  }

  setValue(values) {
    // Clear existing items
    this.items = [];
    this.itemsContainer.innerHTML = "";
    this.nextId = 1;

    // Add new items
    for (const value of values) {
      const itemId = this.nextId++;
      const item = {
        id: itemId,
        data: { ...value },
      };

      this.items.push(item);
      this.renderItem(item);
    }

    this.updateAddButton();
  }

  notifyChange() {
    if (this.options.onItemsChange) {
      this.options.onItemsChange(this.getValue());
    }
  }

  destroy() {
    this.container.innerHTML = "";
    this.items = [];
  }
}

// Export for module systems
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    FileUploadComponent,
    DateRangePickerComponent,
    MultiSelectComponent,
    DynamicFieldsetComponent,
  };
}

// Global access
window.FileUploadComponent = FileUploadComponent;
window.DateRangePickerComponent = DateRangePickerComponent;
window.MultiSelectComponent = MultiSelectComponent;
window.DynamicFieldsetComponent = DynamicFieldsetComponent;
