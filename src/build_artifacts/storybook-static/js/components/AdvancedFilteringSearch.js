/**
 * Advanced Filtering and Search Component
 * Sophisticated filtering, searching, and data exploration capabilities
 */

export class AdvancedFilteringSearch {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      enableFullTextSearch: true,
      enableFacetedSearch: true,
      enableSmartSuggestions: true,
      enableSavedFilters: true,
      enableRealTimeFiltering: true,
      debounceDelay: 300,
      maxSuggestions: 10,
      maxSavedFilters: 20,
      enableRegexSearch: true,
      enableNaturalLanguageQuery: true,
      ...options
    };

    this.data = [];
    this.filteredData = [];
    this.searchHistory = [];
    this.savedFilters = [];
    this.activeFilters = new Map();
    this.searchIndex = null;
    this.debounceTimer = null;
    this.currentQuery = '';
    this.facets = new Map();
    this.filterCallbacks = [];

    this.init();
  }

  init() {
    this.createInterface();
    this.setupEventHandlers();
    this.initializeSearchIndex();
    this.loadSavedFilters();
  }

  createInterface() {
    this.container.innerHTML = `
      <div class="advanced-filtering-container">
        <div class="search-header">
          <div class="search-title">
            <h3>Advanced Search & Filtering</h3>
            <div class="search-stats">
              <span id="total-results">0</span> results
              <span id="filtered-results">0</span> filtered
            </div>
          </div>

          <div class="search-actions">
            <button id="clear-all-filters" class="btn-secondary btn-sm">
              <i class="fas fa-times"></i> Clear All
            </button>
            <button id="save-current-filter" class="btn-primary btn-sm">
              <i class="fas fa-save"></i> Save Filter
            </button>
            <button id="toggle-advanced-mode" class="btn-info btn-sm">
              <i class="fas fa-cog"></i> Advanced
            </button>
          </div>
        </div>

        <div class="search-main">
          <div class="search-input-section">
            <div class="search-input-container">
              <div class="search-input-wrapper">
                <input
                  type="text"
                  id="main-search-input"
                  placeholder="Search anomalies, algorithms, datasets..."
                  class="search-input"
                  autocomplete="off"
                >
                <div class="search-input-actions">
                  <button id="search-options-btn" class="btn-link">
                    <i class="fas fa-chevron-down"></i>
                  </button>
                  <button id="clear-search-btn" class="btn-link">
                    <i class="fas fa-times"></i>
                  </button>
                </div>
              </div>

              <div id="search-suggestions" class="search-suggestions" style="display: none;">
                <div class="suggestions-header">
                  <span class="suggestions-title">Suggestions</span>
                  <button id="close-suggestions" class="btn-link">
                    <i class="fas fa-times"></i>
                  </button>
                </div>
                <div id="suggestions-list" class="suggestions-list"></div>
              </div>
            </div>

            <div id="search-options" class="search-options" style="display: none;">
              <div class="options-row">
                <label class="option-label">
                  <input type="checkbox" id="case-sensitive"> Case Sensitive
                </label>
                <label class="option-label">
                  <input type="checkbox" id="regex-mode"> Regex Mode
                </label>
                <label class="option-label">
                  <input type="checkbox" id="whole-words"> Whole Words
                </label>
                <label class="option-label">
                  <input type="checkbox" id="fuzzy-search" checked> Fuzzy Search
                </label>
              </div>

              <div class="options-row">
                <label for="search-fields">Search in:</label>
                <select id="search-fields" multiple>
                  <option value="all" selected>All Fields</option>
                  <option value="algorithm">Algorithm</option>
                  <option value="dataset">Dataset</option>
                  <option value="timestamp">Timestamp</option>
                  <option value="anomaly_score">Anomaly Score</option>
                  <option value="metadata">Metadata</option>
                </select>
              </div>
            </div>
          </div>

          <div class="natural-language-section" style="display: none;">
            <div class="nl-input-container">
              <textarea
                id="natural-language-input"
                placeholder="Ask a question: 'Show me anomalies from the last week with high confidence scores'"
                class="nl-input"
                rows="3"
              ></textarea>
              <button id="process-nl-query" class="btn-primary btn-sm">
                <i class="fas fa-brain"></i> Process Query
              </button>
            </div>
          </div>
        </div>

        <div class="filtering-section">
          <div class="quick-filters">
            <div class="filter-group">
              <label class="filter-label">Quick Filters:</label>
              <div class="filter-buttons">
                <button class="filter-btn" data-filter="anomalies-only">
                  <i class="fas fa-exclamation-triangle"></i> Anomalies Only
                </button>
                <button class="filter-btn" data-filter="high-confidence">
                  <i class="fas fa-check-circle"></i> High Confidence
                </button>
                <button class="filter-btn" data-filter="recent">
                  <i class="fas fa-clock"></i> Recent
                </button>
                <button class="filter-btn" data-filter="critical">
                  <i class="fas fa-exclamation"></i> Critical
                </button>
              </div>
            </div>
          </div>

          <div class="active-filters">
            <div class="filter-group">
              <label class="filter-label">Active Filters:</label>
              <div id="active-filters-container" class="active-filters-list">
                <div class="no-filters">No active filters</div>
              </div>
            </div>
          </div>
        </div>

        <div class="advanced-filters" style="display: none;">
          <div class="filters-layout">
            <div class="faceted-filters">
              <div class="facet-group">
                <h4>Algorithm</h4>
                <div id="algorithm-facets" class="facet-items"></div>
              </div>

              <div class="facet-group">
                <h4>Dataset</h4>
                <div id="dataset-facets" class="facet-items"></div>
              </div>

              <div class="facet-group">
                <h4>Time Range</h4>
                <div class="time-range-controls">
                  <input type="date" id="start-date" class="date-input">
                  <span class="date-separator">to</span>
                  <input type="date" id="end-date" class="date-input">
                </div>
              </div>

              <div class="facet-group">
                <h4>Anomaly Score</h4>
                <div class="range-slider-container">
                  <input type="range" id="anomaly-score-min" min="0" max="1" step="0.01" value="0" class="range-slider">
                  <input type="range" id="anomaly-score-max" min="0" max="1" step="0.01" value="1" class="range-slider">
                  <div class="range-values">
                    <span id="anomaly-score-min-value">0.00</span> -
                    <span id="anomaly-score-max-value">1.00</span>
                  </div>
                </div>
              </div>

              <div class="facet-group">
                <h4>Confidence</h4>
                <div class="range-slider-container">
                  <input type="range" id="confidence-min" min="0" max="1" step="0.01" value="0" class="range-slider">
                  <input type="range" id="confidence-max" min="0" max="1" step="0.01" value="1" class="range-slider">
                  <div class="range-values">
                    <span id="confidence-min-value">0.00</span> -
                    <span id="confidence-max-value">1.00</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="custom-filters">
              <div class="custom-filter-builder">
                <h4>Custom Filter Builder</h4>
                <div id="filter-builder-container" class="filter-builder">
                  <div class="filter-rule">
                    <select class="field-selector">
                      <option value="algorithm">Algorithm</option>
                      <option value="dataset">Dataset</option>
                      <option value="anomaly_score">Anomaly Score</option>
                      <option value="confidence">Confidence</option>
                      <option value="timestamp">Timestamp</option>
                    </select>
                    <select class="operator-selector">
                      <option value="equals">Equals</option>
                      <option value="not_equals">Not Equals</option>
                      <option value="greater_than">Greater Than</option>
                      <option value="less_than">Less Than</option>
                      <option value="contains">Contains</option>
                      <option value="starts_with">Starts With</option>
                      <option value="ends_with">Ends With</option>
                      <option value="regex">Regex</option>
                    </select>
                    <input type="text" class="value-input" placeholder="Value">
                    <button class="add-rule-btn btn-sm btn-primary">
                      <i class="fas fa-plus"></i>
                    </button>
                  </div>
                </div>
                <div class="filter-builder-actions">
                  <button id="apply-custom-filter" class="btn-primary">Apply Filter</button>
                  <button id="reset-custom-filter" class="btn-secondary">Reset</button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="saved-filters-section">
          <div class="saved-filters-header">
            <h4>Saved Filters</h4>
            <button id="manage-saved-filters" class="btn-sm btn-secondary">
              <i class="fas fa-cog"></i> Manage
            </button>
          </div>
          <div id="saved-filters-list" class="saved-filters-list"></div>
        </div>

        <div class="search-history-section">
          <div class="search-history-header">
            <h4>Search History</h4>
            <button id="clear-search-history" class="btn-sm btn-secondary">
              <i class="fas fa-trash"></i> Clear
            </button>
          </div>
          <div id="search-history-list" class="search-history-list"></div>
        </div>
      </div>

      <!-- Save Filter Modal -->
      <div id="save-filter-modal" class="modal" style="display: none;">
        <div class="modal-content">
          <div class="modal-header">
            <h3>Save Filter</h3>
            <button id="close-save-filter-modal" class="btn-close">&times;</button>
          </div>
          <div class="modal-body">
            <div class="form-group">
              <label for="filter-name">Filter Name:</label>
              <input type="text" id="filter-name" class="form-control" placeholder="Enter filter name">
            </div>
            <div class="form-group">
              <label for="filter-description">Description:</label>
              <textarea id="filter-description" class="form-control" rows="3" placeholder="Optional description"></textarea>
            </div>
            <div class="form-group">
              <label>
                <input type="checkbox" id="filter-public"> Make this filter public
              </label>
            </div>
          </div>
          <div class="modal-footer">
            <button id="save-filter-confirm" class="btn-primary">Save Filter</button>
            <button id="cancel-save-filter" class="btn-secondary">Cancel</button>
          </div>
        </div>
      </div>
    `;
  }

  setupEventHandlers() {
    // Main search input
    const searchInput = document.getElementById('main-search-input');
    searchInput.addEventListener('input', (e) => {
      this.handleSearchInput(e.target.value);
    });

    searchInput.addEventListener('focus', () => {
      this.showSuggestions();
    });

    // Search options
    document.getElementById('search-options-btn').addEventListener('click', () => {
      this.toggleSearchOptions();
    });

    document.getElementById('clear-search-btn').addEventListener('click', () => {
      this.clearSearch();
    });

    // Advanced mode toggle
    document.getElementById('toggle-advanced-mode').addEventListener('click', () => {
      this.toggleAdvancedMode();
    });

    // Natural language query
    document.getElementById('process-nl-query').addEventListener('click', () => {
      this.processNaturalLanguageQuery();
    });

    // Quick filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.applyQuickFilter(e.target.dataset.filter);
      });
    });

    // Range sliders
    this.setupRangeSliders();

    // Filter builder
    document.getElementById('apply-custom-filter').addEventListener('click', () => {
      this.applyCustomFilter();
    });

    document.getElementById('reset-custom-filter').addEventListener('click', () => {
      this.resetCustomFilter();
    });

    // Saved filters
    document.getElementById('save-current-filter').addEventListener('click', () => {
      this.showSaveFilterModal();
    });

    document.getElementById('manage-saved-filters').addEventListener('click', () => {
      this.showManageSavedFiltersModal();
    });

    // Clear actions
    document.getElementById('clear-all-filters').addEventListener('click', () => {
      this.clearAllFilters();
    });

    document.getElementById('clear-search-history').addEventListener('click', () => {
      this.clearSearchHistory();
    });

    // Modal handlers
    this.setupModalHandlers();

    // Date inputs
    document.getElementById('start-date').addEventListener('change', () => {
      this.applyDateFilter();
    });

    document.getElementById('end-date').addEventListener('change', () => {
      this.applyDateFilter();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      this.handleKeyboardShortcuts(e);
    });
  }

  setupRangeSliders() {
    const sliders = [
      { min: 'anomaly-score-min', max: 'anomaly-score-max', valueMin: 'anomaly-score-min-value', valueMax: 'anomaly-score-max-value' },
      { min: 'confidence-min', max: 'confidence-max', valueMin: 'confidence-min-value', valueMax: 'confidence-max-value' }
    ];

    sliders.forEach(slider => {
      const minSlider = document.getElementById(slider.min);
      const maxSlider = document.getElementById(slider.max);
      const minValue = document.getElementById(slider.valueMin);
      const maxValue = document.getElementById(slider.valueMax);

      minSlider.addEventListener('input', (e) => {
        minValue.textContent = parseFloat(e.target.value).toFixed(2);
        if (parseFloat(e.target.value) > parseFloat(maxSlider.value)) {
          maxSlider.value = e.target.value;
          maxValue.textContent = parseFloat(e.target.value).toFixed(2);
        }
        this.applyRangeFilter(slider.min.split('-')[0], minSlider.value, maxSlider.value);
      });

      maxSlider.addEventListener('input', (e) => {
        maxValue.textContent = parseFloat(e.target.value).toFixed(2);
        if (parseFloat(e.target.value) < parseFloat(minSlider.value)) {
          minSlider.value = e.target.value;
          minValue.textContent = parseFloat(e.target.value).toFixed(2);
        }
        this.applyRangeFilter(slider.min.split('-')[0], minSlider.value, maxSlider.value);
      });
    });
  }

  setupModalHandlers() {
    // Save filter modal
    document.getElementById('close-save-filter-modal').addEventListener('click', () => {
      document.getElementById('save-filter-modal').style.display = 'none';
    });

    document.getElementById('save-filter-confirm').addEventListener('click', () => {
      this.saveCurrentFilter();
    });

    document.getElementById('cancel-save-filter').addEventListener('click', () => {
      document.getElementById('save-filter-modal').style.display = 'none';
    });
  }

  setData(data) {
    this.data = data;
    this.filteredData = [...data];
    this.buildSearchIndex();
    this.updateFacets();
    this.updateResultsCount();
  }

  buildSearchIndex() {
    // Create search index for fast full-text search
    this.searchIndex = new Map();

    this.data.forEach((item, index) => {
      const searchableText = this.extractSearchableText(item).toLowerCase();
      const words = searchableText.split(/\s+/);

      words.forEach(word => {
        if (!this.searchIndex.has(word)) {
          this.searchIndex.set(word, new Set());
        }
        this.searchIndex.get(word).add(index);
      });
    });
  }

  extractSearchableText(item) {
    // Extract all searchable text from item
    const searchableFields = ['algorithm', 'dataset', 'metadata', 'description'];
    let text = '';

    searchableFields.forEach(field => {
      if (item[field]) {
        if (typeof item[field] === 'object') {
          text += ' ' + JSON.stringify(item[field]);
        } else {
          text += ' ' + item[field];
        }
      }
    });

    return text;
  }

  updateFacets() {
    // Update faceted search options
    this.facets.clear();

    // Algorithm facets
    const algorithms = new Map();
    this.data.forEach(item => {
      if (item.algorithm) {
        algorithms.set(item.algorithm, (algorithms.get(item.algorithm) || 0) + 1);
      }
    });
    this.facets.set('algorithm', algorithms);

    // Dataset facets
    const datasets = new Map();
    this.data.forEach(item => {
      if (item.dataset) {
        datasets.set(item.dataset, (datasets.get(item.dataset) || 0) + 1);
      }
    });
    this.facets.set('dataset', datasets);

    this.renderFacets();
  }

  renderFacets() {
    // Render algorithm facets
    const algorithmFacets = document.getElementById('algorithm-facets');
    algorithmFacets.innerHTML = '';

    this.facets.get('algorithm')?.forEach((count, algorithm) => {
      const facetItem = document.createElement('div');
      facetItem.className = 'facet-item';
      facetItem.innerHTML = `
        <label class="facet-label">
          <input type="checkbox" class="facet-checkbox" data-facet="algorithm" data-value="${algorithm}">
          <span class="facet-text">${algorithm}</span>
          <span class="facet-count">(${count})</span>
        </label>
      `;
      algorithmFacets.appendChild(facetItem);
    });

    // Render dataset facets
    const datasetFacets = document.getElementById('dataset-facets');
    datasetFacets.innerHTML = '';

    this.facets.get('dataset')?.forEach((count, dataset) => {
      const facetItem = document.createElement('div');
      facetItem.className = 'facet-item';
      facetItem.innerHTML = `
        <label class="facet-label">
          <input type="checkbox" class="facet-checkbox" data-facet="dataset" data-value="${dataset}">
          <span class="facet-text">${dataset}</span>
          <span class="facet-count">(${count})</span>
        </label>
      `;
      datasetFacets.appendChild(facetItem);
    });

    // Add event listeners to facet checkboxes
    document.querySelectorAll('.facet-checkbox').forEach(checkbox => {
      checkbox.addEventListener('change', (e) => {
        this.handleFacetChange(e.target);
      });
    });
  }

  handleSearchInput(query) {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.currentQuery = query;
      this.performSearch(query);
      this.addToSearchHistory(query);
      this.updateSuggestions(query);
    }, this.options.debounceDelay);
  }

  performSearch(query) {
    if (!query.trim()) {
      this.filteredData = [...this.data];
      this.applyActiveFilters();
      return;
    }

    const searchOptions = this.getSearchOptions();
    let results = [];

    if (searchOptions.regexMode) {
      results = this.performRegexSearch(query, searchOptions);
    } else if (searchOptions.fuzzySearch) {
      results = this.performFuzzySearch(query, searchOptions);
    } else {
      results = this.performExactSearch(query, searchOptions);
    }

    this.filteredData = results;
    this.applyActiveFilters();
  }

  getSearchOptions() {
    return {
      caseSensitive: document.getElementById('case-sensitive').checked,
      regexMode: document.getElementById('regex-mode').checked,
      wholeWords: document.getElementById('whole-words').checked,
      fuzzySearch: document.getElementById('fuzzy-search').checked,
      searchFields: Array.from(document.getElementById('search-fields').selectedOptions).map(opt => opt.value)
    };
  }

  performExactSearch(query, options) {
    const searchQuery = options.caseSensitive ? query : query.toLowerCase();
    const results = [];

    this.data.forEach(item => {
      const searchableText = options.caseSensitive ?
        this.extractSearchableText(item) :
        this.extractSearchableText(item).toLowerCase();

      if (options.wholeWords) {
        const regex = new RegExp(`\\b${this.escapeRegExp(searchQuery)}\\b`, 'g');
        if (regex.test(searchableText)) {
          results.push(item);
        }
      } else {
        if (searchableText.includes(searchQuery)) {
          results.push(item);
        }
      }
    });

    return results;
  }

  performRegexSearch(query, options) {
    const results = [];

    try {
      const flags = options.caseSensitive ? 'g' : 'gi';
      const regex = new RegExp(query, flags);

      this.data.forEach(item => {
        const searchableText = this.extractSearchableText(item);
        if (regex.test(searchableText)) {
          results.push(item);
        }
      });
    } catch (error) {
      console.error('Invalid regex pattern:', error);
      return this.performExactSearch(query, options);
    }

    return results;
  }

  performFuzzySearch(query, options) {
    const results = [];
    const threshold = 0.6; // Similarity threshold

    this.data.forEach(item => {
      const searchableText = this.extractSearchableText(item);
      const similarity = this.calculateSimilarity(query, searchableText);

      if (similarity >= threshold) {
        results.push({ ...item, _similarity: similarity });
      }
    });

    // Sort by similarity
    results.sort((a, b) => b._similarity - a._similarity);

    return results;
  }

  calculateSimilarity(str1, str2) {
    // Simple Levenshtein distance-based similarity
    const distance = this.levenshteinDistance(str1.toLowerCase(), str2.toLowerCase());
    const maxLength = Math.max(str1.length, str2.length);
    return 1 - (distance / maxLength);
  }

  levenshteinDistance(str1, str2) {
    const matrix = [];

    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }

    return matrix[str2.length][str1.length];
  }

  updateSuggestions(query) {
    if (!query.trim()) {
      this.hideSuggestions();
      return;
    }

    const suggestions = this.generateSuggestions(query);
    this.renderSuggestions(suggestions);
  }

  generateSuggestions(query) {
    const suggestions = [];
    const queryLower = query.toLowerCase();

    // Search in algorithm names
    this.facets.get('algorithm')?.forEach((count, algorithm) => {
      if (algorithm.toLowerCase().includes(queryLower)) {
        suggestions.push({
          text: algorithm,
          type: 'algorithm',
          count: count
        });
      }
    });

    // Search in dataset names
    this.facets.get('dataset')?.forEach((count, dataset) => {
      if (dataset.toLowerCase().includes(queryLower)) {
        suggestions.push({
          text: dataset,
          type: 'dataset',
          count: count
        });
      }
    });

    // Add search history suggestions
    this.searchHistory.forEach(historyItem => {
      if (historyItem.query.toLowerCase().includes(queryLower)) {
        suggestions.push({
          text: historyItem.query,
          type: 'history',
          timestamp: historyItem.timestamp
        });
      }
    });

    return suggestions.slice(0, this.options.maxSuggestions);
  }

  renderSuggestions(suggestions) {
    const suggestionsList = document.getElementById('suggestions-list');
    suggestionsList.innerHTML = '';

    suggestions.forEach(suggestion => {
      const suggestionItem = document.createElement('div');
      suggestionItem.className = 'suggestion-item';
      suggestionItem.innerHTML = `
        <div class="suggestion-content">
          <div class="suggestion-text">${suggestion.text}</div>
          <div class="suggestion-meta">
            <span class="suggestion-type">${suggestion.type}</span>
            ${suggestion.count ? `<span class="suggestion-count">${suggestion.count} results</span>` : ''}
          </div>
        </div>
      `;

      suggestionItem.addEventListener('click', () => {
        this.applySuggestion(suggestion);
      });

      suggestionsList.appendChild(suggestionItem);
    });

    this.showSuggestions();
  }

  applySuggestion(suggestion) {
    document.getElementById('main-search-input').value = suggestion.text;
    this.performSearch(suggestion.text);
    this.hideSuggestions();
  }

  showSuggestions() {
    document.getElementById('search-suggestions').style.display = 'block';
  }

  hideSuggestions() {
    document.getElementById('search-suggestions').style.display = 'none';
  }

  handleFacetChange(checkbox) {
    const facetType = checkbox.dataset.facet;
    const facetValue = checkbox.dataset.value;

    if (checkbox.checked) {
      this.addActiveFilter(facetType, facetValue);
    } else {
      this.removeActiveFilter(facetType, facetValue);
    }

    this.applyActiveFilters();
  }

  addActiveFilter(type, value) {
    if (!this.activeFilters.has(type)) {
      this.activeFilters.set(type, new Set());
    }
    this.activeFilters.get(type).add(value);
    this.updateActiveFiltersDisplay();
  }

  removeActiveFilter(type, value) {
    if (this.activeFilters.has(type)) {
      this.activeFilters.get(type).delete(value);
      if (this.activeFilters.get(type).size === 0) {
        this.activeFilters.delete(type);
      }
    }
    this.updateActiveFiltersDisplay();
  }

  updateActiveFiltersDisplay() {
    const container = document.getElementById('active-filters-container');
    container.innerHTML = '';

    if (this.activeFilters.size === 0) {
      container.innerHTML = '<div class="no-filters">No active filters</div>';
      return;
    }

    this.activeFilters.forEach((values, type) => {
      values.forEach(value => {
        const filterTag = document.createElement('div');
        filterTag.className = 'filter-tag';
        filterTag.innerHTML = `
          <span class="filter-type">${type}:</span>
          <span class="filter-value">${value}</span>
          <button class="remove-filter" data-type="${type}" data-value="${value}">
            <i class="fas fa-times"></i>
          </button>
        `;

        filterTag.querySelector('.remove-filter').addEventListener('click', (e) => {
          this.removeActiveFilter(e.target.dataset.type, e.target.dataset.value);
          this.applyActiveFilters();
        });

        container.appendChild(filterTag);
      });
    });
  }

  applyActiveFilters() {
    let filtered = [...this.filteredData];

    this.activeFilters.forEach((values, type) => {
      filtered = filtered.filter(item => {
        return Array.from(values).some(value => {
          switch (type) {
            case 'algorithm':
              return item.algorithm === value;
            case 'dataset':
              return item.dataset === value;
            default:
              return true;
          }
        });
      });
    });

    this.filteredData = filtered;
    this.updateResultsCount();
    this.notifyFilterCallbacks();
  }

  applyQuickFilter(filterType) {
    switch (filterType) {
      case 'anomalies-only':
        this.filteredData = this.data.filter(item => item.is_anomaly);
        break;
      case 'high-confidence':
        this.filteredData = this.data.filter(item => item.confidence > 0.8);
        break;
      case 'recent':
        const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        this.filteredData = this.data.filter(item => new Date(item.timestamp) > dayAgo);
        break;
      case 'critical':
        this.filteredData = this.data.filter(item => item.severity === 'critical');
        break;
    }

    this.applyActiveFilters();
  }

  applyRangeFilter(field, min, max) {
    const minVal = parseFloat(min);
    const maxVal = parseFloat(max);

    this.filteredData = this.data.filter(item => {
      const value = item[field];
      return value >= minVal && value <= maxVal;
    });

    this.applyActiveFilters();
  }

  applyDateFilter() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;

    if (!startDate && !endDate) return;

    this.filteredData = this.data.filter(item => {
      const itemDate = new Date(item.timestamp);
      if (startDate && itemDate < new Date(startDate)) return false;
      if (endDate && itemDate > new Date(endDate)) return false;
      return true;
    });

    this.applyActiveFilters();
  }

  processNaturalLanguageQuery() {
    const query = document.getElementById('natural-language-input').value;

    // Simple natural language processing
    const filters = this.parseNaturalLanguageQuery(query);
    this.applyNaturalLanguageFilters(filters);
  }

  parseNaturalLanguageQuery(query) {
    const filters = {};
    const lowerQuery = query.toLowerCase();

    // Time-based filters
    if (lowerQuery.includes('last week')) {
      filters.timeRange = 'last_week';
    } else if (lowerQuery.includes('last month')) {
      filters.timeRange = 'last_month';
    } else if (lowerQuery.includes('today')) {
      filters.timeRange = 'today';
    }

    // Confidence-based filters
    if (lowerQuery.includes('high confidence')) {
      filters.confidence = 'high';
    } else if (lowerQuery.includes('low confidence')) {
      filters.confidence = 'low';
    }

    // Anomaly-based filters
    if (lowerQuery.includes('anomalies') || lowerQuery.includes('outliers')) {
      filters.anomaliesOnly = true;
    }

    return filters;
  }

  applyNaturalLanguageFilters(filters) {
    let filtered = [...this.data];

    if (filters.timeRange) {
      const now = new Date();
      let cutoff;

      switch (filters.timeRange) {
        case 'last_week':
          cutoff = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
          break;
        case 'last_month':
          cutoff = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
          break;
        case 'today':
          cutoff = new Date(now.getFullYear(), now.getMonth(), now.getDate());
          break;
      }

      if (cutoff) {
        filtered = filtered.filter(item => new Date(item.timestamp) > cutoff);
      }
    }

    if (filters.confidence) {
      if (filters.confidence === 'high') {
        filtered = filtered.filter(item => item.confidence > 0.8);
      } else if (filters.confidence === 'low') {
        filtered = filtered.filter(item => item.confidence < 0.3);
      }
    }

    if (filters.anomaliesOnly) {
      filtered = filtered.filter(item => item.is_anomaly);
    }

    this.filteredData = filtered;
    this.updateResultsCount();
    this.notifyFilterCallbacks();
  }

  addToSearchHistory(query) {
    if (!query.trim()) return;

    // Remove if already exists
    this.searchHistory = this.searchHistory.filter(item => item.query !== query);

    // Add to beginning
    this.searchHistory.unshift({
      query: query,
      timestamp: Date.now(),
      results: this.filteredData.length
    });

    // Keep only last 20 searches
    if (this.searchHistory.length > 20) {
      this.searchHistory.pop();
    }

    this.updateSearchHistoryDisplay();
  }

  updateSearchHistoryDisplay() {
    const historyList = document.getElementById('search-history-list');
    historyList.innerHTML = '';

    this.searchHistory.forEach(item => {
      const historyItem = document.createElement('div');
      historyItem.className = 'history-item';
      historyItem.innerHTML = `
        <div class="history-query">${item.query}</div>
        <div class="history-meta">
          <span class="history-time">${new Date(item.timestamp).toLocaleString()}</span>
          <span class="history-results">${item.results} results</span>
        </div>
      `;

      historyItem.addEventListener('click', () => {
        document.getElementById('main-search-input').value = item.query;
        this.performSearch(item.query);
      });

      historyList.appendChild(historyItem);
    });
  }

  showSaveFilterModal() {
    document.getElementById('save-filter-modal').style.display = 'block';
  }

  saveCurrentFilter() {
    const name = document.getElementById('filter-name').value;
    const description = document.getElementById('filter-description').value;
    const isPublic = document.getElementById('filter-public').checked;

    if (!name.trim()) {
      alert('Please enter a filter name');
      return;
    }

    const filter = {
      id: Date.now(),
      name: name,
      description: description,
      isPublic: isPublic,
      query: this.currentQuery,
      activeFilters: Object.fromEntries(this.activeFilters),
      timestamp: Date.now()
    };

    this.savedFilters.push(filter);
    this.saveSavedFilters();
    this.updateSavedFiltersDisplay();

    document.getElementById('save-filter-modal').style.display = 'none';
    document.getElementById('filter-name').value = '';
    document.getElementById('filter-description').value = '';
    document.getElementById('filter-public').checked = false;
  }

  updateSavedFiltersDisplay() {
    const filtersList = document.getElementById('saved-filters-list');
    filtersList.innerHTML = '';

    this.savedFilters.forEach(filter => {
      const filterItem = document.createElement('div');
      filterItem.className = 'saved-filter-item';
      filterItem.innerHTML = `
        <div class="saved-filter-header">
          <div class="saved-filter-name">${filter.name}</div>
          <div class="saved-filter-actions">
            <button class="apply-saved-filter btn-sm btn-primary" data-filter-id="${filter.id}">
              <i class="fas fa-play"></i> Apply
            </button>
            <button class="delete-saved-filter btn-sm btn-danger" data-filter-id="${filter.id}">
              <i class="fas fa-trash"></i>
            </button>
          </div>
        </div>
        <div class="saved-filter-description">${filter.description}</div>
        <div class="saved-filter-meta">
          <span class="saved-filter-date">${new Date(filter.timestamp).toLocaleDateString()}</span>
          ${filter.isPublic ? '<span class="saved-filter-public">Public</span>' : ''}
        </div>
      `;

      filterItem.querySelector('.apply-saved-filter').addEventListener('click', (e) => {
        this.applySavedFilter(filter.id);
      });

      filterItem.querySelector('.delete-saved-filter').addEventListener('click', (e) => {
        this.deleteSavedFilter(filter.id);
      });

      filtersList.appendChild(filterItem);
    });
  }

  applySavedFilter(filterId) {
    const filter = this.savedFilters.find(f => f.id === filterId);
    if (!filter) return;

    // Apply the saved query
    document.getElementById('main-search-input').value = filter.query;
    this.performSearch(filter.query);

    // Apply saved filters
    this.activeFilters.clear();
    Object.entries(filter.activeFilters).forEach(([type, values]) => {
      this.activeFilters.set(type, new Set(values));
    });

    this.applyActiveFilters();
  }

  deleteSavedFilter(filterId) {
    if (confirm('Are you sure you want to delete this saved filter?')) {
      this.savedFilters = this.savedFilters.filter(f => f.id !== filterId);
      this.saveSavedFilters();
      this.updateSavedFiltersDisplay();
    }
  }

  clearAllFilters() {
    this.currentQuery = '';
    this.activeFilters.clear();
    this.filteredData = [...this.data];

    document.getElementById('main-search-input').value = '';
    this.updateActiveFiltersDisplay();
    this.updateResultsCount();
    this.notifyFilterCallbacks();

    // Reset UI elements
    document.querySelectorAll('.facet-checkbox').forEach(checkbox => {
      checkbox.checked = false;
    });

    document.getElementById('start-date').value = '';
    document.getElementById('end-date').value = '';
    document.getElementById('anomaly-score-min').value = 0;
    document.getElementById('anomaly-score-max').value = 1;
    document.getElementById('confidence-min').value = 0;
    document.getElementById('confidence-max').value = 1;
  }

  clearSearch() {
    document.getElementById('main-search-input').value = '';
    this.currentQuery = '';
    this.performSearch('');
  }

  clearSearchHistory() {
    this.searchHistory = [];
    this.updateSearchHistoryDisplay();
  }

  toggleSearchOptions() {
    const options = document.getElementById('search-options');
    options.style.display = options.style.display === 'none' ? 'block' : 'none';
  }

  toggleAdvancedMode() {
    const advancedFilters = document.querySelector('.advanced-filters');
    const naturalLanguage = document.querySelector('.natural-language-section');

    if (advancedFilters.style.display === 'none') {
      advancedFilters.style.display = 'block';
      naturalLanguage.style.display = 'block';
      document.getElementById('toggle-advanced-mode').innerHTML = '<i class="fas fa-times"></i> Basic';
    } else {
      advancedFilters.style.display = 'none';
      naturalLanguage.style.display = 'none';
      document.getElementById('toggle-advanced-mode').innerHTML = '<i class="fas fa-cog"></i> Advanced';
    }
  }

  handleKeyboardShortcuts(e) {
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case 'f':
          e.preventDefault();
          document.getElementById('main-search-input').focus();
          break;
        case 'k':
          e.preventDefault();
          this.clearAllFilters();
          break;
        case 's':
          e.preventDefault();
          this.showSaveFilterModal();
          break;
      }
    }
  }

  updateResultsCount() {
    document.getElementById('total-results').textContent = this.data.length;
    document.getElementById('filtered-results').textContent = this.filteredData.length;
  }

  // Callback system
  onFilter(callback) {
    this.filterCallbacks.push(callback);
  }

  notifyFilterCallbacks() {
    this.filterCallbacks.forEach(callback => {
      callback(this.filteredData);
    });
  }

  // Persistence
  loadSavedFilters() {
    const saved = localStorage.getItem('pynomaly_saved_filters');
    if (saved) {
      this.savedFilters = JSON.parse(saved);
      this.updateSavedFiltersDisplay();
    }
  }

  saveSavedFilters() {
    localStorage.setItem('pynomaly_saved_filters', JSON.stringify(this.savedFilters));
  }

  // Utility methods
  escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  initializeSearchIndex() {
    // Initialize search index with empty data
    this.searchIndex = new Map();
  }

  getFilteredData() {
    return this.filteredData;
  }

  destroy() {
    // Clean up
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.data = [];
    this.filteredData = [];
    this.searchHistory = [];
    this.savedFilters = [];
    this.activeFilters.clear();
    this.facets.clear();
    this.filterCallbacks = [];
  }
}

// Export the class
window.AdvancedFilteringSearch = AdvancedFilteringSearch;
