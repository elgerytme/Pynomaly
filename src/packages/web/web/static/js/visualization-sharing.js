/**
 * Visualization Sharing System
 * Handles sharing of visualizations across teams and users
 */

class VisualizationSharing {
  constructor() {
    this.init();
    this.sharedVisualizationsCache = new Map();
    this.apiEndpoint = '/api/visualizations';
  }

  init() {
    this.createSharingInterface();
    this.setupEventListeners();
    this.loadSharedVisualizations();
  }

  createSharingInterface() {
    // Create sharing modal
    const sharingModal = document.createElement('div');
    sharingModal.id = 'sharing-modal';
    sharingModal.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 hidden items-center justify-center z-50';

    sharingModal.innerHTML = `
      <div class="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-screen overflow-y-auto">
        <div class="px-6 py-4 border-b border-gray-200">
          <div class="flex justify-between items-center">
            <h2 class="text-xl font-semibold text-gray-900">Share Visualizations</h2>
            <button id="close-sharing-modal" class="text-gray-400 hover:text-gray-600">
              <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>

        <div class="p-6">
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Share New Visualization -->
            <div class="border border-gray-200 rounded-lg p-4">
              <h3 class="text-lg font-medium text-gray-900 mb-4">Share Current View</h3>

              <div class="space-y-4">
                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-1">Share Title</label>
                  <input type="text" id="share-title" class="w-full px-3 py-2 border border-gray-300 rounded-md" placeholder="My Anomaly Dashboard">
                </div>

                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-1">Description</label>
                  <textarea id="share-description" rows="3" class="w-full px-3 py-2 border border-gray-300 rounded-md" placeholder="Describe what this visualization shows..."></textarea>
                </div>

                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-1">Share With</label>
                  <select id="share-scope" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                    <option value="public">Public - Anyone can view</option>
                    <option value="team">Team - Team members only</option>
                    <option value="private">Private - Specific users only</option>
                  </select>
                </div>

                <div id="specific-users" class="hidden">
                  <label class="block text-sm font-medium text-gray-700 mb-1">User Emails</label>
                  <input type="text" id="user-emails" class="w-full px-3 py-2 border border-gray-300 rounded-md" placeholder="user1@example.com, user2@example.com">
                </div>

                <div class="flex items-center">
                  <input type="checkbox" id="include-data" class="mr-2">
                  <label for="include-data" class="text-sm text-gray-700">Include current data snapshot</label>
                </div>

                <div class="flex items-center">
                  <input type="checkbox" id="allow-comments" class="mr-2" checked>
                  <label for="allow-comments" class="text-sm text-gray-700">Allow comments</label>
                </div>

                <button id="create-share" class="w-full bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600">
                  Create Share Link
                </button>
              </div>

              <div id="share-result" class="mt-4 p-3 bg-gray-50 rounded hidden">
                <div class="text-sm font-medium text-gray-700 mb-2">Share Link Created!</div>
                <div class="flex">
                  <input type="text" id="share-url" class="flex-1 px-2 py-1 text-sm border border-gray-300 rounded-l" readonly>
                  <button id="copy-share-url" class="px-3 py-1 bg-gray-500 text-white text-sm rounded-r hover:bg-gray-600">Copy</button>
                </div>
              </div>
            </div>

            <!-- Browse Shared Visualizations -->
            <div class="border border-gray-200 rounded-lg p-4">
              <h3 class="text-lg font-medium text-gray-900 mb-4">Browse Shared Visualizations</h3>

              <div class="mb-4">
                <div class="flex space-x-2">
                  <button id="filter-all" class="share-filter-btn px-3 py-1 text-sm bg-blue-100 text-blue-800 rounded">All</button>
                  <button id="filter-team" class="share-filter-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200">Team</button>
                  <button id="filter-public" class="share-filter-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200">Public</button>
                  <button id="filter-mine" class="share-filter-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200">Mine</button>
                </div>
              </div>

              <div class="mb-4">
                <input type="text" id="search-shared" class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm" placeholder="Search shared visualizations...">
              </div>

              <div id="shared-viz-list" class="space-y-3 max-h-64 overflow-y-auto">
                <!-- Shared visualizations will be populated here -->
              </div>
            </div>
          </div>

          <!-- Comments Section -->
          <div id="comments-section" class="mt-6 border-t border-gray-200 pt-6 hidden">
            <h3 class="text-lg font-medium text-gray-900 mb-4">Comments</h3>

            <div id="comments-list" class="space-y-3 mb-4">
              <!-- Comments will be populated here -->
            </div>

            <div class="flex space-x-2">
              <input type="text" id="new-comment" class="flex-1 px-3 py-2 border border-gray-300 rounded-md" placeholder="Add a comment...">
              <button id="post-comment" class="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600">Post</button>
            </div>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(sharingModal);

    // Create gallery view
    this.createSharedVisualizationGallery();
  }

  createSharedVisualizationGallery() {
    const galleryContainer = document.createElement('div');
    galleryContainer.id = 'shared-viz-gallery';
    galleryContainer.className = 'mt-6 bg-white shadow rounded-lg';

    galleryContainer.innerHTML = `
      <div class="px-4 py-5 sm:p-6">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-lg font-medium text-gray-900">Shared Visualizations</h2>
          <button id="open-sharing-modal" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
            Share Current View
          </button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" id="shared-viz-grid">
          <!-- Shared visualizations cards will be populated here -->
        </div>
      </div>
    `;

    // Add to visualizations page
    const visualizationsContainer = document.querySelector('.px-4.py-6.sm\\:px-0');
    if (visualizationsContainer) {
      visualizationsContainer.appendChild(galleryContainer);
    }
  }

  setupEventListeners() {
    // Open/close sharing modal
    document.getElementById('open-sharing-modal').addEventListener('click', () => {
      this.showSharingModal();
    });

    document.getElementById('close-sharing-modal').addEventListener('click', () => {
      this.hideSharingModal();
    });

    // Share scope change
    document.getElementById('share-scope').addEventListener('change', (e) => {
      const specificUsers = document.getElementById('specific-users');
      if (e.target.value === 'private') {
        specificUsers.classList.remove('hidden');
      } else {
        specificUsers.classList.add('hidden');
      }
    });

    // Create share
    document.getElementById('create-share').addEventListener('click', () => {
      this.createShare();
    });

    // Copy share URL
    document.getElementById('copy-share-url').addEventListener('click', () => {
      const shareUrl = document.getElementById('share-url');
      shareUrl.select();
      document.execCommand('copy');

      const button = document.getElementById('copy-share-url');
      const originalText = button.textContent;
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = originalText;
      }, 2000);
    });

    // Filter buttons
    document.querySelectorAll('.share-filter-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const filter = e.target.id.replace('filter-', '');
        this.filterSharedVisualizations(filter);

        // Update active filter button
        document.querySelectorAll('.share-filter-btn').forEach(b => {
          b.className = 'share-filter-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200';
        });
        e.target.className = 'share-filter-btn px-3 py-1 text-sm bg-blue-100 text-blue-800 rounded';
      });
    });

    // Search
    document.getElementById('search-shared').addEventListener('input', (e) => {
      this.searchSharedVisualizations(e.target.value);
    });

    // Post comment
    document.getElementById('post-comment').addEventListener('click', () => {
      this.postComment();
    });

    // Enter key for comment
    document.getElementById('new-comment').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.postComment();
      }
    });
  }

  showSharingModal() {
    document.getElementById('sharing-modal').classList.remove('hidden');
    document.getElementById('sharing-modal').classList.add('flex');

    // Pre-fill with current visualization state
    this.captureCurrentVisualization();
  }

  hideSharingModal() {
    document.getElementById('sharing-modal').classList.add('hidden');
    document.getElementById('sharing-modal').classList.remove('flex');

    // Clear form
    document.getElementById('share-title').value = '';
    document.getElementById('share-description').value = '';
    document.getElementById('share-result').classList.add('hidden');
  }

  captureCurrentVisualization() {
    // Capture current state of all charts
    const currentState = {
      timestamp: new Date().toISOString(),
      url: window.location.href,
      charts: [],
      filters: this.getCurrentFilters(),
      timeRange: this.getCurrentTimeRange()
    };

    // Capture each chart's configuration and data
    if (window.AdvancedViz && window.AdvancedViz.charts) {
      Object.entries(window.AdvancedViz.charts).forEach(([id, chart]) => {
        if (chart.getOption) {
          // ECharts
          currentState.charts.push({
            id,
            type: 'echarts',
            config: chart.getOption(),
            screenshot: chart.getDataURL()
          });
        } else if (chart.svg) {
          // D3.js
          currentState.charts.push({
            id,
            type: 'd3',
            svg: new XMLSerializer().serializeToString(chart.svg.node())
          });
        }
      });
    }

    this.currentVisualizationState = currentState;

    // Pre-fill title
    const title = `Anomaly Dashboard - ${new Date().toLocaleDateString()}`;
    document.getElementById('share-title').value = title;
  }

  getCurrentFilters() {
    // Extract current filter state
    const filters = {};

    // Time range
    const activeTimeBtn = document.querySelector('.viz-control-button.active');
    if (activeTimeBtn) {
      filters.timeRange = activeTimeBtn.dataset.range;
    }

    return filters;
  }

  getCurrentTimeRange() {
    const activeBtn = document.querySelector('.viz-control-button.active');
    return activeBtn ? activeBtn.dataset.range : '24h';
  }

  async createShare() {
    const shareData = {
      title: document.getElementById('share-title').value,
      description: document.getElementById('share-description').value,
      scope: document.getElementById('share-scope').value,
      userEmails: document.getElementById('user-emails').value.split(',').map(email => email.trim()).filter(email => email),
      includeData: document.getElementById('include-data').checked,
      allowComments: document.getElementById('allow-comments').checked,
      visualizationState: this.currentVisualizationState,
      createdBy: this.getCurrentUser(),
      createdAt: new Date().toISOString()
    };

    try {
      // In a real application, this would make an API call
      const shareId = this.generateShareId();
      const shareUrl = `${window.location.origin}/shared/${shareId}`;

      // Store in localStorage for demo
      const shares = JSON.parse(localStorage.getItem('pynomaly_shared_viz') || '[]');
      shares.push({ ...shareData, id: shareId, url: shareUrl });
      localStorage.setItem('pynomaly_shared_viz', JSON.stringify(shares));

      // Show success
      document.getElementById('share-url').value = shareUrl;
      document.getElementById('share-result').classList.remove('hidden');

      // Refresh the shared visualizations list
      this.loadSharedVisualizations();

    } catch (error) {
      console.error('Error creating share:', error);
      alert('Error creating share. Please try again.');
    }
  }

  generateShareId() {
    return Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
  }

  getCurrentUser() {
    // In a real application, this would come from authentication
    return 'current_user@example.com';
  }

  async loadSharedVisualizations() {
    try {
      // In a real application, this would be an API call
      const shares = JSON.parse(localStorage.getItem('pynomaly_shared_viz') || '[]');

      this.allSharedVisualizations = shares;
      this.displaySharedVisualizations(shares);
      this.populateSharedVisualizationsList(shares);

    } catch (error) {
      console.error('Error loading shared visualizations:', error);
    }
  }

  displaySharedVisualizations(visualizations) {
    const grid = document.getElementById('shared-viz-grid');
    if (!grid) return;

    grid.innerHTML = visualizations.length === 0
      ? '<div class="col-span-full text-center text-gray-500 py-8">No shared visualizations found</div>'
      : '';

    visualizations.forEach(viz => {
      const card = document.createElement('div');
      card.className = 'border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer';

      // Get first chart screenshot for preview
      const previewImage = viz.visualizationState.charts.length > 0 && viz.visualizationState.charts[0].screenshot
        ? `<img src="${viz.visualizationState.charts[0].screenshot}" alt="Preview" class="w-full h-32 object-cover rounded mb-3">`
        : '<div class="w-full h-32 bg-gray-100 rounded mb-3 flex items-center justify-center text-gray-500">📊</div>';

      card.innerHTML = `
        ${previewImage}
        <h3 class="font-medium text-gray-900 mb-1">${viz.title}</h3>
        <p class="text-sm text-gray-600 mb-2">${viz.description || 'No description'}</p>
        <div class="flex justify-between items-center text-xs text-gray-500">
          <span>By ${viz.createdBy}</span>
          <span>${new Date(viz.createdAt).toLocaleDateString()}</span>
        </div>
        <div class="mt-2">
          <span class="inline-block px-2 py-1 text-xs rounded-full ${
            viz.scope === 'public' ? 'bg-green-100 text-green-800' :
            viz.scope === 'team' ? 'bg-blue-100 text-blue-800' :
            'bg-gray-100 text-gray-800'
          }">${viz.scope}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        this.openSharedVisualization(viz);
      });

      grid.appendChild(card);
    });
  }

  populateSharedVisualizationsList(visualizations) {
    const list = document.getElementById('shared-viz-list');
    if (!list) return;

    list.innerHTML = '';

    visualizations.forEach(viz => {
      const item = document.createElement('div');
      item.className = 'flex justify-between items-center p-2 border border-gray-200 rounded hover:bg-gray-50 cursor-pointer';

      item.innerHTML = `
        <div>
          <div class="font-medium text-sm">${viz.title}</div>
          <div class="text-xs text-gray-500">by ${viz.createdBy}</div>
        </div>
        <div class="text-xs text-gray-400">${new Date(viz.createdAt).toLocaleDateString()}</div>
      `;

      item.addEventListener('click', () => {
        this.openSharedVisualization(viz);
      });

      list.appendChild(item);
    });
  }

  filterSharedVisualizations(filter) {
    if (!this.allSharedVisualizations) return;

    let filtered = this.allSharedVisualizations;

    switch (filter) {
      case 'team':
        filtered = filtered.filter(viz => viz.scope === 'team');
        break;
      case 'public':
        filtered = filtered.filter(viz => viz.scope === 'public');
        break;
      case 'mine':
        filtered = filtered.filter(viz => viz.createdBy === this.getCurrentUser());
        break;
      // 'all' shows everything
    }

    this.displaySharedVisualizations(filtered);
  }

  searchSharedVisualizations(query) {
    if (!this.allSharedVisualizations) return;

    const filtered = this.allSharedVisualizations.filter(viz =>
      viz.title.toLowerCase().includes(query.toLowerCase()) ||
      viz.description.toLowerCase().includes(query.toLowerCase()) ||
      viz.createdBy.toLowerCase().includes(query.toLowerCase())
    );

    this.displaySharedVisualizations(filtered);
  }

  openSharedVisualization(viz) {
    // Create a modal to display the shared visualization
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50';

    modal.innerHTML = `
      <div class="bg-white rounded-lg shadow-xl max-w-6xl w-full mx-4 max-h-screen overflow-y-auto">
        <div class="px-6 py-4 border-b border-gray-200">
          <div class="flex justify-between items-center">
            <div>
              <h2 class="text-xl font-semibold text-gray-900">${viz.title}</h2>
              <p class="text-sm text-gray-600">${viz.description}</p>
            </div>
            <button class="close-viz-modal text-gray-400 hover:text-gray-600">
              <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>

        <div class="p-6">
          <div id="shared-viz-content" class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Charts will be rendered here -->
          </div>

          ${viz.allowComments ? `
            <div class="mt-6 border-t border-gray-200 pt-6">
              <h3 class="text-lg font-medium text-gray-900 mb-4">Comments</h3>
              <div id="viz-comments" class="space-y-3 mb-4">
                <!-- Comments will be loaded here -->
              </div>
              <div class="flex space-x-2">
                <input type="text" id="viz-comment-input" class="flex-1 px-3 py-2 border border-gray-300 rounded-md" placeholder="Add a comment...">
                <button id="post-viz-comment" class="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600">Post</button>
              </div>
            </div>
          ` : ''}
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    // Render the charts
    this.renderSharedCharts(viz.visualizationState.charts);

    // Load comments if enabled
    if (viz.allowComments) {
      this.loadComments(viz.id);
    }

    // Close modal handler
    modal.querySelector('.close-viz-modal').addEventListener('click', () => {
      document.body.removeChild(modal);
    });

    // Comment posting
    if (viz.allowComments) {
      modal.querySelector('#post-viz-comment').addEventListener('click', () => {
        this.postCommentToVisualization(viz.id);
      });
    }
  }

  renderSharedCharts(charts) {
    const container = document.getElementById('shared-viz-content');
    if (!container) return;

    charts.forEach((chartData, index) => {
      const chartDiv = document.createElement('div');
      chartDiv.className = 'border border-gray-200 rounded-lg p-4';
      chartDiv.style.height = '400px';
      container.appendChild(chartDiv);

      if (chartData.type === 'echarts' && chartData.screenshot) {
        // Display screenshot for ECharts
        chartDiv.innerHTML = `<img src="${chartData.screenshot}" alt="Chart" class="w-full h-full object-contain">`;
      } else if (chartData.type === 'd3' && chartData.svg) {
        // Render SVG for D3
        chartDiv.innerHTML = chartData.svg;
      }
    });
  }

  loadComments(vizId) {
    // Load comments from localStorage
    const comments = JSON.parse(localStorage.getItem(`pynomaly_comments_${vizId}`) || '[]');
    this.displayComments(comments);
  }

  displayComments(comments) {
    const container = document.getElementById('viz-comments');
    if (!container) return;

    container.innerHTML = '';

    comments.forEach(comment => {
      const commentDiv = document.createElement('div');
      commentDiv.className = 'flex space-x-3';

      commentDiv.innerHTML = `
        <div class="flex-shrink-0">
          <div class="h-8 w-8 bg-gray-300 rounded-full flex items-center justify-center text-sm font-medium">
            ${comment.author.charAt(0).toUpperCase()}
          </div>
        </div>
        <div class="flex-1">
          <div class="text-sm font-medium text-gray-900">${comment.author}</div>
          <div class="text-sm text-gray-700">${comment.text}</div>
          <div class="text-xs text-gray-500 mt-1">${new Date(comment.timestamp).toLocaleString()}</div>
        </div>
      `;

      container.appendChild(commentDiv);
    });
  }

  postCommentToVisualization(vizId) {
    const input = document.getElementById('viz-comment-input');
    const text = input.value.trim();

    if (!text) return;

    const comment = {
      author: this.getCurrentUser(),
      text,
      timestamp: new Date().toISOString()
    };

    // Save to localStorage
    const comments = JSON.parse(localStorage.getItem(`pynomaly_comments_${vizId}`) || '[]');
    comments.push(comment);
    localStorage.setItem(`pynomaly_comments_${vizId}`, JSON.stringify(comments));

    // Update display
    this.displayComments(comments);

    // Clear input
    input.value = '';
  }

  postComment() {
    // This would be for comments in the main sharing modal
    const input = document.getElementById('new-comment');
    const text = input.value.trim();

    if (!text) return;

    console.log('Posting comment:', text);
    input.value = '';
  }
}

// Initialize visualization sharing
const vizSharing = new VisualizationSharing();
