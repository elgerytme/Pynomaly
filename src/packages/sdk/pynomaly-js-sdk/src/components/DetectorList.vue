<!--
  Detector List Component (Vue)
  
  Vue component for displaying and managing a list of anomaly detectors
  with pagination, filtering, and selection capabilities.
-->

<template>
  <div :class="`pynomaly-detector-list ${className}`">
    <!-- Error State -->
    <div v-if="error" class="error-container">
      <div class="error-message">
        <h3>Error Loading Detectors</h3>
        <p>{{ error.message }}</p>
        <div class="error-actions">
          <button @click="clearError" class="btn-secondary">
            Dismiss
          </button>
          <button @click="refresh" class="btn-primary">
            Retry
          </button>
        </div>
      </div>
    </div>

    <template v-else>
      <!-- Header -->
      <div class="detector-list-header">
        <div class="header-info">
          <h2>Detectors</h2>
          <span class="total-count">{{ total }} total</span>
        </div>
        <div class="header-actions">
          <button 
            @click="refresh" 
            :disabled="isLoading"
            class="btn-secondary"
          >
            {{ isLoading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Filters -->
      <div v-if="filters" class="detector-filters">
        <span v-if="filters.algorithmName" class="filter-tag">
          Algorithm: {{ filters.algorithmName }}
        </span>
        <span 
          v-for="tag in filters.tags" 
          :key="tag" 
          class="filter-tag"
        >
          Tag: {{ tag }}
        </span>
      </div>

      <!-- Loading State -->
      <div v-if="isLoading && detectors.length === 0" class="loading-container">
        <div class="loading-spinner"></div>
        <p>Loading detectors...</p>
      </div>

      <!-- Empty State -->
      <div v-else-if="!isLoading && detectors.length === 0" class="empty-state">
        <div class="empty-icon">🔍</div>
        <h3>No Detectors Found</h3>
        <p>
          {{ filters ? 
            'No detectors match your current filters.' : 
            'No detectors have been created yet.'
          }}
        </p>
      </div>

      <!-- Detector Grid -->
      <template v-else>
        <div class="detector-grid">
          <div
            v-for="detector in detectors"
            :key="detector.id"
            class="detector-card"
            @click="handleDetectorClick(detector)"
            @keydown="handleKeyDown($event, detector)"
            role="button"
            tabindex="0"
          >
            <div class="detector-header">
              <h3 class="detector-name">{{ detector.name }}</h3>
              <span :class="`detector-status ${detector.is_fitted ? 'trained' : 'untrained'}`">
                {{ detector.is_fitted ? 'Trained' : 'Untrained' }}
              </span>
            </div>
            
            <div class="detector-info">
              <div class="info-item">
                <label>Algorithm:</label>
                <span>{{ detector.algorithm_name }}</span>
              </div>
              <div class="info-item">
                <label>Contamination Rate:</label>
                <span>{{ (detector.contamination_rate * 100).toFixed(1) }}%</span>
              </div>
              <div v-if="detector.description" class="info-item">
                <label>Description:</label>
                <span class="description">{{ detector.description }}</span>
              </div>
            </div>

            <div v-if="detector.tags?.length" class="detector-tags">
              <span 
                v-for="tag in detector.tags" 
                :key="tag" 
                class="tag"
              >
                {{ tag }}
              </span>
            </div>

            <div class="detector-meta">
              <span class="created-date">
                Created: {{ formatDate(detector.created_at) }}
              </span>
            </div>
          </div>
        </div>

        <!-- Pagination -->
        <div v-if="total > pageSize" class="pagination">
          <button
            @click="previousPage"
            :disabled="!hasPrevious || isLoading"
            class="pagination-btn"
          >
            Previous
          </button>

          <div class="page-numbers">
            <button
              v-for="pageNum in pageNumbers"
              :key="pageNum"
              @click="goToPage(pageNum)"
              :disabled="isLoading"
              :class="`page-btn ${pageNum === page ? 'active' : ''}`"
            >
              {{ pageNum }}
            </button>
          </div>

          <button
            @click="nextPage"
            :disabled="!hasNext || isLoading"
            class="pagination-btn"
          >
            Next
          </button>
        </div>

        <!-- Page Info -->
        <div class="page-info">
          Showing {{ Math.min((page - 1) * pageSize + 1, total) }} to
          {{ Math.min(page * pageSize, total) }} of {{ total }} detectors
        </div>
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { PynomalyClient } from '../core/client';
import { DetectorListProps } from '../types';

interface Props extends DetectorListProps {
  client: PynomalyClient;
  pageSize?: number;
  className?: string;
}

const props = withDefaults(defineProps<Props>(), {
  pageSize: 20,
  className: ''
});

const emit = defineEmits<{
  'detector-select': [detector: any];
}>();

// Reactive state
const detectors = ref<any[]>([]);
const total = ref(0);
const page = ref(1);
const hasNext = ref(false);
const hasPrevious = ref(false);
const isLoading = ref(false);
const error = ref<any>(null);

// Computed properties
const pageNumbers = computed(() => {
  const totalPages = Math.ceil(total.value / props.pageSize);
  const current = page.value;
  const delta = 2; // Number of pages to show on each side of current page
  
  const pages: number[] = [];
  const start = Math.max(1, current - delta);
  const end = Math.min(totalPages, current + delta);
  
  for (let i = start; i <= end; i++) {
    pages.push(i);
  }
  
  return pages;
});

// Methods
const loadDetectors = async () => {
  if (!props.client) return;
  
  isLoading.value = true;
  error.value = null;
  
  try {
    const options = {
      page: page.value,
      pageSize: props.pageSize,
      algorithmName: props.filters?.algorithmName,
      tags: props.filters?.tags
    };
    
    const response = await props.client.dataScience.listDetectors(options);
    
    detectors.value = response.items;
    total.value = response.total;
    hasNext.value = response.hasNext;
    hasPrevious.value = response.hasPrevious;
  } catch (err) {
    error.value = err;
  } finally {
    isLoading.value = false;
  }
};

const handleDetectorClick = (detector: any) => {
  emit('detector-select', detector);
};

const handleKeyDown = (event: KeyboardEvent, detector: any) => {
  if (event.key === 'Enter' || event.key === ' ') {
    handleDetectorClick(detector);
  }
};

const nextPage = () => {
  if (hasNext.value) {
    page.value++;
  }
};

const previousPage = () => {
  if (hasPrevious.value) {
    page.value--;
  }
};

const goToPage = (pageNum: number) => {
  page.value = pageNum;
};

const refresh = () => {
  loadDetectors();
};

const clearError = () => {
  error.value = null;
};

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString();
};

// Watchers
watch([page, () => props.filters], () => {
  loadDetectors();
});

// Lifecycle
onMounted(() => {
  loadDetectors();
});
</script>

<style scoped>
/* Add your component styles here */
.pynomaly-detector-list {
  /* Component styles */
}

.error-container {
  padding: 20px;
  background-color: #fee;
  border: 1px solid #fcc;
  border-radius: 4px;
}

.loading-container {
  text-align: center;
  padding: 40px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.detector-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.detector-card {
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.detector-card:hover {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin: 20px 0;
}

.pagination-btn, .page-btn {
  padding: 8px 16px;
  border: 1px solid #ddd;
  background: white;
  cursor: pointer;
  border-radius: 4px;
}

.page-btn.active {
  background: #3498db;
  color: white;
  border-color: #3498db;
}
</style>