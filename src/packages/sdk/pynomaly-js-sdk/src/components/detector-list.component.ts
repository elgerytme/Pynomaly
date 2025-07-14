/**
 * Detector List Component (Angular)
 * 
 * Angular component for displaying and managing a list of anomaly detectors
 * with pagination, filtering, and selection capabilities.
 */

import { Component, Input, Output, EventEmitter, OnInit, OnChanges, SimpleChanges } from '@angular/core';
import { PynomalyClient } from '../core/client';
import { DetectorListProps } from '../types';

interface Detector {
  id: string;
  name: string;
  algorithm_name: string;
  contamination_rate: number;
  is_fitted: boolean;
  description?: string;
  tags?: string[];
  created_at: string;
}

@Component({
  selector: 'pynomaly-detector-list',
  template: `
    <div [class]="'pynomaly-detector-list ' + className">
      <!-- Error State -->
      <div *ngIf="error" class="error-container">
        <div class="error-message">
          <h3>Error Loading Detectors</h3>
          <p>{{ error.message }}</p>
          <div class="error-actions">
            <button (click)="clearError()" class="btn-secondary">
              Dismiss
            </button>
            <button (click)="refresh()" class="btn-primary">
              Retry
            </button>
          </div>
        </div>
      </div>

      <ng-container *ngIf="!error">
        <!-- Header -->
        <div class="detector-list-header">
          <div class="header-info">
            <h2>Detectors</h2>
            <span class="total-count">{{ total }} total</span>
          </div>
          <div class="header-actions">
            <button 
              (click)="refresh()" 
              [disabled]="isLoading"
              class="btn-secondary"
            >
              {{ isLoading ? 'Loading...' : 'Refresh' }}
            </button>
          </div>
        </div>

        <!-- Filters -->
        <div *ngIf="filters" class="detector-filters">
          <span *ngIf="filters.algorithmName" class="filter-tag">
            Algorithm: {{ filters.algorithmName }}
          </span>
          <span 
            *ngFor="let tag of filters.tags" 
            class="filter-tag"
          >
            Tag: {{ tag }}
          </span>
        </div>

        <!-- Loading State -->
        <div *ngIf="isLoading && detectors.length === 0" class="loading-container">
          <div class="loading-spinner"></div>
          <p>Loading detectors...</p>
        </div>

        <!-- Empty State -->
        <div *ngIf="!isLoading && detectors.length === 0" class="empty-state">
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
        <ng-container *ngIf="detectors.length > 0">
          <div class="detector-grid">
            <div
              *ngFor="let detector of detectors"
              class="detector-card"
              (click)="handleDetectorClick(detector)"
              (keydown)="handleKeyDown($event, detector)"
              role="button"
              tabindex="0"
            >
              <div class="detector-header">
                <h3 class="detector-name">{{ detector.name }}</h3>
                <span [class]="'detector-status ' + (detector.is_fitted ? 'trained' : 'untrained')">
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
                <div *ngIf="detector.description" class="info-item">
                  <label>Description:</label>
                  <span class="description">{{ detector.description }}</span>
                </div>
              </div>

              <div *ngIf="detector.tags?.length" class="detector-tags">
                <span 
                  *ngFor="let tag of detector.tags" 
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
          <div *ngIf="total > pageSize" class="pagination">
            <button
              (click)="previousPage()"
              [disabled]="!hasPrevious || isLoading"
              class="pagination-btn"
            >
              Previous
            </button>

            <div class="page-numbers">
              <button
                *ngFor="let pageNum of pageNumbers"
                (click)="goToPage(pageNum)"
                [disabled]="isLoading"
                [class]="'page-btn ' + (pageNum === page ? 'active' : '')"
              >
                {{ pageNum }}
              </button>
            </div>

            <button
              (click)="nextPage()"
              [disabled]="!hasNext || isLoading"
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
        </ng-container>
      </ng-container>
    </div>
  `,
  styles: [`
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

    .detector-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 15px;
    }

    .detector-name {
      margin: 0;
      color: #2c3e50;
    }

    .detector-status {
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.8em;
      font-weight: bold;
    }

    .detector-status.trained {
      background: #d4edda;
      color: #155724;
    }

    .detector-status.untrained {
      background: #f8d7da;
      color: #721c24;
    }

    .info-item {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
    }

    .info-item label {
      font-weight: bold;
      color: #666;
    }

    .detector-tags {
      margin: 15px 0;
    }

    .tag {
      display: inline-block;
      padding: 2px 6px;
      background: #e9ecef;
      border-radius: 3px;
      font-size: 0.8em;
      margin-right: 5px;
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

    .page-info {
      text-align: center;
      color: #666;
      margin-top: 10px;
    }
  `]
})
export class DetectorListComponent implements OnInit, OnChanges {
  @Input() client!: PynomalyClient;
  @Input() filters?: DetectorListProps['filters'];
  @Input() pageSize: number = 20;
  @Input() className: string = '';

  @Output() detectorSelect = new EventEmitter<any>();

  // Component state
  detectors: Detector[] = [];
  total: number = 0;
  page: number = 1;
  hasNext: boolean = false;
  hasPrevious: boolean = false;
  isLoading: boolean = false;
  error: any = null;

  ngOnInit() {
    this.loadDetectors();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['filters'] && !changes['filters'].isFirstChange()) {
      this.page = 1;
      this.loadDetectors();
    }
  }

  get pageNumbers(): number[] {
    const totalPages = Math.ceil(this.total / this.pageSize);
    const current = this.page;
    const delta = 2; // Number of pages to show on each side of current page
    
    const pages: number[] = [];
    const start = Math.max(1, current - delta);
    const end = Math.min(totalPages, current + delta);
    
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    
    return pages;
  }

  async loadDetectors() {
    if (!this.client) return;
    
    this.isLoading = true;
    this.error = null;
    
    try {
      const options = {
        page: this.page,
        pageSize: this.pageSize,
        algorithmName: this.filters?.algorithmName,
        tags: this.filters?.tags
      };
      
      const response = await this.client.dataScience.listDetectors(options);
      
      this.detectors = response.items;
      this.total = response.total;
      this.hasNext = response.hasNext;
      this.hasPrevious = response.hasPrevious;
    } catch (err) {
      this.error = err;
    } finally {
      this.isLoading = false;
    }
  }

  handleDetectorClick(detector: Detector) {
    this.detectorSelect.emit(detector);
  }

  handleKeyDown(event: KeyboardEvent, detector: Detector) {
    if (event.key === 'Enter' || event.key === ' ') {
      this.handleDetectorClick(detector);
    }
  }

  nextPage() {
    if (this.hasNext) {
      this.page++;
      this.loadDetectors();
    }
  }

  previousPage() {
    if (this.hasPrevious) {
      this.page--;
      this.loadDetectors();
    }
  }

  goToPage(pageNum: number) {
    this.page = pageNum;
    this.loadDetectors();
  }

  refresh() {
    this.loadDetectors();
  }

  clearError() {
    this.error = null;
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString();
  }

  // Utility method for Angular's Math functions in template
  Math = Math;
}