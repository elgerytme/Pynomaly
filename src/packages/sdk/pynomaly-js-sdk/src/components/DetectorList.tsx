/**
 * Detector List Component
 * 
 * React component for displaying and managing a list of anomaly detectors
 * with pagination, filtering, and selection capabilities.
 */

import React, { useMemo } from 'react';
import { DetectorListProps } from '../types';
import { PynomalyClient } from '../core/client';
import { useDetectorList } from '../hooks/useDetector';

/**
 * Detector list component with built-in pagination and filtering.
 * 
 * @param props Component properties
 * @returns JSX element
 */
export const DetectorList: React.FC<DetectorListProps & { client: PynomalyClient }> = ({
  client,
  onDetectorSelect,
  filters,
  pageSize = 20,
  className = ''
}) => {
  const {
    detectors,
    total,
    page,
    hasNext,
    hasPrevious,
    isLoading,
    error,
    nextPage,
    previousPage,
    goToPage,
    refresh,
    clearError
  } = useDetectorList(client, {
    algorithmName: filters?.algorithmName,
    tags: filters?.tags,
    pageSize,
    autoLoad: true
  });

  // Calculate page numbers for pagination
  const pageNumbers = useMemo(() => {
    const totalPages = Math.ceil(total / pageSize);
    const current = page;
    const delta = 2; // Number of pages to show on each side of current page
    
    const pages: number[] = [];
    const start = Math.max(1, current - delta);
    const end = Math.min(totalPages, current + delta);
    
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    
    return pages;
  }, [total, pageSize, page]);

  // Handle detector click
  const handleDetectorClick = (detector: any) => {
    if (onDetectorSelect) {
      onDetectorSelect(detector);
    }
  };

  // Error display
  if (error) {
    return (
      <div className={`pynomaly-detector-list ${className}`}>
        <div className="error-container">
          <div className="error-message">
            <h3>Error Loading Detectors</h3>
            <p>{error.message}</p>
            <div className="error-actions">
              <button onClick={clearError} className="btn-secondary">
                Dismiss
              </button>
              <button onClick={() => refresh()} className="btn-primary">
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`pynomaly-detector-list ${className}`}>
      {/* Header */}
      <div className="detector-list-header">
        <div className="header-info">
          <h2>Detectors</h2>
          <span className="total-count">{total} total</span>
        </div>
        <div className="header-actions">
          <button 
            onClick={() => refresh()} 
            disabled={isLoading}
            className="btn-secondary"
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Filters */}
      {filters && (
        <div className="detector-filters">
          {filters.algorithmName && (
            <span className="filter-tag">
              Algorithm: {filters.algorithmName}
            </span>
          )}
          {filters.tags?.map(tag => (
            <span key={tag} className="filter-tag">
              Tag: {tag}
            </span>
          ))}
        </div>
      )}

      {/* Loading State */}
      {isLoading && detectors.length === 0 && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading detectors...</p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && detectors.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">🔍</div>
          <h3>No Detectors Found</h3>
          <p>
            {filters ? 
              'No detectors match your current filters.' : 
              'No detectors have been created yet.'
            }
          </p>
        </div>
      )}

      {/* Detector Grid */}
      {detectors.length > 0 && (
        <>
          <div className="detector-grid">
            {detectors.map((detector) => (
              <div
                key={detector.id}
                className="detector-card"
                onClick={() => handleDetectorClick(detector)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    handleDetectorClick(detector);
                  }
                }}
              >
                <div className="detector-header">
                  <h3 className="detector-name">{detector.name}</h3>
                  <span className={`detector-status ${detector.is_fitted ? 'trained' : 'untrained'}`}>
                    {detector.is_fitted ? 'Trained' : 'Untrained'}
                  </span>
                </div>
                
                <div className="detector-info">
                  <div className="info-item">
                    <label>Algorithm:</label>
                    <span>{detector.algorithm_name}</span>
                  </div>
                  <div className="info-item">
                    <label>Contamination Rate:</label>
                    <span>{(detector.contamination_rate * 100).toFixed(1)}%</span>
                  </div>
                  {detector.description && (
                    <div className="info-item">
                      <label>Description:</label>
                      <span className="description">{detector.description}</span>
                    </div>
                  )}
                </div>

                {detector.tags && detector.tags.length > 0 && (
                  <div className="detector-tags">
                    {detector.tags.map((tag: string) => (
                      <span key={tag} className="tag">{tag}</span>
                    ))}
                  </div>
                )}

                <div className="detector-meta">
                  <span className="created-date">
                    Created: {new Date(detector.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {total > pageSize && (
            <div className="pagination">
              <button
                onClick={previousPage}
                disabled={!hasPrevious || isLoading}
                className="pagination-btn"
              >
                Previous
              </button>

              <div className="page-numbers">
                {pageNumbers.map((pageNum) => (
                  <button
                    key={pageNum}
                    onClick={() => goToPage(pageNum)}
                    disabled={isLoading}
                    className={`page-btn ${pageNum === page ? 'active' : ''}`}
                  >
                    {pageNum}
                  </button>
                ))}
              </div>

              <button
                onClick={nextPage}
                disabled={!hasNext || isLoading}
                className="pagination-btn"
              >
                Next
              </button>
            </div>
          )}

          {/* Page Info */}
          <div className="page-info">
            Showing {Math.min((page - 1) * pageSize + 1, total)} to{' '}
            {Math.min(page * pageSize, total)} of {total} detectors
          </div>
        </>
      )}
    </div>
  );
};

/**
 * Simplified detector list component for basic use cases.
 * 
 * @param props Component properties
 * @returns JSX element
 */
export const SimpleDetectorList: React.FC<{
  client: PynomalyClient;
  onSelect?: (detector: any) => void;
  className?: string;
}> = ({ client, onSelect, className = '' }) => {
  const { detectors, isLoading, error } = useDetectorList(client, {
    pageSize: 50,
    autoLoad: true
  });

  if (isLoading) {
    return (
      <div className={`pynomaly-simple-detector-list ${className}`}>
        <div className="loading">Loading detectors...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`pynomaly-simple-detector-list ${className}`}>
        <div className="error">Error: {error.message}</div>
      </div>
    );
  }

  return (
    <div className={`pynomaly-simple-detector-list ${className}`}>
      <select 
        onChange={(e) => {
          const detector = detectors.find(d => d.id === e.target.value);
          if (detector && onSelect) {
            onSelect(detector);
          }
        }}
        className="detector-select"
      >
        <option value="">Select a detector...</option>
        {detectors.map((detector) => (
          <option key={detector.id} value={detector.id}>
            {detector.name} ({detector.algorithm_name})
          </option>
        ))}
      </select>
    </div>
  );
};