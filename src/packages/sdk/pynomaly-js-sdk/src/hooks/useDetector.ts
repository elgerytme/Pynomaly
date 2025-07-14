/**
 * React Hook for Detector Management
 * 
 * Custom React hook for managing detector operations including
 * creating, fetching, updating, and deleting detectors.
 */

import { useState, useEffect, useCallback } from 'react';
import { UseDetectorState, DetectorConfig, PynomalyError } from '../types';
import { PynomalyClient } from '../core/client';

/**
 * Hook for managing a single detector.
 * 
 * @param client Pynomaly client instance
 * @param detectorId Optional detector ID to load
 * @returns Detector state and management functions
 */
export function useDetector(client: PynomalyClient, detectorId?: string) {
  const [state, setState] = useState<UseDetectorState>({
    detector: null,
    isLoading: false,
    error: null
  });

  // Load detector by ID
  const loadDetector = useCallback(async (id: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const detector = await client.dataScience.getDetector(id);
      setState(prev => ({ ...prev, detector, isLoading: false }));
      return detector;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isLoading: false }));
      throw error;
    }
  }, [client]);

  // Create new detector
  const createDetector = useCallback(async (
    name: string,
    config: DetectorConfig,
    description?: string,
    tags?: string[]
  ) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const detector = await client.dataScience.createDetector(name, config, description, tags);
      setState(prev => ({ ...prev, detector, isLoading: false }));
      return detector;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isLoading: false }));
      throw error;
    }
  }, [client]);

  // Update detector
  const updateDetector = useCallback(async (
    id: string,
    updates: {
      name?: string;
      config?: DetectorConfig;
      description?: string;
      tags?: string[];
    }
  ) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const detector = await client.dataScience.updateDetector(id, updates);
      setState(prev => ({ ...prev, detector, isLoading: false }));
      return detector;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isLoading: false }));
      throw error;
    }
  }, [client]);

  // Delete detector
  const deleteDetector = useCallback(async (id: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await client.dataScience.deleteDetector(id);
      setState(prev => ({ ...prev, detector: null, isLoading: false }));
      return true;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isLoading: false }));
      throw error;
    }
  }, [client]);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Reset state
  const reset = useCallback(() => {
    setState({
      detector: null,
      isLoading: false,
      error: null
    });
  }, []);

  // Load detector on mount if ID provided
  useEffect(() => {
    if (detectorId) {
      loadDetector(detectorId);
    }
  }, [detectorId, loadDetector]);

  return {
    ...state,
    loadDetector,
    createDetector,
    updateDetector,
    deleteDetector,
    clearError,
    reset
  };
}

/**
 * Hook for managing multiple detectors with pagination.
 * 
 * @param client Pynomaly client instance
 * @param options List options
 * @returns Detector list state and management functions
 */
export function useDetectorList(
  client: PynomalyClient,
  options: {
    algorithmName?: string;
    tags?: string[];
    pageSize?: number;
    autoLoad?: boolean;
  } = {}
) {
  const [state, setState] = useState({
    detectors: [] as any[],
    total: 0,
    page: 1,
    pageSize: options.pageSize || 20,
    hasNext: false,
    hasPrevious: false,
    isLoading: false,
    error: null as PynomalyError | null
  });

  // Load detectors
  const loadDetectors = useCallback(async (page: number = 1) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const result = await client.dataScience.listDetectors({
        page,
        pageSize: state.pageSize,
        algorithmName: options.algorithmName,
        tags: options.tags
      });
      
      setState(prev => ({
        ...prev,
        detectors: result.items,
        total: result.total,
        page: result.page,
        hasNext: result.hasNext,
        hasPrevious: result.hasPrevious,
        isLoading: false
      }));
      
      return result;
    } catch (error) {
      const pynomalyError = error as PynomalyError;
      setState(prev => ({ ...prev, error: pynomalyError, isLoading: false }));
      throw error;
    }
  }, [client, state.pageSize, options.algorithmName, options.tags]);

  // Navigate to next page
  const nextPage = useCallback(async () => {
    if (state.hasNext) {
      await loadDetectors(state.page + 1);
    }
  }, [loadDetectors, state.hasNext, state.page]);

  // Navigate to previous page
  const previousPage = useCallback(async () => {
    if (state.hasPrevious) {
      await loadDetectors(state.page - 1);
    }
  }, [loadDetectors, state.hasPrevious, state.page]);

  // Go to specific page
  const goToPage = useCallback(async (page: number) => {
    await loadDetectors(page);
  }, [loadDetectors]);

  // Refresh current page
  const refresh = useCallback(async () => {
    await loadDetectors(state.page);
  }, [loadDetectors, state.page]);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Load detectors on mount if autoLoad is true
  useEffect(() => {
    if (options.autoLoad !== false) {
      loadDetectors();
    }
  }, [loadDetectors, options.autoLoad]);

  return {
    ...state,
    loadDetectors,
    nextPage,
    previousPage,
    goToPage,
    refresh,
    clearError
  };
}