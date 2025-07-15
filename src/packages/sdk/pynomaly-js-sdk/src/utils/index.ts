/**
 * Pynomaly JavaScript SDK Utilities
 * 
 * Helper functions and utilities for data processing, validation,
 * and common operations.
 */

import { Dataset, DetectionResult, ValidationRule } from '../types';

/**
 * Convert various data formats to Dataset format.
 */
export class DatasetConverter {
  /**
   * Convert CSV string to Dataset.
   * 
   * @param name Dataset name
   * @param csvString CSV data as string
   * @param options Conversion options
   * @returns Dataset object
   */
  static fromCSV(
    name: string,
    csvString: string,
    options: {
      delimiter?: string;
      hasHeader?: boolean;
      targetColumn?: string;
    } = {}
  ): Dataset {
    const { delimiter = ',', hasHeader = true, targetColumn } = options;
    
    const lines = csvString.trim().split('\n');
    const headers = hasHeader ? lines[0].split(delimiter) : undefined;
    const dataLines = hasHeader ? lines.slice(1) : lines;
    
    const data = dataLines.map(line => {
      const values = line.split(delimiter).map(val => {
        const num = parseFloat(val.trim());
        return isNaN(num) ? val.trim() : num;
      });
      
      if (headers) {
        const obj: Record<string, any> = {};
        headers.forEach((header, index) => {
          obj[header.trim()] = values[index];
        });
        return obj;
      }
      
      return values;
    });

    return {
      name,
      data,
      featureNames: headers?.map(h => h.trim()),
      targetColumn,
      metadata: {
        source: 'csv',
        rows: data.length,
        columns: headers?.length || (data[0] as any[])?.length || 0
      }
    };
  }

  /**
   * Convert JSON array to Dataset.
   * 
   * @param name Dataset name
   * @param jsonData JSON array data
   * @param targetColumn Optional target column name
   * @returns Dataset object
   */
  static fromJSON(
    name: string,
    jsonData: any[],
    targetColumn?: string
  ): Dataset {
    if (!Array.isArray(jsonData)) {
      throw new Error('JSON data must be an array');
    }

    const data = jsonData;
    const featureNames = data.length > 0 && typeof data[0] === 'object' 
      ? Object.keys(data[0]) 
      : undefined;

    return {
      name,
      data,
      featureNames,
      targetColumn,
      metadata: {
        source: 'json',
        rows: data.length,
        columns: featureNames?.length || 0
      }
    };
  }

  /**
   * Convert 2D array to Dataset.
   * 
   * @param name Dataset name
   * @param arrayData 2D array data
   * @param featureNames Optional feature names
   * @param targetColumn Optional target column name
   * @returns Dataset object
   */
  static fromArray(
    name: string,
    arrayData: any[][],
    featureNames?: string[],
    targetColumn?: string
  ): Dataset {
    if (!Array.isArray(arrayData) || !Array.isArray(arrayData[0])) {
      throw new Error('Array data must be a 2D array');
    }

    return {
      name,
      data: arrayData,
      featureNames,
      targetColumn,
      metadata: {
        source: 'array',
        rows: arrayData.length,
        columns: arrayData[0]?.length || 0
      }
    };
  }

  /**
   * Convert File object to Dataset.
   * 
   * @param file File object
   * @param options Conversion options
   * @returns Promise resolving to Dataset object
   */
  static async fromFile(
    file: File,
    options: {
      delimiter?: string;
      hasHeader?: boolean;
      targetColumn?: string;
    } = {}
  ): Promise<Dataset> {
    const text = await file.text();
    const name = file.name.replace(/\.[^/.]+$/, ''); // Remove extension
    
    if (file.type === 'application/json' || file.name.endsWith('.json')) {
      const jsonData = JSON.parse(text);
      return this.fromJSON(name, jsonData, options.targetColumn);
    } else {
      // Assume CSV
      return this.fromCSV(name, text, options);
    }
  }
}

/**
 * Data validation utilities.
 */
export class DataValidator {
  private static rules: ValidationRule[] = [
    {
      name: 'non-empty',
      validate: (data: any) => Array.isArray(data) && data.length > 0,
      message: 'Dataset cannot be empty'
    },
    {
      name: 'consistent-columns',
      validate: (data: any[]) => {
        if (!Array.isArray(data) || data.length === 0) return true;
        
        const firstRowLength = Array.isArray(data[0]) 
          ? data[0].length 
          : Object.keys(data[0]).length;
        
        return data.every(row => {
          const rowLength = Array.isArray(row) 
            ? row.length 
            : Object.keys(row).length;
          return rowLength === firstRowLength;
        });
      },
      message: 'All rows must have the same number of columns'
    },
    {
      name: 'numeric-data',
      validate: (data: any[]) => {
        if (!Array.isArray(data) || data.length === 0) return true;
        
        return data.every(row => {
          const values = Array.isArray(row) ? row : Object.values(row);
          return values.every(val => typeof val === 'number' && !isNaN(val));
        });
      },
      message: 'All data values must be numeric'
    }
  ];

  /**
   * Validate dataset against common rules.
   * 
   * @param dataset Dataset to validate
   * @param ruleNames Optional specific rules to check
   * @returns Validation result
   */
  static validate(
    dataset: Dataset,
    ruleNames?: string[]
  ): { isValid: boolean; errors: string[] } {
    const rulesToCheck = ruleNames 
      ? this.rules.filter(rule => ruleNames.includes(rule.name))
      : this.rules;

    const errors: string[] = [];

    for (const rule of rulesToCheck) {
      if (!rule.validate(dataset.data)) {
        errors.push(rule.message);
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Add custom validation rule.
   * 
   * @param rule Validation rule to add
   */
  static addRule(rule: ValidationRule): void {
    this.rules.push(rule);
  }

  /**
   * Check if dataset has numeric data only.
   * 
   * @param dataset Dataset to check
   * @returns True if all data is numeric
   */
  static isNumericData(dataset: Dataset): boolean {
    return this.validate(dataset, ['numeric-data']).isValid;
  }

  /**
   * Get dataset statistics.
   * 
   * @param dataset Dataset to analyze
   * @returns Dataset statistics
   */
  static getStatistics(dataset: Dataset): {
    rows: number;
    columns: number;
    numericColumns: number;
    missingValues: number;
    dataTypes: Record<string, string>;
  } {
    const data = dataset.data;
    const rows = data.length;
    
    if (rows === 0) {
      return {
        rows: 0,
        columns: 0,
        numericColumns: 0,
        missingValues: 0,
        dataTypes: {}
      };
    }

    const firstRow = data[0];
    const columns = Array.isArray(firstRow) 
      ? firstRow.length 
      : Object.keys(firstRow).length;

    const isArrayData = Array.isArray(firstRow);
    const columnNames = isArrayData 
      ? dataset.featureNames || Array.from({ length: columns }, (_, i) => `col_${i}`)
      : Object.keys(firstRow);

    const dataTypes: Record<string, string> = {};
    let numericColumns = 0;
    let missingValues = 0;

    columnNames.forEach((colName, colIndex) => {
      const values = data.map(row => 
        isArrayData ? (row as any[])[colIndex] : (row as any)[colName]
      );

      // Count missing values
      const missing = values.filter(val => val == null || val === '').length;
      missingValues += missing;

      // Determine data type
      const nonMissingValues = values.filter(val => val != null && val !== '');
      const numericValues = nonMissingValues.filter(val => 
        typeof val === 'number' || (!isNaN(parseFloat(val)) && isFinite(val))
      );

      if (numericValues.length === nonMissingValues.length) {
        dataTypes[colName] = 'numeric';
        numericColumns++;
      } else {
        dataTypes[colName] = 'string';
      }
    });

    return {
      rows,
      columns,
      numericColumns,
      missingValues,
      dataTypes
    };
  }
}

/**
 * Detection result analysis utilities.
 */
export class ResultAnalyzer {
  /**
   * Calculate basic statistics for detection results.
   * 
   * @param result Detection result
   * @returns Analysis statistics
   */
  static analyze(result: DetectionResult): {
    anomalyRate: number;
    scoreStats: {
      mean: number;
      median: number;
      std: number;
      min: number;
      max: number;
      q1: number;
      q3: number;
    };
    anomalyStats: {
      meanScore: number;
      minScore: number;
      maxScore: number;
    };
    normalStats: {
      meanScore: number;
      minScore: number;
      maxScore: number;
    };
  } {
    const scores = result.anomalyScores;
    const labels = result.anomalyLabels;
    
    // Overall stats
    const anomalyRate = (result.nAnomalies / result.nSamples) * 100;
    
    // Score statistics
    const sortedScores = [...scores].sort((a, b) => a - b);
    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const median = sortedScores[Math.floor(sortedScores.length / 2)];
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
    const std = Math.sqrt(variance);
    const q1 = sortedScores[Math.floor(sortedScores.length * 0.25)];
    const q3 = sortedScores[Math.floor(sortedScores.length * 0.75)];
    
    // Anomaly and normal score statistics
    const anomalyScores = scores.filter((_, i) => labels[i] === 1);
    const normalScores = scores.filter((_, i) => labels[i] === 0);
    
    const anomalyStats = {
      meanScore: anomalyScores.length > 0 
        ? anomalyScores.reduce((sum, score) => sum + score, 0) / anomalyScores.length 
        : 0,
      minScore: anomalyScores.length > 0 ? Math.min(...anomalyScores) : 0,
      maxScore: anomalyScores.length > 0 ? Math.max(...anomalyScores) : 0
    };
    
    const normalStats = {
      meanScore: normalScores.length > 0 
        ? normalScores.reduce((sum, score) => sum + score, 0) / normalScores.length 
        : 0,
      minScore: normalScores.length > 0 ? Math.min(...normalScores) : 0,
      maxScore: normalScores.length > 0 ? Math.max(...normalScores) : 0
    };

    return {
      anomalyRate,
      scoreStats: {
        mean,
        median,
        std,
        min: Math.min(...scores),
        max: Math.max(...scores),
        q1,
        q3
      },
      anomalyStats,
      normalStats
    };
  }

  /**
   * Get anomaly indices sorted by score.
   * 
   * @param result Detection result
   * @param descending Sort in descending order (highest scores first)
   * @returns Array of anomaly indices with scores
   */
  static getAnomaliesByScore(
    result: DetectionResult,
    descending: boolean = true
  ): Array<{ index: number; score: number }> {
    const anomalies = result.anomalyLabels
      .map((label, index) => ({ index, label, score: result.anomalyScores[index] }))
      .filter(item => item.label === 1)
      .map(item => ({ index: item.index, score: item.score }));

    return anomalies.sort((a, b) => 
      descending ? b.score - a.score : a.score - b.score
    );
  }

  /**
   * Compare two detection results.
   * 
   * @param result1 First detection result
   * @param result2 Second detection result
   * @returns Comparison metrics
   */
  static compare(
    result1: DetectionResult,
    result2: DetectionResult
  ): {
    anomalyDifference: number;
    averageScoreDifference: number;
    thresholdDifference: number;
    executionTimeDifference: number;
    agreement: number; // Percentage of labels that agree
  } {
    if (result1.nSamples !== result2.nSamples) {
      throw new Error('Cannot compare results with different sample counts');
    }

    const anomalyDifference = result2.nAnomalies - result1.nAnomalies;
    
    const avg1 = result1.anomalyScores.reduce((sum, score) => sum + score, 0) / result1.anomalyScores.length;
    const avg2 = result2.anomalyScores.reduce((sum, score) => sum + score, 0) / result2.anomalyScores.length;
    const averageScoreDifference = avg2 - avg1;
    
    const thresholdDifference = result2.threshold - result1.threshold;
    const executionTimeDifference = result2.executionTime - result1.executionTime;
    
    // Calculate agreement
    const agreements = result1.anomalyLabels.filter((label, i) => label === result2.anomalyLabels[i]).length;
    const agreement = (agreements / result1.nSamples) * 100;

    return {
      anomalyDifference,
      averageScoreDifference,
      thresholdDifference,
      executionTimeDifference,
      agreement
    };
  }
}

/**
 * General utility functions.
 */
export class Utils {
  /**
   * Debounce function calls.
   * 
   * @param func Function to debounce
   * @param delay Delay in milliseconds
   * @returns Debounced function
   */
  static debounce<T extends (...args: any[]) => any>(
    func: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let timeoutId: NodeJS.Timeout;
    
    return (...args: Parameters<T>) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func(...args), delay);
    };
  }

  /**
   * Throttle function calls.
   * 
   * @param func Function to throttle
   * @param delay Delay in milliseconds
   * @returns Throttled function
   */
  static throttle<T extends (...args: any[]) => any>(
    func: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let lastCall = 0;
    
    return (...args: Parameters<T>) => {
      const now = Date.now();
      if (now - lastCall >= delay) {
        lastCall = now;
        func(...args);
      }
    };
  }

  /**
   * Format file size in human-readable format.
   * 
   * @param bytes File size in bytes
   * @returns Formatted file size string
   */
  static formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Generate a random ID.
   * 
   * @param length ID length
   * @returns Random ID string
   */
  static generateId(length: number = 8): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  }

  /**
   * Deep clone an object.
   * 
   * @param obj Object to clone
   * @returns Cloned object
   */
  static deepClone<T>(obj: T): T {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj.getTime()) as unknown as T;
    if (obj instanceof Array) return obj.map(item => this.deepClone(item)) as unknown as T;
    if (typeof obj === 'object') {
      const cloned = {} as T;
      Object.keys(obj).forEach(key => {
        (cloned as any)[key] = this.deepClone((obj as any)[key]);
      });
      return cloned;
    }
    return obj;
  }
}