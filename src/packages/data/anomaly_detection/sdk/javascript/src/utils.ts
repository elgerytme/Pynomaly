/**
 * Utility functions for the Anomaly Detection SDK
 */

import { AlgorithmType, ValidationError } from './types';

/**
 * Validate that data is in the correct format for anomaly detection
 */
export function validateDataFormat(data: any): data is number[][] {
  if (!Array.isArray(data)) {
    throw new ValidationError('Data must be an array', 'data', data);
  }

  if (data.length === 0) {
    throw new ValidationError('Data cannot be empty', 'data', data);
  }

  // Check if all elements are arrays
  if (!data.every(point => Array.isArray(point))) {
    throw new ValidationError('All data points must be arrays', 'data', data);
  }

  // Check if all data points have the same length
  const firstPointLength = data[0].length;
  if (!data.every(point => point.length === firstPointLength)) {
    throw new ValidationError('All data points must have the same number of features', 'data', data);
  }

  // Check if all values are numbers
  for (let i = 0; i < data.length; i++) {
    const point = data[i];
    for (let j = 0; j < point.length; j++) {
      if (typeof point[j] !== 'number' || !isFinite(point[j])) {
        throw new ValidationError(
          `Invalid value at data[${i}][${j}]: expected finite number, got ${typeof point[j]}`,
          'data',
          point[j]
        );
      }
    }
  }

  return true;
}

/**
 * Normalize data to zero mean and unit variance
 */
export function normalizeData(data: number[][]): {
  normalizedData: number[][];
  means: number[];
  stds: number[];
} {
  validateDataFormat(data);

  const numFeatures = data[0].length;
  const numSamples = data.length;

  // Calculate means
  const means = new Array(numFeatures).fill(0);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      means[j] += data[i][j];
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    means[j] /= numSamples;
  }

  // Calculate standard deviations
  const stds = new Array(numFeatures).fill(0);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      const diff = data[i][j] - means[j];
      stds[j] += diff * diff;
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    stds[j] = Math.sqrt(stds[j] / numSamples);
    // Avoid division by zero
    if (stds[j] === 0) {
      stds[j] = 1;
    }
  }

  // Normalize data
  const normalizedData = data.map(point =>
    point.map((value, j) => (value - means[j]) / stds[j])
  );

  return { normalizedData, means, stds };
}

/**
 * Apply normalization parameters to new data
 */
export function applyNormalization(
  data: number[][],
  means: number[],
  stds: number[]
): number[][] {
  validateDataFormat(data);

  if (data[0].length !== means.length || data[0].length !== stds.length) {
    throw new ValidationError(
      'Data dimensions must match normalization parameters',
      'data',
      data
    );
  }

  return data.map(point =>
    point.map((value, j) => (value - means[j]) / stds[j])
  );
}

/**
 * Split data into training and validation sets
 */
export function trainValidationSplit(
  data: number[][],
  validationRatio = 0.2,
  shuffle = true
): {
  trainData: number[][];
  validationData: number[][];
} {
  validateDataFormat(data);

  if (validationRatio < 0 || validationRatio >= 1) {
    throw new ValidationError(
      'Validation ratio must be between 0 and 1',
      'validationRatio',
      validationRatio
    );
  }

  let workingData = [...data];

  // Shuffle if requested
  if (shuffle) {
    for (let i = workingData.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [workingData[i], workingData[j]] = [workingData[j], workingData[i]];
    }
  }

  const splitIndex = Math.floor(data.length * (1 - validationRatio));
  
  return {
    trainData: workingData.slice(0, splitIndex),
    validationData: workingData.slice(splitIndex),
  };
}

/**
 * Calculate basic statistics for data
 */
export function calculateDataStatistics(data: number[][]): {
  numSamples: number;
  numFeatures: number;
  means: number[];
  stds: number[];
  mins: number[];
  maxs: number[];
} {
  validateDataFormat(data);

  const numSamples = data.length;
  const numFeatures = data[0].length;

  // Initialize arrays
  const means = new Array(numFeatures).fill(0);
  const mins = new Array(numFeatures).fill(Infinity);
  const maxs = new Array(numFeatures).fill(-Infinity);

  // Calculate means, mins, and maxs
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      const value = data[i][j];
      means[j] += value;
      mins[j] = Math.min(mins[j], value);
      maxs[j] = Math.max(maxs[j], value);
    }
  }

  // Finalize means
  for (let j = 0; j < numFeatures; j++) {
    means[j] /= numSamples;
  }

  // Calculate standard deviations
  const stds = new Array(numFeatures).fill(0);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      const diff = data[i][j] - means[j];
      stds[j] += diff * diff;
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    stds[j] = Math.sqrt(stds[j] / numSamples);
  }

  return {
    numSamples,
    numFeatures,
    means,
    stds,
    mins,
    maxs,
  };
}

/**
 * Convert CSV string to data array
 */
export function parseCSV(csvString: string, hasHeader = true): number[][] {
  const lines = csvString.trim().split('\n');
  
  if (lines.length === 0) {
    throw new ValidationError('CSV data is empty', 'csvString', csvString);
  }

  const startIndex = hasHeader ? 1 : 0;
  const data: number[][] = [];

  for (let i = startIndex; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === '') continue;

    const values = line.split(',').map(val => {
      const num = parseFloat(val.trim());
      if (isNaN(num)) {
        throw new ValidationError(
          `Invalid number in CSV at line ${i + 1}: "${val}"`,
          'csvString',
          val
        );
      }
      return num;
    });

    data.push(values);
  }

  validateDataFormat(data);
  return data;
}

/**
 * Convert data array to CSV string
 */
export function toCSV(data: number[][], headers?: string[]): string {
  validateDataFormat(data);

  let csv = '';

  // Add headers if provided
  if (headers) {
    if (headers.length !== data[0].length) {
      throw new ValidationError(
        'Number of headers must match number of features',
        'headers',
        headers
      );
    }
    csv += headers.join(',') + '\n';
  }

  // Add data rows
  for (const row of data) {
    csv += row.join(',') + '\n';
  }

  return csv;
}

/**
 * Generate sample data for testing
 */
export function generateSampleData(
  numSamples: number,
  numFeatures: number,
  anomalyRatio = 0.1
): {
  data: number[][];
  labels: boolean[]; // true for anomalies
} {
  if (numSamples <= 0 || numFeatures <= 0) {
    throw new ValidationError('Number of samples and features must be positive');
  }

  if (anomalyRatio < 0 || anomalyRatio > 1) {
    throw new ValidationError('Anomaly ratio must be between 0 and 1');
  }

  const data: number[][] = [];
  const labels: boolean[] = [];
  const numAnomalies = Math.floor(numSamples * anomalyRatio);

  // Generate normal points (mean=0, std=1)
  for (let i = 0; i < numSamples - numAnomalies; i++) {
    const point: number[] = [];
    for (let j = 0; j < numFeatures; j++) {
      point.push(normalRandom(0, 1));
    }
    data.push(point);
    labels.push(false);
  }

  // Generate anomalous points (mean=3, std=1)
  for (let i = 0; i < numAnomalies; i++) {
    const point: number[] = [];
    for (let j = 0; j < numFeatures; j++) {
      point.push(normalRandom(3, 1));
    }
    data.push(point);
    labels.push(true);
  }

  // Shuffle the data
  for (let i = data.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [data[i], data[j]] = [data[j], data[i]];
    [labels[i], labels[j]] = [labels[j], labels[i]];
  }

  return { data, labels };
}

/**
 * Generate normally distributed random number using Box-Muller transform
 */
function normalRandom(mean = 0, std = 1): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
  while (v === 0) v = Math.random();
  
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return z * std + mean;
}

/**
 * Check if algorithm type is valid
 */
export function isValidAlgorithmType(algorithm: string): algorithm is AlgorithmType {
  return Object.values(AlgorithmType).includes(algorithm as AlgorithmType);
}

/**
 * Get all available algorithm types
 */
export function getAvailableAlgorithms(): AlgorithmType[] {
  return Object.values(AlgorithmType);
}

/**
 * Format execution time for display
 */
export function formatExecutionTime(seconds: number): string {
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)}ms`;
  } else if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  } else {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(2)}s`;
  }
}