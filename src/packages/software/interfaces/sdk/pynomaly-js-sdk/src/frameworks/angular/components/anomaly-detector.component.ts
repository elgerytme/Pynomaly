/**
 * Angular component for anomaly detection
 */

import { 
  Component, 
  Input, 
  Output, 
  EventEmitter, 
  OnInit, 
  OnDestroy 
} from '@angular/core';
import { FormControl, Validators } from '@angular/forms';
import { Subject, takeUntil } from 'rxjs';
import { 
  AnomalyDetectionRequest, 
  AnomalyDetectionResult 
} from '../../../index';
import { PynomalyService } from '../services/pynomaly.service';

@Component({
  selector: 'pynomaly-anomaly-detector',
  template: `
    <div class="pynomaly-anomaly-detector">
      <div class="input-section">
        <h3>Anomaly Detection</h3>
        
        <div class="algorithm-selection">
          <label>
            Algorithm:
            <select [formControl]="algorithmControl">
              <option value="auto">Auto</option>
              <option value="isolation_forest">Isolation Forest</option>
              <option value="local_outlier_factor">Local Outlier Factor</option>
              <option value="one_class_svm">One-Class SVM</option>
            </select>
          </label>
        </div>

        <div class="data-input">
          <label>
            Data (JSON array of arrays):
            <textarea
              [formControl]="dataControl"
              placeholder='[[1, 2, 3], [4, 5, 6], [100, 200, 300]]'
              rows="4"
              style="width: 100%; font-family: monospace;"
            ></textarea>
          </label>
          <div *ngIf="dataControl.errors?.['invalidJson']" class="error">
            Invalid JSON format
          </div>
        </div>

        <div class="actions">
          <button 
            (click)="handleAnalyzeInput()"
            [disabled]="isLoading || !isReady || dataControl.invalid"
          >
            {{ isLoading ? 'Detecting...' : 'Detect Anomalies' }}
          </button>
          
          <button 
            (click)="clear()"
            [disabled]="isLoading"
          >
            Clear
          </button>
        </div>

        <div *ngIf="isLoading" class="progress">
          <div class="progress-bar">
            <div 
              class="progress-fill"
              [style.width.%]="progress"
            ></div>
          </div>
          <span>Progress: {{ progress }}%</span>
        </div>
      </div>

      <div *ngIf="error" class="error" style="color: red; margin-top: 10px;">
        Error: {{ error }}
      </div>

      <div *ngIf="result" class="result" style="margin-top: 20px;">
        <h4>Detection Results</h4>
        <div class="result-summary">
          <p><strong>Algorithm:</strong> {{ result.algorithm }}</p>
          <p><strong>Total Points:</strong> {{ result.metrics.totalPoints }}</p>
          <p><strong>Anomalies Found:</strong> {{ result.metrics.anomalyCount }}</p>
          <p><strong>Anomaly Rate:</strong> {{ (result.metrics.anomalyRate * 100).toFixed(2) }}%</p>
          <p><strong>Processing Time:</strong> {{ result.processingTime }}ms</p>
        </div>

        <div *ngIf="result.anomalies.length > 0" class="anomalies">
          <h5>Detected Anomalies:</h5>
          <table style="width: 100%; border-collapse: collapse;">
            <thead>
              <tr>
                <th style="border: 1px solid #ccc; padding: 8px;">Index</th>
                <th style="border: 1px solid #ccc; padding: 8px;">Score</th>
                <th style="border: 1px solid #ccc; padding: 8px;">Confidence</th>
                <th style="border: 1px solid #ccc; padding: 8px;">Data</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let anomaly of result.anomalies">
                <td style="border: 1px solid #ccc; padding: 8px;">{{ anomaly.index }}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">{{ anomaly.score.toFixed(4) }}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">{{ (anomaly.confidence * 100).toFixed(1) }}%</td>
                <td style="border: 1px solid #ccc; padding: 8px;">
                  {{ formatJson(anomaly.data) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .pynomaly-anomaly-detector {
      max-width: 800px;
      margin: 0 auto;
    }

    .input-section {
      margin-bottom: 20px;
    }

    .algorithm-selection,
    .data-input {
      margin-bottom: 15px;
    }

    .algorithm-selection label,
    .data-input label {
      display: block;
      margin-bottom: 5px;
      font-weight: bold;
    }

    .algorithm-selection select {
      padding: 5px;
      border: 1px solid #ccc;
      border-radius: 4px;
    }

    .actions {
      margin-bottom: 15px;
    }

    .actions button {
      padding: 10px 15px;
      margin-right: 10px;
      border: none;
      border-radius: 4px;
      background-color: #007bff;
      color: white;
      cursor: pointer;
    }

    .actions button:disabled {
      background-color: #6c757d;
      cursor: not-allowed;
    }

    .actions button:not(:disabled):hover {
      background-color: #0056b3;
    }

    .progress {
      margin-top: 10px;
    }

    .progress-bar {
      width: 100%;
      height: 20px;
      background-color: #f0f0f0;
      border-radius: 10px;
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      background-color: #007bff;
      transition: width 0.3s ease;
    }

    .result {
      padding: 15px;
      border: 1px solid #dee2e6;
      border-radius: 4px;
      background-color: #f8f9fa;
    }

    .result-summary p {
      margin: 5px 0;
    }

    .anomalies {
      margin-top: 15px;
    }

    .anomalies table {
      font-size: 14px;
    }

    .error {
      padding: 10px;
      background-color: #f8d7da;
      border: 1px solid #f5c6cb;
      border-radius: 4px;
      color: #721c24;
      font-size: 14px;
    }
  `]
})
export class AnomalyDetectorComponent implements OnInit, OnDestroy {
  @Input() data?: number[][];
  @Input() algorithm: 'isolation_forest' | 'local_outlier_factor' | 'one_class_svm' | 'auto' = 'auto';
  @Input() parameters: Record<string, any> = {};
  @Input() autoDetect: boolean = false;

  @Output() resultEvent = new EventEmitter<AnomalyDetectionResult>();
  @Output() errorEvent = new EventEmitter<Error>();

  algorithmControl = new FormControl(this.algorithm);
  dataControl = new FormControl('', [this.jsonValidator]);

  result: AnomalyDetectionResult | null = null;
  isLoading = false;
  error: string | null = null;
  progress = 0;
  isReady = false;

  private destroy$ = new Subject<void>();

  constructor(private pynomalyService: PynomalyService) {}

  ngOnInit(): void {
    // Subscribe to service state
    this.pynomalyService.isReady$
      .pipe(takeUntil(this.destroy$))
      .subscribe(ready => this.isReady = ready);

    // Auto-detect when data changes
    if (this.autoDetect && this.data) {
      this.handleDetectAnomalies();
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private jsonValidator(control: FormControl) {
    if (!control.value) return null;
    
    try {
      const parsed = JSON.parse(control.value);
      if (Array.isArray(parsed) && parsed.every(row => Array.isArray(row))) {
        return null;
      }
      return { invalidJson: true };
    } catch {
      return { invalidJson: true };
    }
  }

  private parseInputData(): number[][] | null {
    try {
      const parsed = JSON.parse(this.dataControl.value || '');
      if (Array.isArray(parsed) && parsed.every(row => Array.isArray(row))) {
        return parsed;
      }
      return null;
    } catch {
      return null;
    }
  }

  handleDetectAnomalies(dataToAnalyze?: number[][]): void {
    const analysisData = dataToAnalyze || this.data;
    
    if (!analysisData || analysisData.length === 0) {
      return;
    }

    const request: AnomalyDetectionRequest = {
      data: analysisData,
      algorithm: this.algorithmControl.value as any,
      parameters: this.parameters
    };

    this.isLoading = true;
    this.error = null;
    this.progress = 0;

    this.pynomalyService.detectAnomalies(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (detectionResult) => {
          this.progress = 100;
          this.result = detectionResult;
          this.isLoading = false;
          this.resultEvent.emit(detectionResult);
        },
        error: (err) => {
          this.error = err.message;
          this.isLoading = false;
          this.errorEvent.emit(err);
        }
      });

    // Simulate progress
    this.simulateProgress();
  }

  handleAnalyzeInput(): void {
    const parsedData = this.parseInputData();
    if (parsedData) {
      this.handleDetectAnomalies(parsedData);
    }
  }

  clear(): void {
    this.result = null;
    this.error = null;
    this.progress = 0;
  }

  formatJson(data: any): string {
    return JSON.stringify(data);
  }

  private simulateProgress(): void {
    const interval = setInterval(() => {
      if (this.progress < 90 && this.isLoading) {
        this.progress += 10;
      } else {
        clearInterval(interval);
      }
    }, 200);
  }
}