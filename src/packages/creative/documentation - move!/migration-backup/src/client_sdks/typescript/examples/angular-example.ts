/**
 * Angular Example: Pynomaly SDK Integration
 * Demonstrates how to use the Pynomaly TypeScript SDK in Angular applications
 */

import { Component, OnInit, OnDestroy, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

import { PynomaliClient } from '../src/index';
import {
  DetectionRequest,
  DetectionResponse,
  StreamDetectionResult,
  StreamAlert,
  HealthStatus,
  LoginCredentials,
  UserProfile,
} from '../src/types';

// Angular Service for Pynomaly SDK
@Injectable({
  providedIn: 'root'
})
export class PynomaliService {
  private client: PynomaliClient;
  private isAuthenticatedSubject = new BehaviorSubject<boolean>(false);
  private userProfileSubject = new BehaviorSubject<UserProfile | null>(null);
  private isConnectedSubject = new BehaviorSubject<boolean>(false);
  private streamDataSubject = new BehaviorSubject<StreamDetectionResult[]>([]);
  private alertsSubject = new BehaviorSubject<StreamAlert[]>([]);
  private healthStatusSubject = new BehaviorSubject<HealthStatus | null>(null);

  public isAuthenticated$ = this.isAuthenticatedSubject.asObservable();
  public userProfile$ = this.userProfileSubject.asObservable();
  public isConnected$ = this.isConnectedSubject.asObservable();
  public streamData$ = this.streamDataSubject.asObservable();
  public alerts$ = this.alertsSubject.asObservable();
  public healthStatus$ = this.healthStatusSubject.asObservable();

  constructor() {
    this.client = new PynomaliClient({
      baseUrl: process.env['NG_APP_PYNOMALY_API_URL'] || 'https://api.pynomaly.com',
      apiKey: process.env['NG_APP_PYNOMALY_API_KEY'],
      debug: process.env['NODE_ENV'] === 'development',
      websocket: {
        enabled: true,
        autoReconnect: true,
      },
    });

    // Check initial authentication state
    this.checkAuthenticationState();
  }

  private checkAuthenticationState(): void {
    const clientInfo = this.client.getClientInfo();
    this.isAuthenticatedSubject.next(clientInfo.isAuthenticated);
    this.userProfileSubject.next(clientInfo.sessionInfo || null);
    this.isConnectedSubject.next(clientInfo.isConnected);
  }

  async login(credentials: LoginCredentials): Promise<{ token: any; user: UserProfile }> {
    const response = await this.client.auth.login(credentials);
    this.isAuthenticatedSubject.next(true);
    this.userProfileSubject.next(response.user);
    return response;
  }

  async logout(): Promise<void> {
    await this.client.auth.logout();
    this.isAuthenticatedSubject.next(false);
    this.userProfileSubject.next(null);
    this.isConnectedSubject.next(false);
  }

  async detectAnomalies(request: DetectionRequest): Promise<DetectionResponse> {
    return await this.client.detection.detect(request);
  }

  async connectWebSocket(): Promise<void> {
    await this.client.connectWebSocket({
      onConnect: () => {
        this.isConnectedSubject.next(true);
        console.log('WebSocket connected');
      },
      onDisconnect: () => {
        this.isConnectedSubject.next(false);
        console.log('WebSocket disconnected');
      },
      onData: (data: StreamDetectionResult) => {
        const currentData = this.streamDataSubject.value;
        this.streamDataSubject.next([...currentData.slice(-19), data]);
      },
      onAlert: (alert: StreamAlert) => {
        const currentAlerts = this.alertsSubject.value;
        this.alertsSubject.next([...currentAlerts.slice(-9), alert]);
      },
      onError: (error: Error) => {
        console.error('WebSocket error:', error);
      },
    });
  }

  disconnectWebSocket(): void {
    this.client.disconnectWebSocket();
    this.isConnectedSubject.next(false);
  }

  async sendTestData(): Promise<void> {
    const streamingManager = this.client.getStreamingManager();
    if (!streamingManager) return;

    const data = Array.from({ length: 10 }, () => Math.random() * 100);
    await streamingManager.sendStreamData('test-stream', data);
  }

  async getHealth(): Promise<HealthStatus> {
    const health = await this.client.health.getHealth();
    this.healthStatusSubject.next(health);
    return health;
  }

  getClient(): PynomaliClient {
    return this.client;
  }
}

// Authentication Component
@Component({
  selector: 'app-pynomaly-auth',
  template: `
    <div class="auth-container">
      <div *ngIf="isAuthenticated$ | async; else loginForm">
        <h2>Welcome, {{ (userProfile$ | async)?.username }}!</h2>
        <button 
          (click)="logout()" 
          [disabled]="loading"
          class="btn btn-danger">
          {{ loading ? 'Logging out...' : 'Logout' }}
        </button>
      </div>
      
      <ng-template #loginForm>
        <form (ngSubmit)="login()" #authForm="ngForm">
          <h2>Login to Pynomaly</h2>
          
          <div class="form-group">
            <label>Username:</label>
            <input
              type="text"
              [(ngModel)]="credentials.username"
              name="username"
              required
              class="form-control"
            />
          </div>

          <div class="form-group">
            <label>Password:</label>
            <input
              type="password"
              [(ngModel)]="credentials.password"
              name="password"
              required
              class="form-control"
            />
          </div>

          <div class="form-group">
            <label>MFA Code (optional):</label>
            <input
              type="text"
              [(ngModel)]="credentials.mfaCode"
              name="mfaCode"
              class="form-control"
            />
          </div>

          <button 
            type="submit" 
            [disabled]="loading || !authForm.form.valid"
            class="btn btn-primary">
            {{ loading ? 'Logging in...' : 'Login' }}
          </button>
        </form>
      </ng-template>

      <div *ngIf="error" class="alert alert-danger">
        {{ error }}
      </div>
    </div>
  `,
  styles: [`
    .auth-container {
      padding: 20px;
      max-width: 400px;
      margin: 0 auto;
    }
    
    .form-group {
      margin-bottom: 15px;
    }
    
    .form-group label {
      display: block;
      margin-bottom: 5px;
      font-weight: bold;
    }
    
    .form-control {
      width: 100%;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      box-sizing: border-box;
    }
    
    .btn {
      padding: 10px 20px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    
    .btn-primary {
      background-color: #007bff;
      color: white;
    }
    
    .btn-danger {
      background-color: #dc3545;
      color: white;
    }
    
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    
    .alert {
      padding: 10px;
      border-radius: 4px;
      margin: 10px 0;
    }
    
    .alert-danger {
      background-color: #f8d7da;
      color: #721c24;
      border: 1px solid #f5c6cb;
    }
  `]
})
export class PynomaliAuthComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  isAuthenticated$: Observable<boolean>;
  userProfile$: Observable<UserProfile | null>;
  loading = false;
  error: string | null = null;

  credentials: LoginCredentials = {
    username: '',
    password: '',
    mfaCode: '',
  };

  constructor(private pynomaliService: PynomaliService) {
    this.isAuthenticated$ = this.pynomaliService.isAuthenticated$;
    this.userProfile$ = this.pynomaliService.userProfile$;
  }

  ngOnInit(): void {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  async login(): Promise<void> {
    this.loading = true;
    this.error = null;

    try {
      await this.pynomaliService.login(this.credentials);
    } catch (err: any) {
      this.error = err.message || 'Login failed';
    } finally {
      this.loading = false;
    }
  }

  async logout(): Promise<void> {
    this.loading = true;
    try {
      await this.pynomaliService.logout();
    } catch (err: any) {
      this.error = err.message || 'Logout failed';
    } finally {
      this.loading = false;
    }
  }
}

// Anomaly Detection Component
@Component({
  selector: 'app-pynomaly-detection',
  template: `
    <div class="detection-container">
      <h2>Anomaly Detection</h2>
      
      <div class="form-group">
        <label>Input Data (comma-separated):</label>
        <input
          type="text"
          [(ngModel)]="inputData"
          placeholder="1,2,3,4,5,100,6,7,8,9"
          class="form-control"
        />
      </div>

      <div class="form-group">
        <label>Algorithm:</label>
        <select [(ngModel)]="algorithm" class="form-control">
          <option value="isolation_forest">Isolation Forest</option>
          <option value="one_class_svm">One-Class SVM</option>
          <option value="local_outlier_factor">Local Outlier Factor</option>
          <option value="elliptic_envelope">Elliptic Envelope</option>
        </select>
      </div>

      <button 
        (click)="detectAnomalies()" 
        [disabled]="loading"
        class="btn btn-primary">
        {{ loading ? 'Detecting...' : 'Detect Anomalies' }}
      </button>

      <div *ngIf="error" class="alert alert-danger">
        {{ error }}
      </div>

      <div *ngIf="result" class="results">
        <h3>Results</h3>
        <div class="result-item">
          <strong>Anomaly Rate:</strong> {{ (result.anomalyRate * 100).toFixed(2) }}%
        </div>
        <div class="result-item">
          <strong>Processing Time:</strong> {{ result.processingTime }}ms
        </div>
        <div class="result-item">
          <strong>Algorithm:</strong> {{ result.algorithm }}
        </div>
        
        <h4>Scores and Predictions:</h4>
        <table class="results-table">
          <thead>
            <tr>
              <th>Index</th>
              <th>Score</th>
              <th>Prediction</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            <tr 
              *ngFor="let score of result.scores; let i = index" 
              [class.anomaly]="result.predictions[i]">
              <td>{{ i }}</td>
              <td>{{ score.toFixed(4) }}</td>
              <td>{{ result.predictions[i] ? 'Anomaly' : 'Normal' }}</td>
              <td>{{ result.confidence[i]?.toFixed(4) || 'N/A' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `,
  styles: [`
    .detection-container {
      padding: 20px;
    }
    
    .form-group {
      margin-bottom: 15px;
    }
    
    .form-group label {
      display: block;
      margin-bottom: 5px;
      font-weight: bold;
    }
    
    .form-control {
      width: 100%;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      box-sizing: border-box;
    }
    
    .btn {
      padding: 10px 20px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    
    .btn-primary {
      background-color: #007bff;
      color: white;
    }
    
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    
    .alert {
      padding: 10px;
      border-radius: 4px;
      margin: 10px 0;
    }
    
    .alert-danger {
      background-color: #f8d7da;
      color: #721c24;
      border: 1px solid #f5c6cb;
    }
    
    .results {
      margin-top: 20px;
      padding: 20px;
      border: 1px solid #ddd;
      border-radius: 4px;
      background-color: #f8f9fa;
    }
    
    .result-item {
      margin-bottom: 10px;
    }
    
    .results-table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
    }
    
    .results-table th,
    .results-table td {
      border: 1px solid #ddd;
      padding: 8px;
      text-align: left;
    }
    
    .results-table th {
      background-color: #f2f2f2;
      font-weight: bold;
    }
    
    .results-table tr.anomaly {
      background-color: #ffe6e6;
    }
  `]
})
export class PynomaliDetectionComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  inputData = '1,2,3,4,5,100,6,7,8,9';
  algorithm = 'isolation_forest';
  result: DetectionResponse | null = null;
  loading = false;
  error: string | null = null;

  constructor(private pynomaliService: PynomaliService) {}

  ngOnInit(): void {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  async detectAnomalies(): Promise<void> {
    this.loading = true;
    this.error = null;

    try {
      const data = this.inputData.split(',').map(Number);
      
      const request: DetectionRequest = {
        data,
        algorithm: this.algorithm as any,
        parameters: {
          contamination: 0.1,
          n_estimators: 100,
        },
        includeExplanations: true,
      };

      this.result = await this.pynomaliService.detectAnomalies(request);
    } catch (err: any) {
      this.error = err.message || 'Detection failed';
    } finally {
      this.loading = false;
    }
  }
}

// Streaming Component
@Component({
  selector: 'app-pynomaly-streaming',
  template: `
    <div class="streaming-container">
      <h2>Real-time Streaming</h2>
      
      <div class="controls">
        <button 
          *ngIf="!(isConnected$ | async)" 
          (click)="connectWebSocket()" 
          [disabled]="loading"
          class="btn btn-primary">
          {{ loading ? 'Connecting...' : 'Connect WebSocket' }}
        </button>
        
        <ng-container *ngIf="isConnected$ | async">
          <button (click)="disconnectWebSocket()" class="btn btn-danger">
            Disconnect
          </button>
          <button (click)="sendTestData()" class="btn btn-secondary">
            Send Test Data
          </button>
        </ng-container>
      </div>

      <div class="status">
        Status: {{ (isConnected$ | async) ? '🟢 Connected' : '🔴 Disconnected' }}
      </div>

      <div *ngIf="error" class="alert alert-danger">
        {{ error }}
      </div>

      <div *ngIf="(alerts$ | async)?.length" class="alerts">
        <h3>Recent Alerts</h3>
        <div 
          *ngFor="let alert of alerts$ | async" 
          [class]="'alert alert-' + alert.level">
          <strong>{{ alert.level.toUpperCase() }}:</strong> {{ alert.message }}
          <span class="score">Score: {{ alert.score.toFixed(4) }}</span>
        </div>
      </div>

      <div *ngIf="(streamData$ | async)?.length" class="stream-data">
        <h3>Real-time Results</h3>
        <div class="data-grid">
          <div *ngFor="let data of streamData$ | async" class="data-item">
            <div>Stream: {{ data.streamId }}</div>
            <div>Anomalies: {{ data.result.anomalyRate.toFixed(2) }}%</div>
            <div>Time: {{ formatTime(data.timestamp) }}</div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .streaming-container {
      padding: 20px;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 10px;
    }
    
    .btn {
      padding: 10px 20px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    
    .btn-primary {
      background-color: #007bff;
      color: white;
    }
    
    .btn-danger {
      background-color: #dc3545;
      color: white;
    }
    
    .btn-secondary {
      background-color: #6c757d;
      color: white;
    }
    
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    
    .status {
      font-size: 16px;
      font-weight: bold;
      margin-bottom: 20px;
    }
    
    .alert {
      padding: 10px;
      margin: 5px 0;
      border-radius: 4px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .alert-info {
      background-color: #d1ecf1;
      color: #0c5460;
    }
    
    .alert-warning {
      background-color: #fff3cd;
      color: #856404;
    }
    
    .alert-critical {
      background-color: #f8d7da;
      color: #721c24;
    }
    
    .score {
      font-size: 12px;
      opacity: 0.8;
    }
    
    .stream-data {
      margin-top: 20px;
    }
    
    .data-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 10px;
      margin: 10px 0;
    }
    
    .data-item {
      border: 1px solid #ddd;
      padding: 10px;
      border-radius: 4px;
      background-color: #f8f9fa;
    }
  `]
})
export class PynomaliStreamingComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  isConnected$: Observable<boolean>;
  streamData$: Observable<StreamDetectionResult[]>;
  alerts$: Observable<StreamAlert[]>;
  loading = false;
  error: string | null = null;

  constructor(private pynomaliService: PynomaliService) {
    this.isConnected$ = this.pynomaliService.isConnected$;
    this.streamData$ = this.pynomaliService.streamData$;
    this.alerts$ = this.pynomaliService.alerts$;
  }

  ngOnInit(): void {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  async connectWebSocket(): Promise<void> {
    this.loading = true;
    this.error = null;

    try {
      await this.pynomaliService.connectWebSocket();
    } catch (err: any) {
      this.error = err.message || 'Connection failed';
    } finally {
      this.loading = false;
    }
  }

  disconnectWebSocket(): void {
    this.pynomaliService.disconnectWebSocket();
  }

  async sendTestData(): Promise<void> {
    try {
      await this.pynomaliService.sendTestData();
    } catch (err: any) {
      this.error = err.message || 'Failed to send data';
    }
  }

  formatTime(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString();
  }
}

// Health Component
@Component({
  selector: 'app-pynomaly-health',
  template: `
    <div class="health-container">
      <h2>System Health</h2>
      
      <div class="controls">
        <button 
          (click)="fetchHealth()" 
          [disabled]="loading"
          class="btn btn-primary">
          {{ loading ? 'Refreshing...' : 'Refresh' }}
        </button>
        
        <label class="checkbox-label">
          <input 
            type="checkbox" 
            [(ngModel)]="autoRefresh"
            (change)="onAutoRefreshChange()"
          />
          Auto-refresh (5s)
        </label>
      </div>

      <div *ngIf="error" class="alert alert-danger">
        {{ error }}
      </div>

      <div *ngIf="healthStatus$ | async as health" class="health-status">
        <div [class]="'overall-status status-' + health.status">
          <h3>Overall Status: {{ health.status.toUpperCase() }}</h3>
          <div>Version: {{ health.version }}</div>
          <div>Uptime: {{ formatUptime(health.uptime) }}</div>
        </div>

        <div class="services">
          <h4>Services</h4>
          <div 
            *ngFor="let service of health.services" 
            [class]="'service status-' + service.status">
            <strong>{{ service.name }}</strong>
            <span class="status">{{ service.status }}</span>
            <span *ngIf="service.responseTime" class="response-time">
              {{ service.responseTime }}ms
            </span>
            <div *ngIf="service.error" class="service-error">
              {{ service.error }}
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .health-container {
      padding: 20px;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 15px;
    }
    
    .btn {
      padding: 10px 20px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    
    .btn-primary {
      background-color: #007bff;
      color: white;
    }
    
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    
    .checkbox-label {
      display: flex;
      align-items: center;
      gap: 5px;
    }
    
    .alert {
      padding: 10px;
      border-radius: 4px;
      margin: 10px 0;
    }
    
    .alert-danger {
      background-color: #f8d7da;
      color: #721c24;
      border: 1px solid #f5c6cb;
    }
    
    .health-status {
      margin-top: 20px;
    }
    
    .overall-status {
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 20px;
    }
    
    .overall-status.status-healthy {
      background-color: #d4edda;
      border: 1px solid #c3e6cb;
    }
    
    .overall-status.status-degraded {
      background-color: #fff3cd;
      border: 1px solid #ffeaa7;
    }
    
    .overall-status.status-unhealthy {
      background-color: #f8d7da;
      border: 1px solid #f5c6cb;
    }
    
    .services {
      margin-top: 20px;
    }
    
    .service {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px;
      margin: 5px 0;
      border: 1px solid #ddd;
      border-radius: 4px;
      background-color: white;
    }
    
    .service-error {
      color: #dc3545;
      font-size: 12px;
      flex-basis: 100%;
      margin-top: 5px;
    }
    
    .response-time {
      font-size: 12px;
      opacity: 0.7;
    }
    
    .status-healthy { color: #28a745; }
    .status-degraded { color: #ffc107; }
    .status-unhealthy { color: #dc3545; }
  `]
})
export class PynomaliHealthComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();
  private refreshInterval: any;

  healthStatus$: Observable<HealthStatus | null>;
  loading = false;
  error: string | null = null;
  autoRefresh = false;

  constructor(private pynomaliService: PynomaliService) {
    this.healthStatus$ = this.pynomaliService.healthStatus$;
  }

  ngOnInit(): void {
    this.fetchHealth();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  }

  async fetchHealth(): Promise<void> {
    this.loading = true;
    this.error = null;

    try {
      await this.pynomaliService.getHealth();
    } catch (err: any) {
      this.error = err.message || 'Failed to fetch health status';
    } finally {
      this.loading = false;
    }
  }

  onAutoRefreshChange(): void {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = null;
    }

    if (this.autoRefresh) {
      this.refreshInterval = setInterval(() => {
        this.fetchHealth();
      }, 5000);
    }
  }

  formatUptime(uptime: number): string {
    const hours = Math.floor(uptime / 3600);
    const minutes = Math.floor((uptime % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
}

// Main Application Component
@Component({
  selector: 'app-pynomaly-main',
  template: `
    <div class="pynomaly-app">
      <header>
        <h1>Pynomaly Angular Example</h1>
        <nav>
          <button 
            *ngFor="let tab of tabs" 
            [class.active]="activeTab === tab.id"
            (click)="activeTab = tab.id"
            class="nav-button">
            {{ tab.label }}
          </button>
        </nav>
      </header>

      <main>
        <app-pynomaly-auth *ngIf="activeTab === 'auth'"></app-pynomaly-auth>
        <app-pynomaly-detection *ngIf="activeTab === 'detection'"></app-pynomaly-detection>
        <app-pynomaly-streaming *ngIf="activeTab === 'streaming'"></app-pynomaly-streaming>
        <app-pynomaly-health *ngIf="activeTab === 'health'"></app-pynomaly-health>
      </main>
    </div>
  `,
  styles: [`
    .pynomaly-app {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      font-family: Arial, sans-serif;
    }
    
    header {
      border-bottom: 2px solid #eee;
      margin-bottom: 20px;
      padding-bottom: 20px;
    }
    
    nav {
      display: flex;
      gap: 10px;
      margin-top: 15px;
    }
    
    .nav-button {
      padding: 10px 20px;
      border: 1px solid #ddd;
      background: white;
      cursor: pointer;
      border-radius: 4px;
      font-size: 14px;
    }
    
    .nav-button:hover {
      background: #f8f9fa;
    }
    
    .nav-button.active {
      background: #007bff;
      color: white;
    }
    
    .nav-button.active:hover {
      background: #0056b3;
    }
  `]
})
export class PynomaliMainComponent {
  activeTab = 'auth';

  tabs = [
    { id: 'auth', label: 'Authentication' },
    { id: 'detection', label: 'Anomaly Detection' },
    { id: 'streaming', label: 'Real-time Streaming' },
    { id: 'health', label: 'System Health' },
  ];
}

// Angular Module
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

@NgModule({
  declarations: [
    PynomaliMainComponent,
    PynomaliAuthComponent,
    PynomaliDetectionComponent,
    PynomaliStreamingComponent,
    PynomaliHealthComponent,
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
  ],
  providers: [
    PynomaliService,
  ],
  bootstrap: [PynomaliMainComponent]
})
export class PynomaliModule { }

// Export all components for use in other modules
export {
  PynomaliService,
  PynomaliMainComponent,
  PynomaliAuthComponent,
  PynomaliDetectionComponent,
  PynomaliStreamingComponent,
  PynomaliHealthComponent,
  PynomaliModule,
};