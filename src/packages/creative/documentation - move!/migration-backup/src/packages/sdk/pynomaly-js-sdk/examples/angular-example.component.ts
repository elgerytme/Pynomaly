/**
 * Angular example component using Pynomaly SDK
 */

import { Component, OnInit, OnDestroy } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Subject, takeUntil } from 'rxjs';
import { PynomalyService, PynomalyAuthService } from '../src/frameworks/angular';
import { AnomalyDetectionResult } from '../src/index';

@Component({
  selector: 'app-pynomaly-example',
  template: `
    <div style="padding: 20px; font-family: Arial, sans-serif;">
      <h1>Pynomaly Angular SDK Example</h1>

      <!-- Client Status -->
      <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;">
        <h2>Client Status</h2>
        <p><strong>Ready:</strong> {{ isReady ? '✅ Yes' : '❌ No' }}</p>
        <p><strong>Loading:</strong> {{ isLoading ? '⏳ Yes' : '✅ No' }}</p>
        <p><strong>Client:</strong> {{ client ? '✅ Available' : '❌ Not Available' }}</p>
        <p *ngIf="clientError" style="color: red;"><strong>Error:</strong> {{ clientError.message }}</p>
        
        <button (click)="initializeClient()" [disabled]="isLoading">
          {{ isLoading ? 'Initializing...' : 'Initialize Client' }}
        </button>
      </div>

      <!-- Authentication -->
      <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;">
        <h2>Authentication</h2>
        <p><strong>Authenticated:</strong> {{ authState.isAuthenticated ? '✅ Yes' : '❌ No' }}</p>
        <p><strong>User:</strong> {{ authState.user ? authState.user.email : 'None' }}</p>
        <p *ngIf="authError" style="color: red;"><strong>Error:</strong> {{ authError }}</p>

        <div *ngIf="!authState.isAuthenticated">
          <!-- Login Form -->
          <form [formGroup]="loginForm" (ngSubmit)="handleLogin()">
            <h4>Login with Credentials</h4>
            <div style="margin-bottom: 10px;">
              <input
                formControlName="email"
                type="email"
                placeholder="Email"
                style="margin-right: 10px; padding: 5px;"
              />
              <input
                formControlName="password"
                type="password"
                placeholder="Password"
                style="margin-right: 10px; padding: 5px;"
              />
              <button type="submit" [disabled]="authLoading || loginForm.invalid">
                {{ authLoading ? 'Logging in...' : 'Login' }}
              </button>
            </div>
          </form>

          <!-- API Key Login -->
          <form [formGroup]="apiKeyForm" (ngSubmit)="handleApiKeyLogin()">
            <h4>Login with API Key</h4>
            <div>
              <input
                formControlName="apiKey"
                type="text"
                placeholder="API Key"
                style="margin-right: 10px; padding: 5px;"
              />
              <button type="submit" [disabled]="authLoading || apiKeyForm.invalid">
                {{ authLoading ? 'Logging in...' : 'Login with API Key' }}
              </button>
            </div>
          </form>
        </div>

        <div *ngIf="authState.isAuthenticated">
          <button (click)="handleLogout()" [disabled]="authLoading">
            {{ authLoading ? 'Logging out...' : 'Logout' }}
          </button>
        </div>
      </div>

      <!-- Anomaly Detection with Service -->
      <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;">
        <h2>Anomaly Detection with Service</h2>
        
        <div style="margin-bottom: 10px;">
          <button (click)="handleDetectAnomalies()" [disabled]="detectionLoading || !isReady">
            {{ detectionLoading ? 'Detecting...' : 'Detect Anomalies' }}
          </button>
          <button (click)="clearDetection()" [disabled]="detectionLoading" style="margin-left: 10px;">
            Clear
          </button>
        </div>

        <div *ngIf="detectionLoading" style="margin-bottom: 10px;">
          <div style="width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden;">
            <div [style.width.%]="detectionProgress" style="height: 100%; background-color: #007bff; transition: width 0.3s ease;"></div>
          </div>
          <p>Progress: {{ detectionProgress }}%</p>
        </div>

        <p *ngIf="detectionError" style="color: red;">Error: {{ detectionError }}</p>

        <div *ngIf="detectionResult" style="margin-top: 10px;">
          <h4>Results:</h4>
          <p><strong>Algorithm:</strong> {{ detectionResult.algorithm }}</p>
          <p><strong>Anomalies Found:</strong> {{ detectionResult.metrics.anomalyCount }}</p>
          <p><strong>Processing Time:</strong> {{ detectionResult.processingTime }}ms</p>
        </div>

        <div style="margin-top: 10px;">
          <h4>Sample Data:</h4>
          <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 12px;">{{ formatJson(sampleData) }}</pre>
        </div>
      </div>

      <!-- Anomaly Detector Component -->
      <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;">
        <h2>Anomaly Detector Component</h2>
        <pynomaly-anomaly-detector
          algorithm="auto"
          [autoDetect]="false"
          (resultEvent)="handleComponentResult($event)"
          (errorEvent)="handleComponentError($event)"
        ></pynomaly-anomaly-detector>
      </div>
    </div>
  `
})
export class PynomalyExampleComponent implements OnInit, OnDestroy {
  // Client state
  client: any = null;
  isReady = false;
  isLoading = false;
  clientError: Error | null = null;

  // Auth state
  authState = {
    isAuthenticated: false,
    user: null,
    token: null,
    expiresAt: null
  };
  authLoading = false;
  authError: string | null = null;

  // Detection state
  detectionResult: AnomalyDetectionResult | null = null;
  detectionLoading = false;
  detectionError: string | null = null;
  detectionProgress = 0;

  // Forms
  loginForm: FormGroup;
  apiKeyForm: FormGroup;

  // Sample data
  sampleData = [
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [100, 200, 300] // This should be detected as an anomaly
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private pynomalyService: PynomalyService,
    private authService: PynomalyAuthService,
    private fb: FormBuilder
  ) {
    // Initialize forms
    this.loginForm = this.fb.group({
      email: ['demo@example.com', [Validators.required, Validators.email]],
      password: ['password', [Validators.required]]
    });

    this.apiKeyForm = this.fb.group({
      apiKey: ['demo-api-key', [Validators.required]]
    });
  }

  ngOnInit(): void {
    // Subscribe to client state
    this.pynomalyService.client$
      .pipe(takeUntil(this.destroy$))
      .subscribe(client => this.client = client);

    this.pynomalyService.isReady$
      .pipe(takeUntil(this.destroy$))
      .subscribe(ready => this.isReady = ready);

    this.pynomalyService.isLoading$
      .pipe(takeUntil(this.destroy$))
      .subscribe(loading => this.isLoading = loading);

    this.pynomalyService.error$
      .pipe(takeUntil(this.destroy$))
      .subscribe(error => this.clientError = error);

    // Subscribe to auth state
    this.authService.authState$
      .pipe(takeUntil(this.destroy$))
      .subscribe(authState => this.authState = authState);

    this.authService.isLoading$
      .pipe(takeUntil(this.destroy$))
      .subscribe(loading => this.authLoading = loading);

    this.authService.error$
      .pipe(takeUntil(this.destroy$))
      .subscribe(error => this.authError = error);

    // Initialize services
    this.initializeClient();
    this.authService.initialize({}, this.client);
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  initializeClient(): void {
    const config = {
      apiKey: 'demo-api-key',
      baseUrl: 'https://api.pynomaly.com',
      debug: true
    };

    this.pynomalyService.initialize(config, true)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (client) => {
          console.log('Client initialized:', client);
          this.authService.setClient(client);
        },
        error: (error) => {
          console.error('Client initialization failed:', error);
        }
      });
  }

  handleLogin(): void {
    if (this.loginForm.valid) {
      const { email, password } = this.loginForm.value;
      this.authService.login({ email, password })
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (user) => console.log('Login successful:', user),
          error: (error) => console.error('Login failed:', error)
        });
    }
  }

  handleApiKeyLogin(): void {
    if (this.apiKeyForm.valid) {
      const { apiKey } = this.apiKeyForm.value;
      this.authService.loginWithApiKey(apiKey)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (user) => console.log('API key login successful:', user),
          error: (error) => console.error('API key login failed:', error)
        });
    }
  }

  handleLogout(): void {
    this.authService.logout()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => console.log('Logout successful'),
        error: (error) => console.error('Logout failed:', error)
      });
  }

  handleDetectAnomalies(): void {
    this.detectionLoading = true;
    this.detectionError = null;
    this.detectionProgress = 0;

    const request = {
      data: this.sampleData,
      algorithm: 'isolation_forest' as const,
      parameters: { contamination: 0.1 }
    };

    // Simulate progress
    this.simulateProgress();

    this.pynomalyService.detectAnomalies(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          this.detectionProgress = 100;
          this.detectionResult = result;
          this.detectionLoading = false;
          console.log('Detection successful:', result);
        },
        error: (error) => {
          this.detectionError = error.message;
          this.detectionLoading = false;
          console.error('Detection failed:', error);
        }
      });
  }

  clearDetection(): void {
    this.detectionResult = null;
    this.detectionError = null;
    this.detectionProgress = 0;
  }

  handleComponentResult(result: AnomalyDetectionResult): void {
    console.log('Component result:', result);
  }

  handleComponentError(error: Error): void {
    console.error('Component error:', error);
  }

  formatJson(data: any): string {
    return JSON.stringify(data, null, 2);
  }

  private simulateProgress(): void {
    const interval = setInterval(() => {
      if (this.detectionProgress < 90 && this.detectionLoading) {
        this.detectionProgress += 10;
      } else {
        clearInterval(interval);
      }
    }, 200);
  }
}