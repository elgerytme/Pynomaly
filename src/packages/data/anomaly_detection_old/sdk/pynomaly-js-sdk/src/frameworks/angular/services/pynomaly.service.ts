/**
 * Angular service for Pynomaly SDK
 */

import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Observable, from, throwError } from 'rxjs';
import { catchError, map, switchMap } from 'rxjs/operators';
import { 
  PynomalyClient, 
  PynomalyConfig, 
  AnomalyDetectionRequest, 
  AnomalyDetectionResult,
  DataQualityRequest,
  DataQualityResult,
  DataProfilingRequest,
  DataProfilingResult
} from '../../../index';

@Injectable({
  providedIn: 'root'
})
export class PynomalyService implements OnDestroy {
  private clientSubject = new BehaviorSubject<PynomalyClient | null>(null);
  private isReadySubject = new BehaviorSubject<boolean>(false);
  private isLoadingSubject = new BehaviorSubject<boolean>(false);
  private errorSubject = new BehaviorSubject<Error | null>(null);

  public client$ = this.clientSubject.asObservable();
  public isReady$ = this.isReadySubject.asObservable();
  public isLoading$ = this.isLoadingSubject.asObservable();
  public error$ = this.errorSubject.asObservable();

  private currentClient: PynomalyClient | null = null;

  constructor() {}

  initialize(config: PynomalyConfig, autoConnect: boolean = true): Observable<PynomalyClient> {
    this.isLoadingSubject.next(true);
    this.errorSubject.next(null);

    return from(this.initializeClient(config, autoConnect)).pipe(
      map(client => {
        this.currentClient = client;
        this.clientSubject.next(client);
        this.isReadySubject.next(true);
        this.isLoadingSubject.next(false);
        return client;
      }),
      catchError(error => {
        this.errorSubject.next(error);
        this.isLoadingSubject.next(false);
        return throwError(error);
      })
    );
  }

  private async initializeClient(config: PynomalyConfig, autoConnect: boolean): Promise<PynomalyClient> {
    const client = new PynomalyClient({
      apiKey: '',
      baseUrl: 'https://api.pynomaly.com',
      ...config
    });

    // Test connection if autoConnect is enabled
    if (autoConnect) {
      try {
        await client.healthCheck();
      } catch (healthError) {
        throw new Error(`Failed to connect to Pynomaly API: ${healthError.message}`);
      }
    }

    return client;
  }

  detectAnomalies(request: AnomalyDetectionRequest): Observable<AnomalyDetectionResult> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.detectAnomalies(request)).pipe(
      catchError(error => throwError(error))
    );
  }

  detectAnomaliesAsync(request: AnomalyDetectionRequest): Observable<string> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.detectAnomaliesAsync(request)).pipe(
      catchError(error => throwError(error))
    );
  }

  analyzeDataQuality(request: DataQualityRequest): Observable<DataQualityResult> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.analyzeDataQuality(request)).pipe(
      catchError(error => throwError(error))
    );
  }

  analyzeDataQualityAsync(request: DataQualityRequest): Observable<string> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.analyzeDataQualityAsync(request)).pipe(
      catchError(error => throwError(error))
    );
  }

  profileData(request: DataProfilingRequest): Observable<DataProfilingResult> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.profileData(request)).pipe(
      catchError(error => throwError(error))
    );
  }

  profileDataAsync(request: DataProfilingRequest): Observable<string> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.profileDataAsync(request)).pipe(
      catchError(error => throwError(error))
    );
  }

  getJobStatus(jobId: string): Observable<any> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.getJobStatus(jobId)).pipe(
      catchError(error => throwError(error))
    );
  }

  getJobResult(jobId: string): Observable<any> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.getJobResult(jobId)).pipe(
      catchError(error => throwError(error))
    );
  }

  cancelJob(jobId: string): Observable<void> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.cancelJob(jobId)).pipe(
      catchError(error => throwError(error))
    );
  }

  healthCheck(): Observable<any> {
    if (!this.currentClient) {
      return throwError(new Error('Pynomaly client not initialized'));
    }

    return from(this.currentClient.healthCheck()).pipe(
      catchError(error => throwError(error))
    );
  }

  reconnect(config?: PynomalyConfig): Observable<PynomalyClient> {
    if (this.currentClient) {
      this.currentClient.destroy();
      this.currentClient = null;
      this.clientSubject.next(null);
      this.isReadySubject.next(false);
    }

    const clientConfig = config || this.getLastConfig();
    return this.initialize(clientConfig);
  }

  disconnect(): void {
    if (this.currentClient) {
      this.currentClient.destroy();
      this.currentClient = null;
      this.clientSubject.next(null);
      this.isReadySubject.next(false);
    }
  }

  private getLastConfig(): PynomalyConfig {
    // Return a default config or store the last used config
    return {
      apiKey: '',
      baseUrl: 'https://api.pynomaly.com'
    };
  }

  ngOnDestroy(): void {
    this.disconnect();
  }
}