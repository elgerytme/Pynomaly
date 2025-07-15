/**
 * Angular service for Pynomaly authentication
 */

import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Observable, from, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { 
  AuthManager, 
  AuthState, 
  AuthCredentials, 
  SessionConfig,
  PynomalyClient
} from '../../../index';

@Injectable({
  providedIn: 'root'
})
export class PynomalyAuthService implements OnDestroy {
  private authManagerSubject = new BehaviorSubject<AuthManager | null>(null);
  private authStateSubject = new BehaviorSubject<AuthState>({
    isAuthenticated: false,
    user: null,
    token: null,
    expiresAt: null
  });
  private isLoadingSubject = new BehaviorSubject<boolean>(false);
  private errorSubject = new BehaviorSubject<string | null>(null);

  public authManager$ = this.authManagerSubject.asObservable();
  public authState$ = this.authStateSubject.asObservable();
  public isLoading$ = this.isLoadingSubject.asObservable();
  public error$ = this.errorSubject.asObservable();

  public isAuthenticated$ = this.authState$.pipe(
    map(state => state.isAuthenticated)
  );

  public user$ = this.authState$.pipe(
    map(state => state.user)
  );

  public token$ = this.authState$.pipe(
    map(state => state.token)
  );

  private currentAuthManager: AuthManager | null = null;
  private currentClient: PynomalyClient | null = null;

  constructor() {}

  initialize(options: SessionConfig = {}, client?: PynomalyClient): void {
    this.currentClient = client || null;

    const manager = new AuthManager({
      enablePersistence: true,
      autoRefresh: true,
      ...options
    });

    // Set up event listeners
    manager.on('auth:login', ({ user, token }) => {
      this.authStateSubject.next(manager.getAuthState());
      this.errorSubject.next(null);
    });

    manager.on('auth:logout', () => {
      this.authStateSubject.next(manager.getAuthState());
      this.errorSubject.next(null);
    });

    manager.on('auth:refresh', (token) => {
      this.authStateSubject.next(manager.getAuthState());
    });

    manager.on('auth:error', ({ error, type }) => {
      this.errorSubject.next(error);
    });

    manager.on('auth:expired', () => {
      this.authStateSubject.next(manager.getAuthState());
      this.errorSubject.next('Session expired');
    });

    this.currentAuthManager = manager;
    this.authManagerSubject.next(manager);
    this.authStateSubject.next(manager.getAuthState());
  }

  setClient(client: PynomalyClient): void {
    this.currentClient = client;
  }

  login(credentials: AuthCredentials): Observable<any> {
    if (!this.currentAuthManager || !this.currentClient) {
      return throwError(new Error('Auth manager or client not available'));
    }

    this.isLoadingSubject.next(true);
    this.errorSubject.next(null);

    return from(this.currentAuthManager.login(credentials, this.currentClient)).pipe(
      map(user => {
        this.isLoadingSubject.next(false);
        return user;
      }),
      catchError(error => {
        this.isLoadingSubject.next(false);
        this.errorSubject.next(error.message);
        return throwError(error);
      })
    );
  }

  loginWithApiKey(apiKey: string): Observable<any> {
    if (!this.currentAuthManager || !this.currentClient) {
      return throwError(new Error('Auth manager or client not available'));
    }

    this.isLoadingSubject.next(true);
    this.errorSubject.next(null);

    return from(this.currentAuthManager.loginWithApiKey(apiKey, this.currentClient)).pipe(
      map(user => {
        this.isLoadingSubject.next(false);
        return user;
      }),
      catchError(error => {
        this.isLoadingSubject.next(false);
        this.errorSubject.next(error.message);
        return throwError(error);
      })
    );
  }

  logout(): Observable<void> {
    if (!this.currentAuthManager || !this.currentClient) {
      return throwError(new Error('Auth manager or client not available'));
    }

    this.isLoadingSubject.next(true);
    this.errorSubject.next(null);

    return from(this.currentAuthManager.logout(this.currentClient)).pipe(
      map(() => {
        this.isLoadingSubject.next(false);
      }),
      catchError(error => {
        this.isLoadingSubject.next(false);
        this.errorSubject.next(error.message);
        return throwError(error);
      })
    );
  }

  refreshToken(): Observable<any> {
    if (!this.currentAuthManager || !this.currentClient) {
      return throwError(new Error('Auth manager or client not available'));
    }

    this.isLoadingSubject.next(true);
    this.errorSubject.next(null);

    return from(this.currentAuthManager.refreshToken(this.currentClient)).pipe(
      map(token => {
        this.isLoadingSubject.next(false);
        return token;
      }),
      catchError(error => {
        this.isLoadingSubject.next(false);
        this.errorSubject.next(error.message);
        return throwError(error);
      })
    );
  }

  getCurrentAuthState(): AuthState {
    return this.authStateSubject.value;
  }

  isAuthenticated(): boolean {
    return this.authStateSubject.value.isAuthenticated;
  }

  getUser(): any {
    return this.authStateSubject.value.user;
  }

  getToken(): any {
    return this.authStateSubject.value.token;
  }

  ngOnDestroy(): void {
    if (this.currentAuthManager) {
      this.currentAuthManager.destroy();
    }
  }
}