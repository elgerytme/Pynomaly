# Framework Integration Guide

This guide covers how to integrate the Pynomaly JavaScript SDK with popular frontend frameworks.

## Table of Contents

- [React Integration](#react-integration)
- [Vue.js Integration](#vue.js-integration)
- [Angular Integration](#angular-integration)
- [Framework Comparison](#framework-comparison)
- [Best Practices](#best-practices)

## React Integration

### Installation

```bash
npm install @pynomaly/js-sdk react
```

### Basic Setup with Provider

```tsx
import React from 'react';
import { PynomalyProvider } from '@pynomaly/js-sdk/react';

const App: React.FC = () => {
  return (
    <PynomalyProvider
      config={{
        apiKey: 'your-api-key',
        baseUrl: 'https://api.pynomaly.com'
      }}
      autoConnect={true}
      onError={(error) => console.error('Pynomaly error:', error)}
      onReady={(client) => console.log('Pynomaly ready:', client)}
    >
      <YourAppContent />
    </PynomalyProvider>
  );
};
```

### Using Hooks

#### Client Hook

```tsx
import { usePynomaly } from '@pynomaly/js-sdk/react';

const MyComponent: React.FC = () => {
  const { client, isReady, isLoading, error } = usePynomaly();

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!isReady) return <div>Not ready</div>;

  return <div>Pynomaly client is ready!</div>;
};
```

#### Authentication Hook

```tsx
import { usePynomalyAuth } from '@pynomaly/js-sdk/react';

const AuthComponent: React.FC = () => {
  const {
    authState,
    isAuthenticated,
    login,
    logout,
    isLoading,
    error
  } = usePynomalyAuth({ client });

  const handleLogin = async () => {
    try {
      await login({ email: 'user@example.com', password: 'password' });
    } catch (err) {
      console.error('Login failed:', err);
    }
  };

  return (
    <div>
      {isAuthenticated ? (
        <button onClick={logout}>Logout</button>
      ) : (
        <button onClick={handleLogin}>Login</button>
      )}
    </div>
  );
};
```

#### Anomaly Detection Hook

```tsx
import { useAnomalyDetection } from '@pynomaly/js-sdk/react';

const DetectionComponent: React.FC = () => {
  const {
    detectAnomalies,
    result,
    isLoading,
    error,
    progress,
    clear
  } = useAnomalyDetection({ client });

  const handleDetect = async () => {
    try {
      await detectAnomalies({
        data: [[1, 2], [3, 4], [100, 200]],
        algorithm: 'isolation_forest'
      });
    } catch (err) {
      console.error('Detection failed:', err);
    }
  };

  return (
    <div>
      <button onClick={handleDetect} disabled={isLoading}>
        {isLoading ? 'Detecting...' : 'Detect Anomalies'}
      </button>
      {isLoading && <div>Progress: {progress}%</div>}
      {result && <div>Found {result.metrics.anomalyCount} anomalies</div>}
      {error && <div>Error: {error.message}</div>}
    </div>
  );
};
```

### Pre-built Components

```tsx
import { AnomalyDetector } from '@pynomaly/js-sdk/react';

const MyApp: React.FC = () => {
  return (
    <AnomalyDetector
      algorithm="auto"
      autoDetect={false}
      onResult={(result) => console.log('Result:', result)}
      onError={(error) => console.error('Error:', error)}
    />
  );
};
```

## Vue.js Integration

### Installation

```bash
npm install @pynomaly/js-sdk vue@^3.0.0
```

### Basic Setup with Provider

```vue
<template>
  <PynomalyProvider 
    :config="config" 
    :auto-connect="true"
    v-slot="{ client, isReady, isLoading, error }"
  >
    <div v-if="isLoading">Loading...</div>
    <div v-else-if="error">Error: {{ error.message }}</div>
    <div v-else-if="isReady">
      <YourAppContent :client="client" />
    </div>
  </PynomalyProvider>
</template>

<script setup lang="ts">
import { reactive } from 'vue';
import { PynomalyProvider } from '@pynomaly/js-sdk/vue';

const config = reactive({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.pynomaly.com'
});
</script>
```

### Using Composables

#### Client Composable

```vue
<script setup lang="ts">
import { usePynomalyClient } from '@pynomaly/js-sdk/vue';

const { client, isReady, isLoading, error, reconnect, disconnect } = usePynomalyClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.pynomaly.com',
  autoConnect: true
});
</script>
```

#### Authentication Composable

```vue
<script setup lang="ts">
import { usePynomalyAuth } from '@pynomaly/js-sdk/vue';

const {
  authState,
  isLoading,
  error,
  login,
  loginWithApiKey,
  logout,
  refreshToken
} = usePynomalyAuth({ client });

const handleLogin = async () => {
  try {
    await login({ 
      email: 'user@example.com', 
      password: 'password' 
    });
  } catch (err) {
    console.error('Login failed:', err);
  }
};
</script>
```

#### Anomaly Detection Composable

```vue
<script setup lang="ts">
import { useAnomalyDetection } from '@pynomaly/js-sdk/vue';

const {
  result,
  isLoading,
  error,
  progress,
  detectAnomalies,
  detectAnomaliesAsync,
  clear
} = useAnomalyDetection({ client });

const handleDetect = async () => {
  try {
    await detectAnomalies({
      data: [[1, 2], [3, 4], [100, 200]],
      algorithm: 'isolation_forest'
    });
  } catch (err) {
    console.error('Detection failed:', err);
  }
};
</script>
```

### Pre-built Components

```vue
<template>
  <AnomalyDetector
    algorithm="auto"
    :auto-detect="false"
    @result="handleResult"
    @error="handleError"
  />
</template>

<script setup lang="ts">
import { AnomalyDetector } from '@pynomaly/js-sdk/vue';

const handleResult = (result: any) => {
  console.log('Result:', result);
};

const handleError = (error: any) => {
  console.error('Error:', error);
};
</script>
```

## Angular Integration

### Installation

```bash
npm install @pynomaly/js-sdk @angular/core @angular/common
```

### Module Setup

```typescript
import { NgModule } from '@angular/core';
import { PynomalyModule } from '@pynomaly/js-sdk/angular';

@NgModule({
  imports: [
    PynomalyModule.forRoot({
      config: {
        apiKey: 'your-api-key',
        baseUrl: 'https://api.pynomaly.com'
      },
      autoConnect: true
    })
  ],
  // ... other module configuration
})
export class AppModule {}
```

### Using Services

#### Client Service

```typescript
import { Component, OnInit, OnDestroy } from '@angular/core';
import { Subject, takeUntil } from 'rxjs';
import { PynomalyService } from '@pynomaly/js-sdk/angular';

@Component({
  selector: 'app-client',
  template: `
    <div *ngIf="isLoading">Loading...</div>
    <div *ngIf="error">Error: {{ error.message }}</div>
    <div *ngIf="isReady">Pynomaly client is ready!</div>
  `
})
export class ClientComponent implements OnInit, OnDestroy {
  isReady = false;
  isLoading = false;
  error: Error | null = null;
  private destroy$ = new Subject<void>();

  constructor(private pynomalyService: PynomalyService) {}

  ngOnInit(): void {
    this.pynomalyService.isReady$
      .pipe(takeUntil(this.destroy$))
      .subscribe(ready => this.isReady = ready);

    this.pynomalyService.isLoading$
      .pipe(takeUntil(this.destroy$))
      .subscribe(loading => this.isLoading = loading);

    this.pynomalyService.error$
      .pipe(takeUntil(this.destroy$))
      .subscribe(error => this.error = error);

    // Initialize client
    this.pynomalyService.initialize({
      apiKey: 'your-api-key',
      baseUrl: 'https://api.pynomaly.com'
    }).subscribe();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
```

#### Authentication Service

```typescript
import { Component, OnInit, OnDestroy } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Subject, takeUntil } from 'rxjs';
import { PynomalyAuthService } from '@pynomaly/js-sdk/angular';

@Component({
  selector: 'app-auth',
  template: `
    <div *ngIf="!authState.isAuthenticated">
      <form [formGroup]="loginForm" (ngSubmit)="handleLogin()">
        <input formControlName="email" type="email" placeholder="Email" />
        <input formControlName="password" type="password" placeholder="Password" />
        <button type="submit" [disabled]="isLoading">
          {{ isLoading ? 'Logging in...' : 'Login' }}
        </button>
      </form>
    </div>
    <div *ngIf="authState.isAuthenticated">
      <p>Welcome, {{ authState.user?.email }}!</p>
      <button (click)="handleLogout()">Logout</button>
    </div>
  `
})
export class AuthComponent implements OnInit, OnDestroy {
  authState: any = { isAuthenticated: false };
  isLoading = false;
  loginForm: FormGroup;
  private destroy$ = new Subject<void>();

  constructor(
    private authService: PynomalyAuthService,
    private fb: FormBuilder
  ) {
    this.loginForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required]]
    });
  }

  ngOnInit(): void {
    this.authService.authState$
      .pipe(takeUntil(this.destroy$))
      .subscribe(state => this.authState = state);

    this.authService.isLoading$
      .pipe(takeUntil(this.destroy$))
      .subscribe(loading => this.isLoading = loading);
  }

  handleLogin(): void {
    if (this.loginForm.valid) {
      const { email, password } = this.loginForm.value;
      this.authService.login({ email, password }).subscribe();
    }
  }

  handleLogout(): void {
    this.authService.logout().subscribe();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
```

### Pre-built Components

```typescript
@Component({
  selector: 'app-detection',
  template: `
    <pynomaly-anomaly-detector
      algorithm="auto"
      [autoDetect]="false"
      (resultEvent)="handleResult($event)"
      (errorEvent)="handleError($event)"
    ></pynomaly-anomaly-detector>
  `
})
export class DetectionComponent {
  handleResult(result: any): void {
    console.log('Result:', result);
  }

  handleError(error: any): void {
    console.error('Error:', error);
  }
}
```

## Framework Comparison

| Feature | React | Vue 3 | Angular |
|---------|-------|-------|---------|
| **Setup Complexity** | Medium | Easy | Complex |
| **Bundle Size Impact** | Low | Low | Medium |
| **TypeScript Support** | Excellent | Excellent | Native |
| **State Management** | Hooks | Composables | Services + RxJS |
| **Component Architecture** | Functional | Composition API | Class/Functional |
| **Learning Curve** | Medium | Easy | Steep |
| **Performance** | Excellent | Excellent | Good |

### React
- **Pros**: Large ecosystem, flexible, excellent TypeScript support
- **Cons**: Requires understanding of hooks lifecycle
- **Best for**: Modern React applications with functional components

### Vue 3
- **Pros**: Easy to learn, excellent developer experience, Composition API
- **Cons**: Smaller ecosystem compared to React
- **Best for**: Rapid development, easy migration from Vue 2

### Angular
- **Pros**: Full framework, excellent tooling, enterprise-ready
- **Cons**: Steep learning curve, more complex setup
- **Best for**: Large enterprise applications, teams familiar with Angular

## Best Practices

### 1. Error Handling

```typescript
// React
const { client, error } = usePynomaly();

useEffect(() => {
  if (error) {
    // Log to monitoring service
    console.error('Pynomaly error:', error);
    // Show user-friendly message
    showNotification('Connection failed. Please try again.');
  }
}, [error]);
```

### 2. Loading States

```typescript
// Vue
const { isLoading, detectAnomalies } = useAnomalyDetection();

const handleDetect = async () => {
  try {
    // Show loading indicator automatically via isLoading
    await detectAnomalies(data);
  } catch (error) {
    // Handle error
  }
};
```

### 3. Authentication Flow

```typescript
// Angular
ngOnInit(): void {
  // Check for existing session
  this.authService.authState$.subscribe(state => {
    if (state.isAuthenticated) {
      this.router.navigate(['/dashboard']);
    }
  });
}
```

### 4. Memory Management

```typescript
// React
useEffect(() => {
  return () => {
    // Cleanup handled automatically by hooks
  };
}, []);

// Vue
onUnmounted(() => {
  // Cleanup handled automatically by composables
});

// Angular
ngOnDestroy(): void {
  this.destroy$.next();
  this.destroy$.complete();
}
```

### 5. Configuration Management

```typescript
// Environment-based configuration
const config = {
  apiKey: process.env.REACT_APP_PYNOMALY_API_KEY,
  baseUrl: process.env.REACT_APP_PYNOMALY_BASE_URL || 'https://api.pynomaly.com',
  debug: process.env.NODE_ENV === 'development'
};
```

### 6. Type Safety

```typescript
// Use provided TypeScript types
import type { 
  AnomalyDetectionResult, 
  DataQualityResult,
  PynomalyConfig 
} from '@pynomaly/js-sdk';

const config: PynomalyConfig = {
  apiKey: 'your-key',
  baseUrl: 'https://api.pynomaly.com'
};
```

### 7. Performance Optimization

```typescript
// React - Memoize expensive operations
const memoizedData = useMemo(() => 
  processLargeDataset(rawData), [rawData]
);

// Vue - Use computed for derived state
const processedData = computed(() => 
  processLargeDataset(rawData.value)
);

// Angular - Use OnPush change detection
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush
})
```

### 8. Testing

```typescript
// Mock the SDK for testing
jest.mock('@pynomaly/js-sdk', () => ({
  PynomalyClient: jest.fn().mockImplementation(() => ({
    detectAnomalies: jest.fn().mockResolvedValue(mockResult),
    // ... other methods
  }))
}));
```

## Migration Guide

### From Plain JavaScript

1. Install framework-specific package
2. Replace manual client instantiation with provider/service
3. Replace callbacks with reactive state management
4. Update error handling to use framework patterns

### Between Frameworks

The core SDK remains the same, only the integration layer changes:

1. Export data/state from old framework
2. Set up new framework integration
3. Import data/state into new framework
4. Update component patterns as needed

## Troubleshooting

### Common Issues

1. **Client not initializing**: Check API key and network connectivity
2. **Authentication failing**: Verify credentials and API endpoints
3. **Type errors**: Ensure latest SDK and framework types are installed
4. **Memory leaks**: Verify cleanup in component destruction lifecycle
5. **Bundle size**: Use tree shaking and import only needed components

### Debug Mode

Enable debug mode for additional logging:

```typescript
const config = {
  apiKey: 'your-key',
  baseUrl: 'https://api.pynomaly.com',
  debug: true // Enable detailed logging
};
```

This will log all SDK operations to the browser console for debugging.