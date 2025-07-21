/**
 * Angular module for Pynomaly SDK
 */

import { NgModule, ModuleWithProviders } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule } from '@angular/forms';
import { PynomalyConfig } from '../../index';

// Services
import { PynomalyService } from './services/pynomaly.service';
import { PynomalyAuthService } from './services/pynomaly-auth.service';

// Components
import { AnomalyDetectorComponent } from './components/anomaly-detector.component';

// Configuration token
export const PYNOMALY_CONFIG = 'PYNOMALY_CONFIG';

export interface PynomalyModuleConfig {
  config: PynomalyConfig;
  autoConnect?: boolean;
}

@NgModule({
  declarations: [
    AnomalyDetectorComponent
  ],
  imports: [
    CommonModule,
    ReactiveFormsModule
  ],
  exports: [
    AnomalyDetectorComponent
  ],
  providers: [
    PynomalyService,
    PynomalyAuthService
  ]
})
export class PynomalyModule {
  static forRoot(moduleConfig: PynomalyModuleConfig): ModuleWithProviders<PynomalyModule> {
    return {
      ngModule: PynomalyModule,
      providers: [
        {
          provide: PYNOMALY_CONFIG,
          useValue: moduleConfig
        },
        PynomalyService,
        PynomalyAuthService
      ]
    };
  }

  constructor(
    private pynomalyService: PynomalyService,
    private authService: PynomalyAuthService
  ) {
    // Auto-initialize if config is provided
    // This would need to be handled through injection token in a real implementation
  }
}