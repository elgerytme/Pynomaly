# Click-based Modules Inventory

## Overview
This document provides a detailed inventory of all Click-based CLI modules identified for conversion to Typer.

**Total Modules:** 12  
**Total Size:** 318,585 bytes (~311 KB)  
**Backup Location:** `src/pynomaly/presentation/cli/_click_backup/`

## Module Details

### 1. alert.py
- **Size:** 25,331 bytes
- **Description:** Click-based alert management commands
- **Location:** `src/pynomaly/presentation/cli/alert.py`
- **Backup:** ✅ `_click_backup/alert.py`

### 2. benchmarking.py
- **Size:** 30,917 bytes
- **Description:** Click-based benchmarking commands
- **Location:** `src/pynomaly/presentation/cli/benchmarking.py`
- **Backup:** ✅ `_click_backup/benchmarking.py`

### 3. cost_optimization.py
- **Size:** 28,245 bytes
- **Description:** Click-based cost optimization commands
- **Location:** `src/pynomaly/presentation/cli/cost_optimization.py`
- **Backup:** ✅ `_click_backup/cost_optimization.py`

### 4. dashboard.py
- **Size:** 25,226 bytes
- **Description:** Click-based dashboard commands
- **Location:** `src/pynomaly/presentation/cli/dashboard.py`
- **Backup:** ✅ `_click_backup/dashboard.py`

### 5. enhanced_automl.py
- **Size:** 23,566 bytes
- **Description:** Click-based enhanced AutoML commands
- **Location:** `src/pynomaly/presentation/cli/enhanced_automl.py`
- **Backup:** ✅ `_click_backup/enhanced_automl.py`

### 6. ensemble.py
- **Size:** 24,337 bytes
- **Description:** Click-based ensemble methods commands
- **Location:** `src/pynomaly/presentation/cli/ensemble.py`
- **Backup:** ✅ `_click_backup/ensemble.py`

### 7. explain.py
- **Size:** 33,223 bytes
- **Description:** Click-based explanation/explainability commands
- **Location:** `src/pynomaly/presentation/cli/explain.py`
- **Backup:** ✅ `_click_backup/explain.py`

### 8. governance.py
- **Size:** 36,554 bytes (Largest module)
- **Description:** Click-based governance framework commands
- **Location:** `src/pynomaly/presentation/cli/governance.py`
- **Backup:** ✅ `_click_backup/governance.py`

### 9. quality.py
- **Size:** 17,572 bytes (Smallest module)
- **Description:** Click-based quality management commands
- **Location:** `src/pynomaly/presentation/cli/quality.py`
- **Backup:** ✅ `_click_backup/quality.py`

### 10. security.py
- **Size:** 28,482 bytes
- **Description:** Click-based security and compliance commands
- **Location:** `src/pynomaly/presentation/cli/security.py`
- **Backup:** ✅ `_click_backup/security.py`

### 11. tenant.py
- **Size:** 22,222 bytes
- **Description:** Click-based tenant management commands
- **Location:** `src/pynomaly/presentation/cli/tenant.py`
- **Backup:** ✅ `_click_backup/tenant.py`

### 12. training_automation_commands.py
- **Size:** 22,912 bytes
- **Description:** Click-based training automation commands
- **Location:** `src/pynomaly/presentation/cli/training_automation_commands.py`
- **Backup:** ✅ `_click_backup/training_automation_commands.py`

## Conversion Priority
Based on analysis and current app.py imports, suggest conversion order:

### High Priority (Currently disabled in app.py)
1. `security.py` - Referenced in app.py comments
2. `dashboard.py` - Referenced in app.py comments  
3. `governance.py` - Referenced in app.py comments

### Medium Priority (Larger/Complex modules)
4. `explain.py` - 33,223 bytes
5. `benchmarking.py` - 30,917 bytes
6. `cost_optimization.py` - 28,245 bytes
7. `security.py` - 28,482 bytes

### Lower Priority (Smaller/Standalone modules)
8. `alert.py`
9. `tenant.py`
10. `ensemble.py`
11. `enhanced_automl.py`
12. `training_automation_commands.py`
13. `quality.py` - Smallest module

## Backup Verification
All 12 modules have been successfully backed up to the `_click_backup` folder for diff reference during and after conversion.

## Notes
- These modules are currently using Click framework
- All modules need conversion to Typer to maintain consistency with the main application
- The backup serves as a reference for diff comparison post-conversion
- Some modules (security, dashboard, governance) are currently commented out in app.py
