# Automated Test Coverage Report

**Generated**: 20250709_162210
**Project**: /mnt/c/Users/andre/anomaly_detection

## Overview

- **Total Source Files**: 688
- **Total Test Files**: 546
- **Overall Coverage Ratio**: 79.4%

## Coverage by Area

| Area | Coverage | Target | Status |
|------|----------|--------|--------|
| Core | 73.0% | 80% | ⚠️ |
| Sdk | 47.6% | 90% | ❌ |
| Cli | 33.3% | 60% | ❌ |
| Web_Api | 45.2% | 80% | ❌ |
| Web_Ui | 441.7% | 70% | ✅ |

## Coverage by Layer

| Layer | Coverage | Target | Status |
|-------|----------|--------|--------|
| Domain | 54.8% | 90% | ❌ |
| Application | 57.7% | 80% | ⚠️ |
| Infrastructure | 28.4% | 70% | ❌ |
| Presentation | 16.7% | 60% | ❌ |

## Critical Gaps

### Cli (Area)
- **Current Coverage**: 33.3%
- **Target Coverage**: 60%
- **Gap**: 26.7%
- **Description**: Command line interface
- **Recommendations**:
  - Create comprehensive command-specific tests
  - Add CLI workflow integration tests
  - Implement argument validation testing
  - Add help system validation

### Domain (Layer)
- **Current Coverage**: 54.8%
- **Target Coverage**: 90%
- **Gap**: 35.2%
- **Description**: Business logic and domain entities
- **Recommendations**:
  - Add comprehensive entity testing
  - Implement value object validation tests
  - Add domain service testing
  - Create business rule tests

## Prioritized Recommendations

### Critical Priority

- Area cli: Command line interface (Current: 33.3%, Target: 60%)
- Layer domain: Business logic and domain entities (Current: 54.8%, Target: 90%)

### High Priority

- Area core: Domain and application logic (Current: 73.0%, Target: 80%)
- Area web_api: Web API endpoints (Current: 45.2%, Target: 80%)
- Layer application: Application services and use cases (Current: 57.7%, Target: 80%)
- Layer infrastructure: External integrations and persistence (Current: 28.4%, Target: 70%)

### Medium Priority

- Area sdk: SDK and library interfaces (Current: 47.6%, Target: 90%)
- Layer presentation: APIs, CLI, and UI interfaces (Current: 16.7%, Target: 60%)

