# Automated Test Coverage Report

**Generated**: 20250709_160315
**Project**: /mnt/c/Users/andre/Pynomaly

## Overview

- **Total Source Files**: 677
- **Total Test Files**: 523
- **Overall Coverage Ratio**: 77.3%

## Coverage by Area

| Area | Coverage | Target | Status |
|------|----------|--------|--------|
| Core | 68.8% | 80% | ⚠️ |
| Sdk | 47.6% | 90% | ❌ |
| Cli | 34.5% | 60% | ❌ |
| Web_Api | 45.8% | 80% | ❌ |
| Web_Ui | 425.0% | 70% | ✅ |

## Coverage by Layer

| Layer | Coverage | Target | Status |
|-------|----------|--------|--------|
| Domain | 54.4% | 90% | ❌ |
| Application | 51.6% | 80% | ❌ |
| Infrastructure | 27.7% | 70% | ❌ |
| Presentation | 17.3% | 60% | ❌ |

## Critical Gaps

### Cli (Area)
- **Current Coverage**: 34.5%
- **Target Coverage**: 60%
- **Gap**: 25.5%
- **Description**: Command line interface
- **Recommendations**:
  - Create comprehensive command-specific tests
  - Add CLI workflow integration tests
  - Implement argument validation testing
  - Add help system validation

### Domain (Layer)
- **Current Coverage**: 54.4%
- **Target Coverage**: 90%
- **Gap**: 35.6%
- **Description**: Business logic and domain entities
- **Recommendations**:
  - Add comprehensive entity testing
  - Implement value object validation tests
  - Add domain service testing
  - Create business rule tests

## Prioritized Recommendations

### Critical Priority

- Area cli: Command line interface (Current: 34.5%, Target: 60%)
- Layer domain: Business logic and domain entities (Current: 54.4%, Target: 90%)

### High Priority

- Area core: Domain and application logic (Current: 68.8%, Target: 80%)
- Area web_api: Web API endpoints (Current: 45.8%, Target: 80%)
- Layer application: Application services and use cases (Current: 51.6%, Target: 80%)
- Layer infrastructure: External integrations and persistence (Current: 27.7%, Target: 70%)

### Medium Priority

- Area sdk: SDK and library interfaces (Current: 47.6%, Target: 90%)
- Layer presentation: APIs, CLI, and UI interfaces (Current: 17.3%, Target: 60%)

