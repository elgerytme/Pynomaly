# Automated Test Coverage Report

**Generated**: 20250709_155115
**Project**: /mnt/c/Users/andre/Pynomaly

## Overview

- **Total Source Files**: 675
- **Total Test Files**: 513
- **Overall Coverage Ratio**: 76.0%

## Coverage by Area

| Area | Coverage | Target | Status |
|------|----------|--------|--------|
| Core | 67.6% | 80% | ⚠️ |
| Sdk | 47.6% | 90% | ❌ |
| Cli | 34.5% | 60% | ❌ |
| Web_Api | 45.8% | 80% | ❌ |
| Web_Ui | 416.7% | 70% | ✅ |

## Coverage by Layer

| Layer | Coverage | Target | Status |
|-------|----------|--------|--------|
| Domain | 52.9% | 90% | ❌ |
| Application | 50.8% | 80% | ❌ |
| Infrastructure | 26.0% | 70% | ❌ |
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

- **Current Coverage**: 52.9%
- **Target Coverage**: 90%
- **Gap**: 37.1%
- **Description**: Business logic and domain entities
- **Recommendations**:
  - Add comprehensive entity testing
  - Implement value object validation tests
  - Add domain service testing
  - Create business rule tests

## Prioritized Recommendations

### Critical Priority

- Area cli: Command line interface (Current: 34.5%, Target: 60%)
- Layer domain: Business logic and domain entities (Current: 52.9%, Target: 90%)

### High Priority

- Area core: Domain and application logic (Current: 67.6%, Target: 80%)
- Area web_api: Web API endpoints (Current: 45.8%, Target: 80%)
- Layer application: Application services and use cases (Current: 50.8%, Target: 80%)
- Layer infrastructure: External integrations and persistence (Current: 26.0%, Target: 70%)

### Medium Priority

- Area sdk: SDK and library interfaces (Current: 47.6%, Target: 90%)
- Layer presentation: APIs, CLI, and UI interfaces (Current: 17.3%, Target: 60%)
