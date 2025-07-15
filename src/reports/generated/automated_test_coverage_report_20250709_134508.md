# Automated Test Coverage Report

**Generated**: 20250709_134508
**Project**: .

## Overview

- **Total Source Files**: 639
- **Total Test Files**: 474
- **Overall Coverage Ratio**: 74.2%

## Coverage by Area

| Area | Coverage | Target | Status |
|------|----------|--------|--------|
| Core | 61.0% | 80% | ⚠️ |
| Sdk | 47.6% | 90% | ❌ |
| Cli | 24.0% | 60% | ❌ |
| Web_Api | 44.6% | 80% | ❌ |
| Web_Ui | 625.0% | 70% | ✅ |

## Coverage by Layer

| Layer | Coverage | Target | Status |
|-------|----------|--------|--------|
| Domain | 42.6% | 90% | ❌ |
| Application | 46.6% | 80% | ❌ |
| Infrastructure | 23.7% | 70% | ❌ |
| Presentation | 19.1% | 60% | ❌ |

## Critical Gaps

### Cli (Area)
- **Current Coverage**: 24.0%
- **Target Coverage**: 60%
- **Gap**: 36.0%
- **Description**: Command line interface
- **Recommendations**:
  - Create comprehensive command-specific tests
  - Add CLI workflow integration tests
  - Implement argument validation testing
  - Add help system validation

### Domain (Layer)
- **Current Coverage**: 42.6%
- **Target Coverage**: 90%
- **Gap**: 47.4%
- **Description**: Business logic and domain entities
- **Recommendations**:
  - Add comprehensive entity testing
  - Implement value object validation tests
  - Add domain service testing
  - Create business rule tests

## Prioritized Recommendations

### Critical Priority

- Area cli: Command line interface (Current: 24.0%, Target: 60%)
- Layer domain: Business logic and domain entities (Current: 42.6%, Target: 90%)

### High Priority

- Area core: Domain and application logic (Current: 61.0%, Target: 80%)
- Area web_api: Web API endpoints (Current: 44.6%, Target: 80%)
- Layer application: Application services and use cases (Current: 46.6%, Target: 80%)
- Layer infrastructure: External integrations and persistence (Current: 23.7%, Target: 70%)

### Medium Priority

- Area sdk: SDK and library interfaces (Current: 47.6%, Target: 90%)
- Layer presentation: APIs, CLI, and UI interfaces (Current: 19.1%, Target: 60%)

