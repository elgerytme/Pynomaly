# Automated Test Coverage Report

**Generated**: 20250709_152706
**Project**: /mnt/c/Users/andre/Pynomaly

## Overview

- **Total Source Files**: 661
- **Total Test Files**: 499
- **Overall Coverage Ratio**: 75.5%

## Coverage by Area

| Area | Coverage | Target | Status |
|------|----------|--------|--------|
| Core | 65.8% | 80% | ⚠️ |
| Sdk | 47.6% | 90% | ❌ |
| Cli | 32.1% | 60% | ❌ |
| Web_Api | 45.8% | 80% | ❌ |
| Web_Ui | 500.0% | 70% | ✅ |

## Coverage by Layer

| Layer | Coverage | Target | Status |
|-------|----------|--------|--------|
| Domain | 51.9% | 90% | ❌ |
| Application | 48.4% | 80% | ❌ |
| Infrastructure | 23.5% | 70% | ❌ |
| Presentation | 17.9% | 60% | ❌ |

## Critical Gaps

### Cli (Area)
- **Current Coverage**: 32.1%
- **Target Coverage**: 60%
- **Gap**: 27.9%
- **Description**: Command line interface
- **Recommendations**:
  - Create comprehensive command-specific tests
  - Add CLI workflow integration tests
  - Implement argument validation testing
  - Add help system validation

### Domain (Layer)
- **Current Coverage**: 51.9%
- **Target Coverage**: 90%
- **Gap**: 38.1%
- **Description**: Business logic and domain entities
- **Recommendations**:
  - Add comprehensive entity testing
  - Implement value object validation tests
  - Add domain service testing
  - Create business rule tests

## Prioritized Recommendations

### Critical Priority

- Area cli: Command line interface (Current: 32.1%, Target: 60%)
- Layer domain: Business logic and domain entities (Current: 51.9%, Target: 90%)

### High Priority

- Area core: Domain and application logic (Current: 65.8%, Target: 80%)
- Area web_api: Web API endpoints (Current: 45.8%, Target: 80%)
- Layer application: Application services and use cases (Current: 48.4%, Target: 80%)
- Layer infrastructure: External integrations and persistence (Current: 23.5%, Target: 70%)

### Medium Priority

- Area sdk: SDK and library interfaces (Current: 47.6%, Target: 90%)
- Layer presentation: APIs, CLI, and UI interfaces (Current: 17.9%, Target: 60%)

