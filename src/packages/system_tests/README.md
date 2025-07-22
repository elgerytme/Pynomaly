# System Tests Package

This package contains system-level integration tests, end-to-end tests, and cross-package testing.

## Structure

- `conftest.py` - Global pytest configuration
- `data/` - Test data and domain-specific unit tests
- `e2e/` - End-to-end workflow tests
- `performance/` - Performance and benchmark tests  
- `security/` - Security-related tests

## Purpose

These tests validate the system as a whole rather than individual packages, ensuring proper integration between components.