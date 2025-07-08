# Environment Build Matrix Test Report

**Test Date:** 2025-07-08T14:01:50.275850

**Total Tests:** 7
**Successful:** 3
**Failed:** 4

**Success Rate:** 42.9%

## Python 3.11

- **minimal**: SUCCESS
- **server**: SUCCESS
- **production**: SUCCESS
- **deep**: FAILED
  - Errors: ERROR: Exception:; pip._vendor.tomli.TOMLDecodeError: Illegal character '\x00' (at line 34, column 11)
- **deep-cpu**: FAILED
  - Errors: ERROR: Exception:; pip._vendor.tomli.TOMLDecodeError: Illegal character '\x00' (at line 34, column 11)
- **automl**: FAILED
  - Errors: ERROR: Exception:; pip._vendor.tomli.TOMLDecodeError: Illegal character '\x00' (at line 34, column 11)
- **all**: FAILED
  - Errors: ERROR: Exception:; pip._vendor.tomli.TOMLDecodeError: Illegal character '\x00' (at line 34, column 11)

