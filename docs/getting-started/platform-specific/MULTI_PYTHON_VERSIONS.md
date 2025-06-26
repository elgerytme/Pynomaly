# Multi-Version Python Testing - Version Information

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 🚀 [Getting Started](../README.md) > 🖥️ [Platform Setup](README.md) > 📄 Multi_Python_Versions

---


**Last Updated**: December 25, 2024

## 🐍 Supported Python Versions

This document provides the current Python versions tested by the Pynomaly project's multi-version testing infrastructure.

### Current Testing Matrix

| Branch | Specific Version | Status | Release Date | Notes |
|--------|------------------|--------|--------------|--------|
| **3.11.x** | 3.11.4 | ✅ Stable | March 2023 | Compatibility baseline |
| **3.11.x** | 3.11.9 | ✅ Stable | April 2024 | Latest 3.11.x |
| **3.12.x** | 3.12.8 | ✅ Stable | **Dec 3, 2024** | **Latest 3.12.x** |
| **3.13.x** | 3.13.1 | ✅ Stable | **Dec 3, 2024** | **Latest 3.13.x** |
| **3.14.x** | 3.14.0a3 | ⚠️ Alpha | **Dec 17, 2024** | **Development** |

## 📋 Version Status Details

### Python 3.12.8 (Latest Stable)
- **Released**: December 3, 2024
- **Type**: Security and maintenance release
- **Key Updates**:
  - 250+ bug fixes since 3.12.7
  - Security fixes including CVE-2024-50602
  - Upgraded libexpat to 2.6.4
  - Properly quoted template strings in venv activation scripts
- **Support Status**: Security fixes only until October 2028
- **Binary Installers**: No longer provided (source-only releases)

### Python 3.13.1 (Latest Stable)
- **Released**: December 3, 2024  
- **Type**: Bug fix release
- **Key Features** (from 3.13 series):
  - Just-in-Time (JIT) compiler (experimental)
  - Free-threaded mode without GIL (experimental)
  - Enhanced interactive interpreter
  - Colorized tracebacks
  - Multi-line editing and syntax highlighting
  - Performance improvements
- **Recommendation**: Primary version for new development

### Python 3.14.0a3 (Development)
- **Released**: December 17, 2024
- **Type**: Alpha 3 release
- **Development Status**: Active development
- **Upcoming Schedule**:
  - Alpha 4: January 14, 2025
  - Alpha 5: February 11, 2025
  - Alpha 6: March 14, 2025
  - Alpha 7: April 8, 2025
  - Beta 1: May 7, 2025
  - Final Release: October 7, 2025
- **Usage**: Testing only - not for production

## 🔧 Configuration Files Updated

The following files have been updated with the latest Python versions:

### Core Configuration
- **`.python-version`**: Default versions (3.11.9, 3.12.8, 3.13.1)
- **`.python-version-dev`**: All versions including alpha (3.11.4, 3.11.9, 3.12.8, 3.13.1, 3.14.0a3)

### Testing Infrastructure
- **`tox.ini`**: Multi-environment testing with specific versions
- **`.github/workflows/multi-python-testing.yml`**: CI/CD matrix testing
- **`deploy/docker/Dockerfile.multi-python`**: Docker multi-version container

### Scripts and Tools
- **`scripts/setup_multi_python.py`**: Environment setup with latest versions
- **`scripts/test_all_pythons.py`**: Multi-version test runner
- **`scripts/validate_multi_python.py`**: Validation framework

## 🚀 Quick Start with Latest Versions

```bash
# 1. Set up environments with latest versions
python3 scripts/setup_multi_python.py --install

# 2. Test specific latest versions
tox -e py311,py312,py313

# 3. Test with specific version targeting
tox -e py312-specific  # Tests 3.12.8 specifically
tox -e py313           # Tests 3.13.1 specifically

# 4. Run comprehensive validation
python3 scripts/validate_multi_python.py

# 5. Build Docker with latest versions
docker build -f deploy/docker/Dockerfile.multi-python -t pynomaly-multi-python .
```

## 📊 GitHub Actions Matrix

The CI/CD pipeline tests against:

### Matrix Strategy
- **Generic versions**: `"3.11"`, `"3.12"`, `"3.13"`, `"3.14-dev"`
- **Specific versions**: `"3.11.4"`, `"3.11.9"`, `"3.12.8"`, `"3.13.1"`, `"3.14.0a3"`
- **Operating Systems**: Ubuntu, Windows, macOS
- **Special handling**: 3.14-dev excluded on Windows/macOS due to instability

### Test Coverage
- ✅ **Basic functionality**: All versions
- ✅ **Security scanning**: All stable versions  
- ✅ **Performance testing**: 3.11.9, 3.12.8, 3.13.1
- ✅ **Type checking**: All stable versions
- ⚠️ **Advanced testing**: Limited on alpha versions

## 🔍 Version Selection Rationale

### Why These Specific Versions?

1. **Python 3.11.4**: Industry standard baseline for compatibility testing
2. **Python 3.11.9**: Latest stable in 3.11 series for reliability  
3. **Python 3.12.8**: Latest maintenance release with security fixes
4. **Python 3.13.1**: Latest stable with newest features and bug fixes
5. **Python 3.14.0a3**: Forward compatibility testing for upcoming features

### Version Update Policy

We update to the latest patch releases when they include:
- ✅ Security fixes
- ✅ Critical bug fixes  
- ✅ Performance improvements
- ❌ Breaking changes (avoided in patch releases)

## 📅 Update Schedule

### Automatic Updates
- **Security releases**: Updated within 1 week
- **Bug fix releases**: Updated within 2 weeks  
- **Feature releases**: Evaluated for compatibility

### Manual Review Required
- **Alpha/Beta versions**: Reviewed for stability
- **Release candidates**: Tested extensively
- **Breaking changes**: Full compatibility assessment

## 🔗 References

- [Python 3.12.8 Release Notes](https://www.python.org/downloads/release/python-3128/)
- [Python 3.13.1 Release Notes](https://www.python.org/downloads/release/python-3131/) 
- [Python 3.14 Development Schedule (PEP 745)](https://peps.python.org/pep-0745/)
- [What's New in Python 3.13](https://docs.python.org/3/whatsnew/3.13.html)

---

**Note**: This document is automatically updated when Python versions change in the testing infrastructure. Always refer to this file for the most current version information.