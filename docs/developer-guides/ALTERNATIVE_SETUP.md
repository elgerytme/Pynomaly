# Alternative / Legacy Environment Setup

🍞 **Breadcrumb:** 🏠 [Home](../index.md)  👨‍💻 [Developer Guides](./README.md)  🔄 Alternative Setup

---

For cases where Docker is unavailable (such as enterprise environments with restricted permissions), use a virtual environment setup.

## Virtual Environment Setup

### 1. System Prerequisites
- **Python 3.11 or Newer**: Ensure Python is installed on your system.
- **Git**: Required for version control.
- **Node.js 16+**: Needed if working with the web UI.

### 2. Manual Environment Creation

**Create and Activate Virtual Environment**
```bash
mkdir -p environments/.venv
python3 -m venv environments/.venv
source environments/.venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3. Install Core Dependencies
```bash
pip install -r requirements.txt

# Includes all essential libraries for Pynomaly
pip install fastapi uvicorn httpx
```

### 4. Run Development Environment

**Local FastAPI Server**
```bash
PYTHONPATH="src" uvicorn pynomaly.presentation.api:app --reload
```

**CLI Interface**
```bash
PYTHONPATH="src" python -m pynomaly.presentation.cli --help
```

### 5. Testing Workflow

**Run Tests Without Docker**
```bash
# Unit Tests
pytest tests/unit/

# Integration Tests
pytest tests/integration/
```

### 6. Additional Tools

**Install Development Tools**
```bash
pip install mypy black ruff pre-commit

# Code Quality Checks
pip install bandit safety
```

### 7. Configuration Management

**Environment Variables**
Set relevant environment variables for local development in your terminal or in a `.env` file.
```bash
export PYNOMALY_ENVIRONMENT="development"
export PYNOMALY_LOG_LEVEL="DEBUG"
export PYNOMALY_DATABASE_URL="postgresql://localhost:5432/pynomaly_dev"
```

### 8. Known Issues

- **OS Compatibility**: Ensure all scripts and commands are adapted for execution on your specific operating system.
- **Package Dependencies**: Double-check dependency versions in `requirements.txt` for compatibility.

For further assistance and troubleshooting, refer to the main setup documentation or reach out to the maintainers.

---

**Last Updated**: 2025-07-08
**Environment**: Windows 10/11, macOS, Ubuntu (non-containerized)

