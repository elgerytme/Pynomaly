# Running Pynomaly Without Poetry

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Getting Started](README.md) > 📄 Readme_Simple_Setup

---


This guide shows how to run Pynomaly using only Python and pip, without Poetry, Make, or Docker.

## Prerequisites

- Python 3.11 or higher
- pip (comes with Python)

## Quick Setup

### Option 1: Using the setup script

```bash
python scripts/setup_simple.py
```

This script will:
- Create a virtual environment
- Install all dependencies
- Set up Pynomaly in development mode
- Show you how to run the application

### Option 2: Manual setup

1. **Create a virtual environment:**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Install Pynomaly in development mode:**

```bash
pip install -e .
```

## Running the Application

### CLI Commands

After activation, you can use the CLI in several ways:

```bash
# Primary method (recommended after pip install -e .)
pynomaly --help

# Alternative methods
python scripts/cli.py --help
python -m pynomaly.presentation.cli.app --help
```

### Example CLI Usage

```bash
# Show all commands
pynomaly --help

# List available algorithms
pynomaly detector algorithms

# Create a detector
pynomaly detector create --name "My Detector" --algorithm IsolationForest

# Load a dataset
pynomaly dataset load data.csv --name "My Data"

# Start the web server
pynomaly server start
```

### API Server

Start the API server directly:

```bash
# Using uvicorn
python -m uvicorn pynomaly.presentation.api.app:app --reload

# Or using the CLI
python cli.py server start
```

The server will be available at:
- Web UI: http://localhost:8000
- API docs: http://localhost:8000/api/docs
- Health check: http://localhost:8000/api/health

### Python Script Usage

```python
# example.py
from pynomaly.infrastructure.config import create_container
from pynomaly.domain.entities import Detector, Dataset
import pandas as pd
import asyncio

async def main():
    # Initialize container
    container = create_container()

    # Create detector
    detector = Detector(
        name="IForest Detector",
        algorithm="IsolationForest",
        parameters={"contamination": 0.1}
    )
    container.detector_repository().save(detector)

    # Load dataset
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
        'feature2': [1, 2, 3, 4, 100]
    })
    dataset = Dataset(name="My Data", data=data)
    container.dataset_repository().save(dataset)

    # Train and detect
    detection_service = container.detection_service()
    result = await detection_service.train_and_detect(detector.id, dataset)

    print(f"Found {result.anomaly_count} anomalies")

# Run the example
asyncio.run(main())
```

## Entry Points Summary

| Component | Entry Point | Description |
|-----------|-------------|-------------|
| CLI | `/mnt/c/Users/andre/Pynomaly/cli.py` | Main CLI entry point |
| API Server | `/mnt/c/Users/andre/Pynomaly/src/pynomaly/presentation/api/app.py` | FastAPI application |
| CLI Module | `/mnt/c/Users/andre/Pynomaly/src/pynomaly/presentation/cli/app.py` | Typer CLI application |
| Web UI | Served by API at `/web/` | HTMX + Tailwind interface |

## Troubleshooting

### Import errors
Make sure you've activated the virtual environment and installed the package in development mode:

```bash
pip install -e .
```

### Port already in use
The default port is 8000. You can change it:

```bash
python cli.py server start --port 8001
```

### Missing dependencies
If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Without Virtual Environment (Not Recommended)

If you really want to run without a virtual environment:

```bash
# Install dependencies globally
pip install -r requirements.txt

# Run directly
python cli.py --help
```

⚠️ This is not recommended as it can conflict with other Python packages on your system.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
