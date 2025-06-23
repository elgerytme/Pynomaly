"""Minimal setup.py for pip installation"""

from setuptools import setup, find_packages

# Read version from pyproject.toml (simplified approach)
import re
with open("pyproject.toml", "r") as f:
    content = f.read()
    version_match = re.search(r'version = "([^"]+)"', content)
    version = version_match.group(1) if version_match else "0.1.0"

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pynomaly",
    version=version,
    author="Pynomaly Team",
    author_email="team@pynomaly.io",
    description="State-of-the-art Python anomaly detection package with clean architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pynomaly/pynomaly",
    project_urls={
        "Bug Tracker": "https://github.com/pynomaly/pynomaly/issues",
        "Documentation": "https://pynomaly.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies (from requirements.txt)
        "pyod>=2.0.5",
        "scikit-learn>=1.5.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scipy>=1.11.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "structlog>=24.1.0",
        "dependency-injector>=4.41.0",
        "fastapi>=0.109.0",
        "typer[all]>=0.9.0",
        "uvicorn[standard]>=0.27.0",
        "httpx>=0.26.0",
        "rich>=13.7.0",
        "python-multipart>=0.0.6",
        "jinja2>=3.1.0",
        "aiofiles>=23.2.0",
        "pyarrow>=14.0.0",
        "opentelemetry-api>=1.22.0",
        "opentelemetry-sdk>=1.22.0",
        "opentelemetry-instrumentation-fastapi>=0.43b0",
        "prometheus-client>=0.19.0",
        "prometheus-fastapi-instrumentator>=5.9.1",
    ],
    extras_require={
        "torch": ["torch>=2.1.0"],
        "tensorflow": ["tensorflow>=2.15.0"],
        "jax": ["jax>=0.4.23", "jaxlib>=0.4.23"],
        "distributed": ["dask>=2024.1.0"],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.23.0",
            "mypy>=1.8.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pynomaly=pynomaly.presentation.cli.app:app",
        ],
    },
    include_package_data=True,
    package_data={
        "pynomaly": ["presentation/web/templates/**/*", "presentation/web/static/**/*"],
    },
)