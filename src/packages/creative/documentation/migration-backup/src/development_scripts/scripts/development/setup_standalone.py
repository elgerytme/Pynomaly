"""Standalone setup.py for pip installation without pyproject.toml conflicts"""

from setuptools import find_packages, setup

# Version
version = "0.1.0"

# Read long description from README
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = (
        "State-of-the-art Python anomaly detection package with clean architecture"
    )

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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    zip_safe=False,
    install_requires=[
        # Core dependencies only (matches requirements.txt)
        "pyod>=2.0.5",
        "numpy>=1.26.0,<2.2.0",
        "pandas>=2.3.0",
        "polars>=0.20.0",
        "pydantic>=2.9.0",
        "structlog>=24.4.0",
        "dependency-injector>=4.41.0",
    ],
    extras_require={
        # Core functionality extras (match pyproject.toml)
        "minimal": ["scikit-learn>=1.5.0", "scipy>=1.11.0"],
        "api": [
            "fastapi>=0.115.0",
            "uvicorn[standard]>=0.32.0",
            "httpx>=0.28.0",
            "requests>=2.31.0",
            "python-multipart>=0.0.18",
            "jinja2>=3.1.0",
            "aiofiles>=23.2.0",
            "pydantic-settings>=2.1.0",
        ],
        "cli": ["typer[all]>=0.9.0", "rich>=13.7.0"],
        "server": [
            "fastapi>=0.115.0",
            "uvicorn[standard]>=0.32.0",
            "httpx>=0.28.0",
            "requests>=2.31.0",
            "python-multipart>=0.0.18",
            "jinja2>=3.1.0",
            "aiofiles>=23.2.0",
            "pydantic-settings>=2.1.0",
            "typer[all]>=0.9.0",
            "rich>=13.7.0",
            "scikit-learn>=1.5.0",
            "scipy>=1.11.0",
            "pyarrow>=17.0.0",
        ],
        "production": [
            "fastapi>=0.115.0",
            "uvicorn[standard]>=0.32.0",
            "redis>=5.1.0",
            "opentelemetry-api>=1.30.0",
            "opentelemetry-sdk>=1.30.0",
            "opentelemetry-instrumentation-fastapi>=0.51b0",
            "prometheus-client>=0.19.0",
            "psutil>=5.9.0",
            "tenacity>=8.2.0",
            "circuitbreaker>=1.4.0",
            "pydantic-settings>=2.1.0",
            "pyjwt>=2.8.0",
            "passlib[bcrypt]>=1.7.4",
        ],
        # ML backends
        "torch": ["torch>=2.1.0"],
        "tensorflow": ["tensorflow>=2.18.0,<2.20.0", "keras>=3.8.0"],
        "jax": ["jax>=0.4.23", "jaxlib>=0.4.23", "optax>=0.1.7"],
        # Specialized ML
        "graph": ["pygod>=1.1.0", "torch-geometric>=2.4.0"],
        "automl": [
            "optuna>=3.5.0",
            "hyperopt>=0.2.7",
            "auto-sklearn2>=1.0.0",
            "scikit-learn>=1.5.0",
        ],
        "explainability": ["shap>=0.42.0", "lime>=0.2.0"],
        # Data processing
        "data-formats": [
            "pyarrow>=17.0.0",
            "fastparquet>=2024.2.0",
            "openpyxl>=3.1.0",
            "xlsxwriter>=3.1.0",
            "h5py>=3.9.0",
        ],
        "database": ["sqlalchemy>=2.0.0", "psycopg2-binary>=2.9.0"],
        "spark": ["pyspark>=3.5.0"],
        # Development
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
        "pynomaly": [
            "presentation/web/templates/**/*",
            "presentation/web/static/**/*",
            "presentation/web/static/css/*",
            "presentation/web/static/js/*",
            "presentation/web/static/img/*",
            "presentation/web/static/manifest.json",
            "presentation/web/static/sw.js",
        ],
    },
)
