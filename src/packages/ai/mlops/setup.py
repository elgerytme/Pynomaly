from setuptools import setup, find_packages

setup(
    name="mlops",
    version="0.1.0",
    description="ML/MLOps Platform - MLOps Infrastructure",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "structlog>=23.0.0",
        "sqlalchemy>=2.0.0",
        "redis>=4.6.0",
        "prometheus-client>=0.17.0",
        "kafka-python>=2.0.2",
        "mlflow>=2.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
)