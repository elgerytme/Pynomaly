#!/usr/bin/env python3
"""
Setup configuration for the anomaly_detection Python SDK.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text()
    if (this_directory / "README.md").exists()
    else ""
)

# Read version from __init__.py
version = {}
with open("anomaly_detection_client/__init__.py") as fp:
    exec(fp.read(), version)

setup(
    name="anomaly_detection-client",
    version=version["__version__"],
    author="Anomaly Detection Team",
    author_email="support@anomaly_detection.com",
    description="Official Python client library for the anomaly_detection anomaly detection API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anomaly_detection/anomaly_detection-python-sdk",
    project_urls={
        "Bug Tracker": "https://github.com/anomaly_detection/anomaly_detection-python-sdk/issues",
        "Documentation": "https://docs.anomaly_detection.com/sdk/python",
        "Source Code": "https://github.com/anomaly_detection/anomaly_detection-python-sdk",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="anomaly detection machine learning api client sdk",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "aiohttp>=3.8.0",
        "pydantic>=1.8.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=2.12.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.910",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "testing": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=2.12.0",
            "responses>=0.18.0",
            "aioresponses>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anomaly_detection-client=anomaly_detection_client.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "anomaly_detection_client": ["py.typed"],
    },
    zip_safe=False,
)
