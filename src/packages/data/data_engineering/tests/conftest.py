"""Pytest configuration for data engineering package testing."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock

# Import shared test utilities
from test_utilities.fixtures import *
from test_utilities.factories import *
from test_utilities.helpers import *
from test_utilities.markers import *

from data_engineering.domain.entities.data_pipeline import (
    DataPipeline, PipelineStep, PipelineStatus, StepStatus
)
from data_engineering.domain.entities.data_source import (
    DataSource, SourceType, ConnectionConfig, ConnectionStatus
)


@pytest.fixture
def sample_pipeline_step():
    """Sample pipeline step for testing."""
    return PipelineStep(
        name="extract_data",
        step_type="extract",
        config={"source": "database", "table": "users"}
    )


@pytest.fixture
def sample_data_pipeline():
    """Sample data pipeline for testing."""
    return DataPipeline(
        name="User Data ETL",
        description="Extract, transform, and load user data",
        created_by="data_engineer"
    )


@pytest.fixture
def sample_connection_config():
    """Sample connection configuration."""
    return ConnectionConfig(
        host="localhost",
        port=5432,
        username="testuser", 
        password="testpass",
        database="testdb",
        timeout_seconds=30
    )


@pytest.fixture
def sample_data_source(sample_connection_config):
    """Sample data source for testing."""
    return DataSource(
        name="Test Database",
        description="Test database for unit testing",
        source_type=SourceType.DATABASE,
        connection_config=sample_connection_config,
        created_by="data_engineer"
    )


@pytest.fixture
def pipeline_with_steps():
    """Pipeline with multiple steps for testing."""
    pipeline = DataPipeline(
        name="Complex Pipeline",
        description="Pipeline with multiple steps",
        created_by="engineer"
    )
    
    # Add steps with dependencies
    step1 = PipelineStep(name="extract", step_type="extract")
    step2 = PipelineStep(name="transform", step_type="transform", dependencies=[step1.id])
    step3 = PipelineStep(name="load", step_type="load", dependencies=[step2.id])
    
    pipeline.add_step(step1)
    pipeline.add_step(step2)
    pipeline.add_step(step3)
    
    return pipeline