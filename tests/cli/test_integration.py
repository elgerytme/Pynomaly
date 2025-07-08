"""
CLI Integration Testing for Pynomaly
Tests commands including `--help`, detector creation, dataset loading, training, and anomaly detection.
"""

import tempfile
from typer.testing import CliRunner
from pynomaly.presentation.cli.app import app
import pytest


@pytest.fixture

def runner():
    """Create a CLI test runner for the Pynomaly application."""
    return CliRunner()

def test_help_command(runner):
    """Test the CLI --help command for general availability."""
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'Commands:' in result.stdout


def test_detector_creation(runner):
    """Test creation of a detector."""
    result = runner.invoke(app, ['detector', 'create', '--name', 'new_detector', '--algorithm', 'IsolationForest'])
    assert result.exit_code == 0
    assert 'Detector created' in result.stdout


def test_dataset_loading(runner):
    """Test loading a dataset."""
    with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
        result = runner.invoke(app, ['dataset', 'load', temp_file.name, '--name', 'test_dataset'])
        assert result.exit_code == 0
        assert 'Dataset loaded' in result.stdout


def test_training_detector(runner):
    """Test training a detector with a dataset."""
    result = runner.invoke(app, ['detect', 'train', '--detector', 'new_detector', '--dataset', 'test_dataset'])
    assert result.exit_code == 0
    assert 'Training completed' in result.stdout


def test_anomaly_detection(runner):
    """Test running anomaly detection."""
    result = runner.invoke(app, ['detect', 'run', '--detector', 'new_detector', '--dataset', 'test_dataset'])
    assert result.exit_code == 0
    assert 'Anomalies detected' in result.stdout


def test_invalid_input_handling(runner):
    """Test handling of invalid input gracefully."""
    result = runner.invoke(app, ['detector', 'create', '--name'])
    assert result.exit_code != 0
    assert 'Error' in result.stdout
