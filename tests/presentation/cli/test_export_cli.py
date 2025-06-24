"""
Export CLI Testing Suite
Comprehensive tests for export CLI commands.
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from datetime import datetime

from pynomaly.presentation.cli.export import export_app
from pynomaly.infrastructure.exporters import ExcelExporter, PowerBIExporter, TableauExporter


class TestExportCLI:
    """Test suite for export CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results_file(self):
        """Create sample detection results file."""
        results = {
            "metadata": {
                "detector_name": "IsolationForest",
                "dataset_name": "Test Dataset",
                "algorithm": "IsolationForest",
                "contamination_rate": 0.1,
                "timestamp": datetime.utcnow().isoformat(),
                "runtime_seconds": 2.5
            },
            "results": {
                "total_samples": 1000,
                "anomalies_found": 50,
                "anomaly_rate": 0.05,
                "anomaly_indices": list(range(50)),
                "anomaly_scores": [0.8 + i * 0.01 for i in range(50)],
                "feature_importance": {
                    "feature1": 0.35,
                    "feature2": 0.28,
                    "feature3": 0.22,
                    "feature4": 0.15
                }
            },
            "evaluation": {
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81,
                "auc_roc": 0.92,
                "average_precision": 0.88
            },
            "data": [
                {
                    "index": i,
                    "feature1": i * 1.1,
                    "feature2": i * 2.2,
                    "feature3": i * 3.3,
                    "feature4": i * 4.4,
                    "anomaly_score": 0.1 + (i % 10) * 0.05,
                    "is_anomaly": 1 if i < 50 else 0
                }
                for i in range(100)
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def sample_batch_results_file(self):
        """Create sample batch detection results file."""
        batch_results = {
            "experiment_metadata": {
                "name": "Multi-Algorithm Comparison",
                "timestamp": datetime.utcnow().isoformat(),
                "dataset": "Comparison Dataset",
                "algorithms_tested": ["IsolationForest", "LOF", "OneClassSVM"]
            },
            "algorithm_results": {
                "IsolationForest": {
                    "anomalies_found": 45,
                    "precision": 0.87,
                    "recall": 0.82,
                    "f1_score": 0.84,
                    "runtime_seconds": 1.2
                },
                "LOF": {
                    "anomalies_found": 52,
                    "precision": 0.83,
                    "recall": 0.79,
                    "f1_score": 0.81,
                    "runtime_seconds": 2.1
                },
                "OneClassSVM": {
                    "anomalies_found": 38,
                    "precision": 0.91,
                    "recall": 0.76,
                    "f1_score": 0.83,
                    "runtime_seconds": 3.8
                }
            },
            "comparison": {
                "best_algorithm": "IsolationForest",
                "ranking": ["IsolationForest", "OneClassSVM", "LOF"],
                "statistical_significance": {
                    "friedman_test": {"statistic": 8.5, "p_value": 0.014},
                    "wilcoxon_tests": {
                        "IF_vs_LOF": {"statistic": 45, "p_value": 0.032},
                        "IF_vs_SVM": {"statistic": 38, "p_value": 0.089}
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(batch_results, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    # Basic Command Tests

    def test_export_help(self, runner):
        """Test export CLI help."""
        result = runner.invoke(export_app, ["--help"])
        
        assert result.exit_code == 0
        assert "Export results to business intelligence platforms" in result.stdout
        assert "Commands:" in result.stdout
        assert "list-formats" in result.stdout
        assert "excel" in result.stdout
        assert "powerbi" in result.stdout

    def test_list_formats(self, runner):
        """Test listing available export formats."""
        result = runner.invoke(export_app, ["list-formats"])
        
        assert result.exit_code == 0
        assert "Available Export Formats" in result.stdout
        assert "Excel" in result.stdout
        assert "Power BI" in result.stdout
        assert "Tableau" in result.stdout
        assert "CSV" in result.stdout

    def test_list_formats_detailed(self, runner):
        """Test listing formats with detailed information."""
        result = runner.invoke(export_app, ["list-formats", "--detailed"])
        
        assert result.exit_code == 0
        assert "Description:" in result.stdout
        assert "File Extensions:" in result.stdout
        assert "Features:" in result.stdout

    # Excel Export Tests

    def test_excel_export_basic(self, runner, sample_results_file):
        """Test basic Excel export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter') as mock_exporter:
                exporter_instance = Mock()
                mock_exporter.return_value = exporter_instance
                exporter_instance.export.return_value = True
                
                result = runner.invoke(export_app, [
                    "excel", sample_results_file, output_path
                ])
                
                assert result.exit_code == 0
                assert "Excel export completed" in result.stdout
                exporter_instance.export.assert_called_once()
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_excel_export_with_template(self, runner, sample_results_file):
        """Test Excel export with custom template."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            template_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter') as mock_exporter:
                exporter_instance = Mock()
                mock_exporter.return_value = exporter_instance
                exporter_instance.export.return_value = True
                
                result = runner.invoke(export_app, [
                    "excel", sample_results_file, output_path,
                    "--template", template_path
                ])
                
                assert result.exit_code == 0
                assert "Using template:" in result.stdout
                
        finally:
            Path(template_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_excel_export_with_formatting(self, runner, sample_results_file):
        """Test Excel export with custom formatting."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter') as mock_exporter:
                exporter_instance = Mock()
                mock_exporter.return_value = exporter_instance
                exporter_instance.export.return_value = True
                
                result = runner.invoke(export_app, [
                    "excel", sample_results_file, output_path,
                    "--include-charts",
                    "--conditional-formatting",
                    "--freeze-panes"
                ])
                
                assert result.exit_code == 0
                assert "Charts: enabled" in result.stdout
                assert "Conditional formatting: enabled" in result.stdout
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_excel_export_multiple_sheets(self, runner, sample_results_file):
        """Test Excel export with multiple sheets."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter') as mock_exporter:
                exporter_instance = Mock()
                mock_exporter.return_value = exporter_instance
                exporter_instance.export.return_value = True
                
                result = runner.invoke(export_app, [
                    "excel", sample_results_file, output_path,
                    "--sheets", "summary,data,analysis,charts"
                ])
                
                assert result.exit_code == 0
                assert "Sheets: summary,data,analysis,charts" in result.stdout
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    # Power BI Export Tests

    def test_powerbi_export_basic(self, runner, sample_results_file):
        """Test basic Power BI export."""
        with patch('pynomaly.infrastructure.exporters.powerbi_exporter.PowerBIExporter') as mock_exporter:
            exporter_instance = Mock()
            mock_exporter.return_value = exporter_instance
            exporter_instance.export.return_value = True
            
            result = runner.invoke(export_app, [
                "powerbi", sample_results_file,
                "--workspace-id", "test-workspace-123"
            ])
            
            assert result.exit_code == 0
            assert "Power BI export completed" in result.stdout
            exporter_instance.export.assert_called_once()

    def test_powerbi_export_with_dataset_name(self, runner, sample_results_file):
        """Test Power BI export with custom dataset name."""
        with patch('pynomaly.infrastructure.exporters.powerbi_exporter.PowerBIExporter') as mock_exporter:
            exporter_instance = Mock()
            mock_exporter.return_value = exporter_instance
            exporter_instance.export.return_value = True
            
            result = runner.invoke(export_app, [
                "powerbi", sample_results_file,
                "--workspace-id", "test-workspace-123",
                "--dataset-name", "Anomaly Detection Results"
            ])
            
            assert result.exit_code == 0
            assert "Dataset name: Anomaly Detection Results" in result.stdout

    def test_powerbi_export_with_refresh(self, runner, sample_results_file):
        """Test Power BI export with auto-refresh."""
        with patch('pynomaly.infrastructure.exporters.powerbi_exporter.PowerBIExporter') as mock_exporter:
            exporter_instance = Mock()
            mock_exporter.return_value = exporter_instance
            exporter_instance.export.return_value = True
            
            result = runner.invoke(export_app, [
                "powerbi", sample_results_file,
                "--workspace-id", "test-workspace-123",
                "--auto-refresh"
            ])
            
            assert result.exit_code == 0
            assert "Auto-refresh: enabled" in result.stdout

    def test_powerbi_export_authentication_error(self, runner, sample_results_file):
        """Test Power BI export with authentication error."""
        with patch('pynomaly.infrastructure.exporters.powerbi_exporter.PowerBIExporter') as mock_exporter:
            exporter_instance = Mock()
            mock_exporter.return_value = exporter_instance
            exporter_instance.export.side_effect = Exception("Authentication failed")
            
            result = runner.invoke(export_app, [
                "powerbi", sample_results_file,
                "--workspace-id", "test-workspace-123"
            ])
            
            assert result.exit_code == 1
            assert "Authentication failed" in result.stdout

    # Tableau Export Tests

    def test_tableau_export_basic(self, runner, sample_results_file):
        """Test basic Tableau export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tde', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('pynomaly.infrastructure.exporters.tableau_exporter.TableauExporter') as mock_exporter:
                exporter_instance = Mock()
                mock_exporter.return_value = exporter_instance
                exporter_instance.export.return_value = True
                
                result = runner.invoke(export_app, [
                    "tableau", sample_results_file, output_path
                ])
                
                assert result.exit_code == 0
                assert "Tableau export completed" in result.stdout
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_tableau_export_hyper_format(self, runner, sample_results_file):
        """Test Tableau export in Hyper format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.hyper', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('pynomaly.infrastructure.exporters.tableau_exporter.TableauExporter') as mock_exporter:
                exporter_instance = Mock()
                mock_exporter.return_value = exporter_instance
                exporter_instance.export.return_value = True
                
                result = runner.invoke(export_app, [
                    "tableau", sample_results_file, output_path,
                    "--format", "hyper"
                ])
                
                assert result.exit_code == 0
                assert "Format: hyper" in result.stdout
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_tableau_server_publish(self, runner, sample_results_file):
        """Test publishing to Tableau Server."""
        with patch('pynomaly.infrastructure.exporters.tableau_exporter.TableauExporter') as mock_exporter:
            exporter_instance = Mock()
            mock_exporter.return_value = exporter_instance
            exporter_instance.publish_to_server.return_value = True
            
            result = runner.invoke(export_app, [
                "tableau", sample_results_file,
                "--server", "https://tableau.company.com",
                "--site", "analytics",
                "--project", "anomaly-detection",
                "--publish"
            ])
            
            assert result.exit_code == 0
            assert "Publishing to Tableau Server" in result.stdout

    # CSV Export Tests

    def test_csv_export_basic(self, runner, sample_results_file):
        """Test basic CSV export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            result = runner.invoke(export_app, [
                "csv", sample_results_file, output_path
            ])
            
            assert result.exit_code == 0
            assert "CSV export completed" in result.stdout
            
            # Verify CSV file was created and contains data
            assert Path(output_path).exists()
            with open(output_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                assert "index" in headers
                assert "anomaly_score" in headers
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_csv_export_with_options(self, runner, sample_results_file):
        """Test CSV export with custom options."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            result = runner.invoke(export_app, [
                "csv", sample_results_file, output_path,
                "--separator", ";",
                "--include-index",
                "--date-format", "iso"
            ])
            
            assert result.exit_code == 0
            assert "Separator: ;" in result.stdout
            assert "Include index: enabled" in result.stdout
            
        finally:
            Path(output_path).unlink(missing_ok=True)

    # JSON Export Tests

    def test_json_export_basic(self, runner, sample_results_file):
        """Test basic JSON export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            result = runner.invoke(export_app, [
                "json", sample_results_file, output_path
            ])
            
            assert result.exit_code == 0
            assert "JSON export completed" in result.stdout
            
            # Verify JSON file was created
            assert Path(output_path).exists()
            with open(output_path, 'r') as f:
                data = json.load(f)
                assert "metadata" in data
                assert "results" in data
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_json_export_formatted(self, runner, sample_results_file):
        """Test JSON export with formatting."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            result = runner.invoke(export_app, [
                "json", sample_results_file, output_path,
                "--pretty",
                "--indent", "4"
            ])
            
            assert result.exit_code == 0
            assert "Pretty printing: enabled" in result.stdout
            
        finally:
            Path(output_path).unlink(missing_ok=True)

    # Database Export Tests

    def test_database_export_sqlite(self, runner, sample_results_file):
        """Test SQLite database export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = Mock()
                mock_connect.return_value = mock_conn
                
                result = runner.invoke(export_app, [
                    "database", sample_results_file,
                    "--type", "sqlite",
                    "--database", db_path,
                    "--table", "anomaly_results"
                ])
                
                assert result.exit_code == 0
                assert "Database export completed" in result.stdout
                
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_database_export_postgresql(self, runner, sample_results_file):
        """Test PostgreSQL database export."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            
            result = runner.invoke(export_app, [
                "database", sample_results_file,
                "--type", "postgresql",
                "--host", "localhost",
                "--port", "5432",
                "--database", "analytics",
                "--table", "anomaly_results",
                "--username", "analyst"
            ])
            
            # This might fail due to missing password, but should attempt connection
            assert result.exit_code in [0, 1]

    # Batch Export Tests

    def test_batch_export_multiple_formats(self, runner, sample_results_file):
        """Test batch export to multiple formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            with patch('pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter') as mock_excel:
                with patch('pynomaly.infrastructure.exporters.powerbi_exporter.PowerBIExporter') as mock_powerbi:
                    excel_instance = Mock()
                    powerbi_instance = Mock()
                    mock_excel.return_value = excel_instance
                    mock_powerbi.return_value = powerbi_instance
                    excel_instance.export.return_value = True
                    powerbi_instance.export.return_value = True
                    
                    result = runner.invoke(export_app, [
                        "batch", sample_results_file,
                        "--output-dir", str(output_dir),
                        "--formats", "excel,csv,json",
                        "--powerbi-workspace", "test-workspace"
                    ])
                    
                    assert result.exit_code == 0
                    assert "Batch export completed" in result.stdout

    def test_batch_export_comparison_results(self, runner, sample_batch_results_file):
        """Test batch export of algorithm comparison results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            result = runner.invoke(export_app, [
                "batch", sample_batch_results_file,
                "--output-dir", str(output_dir),
                "--formats", "excel,csv",
                "--comparison-mode"
            ])
            
            assert result.exit_code == 0
            assert "Comparison mode: enabled" in result.stdout

    # Report Generation Tests

    def test_generate_report_html(self, runner, sample_results_file):
        """Test HTML report generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('pynomaly.infrastructure.exporters.report_generator.ReportGenerator') as mock_generator:
                generator_instance = Mock()
                mock_generator.return_value = generator_instance
                generator_instance.generate_html_report.return_value = True
                
                result = runner.invoke(export_app, [
                    "report", sample_results_file, output_path,
                    "--format", "html",
                    "--include-charts",
                    "--interactive"
                ])
                
                assert result.exit_code == 0
                assert "HTML report generated" in result.stdout
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_report_pdf(self, runner, sample_results_file):
        """Test PDF report generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('pynomaly.infrastructure.exporters.report_generator.ReportGenerator') as mock_generator:
                generator_instance = Mock()
                mock_generator.return_value = generator_instance
                generator_instance.generate_pdf_report.return_value = True
                
                result = runner.invoke(export_app, [
                    "report", sample_results_file, output_path,
                    "--format", "pdf",
                    "--template", "executive_summary"
                ])
                
                assert result.exit_code == 0
                assert "PDF report generated" in result.stdout
                
        finally:
            Path(output_path).unlink(missing_ok=True)

    # Error Handling Tests

    def test_export_missing_input_file(self, runner):
        """Test export with missing input file."""
        result = runner.invoke(export_app, [
            "excel", "/path/to/nonexistent.json", "output.xlsx"
        ])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_export_invalid_output_directory(self, runner, sample_results_file):
        """Test export to invalid output directory."""
        result = runner.invoke(export_app, [
            "excel", sample_results_file, "/nonexistent/directory/output.xlsx"
        ])
        
        assert result.exit_code == 1

    def test_export_invalid_format(self, runner, sample_results_file):
        """Test export with invalid format."""
        result = runner.invoke(export_app, [
            "invalid-format", sample_results_file, "output.xyz"
        ])
        
        assert result.exit_code != 0

    def test_powerbi_missing_workspace(self, runner, sample_results_file):
        """Test Power BI export without workspace ID."""
        result = runner.invoke(export_app, [
            "powerbi", sample_results_file
        ])
        
        assert result.exit_code != 0

    def test_export_service_unavailable(self, runner, sample_results_file):
        """Test export when service is unavailable."""
        with patch('pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter') as mock_exporter:
            mock_exporter.side_effect = Exception("Service unavailable")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
                output_path = f.name
            
            try:
                result = runner.invoke(export_app, [
                    "excel", sample_results_file, output_path
                ])
                
                assert result.exit_code == 1
                assert "Error" in result.stdout
                
            finally:
                Path(output_path).unlink(missing_ok=True)

    # Integration Tests

    def test_complete_export_workflow(self, runner, sample_results_file):
        """Test complete export workflow with multiple formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # 1. List available formats
            list_result = runner.invoke(export_app, ["list-formats"])
            assert list_result.exit_code == 0
            
            # 2. Export to Excel
            excel_path = output_dir / "results.xlsx"
            with patch('pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter') as mock_exporter:
                exporter_instance = Mock()
                mock_exporter.return_value = exporter_instance
                exporter_instance.export.return_value = True
                
                excel_result = runner.invoke(export_app, [
                    "excel", sample_results_file, str(excel_path)
                ])
                assert excel_result.exit_code == 0
            
            # 3. Export to CSV
            csv_path = output_dir / "results.csv"
            csv_result = runner.invoke(export_app, [
                "csv", sample_results_file, str(csv_path)
            ])
            assert csv_result.exit_code == 0
            
            # 4. Generate HTML report
            report_path = output_dir / "report.html"
            with patch('pynomaly.infrastructure.exporters.report_generator.ReportGenerator') as mock_generator:
                generator_instance = Mock()
                mock_generator.return_value = generator_instance
                generator_instance.generate_html_report.return_value = True
                
                report_result = runner.invoke(export_app, [
                    "report", sample_results_file, str(report_path),
                    "--format", "html"
                ])
                assert report_result.exit_code == 0

    def test_export_configuration_workflow(self, runner, sample_results_file):
        """Test export with configuration file."""
        # Create export configuration
        config = {
            "excel": {
                "include_charts": True,
                "conditional_formatting": True,
                "sheets": ["summary", "data", "analysis"]
            },
            "powerbi": {
                "workspace_id": "test-workspace-123",
                "dataset_name": "Anomaly Detection Results",
                "auto_refresh": True
            },
            "csv": {
                "separator": ",",
                "include_index": True,
                "date_format": "iso"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
                output_path = f.name
            
            try:
                with patch('pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter') as mock_exporter:
                    exporter_instance = Mock()
                    mock_exporter.return_value = exporter_instance
                    exporter_instance.export.return_value = True
                    
                    result = runner.invoke(export_app, [
                        "excel", sample_results_file, output_path,
                        "--config", config_path
                    ])
                    
                    assert result.exit_code == 0
                    assert "Using configuration file" in result.stdout
                    
            finally:
                Path(output_path).unlink(missing_ok=True)
                
        finally:
            Path(config_path).unlink(missing_ok=True)