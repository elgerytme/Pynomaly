"""Tests for Export Options DTO."""

from monorepo.application.dto.export_options import (
    ExportDestination,
    ExportFormat,
    ExportOptions,
)


class TestExportFormat:
    """Test suite for ExportFormat enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ExportFormat.EXCEL.value == "excel"
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.PARQUET.value == "parquet"

    def test_enum_membership(self):
        """Test enum membership."""
        assert ExportFormat.EXCEL in ExportFormat
        assert ExportFormat.CSV in ExportFormat
        assert ExportFormat.JSON in ExportFormat
        assert ExportFormat.PARQUET in ExportFormat

    def test_enum_iteration(self):
        """Test enum iteration."""
        formats = list(ExportFormat)
        assert len(formats) == 4
        assert ExportFormat.EXCEL in formats
        assert ExportFormat.CSV in formats
        assert ExportFormat.JSON in formats
        assert ExportFormat.PARQUET in formats


class TestExportDestination:
    """Test suite for ExportDestination enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ExportDestination.LOCAL_FILE.value == "local_file"
        assert ExportDestination.CLOUD_STORAGE.value == "cloud_storage"
        assert ExportDestination.API_ENDPOINT.value == "api_endpoint"
        assert ExportDestination.EMAIL.value == "email"

    def test_enum_membership(self):
        """Test enum membership."""
        assert ExportDestination.LOCAL_FILE in ExportDestination
        assert ExportDestination.CLOUD_STORAGE in ExportDestination
        assert ExportDestination.API_ENDPOINT in ExportDestination
        assert ExportDestination.EMAIL in ExportDestination

    def test_enum_iteration(self):
        """Test enum iteration."""
        destinations = list(ExportDestination)
        assert len(destinations) == 4
        assert ExportDestination.LOCAL_FILE in destinations
        assert ExportDestination.CLOUD_STORAGE in destinations
        assert ExportDestination.API_ENDPOINT in destinations
        assert ExportDestination.EMAIL in destinations


class TestExportOptions:
    """Test suite for ExportOptions dataclass."""

    def test_default_creation(self):
        """Test creating export options with default values."""
        options = ExportOptions()

        # Basic export settings
        assert options.format == ExportFormat.EXCEL
        assert options.destination == ExportDestination.LOCAL_FILE
        assert options.include_charts is True
        assert options.include_summary is True
        assert options.include_metadata is True
        assert options.use_advanced_formatting is True

        # Excel-specific options
        assert options.create_multiple_sheets is True
        assert options.highlight_anomalies is True
        assert options.add_conditional_formatting is True
        assert options.include_formulas is False
        assert options.sheet_names == ["Results", "Summary", "Charts", "Metadata"]

        # Additional format options
        assert options.parquet_compression == "snappy"
        assert options.json_indent == 2

        # Data filtering and selection
        assert options.include_normal_samples is True
        assert options.include_anomaly_samples is True
        assert options.max_samples is None
        assert options.sample_columns is None

        # Visualization options
        assert options.chart_types == ["scatter", "histogram"]
        assert options.color_scheme == "default"
        assert options.chart_size == (640, 480)

        # Authentication and security
        assert options.credentials is None
        assert options.api_key is None
        assert options.oauth_token is None

        # Performance options
        assert options.batch_size == 1000
        assert options.compression is False
        assert options.parallel_export is False

        # Notification options
        assert options.notify_on_completion is False
        assert options.notification_emails is None
        assert options.webhook_url is None

        # Custom options
        assert options.custom_options == {}

    def test_custom_creation(self):
        """Test creating export options with custom values."""
        custom_sheet_names = ["Data", "Analysis", "Plots"]
        custom_chart_types = ["line", "bar"]
        custom_chart_size = (800, 600)
        custom_credentials = {"username": "test", "password": "secret"}
        custom_options = {"extra_feature": True, "debug": False}

        options = ExportOptions(
            format=ExportFormat.JSON,
            destination=ExportDestination.CLOUD_STORAGE,
            include_charts=False,
            include_summary=False,
            include_metadata=False,
            use_advanced_formatting=False,
            create_multiple_sheets=False,
            highlight_anomalies=False,
            add_conditional_formatting=False,
            include_formulas=True,
            sheet_names=custom_sheet_names,
            parquet_compression="gzip",
            json_indent=4,
            include_normal_samples=False,
            include_anomaly_samples=True,
            max_samples=5000,
            sample_columns=["feature1", "feature2", "anomaly_score"],
            chart_types=custom_chart_types,
            color_scheme="dark",
            chart_size=custom_chart_size,
            credentials=custom_credentials,
            api_key="test_api_key",
            oauth_token="test_oauth_token",
            batch_size=2000,
            compression=True,
            parallel_export=True,
            notify_on_completion=True,
            notification_emails=["admin@example.com", "user@example.com"],
            webhook_url="https://example.com/webhook",
            custom_options=custom_options,
        )

        assert options.format == ExportFormat.JSON
        assert options.destination == ExportDestination.CLOUD_STORAGE
        assert options.include_charts is False
        assert options.include_summary is False
        assert options.include_metadata is False
        assert options.use_advanced_formatting is False
        assert options.create_multiple_sheets is False
        assert options.highlight_anomalies is False
        assert options.add_conditional_formatting is False
        assert options.include_formulas is True
        assert options.sheet_names == custom_sheet_names
        assert options.parquet_compression == "gzip"
        assert options.json_indent == 4
        assert options.include_normal_samples is False
        assert options.include_anomaly_samples is True
        assert options.max_samples == 5000
        assert options.sample_columns == ["feature1", "feature2", "anomaly_score"]
        assert options.chart_types == custom_chart_types
        assert options.color_scheme == "dark"
        assert options.chart_size == custom_chart_size
        assert options.credentials == custom_credentials
        assert options.api_key == "test_api_key"
        assert options.oauth_token == "test_oauth_token"
        assert options.batch_size == 2000
        assert options.compression is True
        assert options.parallel_export is True
        assert options.notify_on_completion is True
        assert options.notification_emails == ["admin@example.com", "user@example.com"]
        assert options.webhook_url == "https://example.com/webhook"
        assert options.custom_options == custom_options

    def test_post_init_validation(self):
        """Test post-initialization validation."""
        # Test sheet names default setting
        options = ExportOptions(sheet_names=None)
        assert options.sheet_names == ["Results", "Summary", "Charts", "Metadata"]

        # Test chart types validation
        options = ExportOptions(
            chart_types=["scatter", "invalid_chart", "histogram", "another_invalid"]
        )
        assert options.chart_types == ["scatter", "histogram"]

        # Test parquet compression validation
        options = ExportOptions(parquet_compression="invalid_compression")
        assert options.parquet_compression == "snappy"

        # Test valid parquet compression
        options = ExportOptions(parquet_compression="gzip")
        assert options.parquet_compression == "gzip"

    def test_valid_chart_types(self):
        """Test valid chart types."""
        valid_charts = ["scatter", "histogram", "line", "bar", "pie"]

        for chart_type in valid_charts:
            options = ExportOptions(chart_types=[chart_type])
            assert chart_type in options.chart_types

    def test_invalid_chart_types_filtered(self):
        """Test that invalid chart types are filtered out."""
        options = ExportOptions(
            chart_types=["scatter", "invalid", "histogram", "another_invalid", "line"]
        )
        assert options.chart_types == ["scatter", "histogram", "line"]

    def test_valid_parquet_compression(self):
        """Test valid parquet compression options."""
        valid_compressions = ["snappy", "gzip", "brotli", "lz4", "zstd"]

        for compression in valid_compressions:
            options = ExportOptions(parquet_compression=compression)
            assert options.parquet_compression == compression

    def test_to_dict(self):
        """Test converting to dictionary."""
        options = ExportOptions(
            format=ExportFormat.CSV,
            destination=ExportDestination.EMAIL,
            include_charts=False,
            max_samples=1000,
            chart_types=["bar", "pie"],
            custom_options={"test": "value"},
        )

        result = options.to_dict()

        assert result["format"] == "csv"
        assert result["destination"] == "email"
        assert result["include_charts"] is False
        assert result["max_samples"] == 1000
        assert result["chart_types"] == ["bar", "pie"]
        assert result["custom_options"] == {"test": "value"}

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "format": "parquet",
            "destination": "cloud_storage",
            "include_charts": True,
            "max_samples": 2000,
            "chart_types": ["scatter", "line"],
            "parquet_compression": "brotli",
            "custom_options": {"feature": "enabled"},
        }

        options = ExportOptions.from_dict(data)

        assert options.format == ExportFormat.PARQUET
        assert options.destination == ExportDestination.CLOUD_STORAGE
        assert options.include_charts is True
        assert options.max_samples == 2000
        assert options.chart_types == ["scatter", "line"]
        assert options.parquet_compression == "brotli"
        assert options.custom_options == {"feature": "enabled"}

    def test_from_dict_with_enum_objects(self):
        """Test creating from dictionary with enum objects."""
        data = {
            "format": ExportFormat.JSON,
            "destination": ExportDestination.API_ENDPOINT,
            "json_indent": 4,
            "api_key": "test_key",
        }

        options = ExportOptions.from_dict(data)

        assert options.format == ExportFormat.JSON
        assert options.destination == ExportDestination.API_ENDPOINT
        assert options.json_indent == 4
        assert options.api_key == "test_key"

    def test_for_excel(self):
        """Test Excel-optimized configuration."""
        options = ExportOptions()
        result = options.for_excel()

        assert result is options  # Should return self
        assert options.format == ExportFormat.EXCEL
        assert options.use_advanced_formatting is True
        assert options.highlight_anomalies is True
        assert options.add_conditional_formatting is True
        assert options.include_charts is True

    def test_for_csv(self):
        """Test CSV-optimized configuration."""
        options = ExportOptions()
        result = options.for_csv()

        assert result is options  # Should return self
        assert options.format == ExportFormat.CSV
        assert options.include_charts is False
        assert options.use_advanced_formatting is False

    def test_for_json(self):
        """Test JSON-optimized configuration."""
        options = ExportOptions()
        result = options.for_json()

        assert result is options  # Should return self
        assert options.format == ExportFormat.JSON
        assert options.include_charts is False
        assert options.use_advanced_formatting is False
        assert options.json_indent == 2

    def test_for_parquet(self):
        """Test Parquet-optimized configuration."""
        options = ExportOptions()
        result = options.for_parquet()

        assert result is options  # Should return self
        assert options.format == ExportFormat.PARQUET
        assert options.include_charts is False
        assert options.use_advanced_formatting is False
        assert options.parquet_compression == "snappy"

    def test_format_optimization_chaining(self):
        """Test that format optimization methods can be chained."""
        options = ExportOptions()

        # Test chaining
        result = options.for_excel().for_csv().for_json()

        assert result is options
        assert options.format == ExportFormat.JSON
        assert options.include_charts is False
        assert options.use_advanced_formatting is False

    def test_excel_specific_options(self):
        """Test Excel-specific options."""
        options = ExportOptions(
            create_multiple_sheets=False,
            highlight_anomalies=False,
            add_conditional_formatting=False,
            include_formulas=True,
            sheet_names=["Custom", "Sheet", "Names"],
        )

        assert options.create_multiple_sheets is False
        assert options.highlight_anomalies is False
        assert options.add_conditional_formatting is False
        assert options.include_formulas is True
        assert options.sheet_names == ["Custom", "Sheet", "Names"]

    def test_data_filtering_options(self):
        """Test data filtering and selection options."""
        options = ExportOptions(
            include_normal_samples=False,
            include_anomaly_samples=True,
            max_samples=10000,
            sample_columns=["id", "timestamp", "value", "anomaly_score"],
        )

        assert options.include_normal_samples is False
        assert options.include_anomaly_samples is True
        assert options.max_samples == 10000
        assert options.sample_columns == ["id", "timestamp", "value", "anomaly_score"]

    def test_visualization_options(self):
        """Test visualization options."""
        options = ExportOptions(
            chart_types=["scatter", "line", "bar"],
            color_scheme="viridis",
            chart_size=(1024, 768),
        )

        assert options.chart_types == ["scatter", "line", "bar"]
        assert options.color_scheme == "viridis"
        assert options.chart_size == (1024, 768)

    def test_authentication_options(self):
        """Test authentication and security options."""
        credentials = {"username": "test_user", "password": "secret_pass"}

        options = ExportOptions(
            credentials=credentials,
            api_key="api_key_123",
            oauth_token="oauth_token_456",
        )

        assert options.credentials == credentials
        assert options.api_key == "api_key_123"
        assert options.oauth_token == "oauth_token_456"

    def test_performance_options(self):
        """Test performance options."""
        options = ExportOptions(batch_size=5000, compression=True, parallel_export=True)

        assert options.batch_size == 5000
        assert options.compression is True
        assert options.parallel_export is True

    def test_notification_options(self):
        """Test notification options."""
        emails = ["admin@company.com", "user@company.com"]
        webhook = "https://hooks.company.com/export"

        options = ExportOptions(
            notify_on_completion=True, notification_emails=emails, webhook_url=webhook
        )

        assert options.notify_on_completion is True
        assert options.notification_emails == emails
        assert options.webhook_url == webhook

    def test_custom_options(self):
        """Test custom options for extensibility."""
        custom_opts = {
            "custom_feature_1": True,
            "custom_feature_2": "value",
            "custom_feature_3": 42,
            "nested_config": {"sub_option": "sub_value"},
        }

        options = ExportOptions(custom_options=custom_opts)

        assert options.custom_options == custom_opts
        assert options.custom_options["custom_feature_1"] is True
        assert options.custom_options["custom_feature_2"] == "value"
        assert options.custom_options["custom_feature_3"] == 42
        assert options.custom_options["nested_config"]["sub_option"] == "sub_value"

    def test_json_indent_options(self):
        """Test JSON indent options."""
        # Test with indent
        options = ExportOptions(json_indent=4)
        assert options.json_indent == 4

        # Test with no indent
        options = ExportOptions(json_indent=None)
        assert options.json_indent is None

    def test_email_list_handling(self):
        """Test email list handling."""
        # Single email
        options = ExportOptions(notification_emails=["single@example.com"])
        assert options.notification_emails == ["single@example.com"]

        # Multiple emails
        emails = ["first@example.com", "second@example.com", "third@example.com"]
        options = ExportOptions(notification_emails=emails)
        assert options.notification_emails == emails

        # Empty list
        options = ExportOptions(notification_emails=[])
        assert options.notification_emails == []

    def test_batch_size_values(self):
        """Test different batch size values."""
        # Small batch size
        options = ExportOptions(batch_size=100)
        assert options.batch_size == 100

        # Large batch size
        options = ExportOptions(batch_size=50000)
        assert options.batch_size == 50000

        # Default batch size
        options = ExportOptions()
        assert options.batch_size == 1000

    def test_chart_size_tuples(self):
        """Test chart size tuple handling."""
        # Standard sizes
        options = ExportOptions(chart_size=(800, 600))
        assert options.chart_size == (800, 600)

        # Square chart
        options = ExportOptions(chart_size=(600, 600))
        assert options.chart_size == (600, 600)

        # Large chart
        options = ExportOptions(chart_size=(1920, 1080))
        assert options.chart_size == (1920, 1080)

    def test_none_values(self):
        """Test handling of None values."""
        options = ExportOptions(
            max_samples=None,
            sample_columns=None,
            credentials=None,
            api_key=None,
            oauth_token=None,
            notification_emails=None,
            webhook_url=None,
        )

        assert options.max_samples is None
        assert options.sample_columns is None
        assert options.credentials is None
        assert options.api_key is None
        assert options.oauth_token is None
        assert options.notification_emails is None
        assert options.webhook_url is None

    def test_boolean_flags(self):
        """Test boolean flag combinations."""
        # All flags enabled
        options = ExportOptions(
            include_charts=True,
            include_summary=True,
            include_metadata=True,
            use_advanced_formatting=True,
            create_multiple_sheets=True,
            highlight_anomalies=True,
            add_conditional_formatting=True,
            include_formulas=True,
            include_normal_samples=True,
            include_anomaly_samples=True,
            compression=True,
            parallel_export=True,
            notify_on_completion=True,
        )

        assert options.include_charts is True
        assert options.include_summary is True
        assert options.include_metadata is True
        assert options.use_advanced_formatting is True
        assert options.create_multiple_sheets is True
        assert options.highlight_anomalies is True
        assert options.add_conditional_formatting is True
        assert options.include_formulas is True
        assert options.include_normal_samples is True
        assert options.include_anomaly_samples is True
        assert options.compression is True
        assert options.parallel_export is True
        assert options.notify_on_completion is True

        # All flags disabled
        options = ExportOptions(
            include_charts=False,
            include_summary=False,
            include_metadata=False,
            use_advanced_formatting=False,
            create_multiple_sheets=False,
            highlight_anomalies=False,
            add_conditional_formatting=False,
            include_formulas=False,
            include_normal_samples=False,
            include_anomaly_samples=False,
            compression=False,
            parallel_export=False,
            notify_on_completion=False,
        )

        assert options.include_charts is False
        assert options.include_summary is False
        assert options.include_metadata is False
        assert options.use_advanced_formatting is False
        assert options.create_multiple_sheets is False
        assert options.highlight_anomalies is False
        assert options.add_conditional_formatting is False
        assert options.include_formulas is False
        assert options.include_normal_samples is False
        assert options.include_anomaly_samples is False
        assert options.compression is False
        assert options.parallel_export is False
        assert options.notify_on_completion is False


class TestExportOptionsIntegration:
    """Test integration scenarios for ExportOptions."""

    def test_excel_export_configuration(self):
        """Test complete Excel export configuration."""
        options = ExportOptions().for_excel()

        options.sheet_names = ["Anomalies", "Normal_Data", "Summary", "Charts"]
        options.max_samples = 10000
        options.include_normal_samples = True
        options.include_anomaly_samples = True
        options.chart_types = ["scatter", "histogram", "bar"]
        options.color_scheme = "tableau"
        options.chart_size = (1024, 768)
        options.notify_on_completion = True
        options.notification_emails = ["analyst@company.com"]

        # Verify Excel-specific configuration
        assert options.format == ExportFormat.EXCEL
        assert options.use_advanced_formatting is True
        assert options.highlight_anomalies is True
        assert options.add_conditional_formatting is True
        assert options.include_charts is True

        # Verify custom configuration
        assert options.sheet_names == ["Anomalies", "Normal_Data", "Summary", "Charts"]
        assert options.max_samples == 10000
        assert options.chart_types == ["scatter", "histogram", "bar"]
        assert options.color_scheme == "tableau"
        assert options.chart_size == (1024, 768)
        assert options.notify_on_completion is True
        assert options.notification_emails == ["analyst@company.com"]

    def test_cloud_storage_export_configuration(self):
        """Test cloud storage export configuration."""
        options = ExportOptions(
            format=ExportFormat.PARQUET,
            destination=ExportDestination.CLOUD_STORAGE,
            parquet_compression="brotli",
            batch_size=50000,
            parallel_export=True,
            compression=True,
            credentials={
                "access_key": "AWS_ACCESS_KEY",
                "secret_key": "AWS_SECRET_KEY",
                "bucket": "anomaly-detection-exports",
                "region": "us-east-1",
            },
            custom_options={
                "s3_path": "exports/anomalies/",
                "metadata_format": "json",
                "encryption": "AES256",
            },
        )

        assert options.format == ExportFormat.PARQUET
        assert options.destination == ExportDestination.CLOUD_STORAGE
        assert options.parquet_compression == "brotli"
        assert options.batch_size == 50000
        assert options.parallel_export is True
        assert options.compression is True
        assert options.credentials["bucket"] == "anomaly-detection-exports"
        assert options.custom_options["s3_path"] == "exports/anomalies/"

    def test_api_endpoint_export_configuration(self):
        """Test API endpoint export configuration."""
        options = ExportOptions(
            format=ExportFormat.JSON,
            destination=ExportDestination.API_ENDPOINT,
            json_indent=None,  # Compact JSON for API
            api_key="api_key_12345",
            batch_size=1000,
            include_charts=False,
            include_metadata=True,
            max_samples=50000,
            sample_columns=[
                "timestamp",
                "feature_vector",
                "anomaly_score",
                "prediction",
            ],
            webhook_url="https://api.company.com/anomaly-results",
            custom_options={
                "endpoint_url": "https://api.company.com/upload",
                "timeout": 30,
                "retry_count": 3,
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": "Pynomaly-Exporter/1.0",
                },
            },
        )

        assert options.format == ExportFormat.JSON
        assert options.destination == ExportDestination.API_ENDPOINT
        assert options.json_indent is None
        assert options.api_key == "api_key_12345"
        assert options.batch_size == 1000
        assert options.include_charts is False
        assert options.include_metadata is True
        assert options.max_samples == 50000
        assert options.sample_columns == [
            "timestamp",
            "feature_vector",
            "anomaly_score",
            "prediction",
        ]
        assert options.webhook_url == "https://api.company.com/anomaly-results"
        assert (
            options.custom_options["endpoint_url"] == "https://api.company.com/upload"
        )
        assert options.custom_options["timeout"] == 30

    def test_email_export_configuration(self):
        """Test email export configuration."""
        options = ExportOptions(
            format=ExportFormat.CSV,
            destination=ExportDestination.EMAIL,
            include_charts=False,
            include_summary=True,
            include_metadata=False,
            max_samples=1000,
            include_normal_samples=False,
            include_anomaly_samples=True,
            notify_on_completion=True,
            notification_emails=[
                "security@company.com",
                "ops@company.com",
                "data-science@company.com",
            ],
            custom_options={
                "subject": "Anomaly Detection Results - {timestamp}",
                "body_template": "automated_report.html",
                "attachment_name": "anomaly_results_{date}.csv",
                "smtp_server": "smtp.company.com",
                "smtp_port": 587,
                "use_tls": True,
            },
        )

        assert options.format == ExportFormat.CSV
        assert options.destination == ExportDestination.EMAIL
        assert options.include_charts is False
        assert options.include_summary is True
        assert options.include_metadata is False
        assert options.max_samples == 1000
        assert options.include_normal_samples is False
        assert options.include_anomaly_samples is True
        assert options.notify_on_completion is True
        assert len(options.notification_emails) == 3
        assert "security@company.com" in options.notification_emails
        assert (
            options.custom_options["subject"]
            == "Anomaly Detection Results - {timestamp}"
        )
        assert options.custom_options["use_tls"] is True

    def test_high_performance_export_configuration(self):
        """Test high-performance export configuration."""
        options = ExportOptions(
            format=ExportFormat.PARQUET,
            destination=ExportDestination.LOCAL_FILE,
            parquet_compression="zstd",
            batch_size=100000,
            parallel_export=True,
            compression=True,
            include_charts=False,
            include_summary=False,
            include_metadata=False,
            use_advanced_formatting=False,
            max_samples=None,  # No limit
            custom_options={
                "output_directory": "/fast_storage/exports/",
                "partition_by": "date",
                "max_file_size_mb": 500,
                "thread_count": 8,
                "memory_limit_gb": 16,
            },
        )

        assert options.format == ExportFormat.PARQUET
        assert options.destination == ExportDestination.LOCAL_FILE
        assert options.parquet_compression == "zstd"
        assert options.batch_size == 100000
        assert options.parallel_export is True
        assert options.compression is True
        assert options.include_charts is False
        assert options.include_summary is False
        assert options.include_metadata is False
        assert options.use_advanced_formatting is False
        assert options.max_samples is None
        assert options.custom_options["thread_count"] == 8
        assert options.custom_options["memory_limit_gb"] == 16

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original_options = ExportOptions(
            format=ExportFormat.JSON,
            destination=ExportDestination.CLOUD_STORAGE,
            include_charts=True,
            max_samples=5000,
            chart_types=["scatter", "line"],
            credentials={"key": "value"},
            custom_options={"feature": "enabled"},
        )

        # Serialize to dictionary
        options_dict = original_options.to_dict()

        # Deserialize from dictionary
        restored_options = ExportOptions.from_dict(options_dict)

        # Verify all fields match
        assert restored_options.format == original_options.format
        assert restored_options.destination == original_options.destination
        assert restored_options.include_charts == original_options.include_charts
        assert restored_options.max_samples == original_options.max_samples
        assert restored_options.chart_types == original_options.chart_types
        assert restored_options.credentials == original_options.credentials
        assert restored_options.custom_options == original_options.custom_options

    def test_format_specific_optimizations(self):
        """Test format-specific optimizations."""
        # Start with base options
        base_options = ExportOptions(
            include_charts=True,
            use_advanced_formatting=True,
            batch_size=2000,
            notify_on_completion=True,
        )

        # Test Excel optimization
        excel_options = ExportOptions(**base_options.__dict__).for_excel()
        assert excel_options.format == ExportFormat.EXCEL
        assert excel_options.include_charts is True
        assert excel_options.use_advanced_formatting is True
        assert excel_options.highlight_anomalies is True

        # Test CSV optimization
        csv_options = ExportOptions(**base_options.__dict__).for_csv()
        assert csv_options.format == ExportFormat.CSV
        assert csv_options.include_charts is False
        assert csv_options.use_advanced_formatting is False
        assert csv_options.batch_size == 2000  # Should preserve other settings

        # Test JSON optimization
        json_options = ExportOptions(**base_options.__dict__).for_json()
        assert json_options.format == ExportFormat.JSON
        assert json_options.include_charts is False
        assert json_options.use_advanced_formatting is False
        assert json_options.json_indent == 2

        # Test Parquet optimization
        parquet_options = ExportOptions(**base_options.__dict__).for_parquet()
        assert parquet_options.format == ExportFormat.PARQUET
        assert parquet_options.include_charts is False
        assert parquet_options.use_advanced_formatting is False
        assert parquet_options.parquet_compression == "snappy"
