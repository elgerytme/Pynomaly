"""
Unit tests for StreamProcessor domain entities.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from enterprise_scalability.domain.entities.stream_processor import (
    StreamSource, StreamSink, ProcessingWindow, StreamProcessor,
    StreamType, ProcessorStatus, ProcessingMode, WindowType
)


class TestStreamSource:
    """Test cases for StreamSource entity."""
    
    def test_stream_source_creation_basic(self):
        """Test basic stream source creation."""
        source = StreamSource(
            name="user-events-source",
            stream_type=StreamType.KAFKA,
            connection_string="localhost:9092",
            consumer_group="analytics-group"
        )
        
        assert isinstance(source.id, UUID)
        assert source.name == "user-events-source"
        assert source.stream_type == StreamType.KAFKA
        assert source.description == ""
        assert source.connection_string == "localhost:9092"
        assert source.topics == []
        assert source.partition_key is None
        assert source.data_format == "json"
        assert source.schema_registry_url is None
        assert source.schema_id is None
        assert source.consumer_group == "analytics-group"
        assert source.max_poll_records == 1000
        assert source.poll_timeout_ms == 5000
        assert source.auto_offset_reset == "latest"
        assert source.enable_auto_commit is False
        assert source.batch_size == 100
        assert source.buffer_size == 10000
        assert source.security_protocol == "PLAINTEXT"
        assert source.auth_config == {}
        assert source.metrics_enabled is True
        assert source.health_check_interval_seconds == 30
        assert source.tags == {}
        assert isinstance(source.created_at, datetime)
        assert isinstance(source.updated_at, datetime)
        
    def test_stream_source_creation_comprehensive(self):
        """Test comprehensive stream source creation."""
        topics = ["user-events", "page-views", "clicks"]
        auth_config = {"username": "kafka-user", "password": "secret"}
        tags = {"environment": "production", "team": "analytics"}
        
        source = StreamSource(
            name="kafka-analytics-source",
            stream_type=StreamType.KAFKA,
            description="Kafka source for analytics events",
            connection_string="kafka-cluster:9092",
            topics=topics,
            partition_key="user_id",
            data_format="avro",
            schema_registry_url="http://schema-registry:8081",
            schema_id="user-events-v1",
            consumer_group="analytics-consumers",
            max_poll_records=500,
            poll_timeout_ms=3000,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            batch_size=200,
            buffer_size=5000,
            security_protocol="SASL_SSL",
            auth_config=auth_config,
            metrics_enabled=False,
            health_check_interval_seconds=60,
            tags=tags
        )
        
        assert source.description == "Kafka source for analytics events"
        assert source.topics == topics
        assert source.partition_key == "user_id"
        assert source.data_format == "avro"
        assert source.schema_registry_url == "http://schema-registry:8081"
        assert source.schema_id == "user-events-v1"
        assert source.max_poll_records == 500
        assert source.poll_timeout_ms == 3000
        assert source.auto_offset_reset == "earliest"
        assert source.enable_auto_commit is True
        assert source.batch_size == 200
        assert source.buffer_size == 5000
        assert source.security_protocol == "SASL_SSL"
        assert source.auth_config == auth_config
        assert source.metrics_enabled is False
        assert source.health_check_interval_seconds == 60
        assert source.tags == tags
        
    def test_validate_connection_valid(self):
        """Test connection validation with valid configuration."""
        source = StreamSource(
            name="test-source",
            stream_type=StreamType.KAFKA,
            connection_string="localhost:9092",
            consumer_group="test-group"
        )
        
        assert source.validate_connection() is True
        
    def test_validate_connection_invalid_no_connection_string(self):
        """Test connection validation with missing connection string."""
        source = StreamSource(
            name="test-source",
            stream_type=StreamType.KAFKA,
            connection_string="",
            consumer_group="test-group"
        )
        
        assert source.validate_connection() is False
        
    def test_validate_connection_invalid_no_consumer_group(self):
        """Test connection validation with missing consumer group."""
        source = StreamSource(
            name="test-source",
            stream_type=StreamType.KAFKA,
            connection_string="localhost:9092",
            consumer_group=""
        )
        
        assert source.validate_connection() is False
        
    def test_get_consumer_config(self):
        """Test getting consumer configuration."""
        auth_config = {"sasl_username": "user", "sasl_password": "pass"}
        
        source = StreamSource(
            name="test-source",
            stream_type=StreamType.KAFKA,
            connection_string="kafka-cluster:9092",
            consumer_group="test-consumers",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            max_poll_records=500,
            security_protocol="SASL_PLAINTEXT",
            auth_config=auth_config
        )
        
        config = source.get_consumer_config()
        
        assert config["bootstrap_servers"] == "kafka-cluster:9092"
        assert config["group_id"] == "test-consumers"
        assert config["auto_offset_reset"] == "earliest"
        assert config["enable_auto_commit"] is True
        assert config["max_poll_records"] == 500
        assert config["security_protocol"] == "SASL_PLAINTEXT"
        assert config["sasl_username"] == "user"
        assert config["sasl_password"] == "pass"


class TestStreamSink:
    """Test cases for StreamSink entity."""
    
    def test_stream_sink_creation_basic(self):
        """Test basic stream sink creation."""
        sink = StreamSink(
            name="processed-events-sink",
            sink_type="kafka",
            connection_string="localhost:9092",
            destination="processed-events"
        )
        
        assert isinstance(sink.id, UUID)
        assert sink.name == "processed-events-sink"
        assert sink.sink_type == "kafka"
        assert sink.description == ""
        assert sink.connection_string == "localhost:9092"
        assert sink.destination == "processed-events"
        assert sink.data_format == "json"
        assert sink.compression is None
        assert sink.partition_by is None
        assert sink.batch_size == 1000
        assert sink.batch_timeout_ms == 10000
        assert sink.max_batch_bytes == 1048576
        assert sink.delivery_guarantee == ProcessingMode.AT_LEAST_ONCE
        assert sink.retry_attempts == 3
        assert sink.retry_backoff_ms == 1000
        assert sink.error_handling == "log"
        assert sink.dead_letter_topic is None
        assert sink.parallelism == 1
        assert sink.buffer_size == 10000
        assert sink.auth_config == {}
        assert sink.tags == {}
        assert isinstance(sink.created_at, datetime)
        assert isinstance(sink.updated_at, datetime)
        
    def test_stream_sink_creation_comprehensive(self):
        """Test comprehensive stream sink creation."""
        auth_config = {"api_key": "secret-key", "api_secret": "secret"}
        tags = {"environment": "staging", "purpose": "analytics"}
        
        sink = StreamSink(
            name="s3-analytics-sink",
            sink_type="s3",
            description="S3 sink for analytics data",
            connection_string="s3://analytics-bucket/",
            destination="events/year={year}/month={month}/day={day}/",
            data_format="parquet",
            compression="gzip",
            partition_by="event_date",
            batch_size=5000,
            batch_timeout_ms=30000,
            max_batch_bytes=10485760,
            delivery_guarantee=ProcessingMode.EXACTLY_ONCE,
            retry_attempts=5,
            retry_backoff_ms=2000,
            error_handling="dead_letter",
            dead_letter_topic="failed-events",
            parallelism=4,
            buffer_size=50000,
            auth_config=auth_config,
            tags=tags
        )
        
        assert sink.description == "S3 sink for analytics data"
        assert sink.data_format == "parquet"
        assert sink.compression == "gzip"
        assert sink.partition_by == "event_date"
        assert sink.batch_size == 5000
        assert sink.batch_timeout_ms == 30000
        assert sink.max_batch_bytes == 10485760
        assert sink.delivery_guarantee == ProcessingMode.EXACTLY_ONCE
        assert sink.retry_attempts == 5
        assert sink.retry_backoff_ms == 2000
        assert sink.error_handling == "dead_letter"
        assert sink.dead_letter_topic == "failed-events"
        assert sink.parallelism == 4
        assert sink.buffer_size == 50000
        assert sink.auth_config == auth_config
        assert sink.tags == tags
        
    def test_get_producer_config(self):
        """Test getting producer configuration."""
        auth_config = {"username": "producer", "password": "secret"}
        
        sink = StreamSink(
            name="test-sink",
            sink_type="kafka",
            connection_string="kafka-cluster:9092",
            destination="output-topic",
            batch_size=2000,
            batch_timeout_ms=5000,
            retry_attempts=5,
            retry_backoff_ms=1500,
            auth_config=auth_config
        )
        
        config = sink.get_producer_config()
        
        assert config["bootstrap_servers"] == "kafka-cluster:9092"
        assert config["batch_size"] == 2000
        assert config["linger_ms"] == 5000
        assert config["retries"] == 5
        assert config["retry_backoff_ms"] == 1500
        assert config["username"] == "producer"
        assert config["password"] == "secret"


class TestProcessingWindow:
    """Test cases for ProcessingWindow entity."""
    
    def test_processing_window_creation_tumbling(self):
        """Test tumbling window creation."""
        window = ProcessingWindow(
            window_type=WindowType.TUMBLING,
            size_seconds=300,
            watermark_delay_seconds=30,
            aggregation_functions=["count", "sum", "avg"],
            group_by_fields=["user_id", "event_type"]
        )
        
        assert isinstance(window.id, UUID)
        assert window.window_type == WindowType.TUMBLING
        assert window.size_seconds == 300
        assert window.slide_seconds is None
        assert window.session_timeout_seconds is None
        assert window.size_count is None
        assert window.slide_count is None
        assert window.watermark_delay_seconds == 30
        assert window.late_data_handling == "drop"
        assert window.trigger_type == "processing_time"
        assert window.trigger_interval_seconds == 60
        assert window.early_firing is False
        assert window.aggregation_functions == ["count", "sum", "avg"]
        assert window.group_by_fields == ["user_id", "event_type"]
        
    def test_processing_window_creation_sliding(self):
        """Test sliding window creation."""
        window = ProcessingWindow(
            window_type=WindowType.SLIDING,
            size_seconds=600,
            slide_seconds=300,
            watermark_delay_seconds=60,
            late_data_handling="include",
            trigger_type="event_time",
            trigger_interval_seconds=30,
            early_firing=True
        )
        
        assert window.window_type == WindowType.SLIDING
        assert window.size_seconds == 600
        assert window.slide_seconds == 300
        assert window.watermark_delay_seconds == 60
        assert window.late_data_handling == "include"
        assert window.trigger_type == "event_time"
        assert window.trigger_interval_seconds == 30
        assert window.early_firing is True
        
    def test_processing_window_creation_session(self):
        """Test session window creation."""
        window = ProcessingWindow(
            window_type=WindowType.SESSION,
            session_timeout_seconds=1800,
            watermark_delay_seconds=120
        )
        
        assert window.window_type == WindowType.SESSION
        assert window.session_timeout_seconds == 1800
        assert window.watermark_delay_seconds == 120
        
    def test_processing_window_creation_count_based(self):
        """Test count-based window creation."""
        window = ProcessingWindow(
            window_type=WindowType.TUMBLING,
            size_count=1000,
            slide_count=500,
            aggregation_functions=["count", "max", "min"]
        )
        
        assert window.size_count == 1000
        assert window.slide_count == 500
        assert window.aggregation_functions == ["count", "max", "min"]
        
    def test_slide_validation_valid(self):
        """Test valid slide interval validation."""
        # This should not raise an error
        ProcessingWindow(
            window_type=WindowType.SLIDING,
            size_seconds=600,
            slide_seconds=300  # Valid: slide <= size
        )
        
    def test_slide_validation_invalid(self):
        """Test invalid slide interval validation."""
        with pytest.raises(ValueError, match="Slide interval cannot be greater than window size"):
            ProcessingWindow(
                window_type=WindowType.SLIDING,
                size_seconds=300,
                slide_seconds=600  # Invalid: slide > size
            )
            
    def test_is_time_based_true(self):
        """Test is_time_based returns True for time-based window."""
        window = ProcessingWindow(
            window_type=WindowType.TUMBLING,
            size_seconds=300
        )
        
        assert window.is_time_based() is True
        
    def test_is_time_based_false(self):
        """Test is_time_based returns False for count-based window."""
        window = ProcessingWindow(
            window_type=WindowType.TUMBLING,
            size_count=1000
        )
        
        assert window.is_time_based() is False
        
    def test_is_count_based_true(self):
        """Test is_count_based returns True for count-based window."""
        window = ProcessingWindow(
            window_type=WindowType.TUMBLING,
            size_count=1000
        )
        
        assert window.is_count_based() is True
        
    def test_is_count_based_false(self):
        """Test is_count_based returns False for time-based window."""
        window = ProcessingWindow(
            window_type=WindowType.TUMBLING,
            size_seconds=300
        )
        
        assert window.is_count_based() is False


class TestStreamProcessor:
    """Test cases for StreamProcessor entity."""
    
    def test_stream_processor_creation_basic(self):
        """Test basic stream processor creation."""
        tenant_id = uuid4()
        created_by = uuid4()
        sources = [uuid4(), uuid4()]
        sinks = [uuid4()]
        
        processor = StreamProcessor(
            name="analytics-processor",
            tenant_id=tenant_id,
            created_by=created_by,
            sources=sources,
            sinks=sinks,
            processing_logic="def process(record): return record * 2",
            reporting_frequency="hourly"
        )
        
        assert isinstance(processor.id, UUID)
        assert processor.name == "analytics-processor"
        assert processor.description == ""
        assert processor.version == "1.0.0"
        assert processor.tenant_id == tenant_id
        assert processor.created_by == created_by
        assert processor.sources == sources
        assert processor.sinks == sinks
        assert processor.processing_logic == "def process(record): return record * 2"
        assert processor.processing_language == "python"
        assert processor.processing_mode == ProcessingMode.AT_LEAST_ONCE
        assert processor.parallelism == 1
        assert processor.max_parallelism == 10
        assert processor.checkpoint_interval_ms == 60000
        assert processor.windowing_enabled is False
        assert processor.windows == []
        assert processor.cpu_request == 0.5
        assert processor.memory_request_gb == 1.0
        assert processor.cpu_limit == 2.0
        assert processor.memory_limit_gb == 4.0
        assert processor.auto_scaling_enabled is False
        assert processor.scale_up_threshold == 80.0
        assert processor.scale_down_threshold == 20.0
        assert processor.scale_up_cooldown_seconds == 300
        assert processor.scale_down_cooldown_seconds == 600
        assert processor.status == ProcessorStatus.PENDING
        assert processor.current_parallelism == 1
        assert processor.current_throughput_per_second == 0.0
        assert processor.current_latency_ms == 0.0
        assert processor.records_processed == 0
        assert processor.records_failed == 0
        assert processor.bytes_processed == 0
        assert processor.processing_errors == 0
        assert processor.avg_processing_time_ms == 0.0
        assert processor.health_score == 100.0
        assert processor.last_checkpoint is None
        assert processor.last_scaling_action is None
        assert processor.error_message is None
        assert processor.environment_variables == {}
        assert processor.secrets == []
        assert processor.tags == {}
        assert isinstance(processor.created_at, datetime)
        assert isinstance(processor.updated_at, datetime)
        assert processor.started_at is None
        assert processor.stopped_at is None
        
    def test_stream_processor_creation_comprehensive(self):
        """Test comprehensive stream processor creation."""
        tenant_id = uuid4()
        created_by = uuid4()
        sources = [uuid4()]
        sinks = [uuid4(), uuid4()]
        windows = [uuid4()]
        environment_variables = {"LOG_LEVEL": "DEBUG", "BATCH_SIZE": "100"}
        secrets = ["api-key-secret", "db-password-secret"]
        tags = {"environment": "production", "version": "v2.1.0"}
        
        processor = StreamProcessor(
            name="real-time-analytics",
            description="Real-time analytics stream processor",
            version="2.1.0",
            tenant_id=tenant_id,
            created_by=created_by,
            sources=sources,
            sinks=sinks,
            processing_logic="complex_analytics_function()",
            processing_language="scala",
            processing_mode=ProcessingMode.EXACTLY_ONCE,
            parallelism=4,
            max_parallelism=20,
            checkpoint_interval_ms=30000,
            windowing_enabled=True,
            windows=windows,
            cpu_request=2.0,
            memory_request_gb=8.0,
            cpu_limit=8.0,
            memory_limit_gb=32.0,
            auto_scaling_enabled=True,
            scale_up_threshold=75.0,
            scale_down_threshold=25.0,
            scale_up_cooldown_seconds=180,
            scale_down_cooldown_seconds=900,
            status=ProcessorStatus.RUNNING,
            current_parallelism=6,
            current_throughput_per_second=1500.0,
            current_latency_ms=45.0,
            records_processed=1000000,
            records_failed=50,
            bytes_processed=10737418240,
            processing_errors=5,
            avg_processing_time_ms=12.5,
            health_score=95.0,
            environment_variables=environment_variables,
            secrets=secrets,
            tags=tags,
            reporting_frequency="real-time"
        )
        
        assert processor.description == "Real-time analytics stream processor"
        assert processor.version == "2.1.0"
        assert processor.processing_language == "scala"
        assert processor.processing_mode == ProcessingMode.EXACTLY_ONCE
        assert processor.parallelism == 4
        assert processor.max_parallelism == 20
        assert processor.checkpoint_interval_ms == 30000
        assert processor.windowing_enabled is True
        assert processor.windows == windows
        assert processor.cpu_request == 2.0
        assert processor.memory_request_gb == 8.0
        assert processor.cpu_limit == 8.0
        assert processor.memory_limit_gb == 32.0
        assert processor.auto_scaling_enabled is True
        assert processor.scale_up_threshold == 75.0
        assert processor.scale_down_threshold == 25.0
        assert processor.scale_up_cooldown_seconds == 180
        assert processor.scale_down_cooldown_seconds == 900
        assert processor.status == ProcessorStatus.RUNNING
        assert processor.current_parallelism == 6
        assert processor.current_throughput_per_second == 1500.0
        assert processor.current_latency_ms == 45.0
        assert processor.records_processed == 1000000
        assert processor.records_failed == 50
        assert processor.bytes_processed == 10737418240
        assert processor.processing_errors == 5
        assert processor.avg_processing_time_ms == 12.5
        assert processor.health_score == 95.0
        assert processor.environment_variables == environment_variables
        assert processor.secrets == secrets
        assert processor.tags == tags
        
    def test_max_parallelism_validation(self):
        """Test max_parallelism validation."""
        with pytest.raises(ValueError, match="max_parallelism must be >= parallelism"):
            StreamProcessor(
                name="test-processor",
                tenant_id=uuid4(),
                created_by=uuid4(),
                sources=[uuid4()],
                sinks=[uuid4()],
                processing_logic="test",
                parallelism=5,
                max_parallelism=3,  # Invalid: less than parallelism
                reporting_frequency="hourly"
            )
            
    def test_cpu_limit_validation(self):
        """Test CPU limit validation."""
        with pytest.raises(ValueError, match="cpu_limit must be >= cpu_request"):
            StreamProcessor(
                name="test-processor",
                tenant_id=uuid4(),
                created_by=uuid4(),
                sources=[uuid4()],
                sinks=[uuid4()],
                processing_logic="test",
                cpu_request=4.0,
                cpu_limit=2.0,  # Invalid: less than request
                reporting_frequency="hourly"
            )
            
    def test_memory_limit_validation(self):
        """Test memory limit validation."""
        with pytest.raises(ValueError, match="memory_limit_gb must be >= memory_request_gb"):
            StreamProcessor(
                name="test-processor",
                tenant_id=uuid4(),
                created_by=uuid4(),
                sources=[uuid4()],
                sinks=[uuid4()],
                processing_logic="test",
                memory_request_gb=8.0,
                memory_limit_gb=4.0,  # Invalid: less than request
                reporting_frequency="hourly"
            )
            
    def test_is_running_true(self):
        """Test is_running returns True for running processor."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.RUNNING,
            reporting_frequency="hourly"
        )
        
        assert processor.is_running() is True
        
    def test_is_running_false(self):
        """Test is_running returns False for non-running processor."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.STOPPED,
            reporting_frequency="hourly"
        )
        
        assert processor.is_running() is False
        
    def test_is_healthy_true(self):
        """Test is_healthy returns True for healthy running processor."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.RUNNING,
            health_score=85.0,
            reporting_frequency="hourly"
        )
        
        assert processor.is_healthy() is True
        
    def test_is_healthy_false_not_running(self):
        """Test is_healthy returns False for non-running processor."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.STOPPED,
            health_score=85.0,
            reporting_frequency="hourly"
        )
        
        assert processor.is_healthy() is False
        
    def test_is_healthy_false_low_score(self):
        """Test is_healthy returns False for low health score."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.RUNNING,
            health_score=50.0,
            reporting_frequency="hourly"
        )
        
        assert processor.is_healthy() is False
        
    def test_get_success_rate(self):
        """Test calculating processing success rate."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            records_processed=950,
            records_failed=50,
            reporting_frequency="hourly"
        )
        
        success_rate = processor.get_success_rate()
        assert success_rate == 95.0  # 950/(950+50) * 100
        
    def test_get_success_rate_no_records(self):
        """Test success rate with no processed records."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            records_processed=0,
            records_failed=0,
            reporting_frequency="hourly"
        )
        
        success_rate = processor.get_success_rate()
        assert success_rate == 100.0
        
    def test_should_scale_up_true(self):
        """Test should_scale_up returns True when conditions met."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            auto_scaling_enabled=True,
            current_parallelism=5,
            max_parallelism=10,
            scale_up_threshold=80.0,
            reporting_frequency="hourly"
        )
        
        # Mock get_current_cpu_usage to return high value
        def mock_get_cpu_usage():
            return 85.0
        processor.get_current_cpu_usage = mock_get_cpu_usage
        
        assert processor.should_scale_up() is True
        
    def test_should_scale_up_false_disabled(self):
        """Test should_scale_up returns False when auto-scaling disabled."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            auto_scaling_enabled=False,
            current_parallelism=5,
            max_parallelism=10,
            reporting_frequency="hourly"
        )
        
        assert processor.should_scale_up() is False
        
    def test_should_scale_up_false_at_max(self):
        """Test should_scale_up returns False when at max parallelism."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            auto_scaling_enabled=True,
            current_parallelism=10,
            max_parallelism=10,
            reporting_frequency="hourly"
        )
        
        assert processor.should_scale_up() is False
        
    def test_should_scale_down_true(self):
        """Test should_scale_down returns True when conditions met."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            auto_scaling_enabled=True,
            current_parallelism=5,
            scale_down_threshold=20.0,
            reporting_frequency="hourly"
        )
        
        # Mock get_current_cpu_usage to return low value
        def mock_get_cpu_usage():
            return 15.0
        processor.get_current_cpu_usage = mock_get_cpu_usage
        
        assert processor.should_scale_down() is True
        
    def test_should_scale_down_false_disabled(self):
        """Test should_scale_down returns False when auto-scaling disabled."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            auto_scaling_enabled=False,
            current_parallelism=5,
            reporting_frequency="hourly"
        )
        
        assert processor.should_scale_down() is False
        
    def test_should_scale_down_false_at_min(self):
        """Test should_scale_down returns False when at minimum parallelism."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            auto_scaling_enabled=True,
            current_parallelism=1,
            reporting_frequency="hourly"
        )
        
        assert processor.should_scale_down() is False
        
    def test_get_current_cpu_usage(self):
        """Test getting current CPU usage."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            reporting_frequency="hourly"
        )
        
        # Default implementation returns 50.0
        cpu_usage = processor.get_current_cpu_usage()
        assert cpu_usage == 50.0
        
    def test_can_scale_no_previous_action(self):
        """Test can_scale returns True with no previous scaling action."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            last_scaling_action=None,
            reporting_frequency="hourly"
        )
        
        assert processor.can_scale("up") is True
        assert processor.can_scale("down") is True
        
    def test_can_scale_during_cooldown(self):
        """Test can_scale returns False during cooldown period."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            last_scaling_action=datetime.utcnow() - timedelta(minutes=2),
            scale_up_cooldown_seconds=300,
            scale_down_cooldown_seconds=600,
            reporting_frequency="hourly"
        )
        
        assert processor.can_scale("up") is False
        assert processor.can_scale("down") is False
        
    def test_can_scale_after_cooldown(self):
        """Test can_scale returns True after cooldown period."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            last_scaling_action=datetime.utcnow() - timedelta(minutes=15),
            scale_up_cooldown_seconds=300,
            scale_down_cooldown_seconds=600,
            reporting_frequency="hourly"
        )
        
        assert processor.can_scale("up") is True
        assert processor.can_scale("down") is True
        
    def test_start(self):
        """Test starting stream processor."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.PENDING,
            error_message="Previous error",
            reporting_frequency="hourly"
        )
        
        original_updated_at = processor.updated_at
        
        processor.start()
        
        assert processor.status == ProcessorStatus.RUNNING
        assert processor.started_at is not None
        assert processor.updated_at > original_updated_at
        assert processor.error_message is None
        
    def test_stop(self):
        """Test stopping stream processor."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.RUNNING,
            reporting_frequency="hourly"
        )
        
        original_updated_at = processor.updated_at
        
        processor.stop()
        
        assert processor.status == ProcessorStatus.STOPPED
        assert processor.stopped_at is not None
        assert processor.updated_at > original_updated_at
        
    def test_pause(self):
        """Test pausing stream processor."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.RUNNING,
            reporting_frequency="hourly"
        )
        
        original_updated_at = processor.updated_at
        
        processor.pause()
        
        assert processor.status == ProcessorStatus.PAUSED
        assert processor.updated_at > original_updated_at
        
    def test_set_error(self):
        """Test setting processor error state."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            status=ProcessorStatus.RUNNING,
            reporting_frequency="hourly"
        )
        
        original_updated_at = processor.updated_at
        error_msg = "Stream connection failed"
        
        processor.set_error(error_msg)
        
        assert processor.status == ProcessorStatus.ERROR
        assert processor.error_message == error_msg
        assert processor.updated_at > original_updated_at
        
    def test_update_metrics(self):
        """Test updating processor metrics."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            reporting_frequency="hourly"
        )
        
        original_updated_at = processor.updated_at
        
        processor.update_metrics(
            throughput=1500.0,
            latency=25.5,
            processed_count=1000,
            failed_count=10
        )
        
        assert processor.current_throughput_per_second == 1500.0
        assert processor.current_latency_ms == 25.5
        assert processor.records_processed == 1000
        assert processor.records_failed == 10
        assert processor.updated_at > original_updated_at
        
    def test_record_checkpoint(self):
        """Test recording successful checkpoint."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            reporting_frequency="hourly"
        )
        
        original_updated_at = processor.updated_at
        
        processor.record_checkpoint()
        
        assert processor.last_checkpoint is not None
        assert processor.updated_at > original_updated_at
        
    def test_scale(self):
        """Test scaling processor to new parallelism level."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            current_parallelism=3,
            max_parallelism=10,
            reporting_frequency="hourly"
        )
        
        original_updated_at = processor.updated_at
        
        processor.scale(5)
        
        assert processor.current_parallelism == 5
        assert processor.last_scaling_action is not None
        assert processor.updated_at > original_updated_at
        
    def test_scale_invalid_parallelism(self):
        """Test scaling with invalid parallelism values."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            max_parallelism=10,
            reporting_frequency="hourly"
        )
        
        # Test below minimum
        with pytest.raises(ValueError, match="Invalid parallelism: 0"):
            processor.scale(0)
            
        # Test above maximum
        with pytest.raises(ValueError, match="Invalid parallelism: 15"):
            processor.scale(15)
            
    def test_get_processor_summary(self):
        """Test getting processor summary."""
        processor = StreamProcessor(
            name="analytics-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4(), uuid4()],
            sinks=[uuid4()],
            processing_logic="analytics_function()",
            status=ProcessorStatus.RUNNING,
            current_parallelism=4,
            max_parallelism=8,
            auto_scaling_enabled=True,
            current_throughput_per_second=2500.0,
            current_latency_ms=15.0,
            records_processed=5000000,
            records_failed=250,
            cpu_request=2.0,
            memory_request_gb=8.0,
            cpu_limit=6.0,
            memory_limit_gb=24.0,
            health_score=90.0,
            started_at=datetime.utcnow() - timedelta(hours=6),
            reporting_frequency="hourly"
        )
        
        summary = processor.get_processor_summary()
        
        assert summary["id"] == str(processor.id)
        assert summary["name"] == "analytics-processor"
        assert summary["status"] == ProcessorStatus.RUNNING
        assert summary["parallelism"]["current"] == 4
        assert summary["parallelism"]["max"] == 8
        assert summary["parallelism"]["auto_scaling"] is True
        assert summary["performance"]["throughput_per_second"] == 2500.0
        assert summary["performance"]["latency_ms"] == 15.0
        assert summary["performance"]["success_rate"] == pytest.approx(95.238, rel=0.01)  # 5000000/(5000000+250)*100
        assert summary["performance"]["records_processed"] == 5000000
        assert summary["performance"]["records_failed"] == 250
        assert summary["resources"]["cpu_request"] == 2.0
        assert summary["resources"]["memory_request_gb"] == 8.0
        assert summary["resources"]["cpu_limit"] == 6.0
        assert summary["resources"]["memory_limit_gb"] == 24.0
        assert summary["health_score"] == 90.0
        assert summary["uptime_hours"] == pytest.approx(6.0, rel=0.1)
        assert summary["sources_count"] == 2
        assert summary["sinks_count"] == 1
        
    def test_get_processor_summary_no_start_time(self):
        """Test processor summary with no start time."""
        processor = StreamProcessor(
            name="test-processor",
            tenant_id=uuid4(),
            created_by=uuid4(),
            sources=[uuid4()],
            sinks=[uuid4()],
            processing_logic="test",
            started_at=None,
            reporting_frequency="hourly"
        )
        
        summary = processor.get_processor_summary()
        
        assert summary["uptime_hours"] == 0.0