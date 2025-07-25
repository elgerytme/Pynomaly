"""
Centralized Log Management System

Provides comprehensive log aggregation, processing, and analysis
using ELK stack (Elasticsearch, Logstash, Kibana) or equivalent solutions.
"""

import asyncio
import logging
import json
import gzip
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles
import aiohttp
from elasticsearch import AsyncElasticsearch
import structlog

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogSource(Enum):
    """Log sources."""
    APPLICATION = "application"
    SYSTEM = "system"
    SECURITY = "security"
    AUDIT = "audit"
    ML_MODEL = "ml_model"
    API = "api"
    DATABASE = "database"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    source: LogSource
    service_name: str
    logger_name: str
    thread_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class LogQuery:
    """Log query parameters."""
    start_time: datetime
    end_time: datetime
    level: Optional[LogLevel] = None
    source: Optional[LogSource] = None
    service_name: Optional[str] = None
    search_text: Optional[str] = None
    fields: Optional[List[str]] = None
    limit: int = 1000
    offset: int = 0
    sort_order: str = "desc"  # desc or asc


@dataclass
class LogPattern:
    """Log pattern for parsing unstructured logs."""
    name: str
    regex: str
    fields: List[str]
    source: LogSource
    sample: str


class LogProcessor:
    """Process and enrich log entries."""
    
    def __init__(self):
        self.processors: List[Callable[[LogEntry], LogEntry]] = []
        self.filters: List[Callable[[LogEntry], bool]] = []
    
    def add_processor(self, processor: Callable[[LogEntry], LogEntry]) -> None:
        """Add a log processor."""
        self.processors.append(processor)
    
    def add_filter(self, filter_func: Callable[[LogEntry], bool]) -> None:
        """Add a log filter."""
        self.filters.append(filter_func)
    
    def process(self, log_entry: LogEntry) -> Optional[LogEntry]:
        """Process a log entry through all processors and filters."""
        # Apply filters first
        for filter_func in self.filters:
            if not filter_func(log_entry):
                return None  # Entry filtered out
        
        # Apply processors
        processed_entry = log_entry
        for processor in self.processors:
            try:
                processed_entry = processor(processed_entry)
            except Exception as e:
                logger.warning(f"Error in log processor: {e}")
        
        return processed_entry


class LogManager:
    """
    Comprehensive log management system with support for multiple
    storage backends, real-time processing, and advanced querying.
    """
    
    def __init__(
        self,
        elasticsearch_hosts: Optional[List[str]] = None,
        index_prefix: str = "mlops-logs",
        retention_days: int = 30,
        enable_compression: bool = True,
        enable_real_time_processing: bool = True,
        log_file_paths: Optional[List[str]] = None
    ):
        self.elasticsearch_hosts = elasticsearch_hosts or ["http://localhost:9200"]
        self.index_prefix = index_prefix
        self.retention_days = retention_days
        self.enable_compression = enable_compression
        self.enable_real_time_processing = enable_real_time_processing
        self.log_file_paths = log_file_paths or []
        
        # Elasticsearch client
        self.es_client = None
        
        # Log processing
        self.log_processor = LogProcessor()
        self.log_patterns: Dict[str, LogPattern] = {}
        
        # Storage and buffering
        self.log_buffer: List[LogEntry] = []
        self.buffer_size = 1000
        self.local_storage_path = Path("./logs")
        self.local_storage_path.mkdir(exist_ok=True)
        
        # Background tasks
        self.running = False
        self.ingestion_task = None
        self.processing_task = None
        self.cleanup_task = None
        
        # Metrics
        self.metrics = {
            "logs_ingested": 0,
            "logs_processed": 0,
            "logs_filtered": 0,
            "errors": 0
        }
        
        # Setup default log patterns
        self._setup_default_patterns()
        
        # Setup default processors
        self._setup_default_processors()
    
    async def initialize(self) -> None:
        """Initialize the log management system."""
        try:
            # Initialize Elasticsearch client
            if self.elasticsearch_hosts:
                self.es_client = AsyncElasticsearch(
                    hosts=self.elasticsearch_hosts,
                    retry_on_timeout=True,
                    max_retries=3
                )
                
                # Test connection
                await self.es_client.ping()
                logger.info("Connected to Elasticsearch")
                
                # Create index templates
                await self._create_index_templates()
            
            # Setup structured logging
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    self._structlog_processor,
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            
            logger.info("Log management system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize log management: {e}")
            raise
    
    async def start(self) -> None:
        """Start log management services."""
        self.running = True
        
        # Start background tasks
        if self.log_file_paths:
            self.ingestion_task = asyncio.create_task(self._log_ingestion_loop())
        
        if self.enable_real_time_processing:
            self.processing_task = asyncio.create_task(self._log_processing_loop())
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Log management services started")
    
    async def stop(self) -> None:
        """Stop log management services."""
        self.running = False
        
        # Cancel background tasks
        if self.ingestion_task:
            self.ingestion_task.cancel()
        
        if self.processing_task:
            self.processing_task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Flush remaining logs
        await self._flush_buffer()
        
        # Close Elasticsearch client
        if self.es_client:
            await self.es_client.close()
        
        logger.info("Log management services stopped")
    
    def _setup_default_patterns(self) -> None:
        """Setup default log parsing patterns."""
        patterns = [
            LogPattern(
                name="nginx_access",
                regex=r'(?P<remote_addr>\S+) - (?P<remote_user>\S+) \[(?P<time_local>.+)\] "(?P<request>.+)" (?P<status>\d+) (?P<body_bytes_sent>\d+) "(?P<http_referer>.*)" "(?P<http_user_agent>.*)"',
                fields=["remote_addr", "remote_user", "time_local", "request", "status", "body_bytes_sent", "http_referer", "http_user_agent"],
                source=LogSource.API,
                sample='192.168.1.100 - - [22/Dec/2023:10:30:45 +0000] "GET /api/v1/models HTTP/1.1" 200 1234 "-" "Python/3.9 requests/2.25.1"'
            ),
            LogPattern(
                name="python_application",
                regex=r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<logger_name>\S+) - (?P<level>\w+) - (?P<message>.*)',
                fields=["timestamp", "logger_name", "level", "message"],
                source=LogSource.APPLICATION,
                sample="2023-12-22 10:30:45,123 - mlops.model - INFO - Model prediction completed successfully"
            ),
            LogPattern(
                name="kubernetes_container",
                regex=r'(?P<timestamp>\S+) (?P<stream>\w+) (?P<log>.*)',
                fields=["timestamp", "stream", "log"],
                source=LogSource.INFRASTRUCTURE,
                sample="2023-12-22T10:30:45.123456789Z stdout Model inference request received"
            ),
            LogPattern(
                name="ml_model_inference",
                regex=r'(?P<timestamp>\S+) - (?P<model_id>\S+) - (?P<model_version>\S+) - (?P<request_id>\S+) - (?P<latency>\d+\.\d+) - (?P<status>\w+) - (?P<message>.*)',
                fields=["timestamp", "model_id", "model_version", "request_id", "latency", "status", "message"],
                source=LogSource.ML_MODEL,
                sample="2023-12-22T10:30:45.123Z - model_v1 - 1.0.0 - req_123 - 0.045 - SUCCESS - Prediction completed"
            )
        ]
        
        for pattern in patterns:
            self.log_patterns[pattern.name] = pattern
    
    def _setup_default_processors(self) -> None:
        """Setup default log processors."""
        # Add trace context enrichment
        self.log_processor.add_processor(self._enrich_trace_context)
        
        # Add geo-location enrichment for IP addresses
        self.log_processor.add_processor(self._enrich_geo_location)
        
        # Add security analysis
        self.log_processor.add_processor(self._analyze_security)
        
        # Filter out health check logs (optional)
        self.log_processor.add_filter(lambda log: "health" not in log.message.lower() or log.level != LogLevel.INFO)
    
    def _enrich_trace_context(self, log_entry: LogEntry) -> LogEntry:
        """Enrich log entry with trace context from current span."""
        try:
            from opentelemetry import trace
            
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                log_entry.trace_id = format(span_context.trace_id, '032x')
                log_entry.span_id = format(span_context.span_id, '016x')
        except Exception:
            pass  # Ignore if OpenTelemetry not available
        
        return log_entry
    
    def _enrich_geo_location(self, log_entry: LogEntry) -> LogEntry:
        """Enrich log entry with geo-location data for IP addresses."""
        # Extract IP addresses from log fields
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        
        # Check common fields for IP addresses
        for field_name in ['remote_addr', 'client_ip', 'source_ip']:
            if field_name in log_entry.extra_fields:
                ip_value = log_entry.extra_fields[field_name]
                if re.match(ip_pattern, str(ip_value)):
                    # In a real implementation, you would use a GeoIP database
                    # For now, we'll just mark it as an IP address
                    log_entry.extra_fields[f"{field_name}_is_ip"] = True
                    log_entry.extra_fields[f"{field_name}_geo"] = {"country": "Unknown", "city": "Unknown"}
        
        return log_entry
    
    def _analyze_security(self, log_entry: LogEntry) -> LogEntry:
        """Analyze log entry for security events."""
        security_keywords = [
            "failed login", "unauthorized", "forbidden", "sql injection",
            "xss", "csrf", "brute force", "suspicious", "anomaly"
        ]
        
        message_lower = log_entry.message.lower()
        security_indicators = [kw for kw in security_keywords if kw in message_lower]
        
        if security_indicators:
            log_entry.tags.extend(["security", "suspicious"])
            log_entry.extra_fields["security_indicators"] = security_indicators
            
            # Escalate to ERROR level for security events
            if log_entry.level in [LogLevel.INFO, LogLevel.WARNING]:
                log_entry.level = LogLevel.ERROR
        
        return log_entry
    
    def _structlog_processor(self, logger, method_name, event_dict):
        """Custom structlog processor to convert to LogEntry."""
        try:
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel(event_dict.get("level", "INFO").upper()),
                message=event_dict.get("event", ""),
                source=LogSource.APPLICATION,
                service_name=event_dict.get("service", "unknown"),
                logger_name=event_dict.get("logger", ""),
                extra_fields=event_dict
            )
            
            # Process the log entry
            processed_entry = self.log_processor.process(log_entry)
            if processed_entry:
                asyncio.create_task(self.ingest_log(processed_entry))
        
        except Exception as e:
            logger.error(f"Error in structlog processor: {e}")
        
        return event_dict
    
    async def _create_index_templates(self) -> None:
        """Create Elasticsearch index templates."""
        template = {
            "index_patterns": [f"{self.index_prefix}-*"],
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "index.refresh_interval": "5s",
                    "index.codec": "best_compression" if self.enable_compression else "default"
                },
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "message": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "service_name": {"type": "keyword"},
                        "logger_name": {"type": "keyword"},
                        "thread_id": {"type": "keyword"},
                        "trace_id": {"type": "keyword"},
                        "span_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "session_id": {"type": "keyword"},
                        "request_id": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "extra_fields": {"type": "object", "dynamic": True}
                    }
                }
            }
        }
        
        try:
            await self.es_client.indices.put_index_template(
                name=f"{self.index_prefix}-template",
                body=template
            )
            logger.info(f"Created Elasticsearch index template: {self.index_prefix}-template")
        except Exception as e:
            logger.error(f"Failed to create index template: {e}")
    
    async def ingest_log(self, log_entry: LogEntry) -> None:
        """Ingest a single log entry."""
        try:
            # Process the log entry
            processed_entry = self.log_processor.process(log_entry)
            if not processed_entry:
                self.metrics["logs_filtered"] += 1
                return
            
            # Add to buffer
            self.log_buffer.append(processed_entry)
            self.metrics["logs_ingested"] += 1
            
            # Flush buffer if it's full
            if len(self.log_buffer) >= self.buffer_size:
                await self._flush_buffer()
        
        except Exception as e:
            logger.error(f"Error ingesting log: {e}")
            self.metrics["errors"] += 1
    
    async def ingest_logs_batch(self, log_entries: List[LogEntry]) -> None:
        """Ingest multiple log entries."""
        for log_entry in log_entries:
            await self.ingest_log(log_entry)
    
    async def parse_log_line(self, line: str, source: LogSource = LogSource.APPLICATION) -> Optional[LogEntry]:
        """Parse a raw log line using configured patterns."""
        for pattern_name, pattern in self.log_patterns.items():
            if pattern.source != source:
                continue
            
            match = re.match(pattern.regex, line)
            if match:
                fields = match.groupdict()
                
                # Create log entry
                log_entry = LogEntry(
                    timestamp=self._parse_timestamp(fields.get("timestamp", fields.get("time_local"))),
                    level=LogLevel(fields.get("level", "INFO").upper()),
                    message=fields.get("message", fields.get("log", line)),
                    source=source,
                    service_name=fields.get("service", "unknown"),
                    logger_name=fields.get("logger_name", pattern_name),
                    extra_fields=fields
                )
                
                return log_entry
        
        # If no pattern matches, create a basic log entry
        return LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.INFO,
            message=line,
            source=source,
            service_name="unknown",
            logger_name="unparsed"
        )
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp from various formats."""
        if not timestamp_str:
            return datetime.utcnow()
        
        # Common timestamp formats
        formats = [
            "%Y-%m-%d %H:%M:%S,%f",  # Python logging
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO with microseconds
            "%Y-%m-%dT%H:%M:%SZ",     # ISO without microseconds
            "%d/%b/%Y:%H:%M:%S %z",   # Apache/Nginx
            "%Y-%m-%d %H:%M:%S"       # Simple format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # If all formats fail, return current time
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return datetime.utcnow()
    
    async def _flush_buffer(self) -> None:
        """Flush log buffer to storage."""
        if not self.log_buffer:
            return
        
        try:
            # Write to Elasticsearch
            if self.es_client:
                await self._write_to_elasticsearch(self.log_buffer)
            
            # Write to local storage as backup
            await self._write_to_local_storage(self.log_buffer)
            
            self.metrics["logs_processed"] += len(self.log_buffer)
            self.log_buffer.clear()
        
        except Exception as e:
            logger.error(f"Error flushing log buffer: {e}")
            self.metrics["errors"] += 1
    
    async def _write_to_elasticsearch(self, log_entries: List[LogEntry]) -> None:
        """Write log entries to Elasticsearch."""
        if not self.es_client:
            return
        
        # Prepare bulk operations
        operations = []
        
        for log_entry in log_entries:
            # Generate index name with date
            index_name = f"{self.index_prefix}-{log_entry.timestamp.strftime('%Y.%m.%d')}"
            
            # Convert log entry to document
            doc = {
                "timestamp": log_entry.timestamp.isoformat(),
                "level": log_entry.level.value,
                "message": log_entry.message,
                "source": log_entry.source.value,
                "service_name": log_entry.service_name,
                "logger_name": log_entry.logger_name,
                "thread_id": log_entry.thread_id,
                "trace_id": log_entry.trace_id,
                "span_id": log_entry.span_id,
                "user_id": log_entry.user_id,
                "session_id": log_entry.session_id,
                "request_id": log_entry.request_id,
                "tags": log_entry.tags,
                "extra_fields": log_entry.extra_fields
            }
            
            # Add to bulk operations
            operations.append({"index": {"_index": index_name}})
            operations.append(doc)
        
        # Execute bulk operation
        try:
            response = await self.es_client.bulk(operations=operations)
            
            # Check for errors
            if response.get("errors"):
                error_count = sum(1 for item in response["items"] if "error" in item.get("index", {}))
                logger.warning(f"Elasticsearch bulk operation had {error_count} errors")
        
        except Exception as e:
            logger.error(f"Error writing to Elasticsearch: {e}")
            raise
    
    async def _write_to_local_storage(self, log_entries: List[LogEntry]) -> None:
        """Write log entries to local storage as backup."""
        try:
            date_str = datetime.utcnow().strftime('%Y-%m-%d')
            log_file = self.local_storage_path / f"logs-{date_str}.jsonl"
            
            async with aiofiles.open(log_file, 'a') as f:
                for log_entry in log_entries:
                    log_doc = {
                        "timestamp": log_entry.timestamp.isoformat(),
                        "level": log_entry.level.value,
                        "message": log_entry.message,
                        "source": log_entry.source.value,
                        "service_name": log_entry.service_name,
                        "logger_name": log_entry.logger_name,
                        "thread_id": log_entry.thread_id,
                        "trace_id": log_entry.trace_id,
                        "span_id": log_entry.span_id,
                        "user_id": log_entry.user_id,
                        "session_id": log_entry.session_id,
                        "request_id": log_entry.request_id,
                        "tags": log_entry.tags,
                        "extra_fields": log_entry.extra_fields
                    }
                    await f.write(json.dumps(log_doc) + '\n')
        
        except Exception as e:
            logger.error(f"Error writing to local storage: {e}")
    
    async def _log_ingestion_loop(self) -> None:
        """Background loop for ingesting logs from files."""
        file_positions = {path: 0 for path in self.log_file_paths}
        
        while self.running:
            try:
                for log_file_path in self.log_file_paths:
                    await self._read_log_file(log_file_path, file_positions)
                
                await asyncio.sleep(5)  # Check every 5 seconds
            
            except Exception as e:
                logger.error(f"Error in log ingestion loop: {e}")
                await asyncio.sleep(5)
    
    async def _read_log_file(self, file_path: str, file_positions: Dict[str, int]) -> None:
        """Read new lines from a log file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return
            
            current_position = file_positions.get(file_path, 0)
            
            async with aiofiles.open(path, 'r') as f:
                await f.seek(current_position)
                
                async for line in f:
                    line = line.strip()
                    if line:
                        # Determine source based on file path
                        source = self._determine_source_from_path(file_path)
                        
                        # Parse and ingest log line
                        log_entry = await self.parse_log_line(line, source)
                        if log_entry:
                            await self.ingest_log(log_entry)
                
                # Update file position
                file_positions[file_path] = await f.tell()
        
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {e}")
    
    def _determine_source_from_path(self, file_path: str) -> LogSource:
        """Determine log source from file path."""
        path_lower = file_path.lower()
        
        if "security" in path_lower or "auth" in path_lower:
            return LogSource.SECURITY
        elif "audit" in path_lower:
            return LogSource.AUDIT
        elif "nginx" in path_lower or "apache" in path_lower or "api" in path_lower:
            return LogSource.API
        elif "system" in path_lower or "syslog" in path_lower:
            return LogSource.SYSTEM
        elif "model" in path_lower or "ml" in path_lower:
            return LogSource.ML_MODEL
        elif "database" in path_lower or "db" in path_lower:
            return LogSource.DATABASE
        else:
            return LogSource.APPLICATION
    
    async def _log_processing_loop(self) -> None:
        """Background loop for real-time log processing."""
        while self.running:
            try:
                # Flush buffer periodically
                if self.log_buffer:
                    await self._flush_buffer()
                
                await asyncio.sleep(10)  # Flush every 10 seconds
            
            except Exception as e:
                logger.error(f"Error in log processing loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up old logs."""
        while self.running:
            try:
                await self._cleanup_old_logs()
                await asyncio.sleep(3600)  # Check every hour
            
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_logs(self) -> None:
        """Clean up old logs based on retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        # Clean up Elasticsearch indices
        if self.es_client:
            try:
                # Get all indices matching our pattern
                indices = await self.es_client.indices.get(index=f"{self.index_prefix}-*")
                
                for index_name in indices.keys():
                    # Extract date from index name
                    try:
                        date_part = index_name.split('-')[-1]
                        index_date = datetime.strptime(date_part, '%Y.%m.%d')
                        
                        if index_date < cutoff_date:
                            await self.es_client.indices.delete(index=index_name)
                            logger.info(f"Deleted old index: {index_name}")
                    
                    except (ValueError, IndexError):
                        continue  # Skip indices that don't match expected format
            
            except Exception as e:
                logger.error(f"Error cleaning up Elasticsearch indices: {e}")
        
        # Clean up local log files
        try:
            for log_file in self.local_storage_path.glob("logs-*.jsonl"):
                try:
                    date_part = log_file.stem.split('-', 1)[1]
                    file_date = datetime.strptime(date_part, '%Y-%m-%d')
                    
                    if file_date < cutoff_date:
                        log_file.unlink()
                        logger.info(f"Deleted old log file: {log_file}")
                
                except (ValueError, IndexError):
                    continue
        
        except Exception as e:
            logger.error(f"Error cleaning up local log files: {e}")
    
    async def search_logs(self, query: LogQuery) -> List[Dict[str, Any]]:
        """Search logs with the given query."""
        if not self.es_client:
            return await self._search_local_logs(query)
        
        try:
            # Build Elasticsearch query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "timestamp": {
                                        "gte": query.start_time.isoformat(),
                                        "lte": query.end_time.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [{"timestamp": {"order": query.sort_order}}],
                "size": query.limit,
                "from": query.offset
            }
            
            # Add filters
            filters = []
            
            if query.level:
                filters.append({"term": {"level": query.level.value}})
            
            if query.source:
                filters.append({"term": {"source": query.source.value}})
            
            if query.service_name:
                filters.append({"term": {"service_name": query.service_name}})
            
            if query.search_text:
                filters.append({
                    "multi_match": {
                        "query": query.search_text,
                        "fields": ["message", "extra_fields.*"]
                    }
                })
            
            if filters:
                es_query["query"]["bool"]["must"].extend(filters)
            
            # Select fields
            if query.fields:
                es_query["_source"] = query.fields
            
            # Execute search
            response = await self.es_client.search(
                index=f"{self.index_prefix}-*",
                body=es_query
            )
            
            # Extract results
            results = []
            for hit in response["hits"]["hits"]:
                results.append(hit["_source"])
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return []
    
    async def _search_local_logs(self, query: LogQuery) -> List[Dict[str, Any]]:
        """Search logs in local storage."""
        results = []
        
        try:
            # Generate list of files to search based on date range
            current_date = query.start_time.date()
            end_date = query.end_time.date()
            
            while current_date <= end_date:
                log_file = self.local_storage_path / f"logs-{current_date.strftime('%Y-%m-%d')}.jsonl"
                
                if log_file.exists():
                    async with aiofiles.open(log_file, 'r') as f:
                        async for line in f:
                            try:
                                log_doc = json.loads(line.strip())
                                
                                # Apply filters
                                log_time = datetime.fromisoformat(log_doc["timestamp"])
                                
                                if not (query.start_time <= log_time <= query.end_time):
                                    continue
                                
                                if query.level and log_doc["level"] != query.level.value:
                                    continue
                                
                                if query.source and log_doc["source"] != query.source.value:
                                    continue
                                
                                if query.service_name and log_doc["service_name"] != query.service_name:
                                    continue
                                
                                if query.search_text and query.search_text.lower() not in log_doc["message"].lower():
                                    continue
                                
                                # Apply field selection
                                if query.fields:
                                    filtered_doc = {field: log_doc.get(field) for field in query.fields}
                                    results.append(filtered_doc)
                                else:
                                    results.append(log_doc)
                                
                                # Apply limit
                                if len(results) >= query.limit + query.offset:
                                    break
                            
                            except json.JSONDecodeError:
                                continue
                
                current_date += timedelta(days=1)
            
            # Apply sorting and pagination
            if query.sort_order == "desc":
                results.sort(key=lambda x: x["timestamp"], reverse=True)
            else:
                results.sort(key=lambda x: x["timestamp"])
            
            return results[query.offset:query.offset + query.limit]
        
        except Exception as e:
            logger.error(f"Error searching local logs: {e}")
            return []
    
    def add_log_pattern(self, pattern: LogPattern) -> None:
        """Add a custom log parsing pattern."""
        self.log_patterns[pattern.name] = pattern
        logger.info(f"Added log pattern: {pattern.name}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get log management statistics."""
        return {
            "metrics": self.metrics.copy(),
            "buffer_size": len(self.log_buffer),
            "patterns_configured": len(self.log_patterns),
            "processors_configured": len(self.log_processor.processors),
            "filters_configured": len(self.log_processor.filters),
            "elasticsearch_connected": self.es_client is not None,
            "retention_days": self.retention_days,
            "running": self.running
        }


# Default log patterns for common applications
DEFAULT_LOG_PATTERNS = [
    LogPattern(
        name="fastapi_uvicorn",
        regex=r'(?P<timestamp>\S+ \S+) - (?P<level>\w+) - (?P<message>.*)',
        fields=["timestamp", "level", "message"],
        source=LogSource.API,
        sample="2023-12-22 10:30:45.123 - INFO - Application startup complete"
    ),
    LogPattern(
        name="docker_container",
        regex=r'(?P<timestamp>\S+) (?P<container_id>\w+)\[(?P<pid>\d+)\]: (?P<message>.*)',
        fields=["timestamp", "container_id", "pid", "message"],
        source=LogSource.INFRASTRUCTURE,
        sample="2023-12-22T10:30:45.123456Z abc123def456[1234]: Application started successfully"
    ),
    LogPattern(
        name="postgresql",
        regex=r'(?P<timestamp>\S+ \S+) \[(?P<pid>\d+)\] (?P<level>\w+): (?P<message>.*)',
        fields=["timestamp", "pid", "level", "message"],
        source=LogSource.DATABASE,
        sample="2023-12-22 10:30:45.123 [1234] LOG: database system is ready to accept connections"
    )
]