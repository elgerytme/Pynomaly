#!/usr/bin/env python3
"""
Advanced Reporting Engine
Comprehensive business intelligence and reporting system with dynamic report generation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
import jinja2
from concurrent.futures import ThreadPoolExecutor
import io
import base64

import asyncpg
import redis.asyncio as redis
import aiohttp
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge


# Metrics
REPORTS_GENERATED = Counter('reports_generated_total', 'Total reports generated', ['report_type', 'format', 'status'])
REPORT_GENERATION_TIME = Histogram('report_generation_seconds', 'Time spent generating reports', ['report_type'])
ACTIVE_REPORTS = Gauge('active_reports', 'Number of active reports', ['status'])
DASHBOARD_VIEWS = Counter('dashboard_views_total', 'Total dashboard views', ['dashboard_name'])
DATA_QUERIES = Counter('data_queries_total', 'Total data queries executed', ['query_type', 'status'])


class ReportType(Enum):
    """Report types."""
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    EXECUTIVE = "executive"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    POWERPOINT = "powerpoint"


class ReportFrequency(Enum):
    """Report generation frequencies."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    ON_DEMAND = "on_demand"


class ReportStatus(Enum):
    """Report generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChartType(Enum):
    """Chart types for visualizations."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    GAUGE = "gauge"


@dataclass
class DataSource:
    """Data source configuration."""
    name: str
    type: str  # postgresql, redis, http_api, file
    connection_config: Dict[str, Any]
    query_template: Optional[str] = None
    cache_ttl_seconds: int = 300
    timeout_seconds: int = 30


@dataclass
class ChartConfig:
    """Chart configuration."""
    name: str
    chart_type: ChartType
    title: str
    data_source: str
    x_column: str
    y_column: str
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReportSection:
    """Report section configuration."""
    name: str
    title: str
    description: Optional[str] = None
    charts: List[ChartConfig] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    custom_html: Optional[str] = None
    page_break: bool = False


@dataclass
class ReportTemplate:
    """Report template configuration."""
    id: str
    name: str
    description: str
    report_type: ReportType
    sections: List[ReportSection]
    parameters: Dict[str, Any] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReportRequest:
    """Report generation request."""
    id: str
    template_id: str
    parameters: Dict[str, Any]
    output_format: ReportFormat
    recipient_emails: List[str] = field(default_factory=list)
    schedule: Optional[Dict[str, Any]] = None
    priority: int = 5  # 1-10, higher = more priority
    requested_by: str = ""
    requested_at: datetime = field(default_factory=datetime.utcnow)
    status: ReportStatus = ReportStatus.PENDING


@dataclass
class GeneratedReport:
    """Generated report metadata."""
    id: str
    request_id: str
    template_id: str
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    generation_time_seconds: float = 0
    status: ReportStatus = ReportStatus.COMPLETED
    error_message: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedReportingEngine:
    """Comprehensive reporting and business intelligence engine."""
    
    def __init__(self, postgres_url: str, redis_url: str = "redis://localhost:6379/4", 
                 output_directory: str = "/tmp/reports"):
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.output_directory = Path(output_directory)
        
        # Initialize connections
        self.postgres_pool = None
        self.redis_client = None
        
        # Configuration
        self.data_sources: Dict[str, DataSource] = {}
        self.report_templates: Dict[str, ReportTemplate] = {}
        self.active_requests: Dict[str, ReportRequest] = {}
        self.generated_reports: Dict[str, GeneratedReport] = {}
        
        # Processing
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.generation_queue: asyncio.Queue = asyncio.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Templates and rendering
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates'),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Monitoring
        self.logger = logging.getLogger("reporting_engine")
        self.performance_cache: Dict[str, Any] = {}
        
        # Chart styling
        self.chart_themes = {
            'corporate': {
                'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                'background': '#ffffff',
                'grid': '#f0f0f0',
                'font_family': 'Arial, sans-serif'
            },
            'dark': {
                'colors': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'],
                'background': '#2F3136',
                'grid': '#444444',
                'font_family': 'Arial, sans-serif'
            }
        }
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the reporting engine."""
        try:
            self.logger.info("Initializing advanced reporting engine...")
            
            # Initialize PostgreSQL connection pool
            self.postgres_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Initialize Redis
            self.redis_client = redis.Redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load existing templates and data sources
            await self._load_configurations()
            
            # Initialize default templates
            await self._create_default_templates()
            
            self.logger.info("Advanced reporting engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reporting engine: {e}")
            raise
    
    async def _load_configurations(self) -> None:
        """Load configurations from database."""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Load data sources
                data_sources = await conn.fetch(
                    "SELECT * FROM report_data_sources WHERE active = true"
                )
                for row in data_sources:
                    source = DataSource(
                        name=row['name'],
                        type=row['type'],
                        connection_config=json.loads(row['connection_config']),
                        query_template=row['query_template'],
                        cache_ttl_seconds=row['cache_ttl_seconds'],
                        timeout_seconds=row['timeout_seconds']
                    )
                    self.data_sources[source.name] = source
                
                # Load report templates
                templates = await conn.fetch(
                    "SELECT * FROM report_templates WHERE active = true"
                )
                for row in templates:
                    template = ReportTemplate(
                        id=row['id'],
                        name=row['name'],
                        description=row['description'],
                        report_type=ReportType(row['report_type']),
                        sections=json.loads(row['sections']),
                        parameters=json.loads(row['parameters']),
                        styling=json.loads(row['styling']),
                        metadata=json.loads(row['metadata']),
                        created_by=row['created_by'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    self.report_templates[template.id] = template
            
            self.logger.info(f"Loaded {len(self.data_sources)} data sources and {len(self.report_templates)} templates")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
    
    async def _create_default_templates(self) -> None:
        """Create default report templates."""
        try:
            # Executive Dashboard Template
            executive_template = ReportTemplate(
                id="executive_dashboard",
                name="Executive Dashboard",
                description="High-level executive metrics and KPIs",
                report_type=ReportType.EXECUTIVE,
                sections=[
                    ReportSection(
                        name="kpi_overview",
                        title="Key Performance Indicators",
                        charts=[
                            ChartConfig(
                                name="revenue_trend",
                                chart_type=ChartType.LINE,
                                title="Revenue Trend",
                                data_source="financial_db",
                                x_column="date",
                                y_column="revenue"
                            ),
                            ChartConfig(
                                name="user_growth",
                                chart_type=ChartType.BAR,
                                title="User Growth",
                                data_source="analytics_db",
                                x_column="month",
                                y_column="active_users"
                            )
                        ]
                    ),
                    ReportSection(
                        name="performance_metrics",
                        title="System Performance",
                        charts=[
                            ChartConfig(
                                name="response_time",
                                chart_type=ChartType.LINE,
                                title="Average Response Time",
                                data_source="metrics_db",
                                x_column="timestamp",
                                y_column="avg_response_time"
                            )
                        ]
                    )
                ],
                styling={
                    'theme': 'corporate',
                    'font_size': 12,
                    'include_summary': True
                }
            )
            
            await self._save_template(executive_template)
            
            # Operational Report Template
            operational_template = ReportTemplate(
                id="operational_report",
                name="Daily Operations Report",
                description="Daily operational metrics and alerts",
                report_type=ReportType.OPERATIONAL,
                sections=[
                    ReportSection(
                        name="system_health",
                        title="System Health Overview",
                        charts=[
                            ChartConfig(
                                name="error_rates",
                                chart_type=ChartType.LINE,
                                title="Error Rates by Service",
                                data_source="logs_db",
                                x_column="hour",
                                y_column="error_count",
                                color_column="service"
                            ),
                            ChartConfig(
                                name="resource_usage",
                                chart_type=ChartType.HEATMAP,
                                title="Resource Usage Heatmap",
                                data_source="metrics_db",
                                x_column="time",
                                y_column="service"
                            )
                        ]
                    )
                ]
            )
            
            await self._save_template(operational_template)
            
            self.logger.info("Created default report templates")
            
        except Exception as e:
            self.logger.error(f"Failed to create default templates: {e}")
    
    async def _save_template(self, template: ReportTemplate) -> None:
        """Save report template to database."""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO report_templates (
                        id, name, description, report_type, sections, 
                        parameters, styling, metadata, created_by, created_at, updated_at, active
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, true)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        sections = EXCLUDED.sections,
                        parameters = EXCLUDED.parameters,
                        styling = EXCLUDED.styling,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                """, 
                template.id, template.name, template.description, template.report_type.value,
                json.dumps([asdict(section) for section in template.sections]),
                json.dumps(template.parameters), json.dumps(template.styling),
                json.dumps(template.metadata), template.created_by,
                template.created_at, template.updated_at)
                
            self.report_templates[template.id] = template
            
        except Exception as e:
            self.logger.error(f"Failed to save template: {e}")
            raise
    
    def register_data_source(self, data_source: DataSource) -> None:
        """Register a data source."""
        self.data_sources[data_source.name] = data_source
        
        # Save to database
        asyncio.create_task(self._save_data_source(data_source))
        
        self.logger.info(f"Registered data source: {data_source.name}")
    
    async def _save_data_source(self, data_source: DataSource) -> None:
        """Save data source to database."""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO report_data_sources (
                        name, type, connection_config, query_template, 
                        cache_ttl_seconds, timeout_seconds, active
                    ) VALUES ($1, $2, $3, $4, $5, $6, true)
                    ON CONFLICT (name) DO UPDATE SET
                        type = EXCLUDED.type,
                        connection_config = EXCLUDED.connection_config,
                        query_template = EXCLUDED.query_template,
                        cache_ttl_seconds = EXCLUDED.cache_ttl_seconds,
                        timeout_seconds = EXCLUDED.timeout_seconds
                """,
                data_source.name, data_source.type, 
                json.dumps(data_source.connection_config),
                data_source.query_template, data_source.cache_ttl_seconds,
                data_source.timeout_seconds)
                
        except Exception as e:
            self.logger.error(f"Failed to save data source: {e}")
    
    async def start(self) -> None:
        """Start the reporting engine."""
        if self.is_running:
            self.logger.warning("Reporting engine is already running")
            return
        
        self.is_running = True
        self.logger.info("Starting advanced reporting engine...")
        
        # Start worker tasks
        self.worker_tasks = [
            asyncio.create_task(self._report_generation_worker()),
            asyncio.create_task(self._scheduled_reports_worker()),
            asyncio.create_task(self._cleanup_worker()),
            asyncio.create_task(self._performance_monitoring_worker())
        ]
        
        self.logger.info(f"Started {len(self.worker_tasks)} reporting worker tasks")
    
    async def generate_report(self, request: ReportRequest) -> str:
        """Generate a report."""
        try:
            # Validate request
            if request.template_id not in self.report_templates:
                raise ValueError(f"Template {request.template_id} not found")
            
            # Add to queue
            self.active_requests[request.id] = request
            await self.generation_queue.put(request)
            
            ACTIVE_REPORTS.labels(status="pending").inc()
            self.logger.info(f"Queued report generation: {request.id}")
            
            return request.id
            
        except Exception as e:
            self.logger.error(f"Failed to queue report generation: {e}")
            raise
    
    async def _report_generation_worker(self) -> None:
        """Report generation worker."""
        self.logger.info("Started report generation worker")
        
        while self.is_running:
            try:
                # Get request from queue
                request = await asyncio.wait_for(self.generation_queue.get(), timeout=1.0)
                
                # Generate report
                await self._generate_single_report(request)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in report generation worker: {e}")
                await asyncio.sleep(5)
    
    async def _generate_single_report(self, request: ReportRequest) -> None:
        """Generate a single report."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating report: {request.id}")
            
            # Update status
            request.status = ReportStatus.GENERATING
            ACTIVE_REPORTS.labels(status="generating").inc()
            ACTIVE_REPORTS.labels(status="pending").dec()
            
            # Get template
            template = self.report_templates[request.template_id]
            
            # Generate report content
            report_data = await self._collect_report_data(template, request.parameters)
            charts = await self._generate_charts(template, report_data)
            
            # Render report
            output_path = await self._render_report(
                template, report_data, charts, request.output_format, request.id
            )
            
            # Create report record
            generation_time = time.time() - start_time
            generated_report = GeneratedReport(
                id=str(uuid.uuid4()),
                request_id=request.id,
                template_id=request.template_id,
                file_path=str(output_path),
                file_size_bytes=output_path.stat().st_size if output_path.exists() else 0,
                generation_time_seconds=generation_time,
                status=ReportStatus.COMPLETED,
                metadata={
                    'parameters': request.parameters,
                    'output_format': request.output_format.value,
                    'charts_generated': len(charts)
                }
            )
            
            # Store report
            self.generated_reports[generated_report.id] = generated_report
            await self._save_generated_report(generated_report)
            
            # Send if recipients specified
            if request.recipient_emails:
                await self._send_report(generated_report, request.recipient_emails)
            
            # Update metrics
            request.status = ReportStatus.COMPLETED
            REPORTS_GENERATED.labels(
                report_type=template.report_type.value,
                format=request.output_format.value,
                status="success"
            ).inc()
            REPORT_GENERATION_TIME.labels(report_type=template.report_type.value).observe(generation_time)
            ACTIVE_REPORTS.labels(status="generating").dec()
            ACTIVE_REPORTS.labels(status="completed").inc()
            
            self.logger.info(f"Report generated successfully: {request.id} in {generation_time:.2f}s")
            
        except Exception as e:
            # Handle generation error
            generation_time = time.time() - start_time
            error_message = str(e)
            
            generated_report = GeneratedReport(
                id=str(uuid.uuid4()),
                request_id=request.id,
                template_id=request.template_id,
                generation_time_seconds=generation_time,
                status=ReportStatus.FAILED,
                error_message=error_message
            )
            
            self.generated_reports[generated_report.id] = generated_report
            await self._save_generated_report(generated_report)
            
            request.status = ReportStatus.FAILED
            REPORTS_GENERATED.labels(
                report_type=template.report_type.value,
                format=request.output_format.value,
                status="error"
            ).inc()
            ACTIVE_REPORTS.labels(status="generating").dec()
            
            self.logger.error(f"Report generation failed: {request.id} - {error_message}")
    
    async def _collect_report_data(self, template: ReportTemplate, parameters: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Collect data for report generation."""
        report_data = {}
        
        try:
            # Get unique data sources from template
            data_source_names = set()
            for section in template.sections:
                for chart in section.charts:
                    data_source_names.add(chart.data_source)
                for table in section.tables:
                    data_source_names.add(table.get('data_source'))
            
            # Collect data from each source
            for source_name in data_source_names:
                if source_name and source_name in self.data_sources:
                    data = await self._query_data_source(source_name, parameters)
                    if data is not None:
                        report_data[source_name] = data
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect report data: {e}")
            raise
    
    async def _query_data_source(self, source_name: str, parameters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Query data from a specific data source."""
        try:
            data_source = self.data_sources[source_name]
            
            # Check cache first
            cache_key = f"data_source:{source_name}:{hash(str(parameters))}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                DATA_QUERIES.labels(query_type=data_source.type, status="cache_hit").inc()
                return pd.read_json(cached_data)
            
            # Query based on data source type
            if data_source.type == "postgresql":
                data = await self._query_postgresql(data_source, parameters)
            elif data_source.type == "redis":
                data = await self._query_redis(data_source, parameters)
            elif data_source.type == "http_api":
                data = await self._query_http_api(data_source, parameters)
            elif data_source.type == "file":
                data = await self._query_file(data_source, parameters)
            else:
                self.logger.warning(f"Unsupported data source type: {data_source.type}")
                return None
            
            # Cache the result
            if data is not None:
                await self.redis_client.setex(
                    cache_key, 
                    data_source.cache_ttl_seconds, 
                    data.to_json()
                )
                DATA_QUERIES.labels(query_type=data_source.type, status="success").inc()
            
            return data
            
        except Exception as e:
            DATA_QUERIES.labels(query_type=data_source.type, status="error").inc()
            self.logger.error(f"Failed to query data source {source_name}: {e}")
            return None
    
    async def _query_postgresql(self, data_source: DataSource, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Query PostgreSQL data source."""
        try:
            # Render query template
            if data_source.query_template:
                query = jinja2.Template(data_source.query_template).render(**parameters)
            else:
                # Default query
                table_name = parameters.get('table_name', 'data')
                query = f"SELECT * FROM {table_name} ORDER BY created_at DESC LIMIT 1000"
            
            # Execute query
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query)
                
                if rows:
                    # Convert to DataFrame
                    data = pd.DataFrame([dict(row) for row in rows])
                    return data
                else:
                    return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"PostgreSQL query failed: {e}")
            raise
    
    async def _query_redis(self, data_source: DataSource, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Query Redis data source."""
        try:
            redis_client = redis.Redis.from_url(data_source.connection_config['url'])
            
            # Example: Get time series data
            key_pattern = parameters.get('key_pattern', '*')
            keys = await redis_client.keys(key_pattern)
            
            data_rows = []
            for key in keys[:1000]:  # Limit results
                value = await redis_client.get(key)
                if value:
                    try:
                        data_rows.append(json.loads(value))
                    except json.JSONDecodeError:
                        data_rows.append({'key': key.decode(), 'value': value.decode()})
            
            return pd.DataFrame(data_rows)
            
        except Exception as e:
            self.logger.error(f"Redis query failed: {e}")
            raise
    
    async def _query_http_api(self, data_source: DataSource, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Query HTTP API data source."""
        try:
            config = data_source.connection_config
            url = config['base_url']
            
            # Add parameters to URL or headers
            if 'endpoint' in parameters:
                url = f"{url}/{parameters['endpoint']}"
            
            headers = config.get('headers', {})
            timeout = aiohttp.ClientTimeout(total=data_source.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Convert to DataFrame
                        if isinstance(data, list):
                            return pd.DataFrame(data)
                        elif isinstance(data, dict) and 'data' in data:
                            return pd.DataFrame(data['data'])
                        else:
                            return pd.DataFrame([data])
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
            
        except Exception as e:
            self.logger.error(f"HTTP API query failed: {e}")
            raise
    
    async def _query_file(self, data_source: DataSource, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Query file data source."""
        try:
            file_path = Path(data_source.connection_config['file_path'])
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read based on file extension
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                return pd.read_json(file_path)
            elif file_path.suffix.lower() == '.parquet':
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        except Exception as e:
            self.logger.error(f"File query failed: {e}")
            raise
    
    async def _generate_charts(self, template: ReportTemplate, report_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Generate charts for the report."""
        charts = {}
        
        try:
            theme = template.styling.get('theme', 'corporate')
            theme_config = self.chart_themes.get(theme, self.chart_themes['corporate'])
            
            for section in template.sections:
                for chart_config in section.charts:
                    if chart_config.data_source in report_data:
                        data = report_data[chart_config.data_source]
                        
                        if not data.empty:
                            chart_html = await self._generate_single_chart(chart_config, data, theme_config)
                            charts[chart_config.name] = chart_html
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Failed to generate charts: {e}")
            return {}
    
    async def _generate_single_chart(self, chart_config: ChartConfig, data: pd.DataFrame, theme_config: Dict[str, Any]) -> str:
        """Generate a single chart."""
        try:
            # Apply filters
            filtered_data = data.copy()
            for filter_column, filter_value in chart_config.filters.items():
                if filter_column in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[filter_column] == filter_value]
            
            # Create chart based on type
            fig = None
            
            if chart_config.chart_type == ChartType.LINE:
                fig = px.line(
                    filtered_data,
                    x=chart_config.x_column,
                    y=chart_config.y_column,
                    color=chart_config.color_column,
                    title=chart_config.title
                )
            
            elif chart_config.chart_type == ChartType.BAR:
                fig = px.bar(
                    filtered_data,
                    x=chart_config.x_column,
                    y=chart_config.y_column,
                    color=chart_config.color_column,
                    title=chart_config.title
                )
            
            elif chart_config.chart_type == ChartType.PIE:
                fig = px.pie(
                    filtered_data,
                    names=chart_config.x_column,
                    values=chart_config.y_column,
                    title=chart_config.title
                )
            
            elif chart_config.chart_type == ChartType.SCATTER:
                fig = px.scatter(
                    filtered_data,
                    x=chart_config.x_column,
                    y=chart_config.y_column,
                    color=chart_config.color_column,
                    size=chart_config.size_column,
                    title=chart_config.title
                )
            
            elif chart_config.chart_type == ChartType.HEATMAP:
                # Pivot data for heatmap
                heatmap_data = filtered_data.pivot_table(
                    index=chart_config.y_column,
                    columns=chart_config.x_column,
                    values=chart_config.color_column or filtered_data.columns[0],
                    aggfunc='mean'
                )
                fig = px.imshow(heatmap_data, title=chart_config.title)
            
            elif chart_config.chart_type == ChartType.HISTOGRAM:
                fig = px.histogram(
                    filtered_data,
                    x=chart_config.x_column,
                    title=chart_config.title
                )
            
            elif chart_config.chart_type == ChartType.BOX:
                fig = px.box(
                    filtered_data,
                    x=chart_config.x_column,
                    y=chart_config.y_column,
                    title=chart_config.title
                )
            
            # Apply theme
            if fig:
                fig.update_layout(
                    plot_bgcolor=theme_config['background'],
                    paper_bgcolor=theme_config['background'],
                    font_family=theme_config['font_family'],
                    colorway=theme_config['colors']
                )
                
                # Add annotations
                for annotation in chart_config.annotations:
                    fig.add_annotation(**annotation)
                
                # Convert to HTML
                return fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{chart_config.name}")
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to generate chart {chart_config.name}: {e}")
            return f"<div>Error generating chart: {chart_config.name}</div>"
    
    async def _render_report(self, template: ReportTemplate, report_data: Dict[str, pd.DataFrame], 
                           charts: Dict[str, str], output_format: ReportFormat, request_id: str) -> Path:
        """Render the final report."""
        try:
            # Generate HTML content first
            html_content = await self._generate_html_content(template, report_data, charts)
            
            # Output file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{template.name.replace(' ', '_')}_{timestamp}_{request_id[:8]}"
            
            if output_format == ReportFormat.HTML:
                output_path = self.output_directory / f"{filename}.html"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
            elif output_format == ReportFormat.PDF:
                # Convert HTML to PDF using weasyprint or similar
                output_path = self.output_directory / f"{filename}.pdf"
                await self._html_to_pdf(html_content, output_path)
                
            elif output_format == ReportFormat.EXCEL:
                output_path = self.output_directory / f"{filename}.xlsx"
                await self._generate_excel_report(template, report_data, output_path)
                
            elif output_format == ReportFormat.CSV:
                output_path = self.output_directory / f"{filename}.csv"
                await self._generate_csv_report(report_data, output_path)
                
            elif output_format == ReportFormat.JSON:
                output_path = self.output_directory / f"{filename}.json"
                await self._generate_json_report(template, report_data, charts, output_path)
                
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to render report: {e}")
            raise
    
    async def _generate_html_content(self, template: ReportTemplate, report_data: Dict[str, pd.DataFrame], 
                                   charts: Dict[str, str]) -> str:
        """Generate HTML content for the report."""
        try:
            # Create Jinja2 template
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ template.name }}</title>
                <meta charset="utf-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .section { margin-bottom: 40px; page-break-inside: avoid; }
                    .chart { margin: 20px 0; }
                    .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    .table th { background-color: #f2f2f2; }
                    .summary { background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }
                    {% if template.styling.get('theme') == 'dark' %}
                    body { background-color: #2F3136; color: white; }
                    .table th { background-color: #444444; }
                    .summary { background-color: #404040; }
                    {% endif %}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div class="header">
                    <h1>{{ template.name }}</h1>
                    <p>{{ template.description }}</p>
                    <p><strong>Generated:</strong> {{ generated_at }}</p>
                </div>
                
                {% for section in template.sections %}
                <div class="section">
                    <h2>{{ section.title }}</h2>
                    {% if section.description %}
                    <p>{{ section.description }}</p>
                    {% endif %}
                    
                    {% for chart in section.charts %}
                    <div class="chart">
                        {{ charts.get(chart.name, '') | safe }}
                    </div>
                    {% endfor %}
                    
                    {% for table_config in section.tables %}
                    <div class="table-container">
                        <h3>{{ table_config.title }}</h3>
                        {{ render_table(report_data.get(table_config.data_source)) }}
                    </div>
                    {% endfor %}
                    
                    {% for text_block in section.text_blocks %}
                    <div class="text-block">
                        <h3>{{ text_block.title }}</h3>
                        <p>{{ text_block.content }}</p>
                    </div>
                    {% endfor %}
                    
                    {% if section.custom_html %}
                    {{ section.custom_html | safe }}
                    {% endif %}
                </div>
                {% endfor %}
                
                <div class="footer">
                    <p><small>Report generated by Advanced Reporting Engine on {{ generated_at }}</small></p>
                </div>
            </body>
            </html>
            """
            
            def render_table(data):
                if data is None or data.empty:
                    return "<p>No data available</p>"
                
                html = '<table class="table"><thead><tr>'
                for col in data.columns:
                    html += f'<th>{col}</th>'
                html += '</tr></thead><tbody>'
                
                for _, row in data.head(100).iterrows():  # Limit to 100 rows
                    html += '<tr>'
                    for val in row:
                        html += f'<td>{val}</td>'
                    html += '</tr>'
                
                html += '</tbody></table>'
                return html
            
            # Render template
            jinja_template = jinja2.Template(html_template)
            html_content = jinja_template.render(
                template=template,
                report_data=report_data,
                charts=charts,
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                render_table=render_table
            )
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML content: {e}")
            raise
    
    async def _html_to_pdf(self, html_content: str, output_path: Path) -> None:
        """Convert HTML to PDF."""
        try:
            # Use weasyprint for HTML to PDF conversion
            import weasyprint
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.thread_pool,
                lambda: weasyprint.HTML(string=html_content).write_pdf(str(output_path))
            )
            
        except ImportError:
            self.logger.error("weasyprint not installed, cannot generate PDF")
            raise Exception("PDF generation not available")
        except Exception as e:
            self.logger.error(f"Failed to convert HTML to PDF: {e}")
            raise
    
    async def _generate_excel_report(self, template: ReportTemplate, report_data: Dict[str, pd.DataFrame], 
                                   output_path: Path) -> None:
        """Generate Excel report."""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Write summary sheet
                summary_data = {
                    'Report Name': [template.name],
                    'Description': [template.description],
                    'Generated At': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Data Sources': [', '.join(report_data.keys())]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Write data sheets
                for source_name, data in report_data.items():
                    if not data.empty:
                        sheet_name = source_name[:30]  # Excel sheet name limit
                        data.to_excel(writer, sheet_name=sheet_name, index=False)
            
        except Exception as e:
            self.logger.error(f"Failed to generate Excel report: {e}")
            raise
    
    async def _generate_csv_report(self, report_data: Dict[str, pd.DataFrame], output_path: Path) -> None:
        """Generate CSV report."""
        try:
            # Combine all data sources
            combined_data = pd.DataFrame()
            
            for source_name, data in report_data.items():
                if not data.empty:
                    data_copy = data.copy()
                    data_copy['data_source'] = source_name
                    combined_data = pd.concat([combined_data, data_copy], ignore_index=True)
            
            combined_data.to_csv(output_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Failed to generate CSV report: {e}")
            raise
    
    async def _generate_json_report(self, template: ReportTemplate, report_data: Dict[str, pd.DataFrame], 
                                  charts: Dict[str, str], output_path: Path) -> None:
        """Generate JSON report."""
        try:
            report_json = {
                'template': asdict(template),
                'generated_at': datetime.now().isoformat(),
                'data': {},
                'charts': charts
            }
            
            # Convert DataFrames to JSON
            for source_name, data in report_data.items():
                if not data.empty:
                    report_json['data'][source_name] = data.to_dict('records')
            
            with open(output_path, 'w') as f:
                json.dump(report_json, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            raise
    
    async def _save_generated_report(self, report: GeneratedReport) -> None:
        """Save generated report metadata to database."""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO generated_reports (
                        id, request_id, template_id, file_path, file_size_bytes,
                        generation_time_seconds, status, error_message, generated_at, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                report.id, report.request_id, report.template_id, report.file_path,
                report.file_size_bytes, report.generation_time_seconds, report.status.value,
                report.error_message, report.generated_at, json.dumps(report.metadata))
                
        except Exception as e:
            self.logger.error(f"Failed to save generated report: {e}")
    
    async def _send_report(self, report: GeneratedReport, recipients: List[str]) -> None:
        """Send report to recipients."""
        try:
            # Placeholder for email sending logic
            # In production, integrate with email service (SMTP, SendGrid, etc.)
            self.logger.info(f"Would send report {report.id} to {len(recipients)} recipients")
            
        except Exception as e:
            self.logger.error(f"Failed to send report: {e}")
    
    async def _scheduled_reports_worker(self) -> None:
        """Worker for scheduled report generation."""
        while self.is_running:
            try:
                # Check for scheduled reports (placeholder)
                # In production, implement scheduling logic
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in scheduled reports worker: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_worker(self) -> None:
        """Cleanup worker for old reports and cache."""
        while self.is_running:
            try:
                # Clean up old report files
                cutoff_time = datetime.now() - timedelta(days=30)
                
                for report_file in self.output_directory.glob("*"):
                    if report_file.is_file():
                        file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                        if file_time < cutoff_time:
                            report_file.unlink()
                            self.logger.debug(f"Cleaned up old report file: {report_file}")
                
                # Clean up old database records
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM generated_reports WHERE generated_at < $1",
                        cutoff_time
                    )
                
                await asyncio.sleep(86400)  # Run daily
                
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_monitoring_worker(self) -> None:
        """Monitor performance and cache metrics."""
        while self.is_running:
            try:
                # Update performance cache
                self.performance_cache.update({
                    'timestamp': datetime.now().isoformat(),
                    'active_requests': len(self.active_requests),
                    'generated_reports': len(self.generated_reports),
                    'data_sources': len(self.data_sources),
                    'templates': len(self.report_templates),
                    'queue_size': self.generation_queue.qsize()
                })
                
                # Store in Redis
                await self.redis_client.setex(
                    "reporting_engine_metrics",
                    300,
                    json.dumps(self.performance_cache)
                )
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(120)
    
    async def get_report_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get report generation status."""
        try:
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                return {
                    'request_id': request_id,
                    'status': request.status.value,
                    'template_id': request.template_id,
                    'requested_at': request.requested_at.isoformat(),
                    'output_format': request.output_format.value
                }
            
            # Check in generated reports
            for report in self.generated_reports.values():
                if report.request_id == request_id:
                    return {
                        'request_id': request_id,
                        'status': report.status.value,
                        'file_path': report.file_path,
                        'file_size_bytes': report.file_size_bytes,
                        'generation_time_seconds': report.generation_time_seconds,
                        'generated_at': report.generated_at.isoformat(),
                        'error_message': report.error_message
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get report status: {e}")
            return None
    
    async def stop(self) -> None:
        """Stop the reporting engine."""
        self.logger.info("Stopping advanced reporting engine...")
        self.is_running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close connections
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Advanced reporting engine stopped")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get reporting engine statistics."""
        return {
            'is_running': self.is_running,
            'active_requests': len(self.active_requests),
            'generated_reports': len(self.generated_reports),
            'data_sources': len(self.data_sources),
            'report_templates': len(self.report_templates),
            'queue_size': self.generation_queue.qsize(),
            'worker_tasks': len(self.worker_tasks),
            'performance_cache': self.performance_cache
        }


# Example usage
async def create_reporting_engine():
    """Create and configure reporting engine."""
    postgres_url = "postgresql://reporting:password@localhost:5432/reporting_db"
    
    engine = AdvancedReportingEngine(postgres_url)
    await engine.initialize()
    
    # Register example data sources
    financial_db = DataSource(
        name="financial_db",
        type="postgresql",
        connection_config={"database_url": postgres_url},
        query_template="SELECT date, revenue FROM financial_metrics WHERE date >= '{{ start_date }}' AND date <= '{{ end_date }}' ORDER BY date",
        cache_ttl_seconds=1800
    )
    engine.register_data_source(financial_db)
    
    # Start engine
    await engine.start()
    
    return engine