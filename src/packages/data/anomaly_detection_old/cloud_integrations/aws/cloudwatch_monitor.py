"""
AWS CloudWatch Monitor for Pynomaly Detection
==============================================

Comprehensive monitoring, logging, and alerting using CloudWatch.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CloudWatchConfig:
    """CloudWatch configuration."""
    region: str = "us-east-1"
    namespace: str = "Pynomaly/Detection"
    log_group_name: str = "/aws/pynomaly/detection"
    log_stream_prefix: str = "detection-"
    retention_days: int = 30
    enable_detailed_monitoring: bool = True
    alert_sns_topic_arn: Optional[str] = None
    dashboard_name: str = "Pynomaly-Detection-Dashboard"
    
@dataclass
class MetricData:
    """CloudWatch metric data."""
    metric_name: str
    value: Union[int, float]
    unit: str = "Count"
    dimensions: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
class CloudWatchMonitor:
    """AWS CloudWatch monitoring for Pynomaly Detection."""
    
    def __init__(self, config: CloudWatchConfig, profile_name: Optional[str] = None):
        """Initialize CloudWatch monitor.
        
        Args:
            config: CloudWatch configuration
            profile_name: AWS profile name (optional)
        """
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK (boto3) is required for CloudWatch integration")
        
        self.config = config
        self.profile_name = profile_name
        
        # Initialize AWS clients
        session = boto3.Session(profile_name=profile_name)
        self.cloudwatch = session.client('cloudwatch', region_name=config.region)
        self.logs = session.client('logs', region_name=config.region)
        
        # Initialize monitoring
        self._setup_log_group()
        self._setup_dashboard()
        
        logger.info(f"CloudWatch Monitor initialized for namespace: {config.namespace}")
    
    def publish_metric(self, metric: MetricData) -> bool:
        """Publish metric to CloudWatch.
        
        Args:
            metric: Metric data to publish
            
        Returns:
            True if successful
        """
        try:
            metric_data = {
                'MetricName': metric.metric_name,
                'Value': metric.value,
                'Unit': metric.unit
            }
            
            if metric.dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in metric.dimensions.items()
                ]
            
            if metric.timestamp:
                metric_data['Timestamp'] = metric.timestamp
            
            self.cloudwatch.put_metric_data(
                Namespace=self.config.namespace,
                MetricData=[metric_data]
            )
            
            logger.debug(f"Published metric: {metric.metric_name} = {metric.value}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to publish metric: {e}")
            return False
    
    def publish_metrics_batch(self, metrics: List[MetricData]) -> bool:
        """Publish multiple metrics to CloudWatch.
        
        Args:
            metrics: List of metrics to publish
            
        Returns:
            True if successful
        """
        try:
            # CloudWatch allows max 20 metrics per request
            batch_size = 20
            
            for i in range(0, len(metrics), batch_size):
                batch = metrics[i:i + batch_size]
                metric_data = []
                
                for metric in batch:
                    data = {
                        'MetricName': metric.metric_name,
                        'Value': metric.value,
                        'Unit': metric.unit
                    }
                    
                    if metric.dimensions:
                        data['Dimensions'] = [
                            {'Name': k, 'Value': v} for k, v in metric.dimensions.items()
                        ]
                    
                    if metric.timestamp:
                        data['Timestamp'] = metric.timestamp
                    
                    metric_data.append(data)
                
                self.cloudwatch.put_metric_data(
                    Namespace=self.config.namespace,
                    MetricData=metric_data
                )
            
            logger.info(f"Published {len(metrics)} metrics to CloudWatch")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to publish metrics batch: {e}")
            return False
    
    def log_detection_event(self, event_data: Dict[str, Any], 
                           log_stream_name: Optional[str] = None) -> bool:
        """Log detection event to CloudWatch Logs.
        
        Args:
            event_data: Event data to log
            log_stream_name: Log stream name (auto-generated if None)
            
        Returns:
            True if successful
        """
        try:
            if log_stream_name is None:
                log_stream_name = f"{self.config.log_stream_prefix}{datetime.now().strftime('%Y%m%d')}"
            
            # Ensure log stream exists
            self._ensure_log_stream(log_stream_name)
            
            # Prepare log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'detection',
                'data': event_data
            }
            
            # Put log event
            self.logs.put_log_events(
                logGroupName=self.config.log_group_name,
                logStreamName=log_stream_name,
                logEvents=[
                    {
                        'timestamp': int(time.time() * 1000),
                        'message': json.dumps(log_entry)
                    }
                ]
            )
            
            logger.debug(f"Logged detection event to: {log_stream_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to log detection event: {e}")
            return False
    
    def create_alarm(self, alarm_name: str, metric_name: str, 
                    threshold: float, comparison_operator: str,
                    evaluation_periods: int = 2, period: int = 300,
                    statistic: str = "Average", dimensions: Optional[Dict] = None,
                    alarm_description: Optional[str] = None) -> bool:
        """Create CloudWatch alarm.
        
        Args:
            alarm_name: Alarm name
            metric_name: Metric name to monitor
            threshold: Alarm threshold
            comparison_operator: Comparison operator (e.g., 'GreaterThanThreshold')
            evaluation_periods: Number of periods to evaluate
            period: Period in seconds
            statistic: Statistic to use (Average, Sum, etc.)
            dimensions: Metric dimensions
            alarm_description: Alarm description
            
        Returns:
            True if successful
        """
        try:
            alarm_config = {
                'AlarmName': alarm_name,
                'AlarmDescription': alarm_description or f"Alarm for {metric_name}",
                'MetricName': metric_name,
                'Namespace': self.config.namespace,
                'Statistic': statistic,
                'Period': period,
                'EvaluationPeriods': evaluation_periods,
                'Threshold': threshold,
                'ComparisonOperator': comparison_operator,
                'TreatMissingData': 'notBreaching'
            }
            
            if dimensions:
                alarm_config['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            if self.config.alert_sns_topic_arn:
                alarm_config['AlarmActions'] = [self.config.alert_sns_topic_arn]
                alarm_config['OKActions'] = [self.config.alert_sns_topic_arn]
            
            self.cloudwatch.put_metric_alarm(**alarm_config)
            
            logger.info(f"Created CloudWatch alarm: {alarm_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to create alarm: {e}")
            return False
    
    def setup_detection_alarms(self) -> bool:
        """Setup standard detection alarms.
        
        Returns:
            True if successful
        """
        try:
            alarms = [
                {
                    'name': 'Pynomaly-HighErrorRate',
                    'metric': 'ErrorRate',
                    'threshold': 0.05,
                    'operator': 'GreaterThanThreshold',
                    'description': 'High error rate detected'
                },
                {
                    'name': 'Pynomaly-HighLatency',
                    'metric': 'DetectionLatency',
                    'threshold': 1000,
                    'operator': 'GreaterThanThreshold',
                    'description': 'High detection latency'
                },
                {
                    'name': 'Pynomaly-LowThroughput',
                    'metric': 'DetectionThroughput',
                    'threshold': 10,
                    'operator': 'LessThanThreshold',
                    'description': 'Low detection throughput'
                },
                {
                    'name': 'Pynomaly-HighMemoryUsage',
                    'metric': 'MemoryUtilization',
                    'threshold': 85,
                    'operator': 'GreaterThanThreshold',
                    'description': 'High memory usage'
                }
            ]
            
            for alarm in alarms:
                self.create_alarm(
                    alarm_name=alarm['name'],
                    metric_name=alarm['metric'],
                    threshold=alarm['threshold'],
                    comparison_operator=alarm['operator'],
                    alarm_description=alarm['description']
                )
            
            logger.info("Setup detection alarms completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup detection alarms: {e}")
            return False
    
    def get_metrics(self, metric_name: str, start_time: datetime, 
                   end_time: datetime, period: int = 300,
                   statistic: str = "Average", dimensions: Optional[Dict] = None) -> List[Dict]:
        """Get metric statistics from CloudWatch.
        
        Args:
            metric_name: Metric name
            start_time: Start time
            end_time: End time
            period: Period in seconds
            statistic: Statistic to retrieve
            dimensions: Metric dimensions
            
        Returns:
            List of metric data points
        """
        try:
            get_metric_config = {
                'Namespace': self.config.namespace,
                'MetricName': metric_name,
                'StartTime': start_time,
                'EndTime': end_time,
                'Period': period,
                'Statistics': [statistic]
            }
            
            if dimensions:
                get_metric_config['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            response = self.cloudwatch.get_metric_statistics(**get_metric_config)
            
            datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
            
            return [
                {
                    'timestamp': dp['Timestamp'].isoformat(),
                    'value': dp[statistic],
                    'unit': dp['Unit']
                }
                for dp in datapoints
            ]
            
        except ClientError as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
    
    def create_detection_dashboard(self) -> bool:
        """Create CloudWatch dashboard for detection monitoring.
        
        Returns:
            True if successful
        """
        try:
            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [self.config.namespace, "DetectionCount"],
                                [self.config.namespace, "AnomalyCount"],
                                [self.config.namespace, "ErrorCount"]
                            ],
                            "period": 300,
                            "stat": "Sum",
                            "region": self.config.region,
                            "title": "Detection Metrics"
                        }
                    },
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [self.config.namespace, "DetectionLatency"],
                                [self.config.namespace, "ProcessingTime"]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config.region,
                            "title": "Performance Metrics"
                        }
                    },
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [self.config.namespace, "MemoryUtilization"],
                                [self.config.namespace, "CPUUtilization"]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config.region,
                            "title": "Resource Utilization"
                        }
                    },
                    {
                        "type": "log",
                        "x": 12,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "query": f"SOURCE '{self.config.log_group_name}' | fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 20",
                            "region": self.config.region,
                            "title": "Recent Errors"
                        }
                    }
                ]
            }
            
            self.cloudwatch.put_dashboard(
                DashboardName=self.config.dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            logger.info(f"Created CloudWatch dashboard: {self.config.dashboard_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to create dashboard: {e}")
            return False
    
    def get_dashboard_url(self) -> str:
        """Get CloudWatch dashboard URL.
        
        Returns:
            Dashboard URL
        """
        return (
            f"https://{self.config.region}.console.aws.amazon.com/cloudwatch/home"
            f"?region={self.config.region}#dashboards:name={self.config.dashboard_name}"
        )
    
    def query_logs(self, query: str, start_time: datetime, 
                  end_time: datetime, limit: int = 1000) -> List[Dict]:
        """Query CloudWatch Logs.
        
        Args:
            query: CloudWatch Logs Insights query
            start_time: Start time
            end_time: End time
            limit: Maximum number of results
            
        Returns:
            Query results
        """
        try:
            response = self.logs.start_query(
                logGroupName=self.config.log_group_name,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query,
                limit=limit
            )
            
            query_id = response['queryId']
            
            # Wait for query to complete
            while True:
                result = self.logs.get_query_results(queryId=query_id)
                
                if result['status'] == 'Complete':
                    break
                elif result['status'] == 'Failed':
                    logger.error("Log query failed")
                    return []
                
                time.sleep(1)
            
            return result['results']
            
        except ClientError as e:
            logger.error(f"Failed to query logs: {e}")
            return []
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary.
        
        Returns:
            Monitoring summary
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            summary = {
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'metrics': {},
                'alarms': {},
                'logs': {}
            }
            
            # Get key metrics
            key_metrics = ['DetectionCount', 'AnomalyCount', 'ErrorCount', 'DetectionLatency']
            
            for metric in key_metrics:
                data = self.get_metrics(metric, start_time, end_time)
                if data:
                    summary['metrics'][metric] = {
                        'latest_value': data[-1]['value'] if data else 0,
                        'data_points': len(data)
                    }
            
            # Get alarm status
            alarms_response = self.cloudwatch.describe_alarms(
                AlarmNamePrefix='Pynomaly-'
            )
            
            for alarm in alarms_response['MetricAlarms']:
                summary['alarms'][alarm['AlarmName']] = {
                    'state': alarm['StateValue'],
                    'reason': alarm['StateReason'],
                    'updated': alarm['StateUpdatedTimestamp'].isoformat()
                }
            
            # Get log statistics
            log_groups_response = self.logs.describe_log_groups(
                logGroupNamePrefix=self.config.log_group_name
            )
            
            if log_groups_response['logGroups']:
                log_group = log_groups_response['logGroups'][0]
                summary['logs'] = {
                    'size_bytes': log_group.get('storedBytes', 0),
                    'retention_days': log_group.get('retentionInDays', 0)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {e}")
            return {}
    
    def _setup_log_group(self):
        """Setup CloudWatch log group."""
        try:
            self.logs.create_log_group(
                logGroupName=self.config.log_group_name,
                tags={
                    'Application': 'Pynomaly-Detection',
                    'Environment': 'Production'
                }
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.error(f"Failed to create log group: {e}")
                raise
        
        # Set retention policy
        try:
            self.logs.put_retention_policy(
                logGroupName=self.config.log_group_name,
                retentionInDays=self.config.retention_days
            )
        except ClientError as e:
            logger.warning(f"Failed to set retention policy: {e}")
    
    def _ensure_log_stream(self, log_stream_name: str):
        """Ensure log stream exists."""
        try:
            self.logs.create_log_stream(
                logGroupName=self.config.log_group_name,
                logStreamName=log_stream_name
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.error(f"Failed to create log stream: {e}")
                raise
    
    def _setup_dashboard(self):
        """Setup initial dashboard."""
        try:
            self.create_detection_dashboard()
        except Exception as e:
            logger.warning(f"Failed to setup dashboard: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass