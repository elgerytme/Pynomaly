"""Log aggregation and analysis infrastructure."""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class LogAggregator:
    """Log aggregation and analysis service."""
    
    def __init__(self, elasticsearch_url: str = "http://localhost:9200"):
        """Initialize log aggregator.
        
        Args:
            elasticsearch_url: Elasticsearch connection URL
        """
        self.es_client = Elasticsearch([elasticsearch_url])
        self.index_prefix = "pynomaly-logs"
    
    def setup_index_templates(self) -> None:
        """Set up Elasticsearch index templates for log data."""
        template = {
            "index_patterns": [f"{self.index_prefix}-*"],
            "template": {
                "mappings": {
                    "properties": {
                        "timestamp": {
                            "type": "date",
                            "format": "yyyy-MM-dd HH:mm:ss||epoch_millis"
                        },
                        "level": {
                            "type": "keyword"
                        },
                        "logger": {
                            "type": "keyword"
                        },
                        "module": {
                            "type": "keyword"
                        },
                        "function": {
                            "type": "keyword"
                        },
                        "message": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "correlation_id": {
                            "type": "keyword"
                        },
                        "user_id": {
                            "type": "keyword"
                        },
                        "request_id": {
                            "type": "keyword"
                        },
                        "exception": {
                            "type": "text"
                        },
                        "service": {
                            "type": "keyword"
                        },
                        "environment": {
                            "type": "keyword"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "index.lifecycle.name": "pynomaly-logs-policy",
                    "index.lifecycle.rollover_alias": f"{self.index_prefix}-alias"
                }
            }
        }
        
        try:
            self.es_client.indices.put_index_template(
                name=f"{self.index_prefix}-template",
                body=template
            )
        except Exception as e:
            print(f"Failed to create index template: {e}")
    
    def index_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Index a single log entry.
        
        Args:
            log_entry: Log entry to index
        """
        index_name = self._get_index_name(log_entry.get("timestamp"))
        
        try:
            self.es_client.index(
                index=index_name,
                body=log_entry
            )
        except Exception as e:
            print(f"Failed to index log entry: {e}")
    
    def bulk_index_logs(self, log_entries: List[Dict[str, Any]]) -> None:
        """Bulk index multiple log entries.
        
        Args:
            log_entries: List of log entries to index
        """
        actions = []
        
        for entry in log_entries:
            index_name = self._get_index_name(entry.get("timestamp"))
            action = {
                "_index": index_name,
                "_source": entry
            }
            actions.append(action)
        
        try:
            bulk(self.es_client, actions)
        except Exception as e:
            print(f"Failed to bulk index logs: {e}")
    
    def search_logs(
        self,
        query: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        size: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs with various filters.
        
        Args:
            query: Text query for log messages
            start_time: Start time for search
            end_time: End time for search
            level: Log level filter
            correlation_id: Correlation ID filter
            user_id: User ID filter
            size: Maximum number of results
            
        Returns:
            List of matching log entries
        """
        search_body = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "sort": [
                {"timestamp": {"order": "desc"}}
            ],
            "size": size
        }
        
        # Add text query
        if query:
            search_body["query"]["bool"]["must"].append({
                "match": {
                    "message": query
                }
            })
        
        # Add time range filter
        if start_time or end_time:
            time_filter = {"range": {"timestamp": {}}}
            if start_time:
                time_filter["range"]["timestamp"]["gte"] = start_time.isoformat()
            if end_time:
                time_filter["range"]["timestamp"]["lte"] = end_time.isoformat()
            search_body["query"]["bool"]["filter"].append(time_filter)
        
        # Add level filter
        if level:
            search_body["query"]["bool"]["filter"].append({
                "term": {"level": level}
            })
        
        # Add correlation ID filter
        if correlation_id:
            search_body["query"]["bool"]["filter"].append({
                "term": {"correlation_id": correlation_id}
            })
        
        # Add user ID filter
        if user_id:
            search_body["query"]["bool"]["filter"].append({
                "term": {"user_id": user_id}
            })
        
        try:
            result = self.es_client.search(
                index=f"{self.index_prefix}-*",
                body=search_body
            )
            
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception as e:
            print(f"Failed to search logs: {e}")
            return []
    
    def get_error_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze errors over the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Error analysis results
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        search_body = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        },
                        {
                            "terms": {"level": ["ERROR", "CRITICAL"]}
                        }
                    ]
                }
            },
            "aggs": {
                "error_by_module": {
                    "terms": {
                        "field": "module",
                        "size": 10
                    }
                },
                "error_by_level": {
                    "terms": {
                        "field": "level"
                    }
                },
                "error_timeline": {
                    "date_histogram": {
                        "field": "timestamp",
                        "interval": "1h"
                    }
                },
                "common_errors": {
                    "significant_text": {
                        "field": "message",
                        "size": 5
                    }
                }
            },
            "size": 0
        }
        
        try:
            result = self.es_client.search(
                index=f"{self.index_prefix}-*",
                body=search_body
            )
            
            return {
                "total_errors": result["hits"]["total"]["value"],
                "error_by_module": result["aggregations"]["error_by_module"]["buckets"],
                "error_by_level": result["aggregations"]["error_by_level"]["buckets"],
                "error_timeline": result["aggregations"]["error_timeline"]["buckets"],
                "common_errors": result["aggregations"]["common_errors"]["buckets"]
            }
        except Exception as e:
            print(f"Failed to analyze errors: {e}")
            return {}
    
    def get_performance_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance logs over the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Performance analysis results
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Search for performance-related log entries
        search_body = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ],
                    "should": [
                        {"match": {"message": "duration"}},
                        {"match": {"message": "response_time"}},
                        {"match": {"message": "processing_time"}},
                        {"match": {"message": "slow"}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "aggs": {
                "slow_operations": {
                    "terms": {
                        "field": "function",
                        "size": 10
                    }
                },
                "performance_timeline": {
                    "date_histogram": {
                        "field": "timestamp",
                        "interval": "1h"
                    }
                }
            },
            "size": 100
        }
        
        try:
            result = self.es_client.search(
                index=f"{self.index_prefix}-*",
                body=search_body
            )
            
            # Extract timing information from log messages
            timing_data = []
            for hit in result["hits"]["hits"]:
                message = hit["_source"].get("message", "")
                timing_match = re.search(r'(\d+\.?\d*)\s*(ms|seconds?)', message)
                if timing_match:
                    value = float(timing_match.group(1))
                    unit = timing_match.group(2)
                    if unit.startswith('s'):
                        value *= 1000  # Convert to milliseconds
                    timing_data.append({
                        "timestamp": hit["_source"].get("timestamp"),
                        "function": hit["_source"].get("function"),
                        "duration_ms": value,
                        "message": message
                    })
            
            return {
                "slow_operations": result["aggregations"]["slow_operations"]["buckets"],
                "performance_timeline": result["aggregations"]["performance_timeline"]["buckets"],
                "timing_data": timing_data
            }
        except Exception as e:
            print(f"Failed to analyze performance: {e}")
            return {}
    
    def get_user_activity_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze user activity from logs.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            User activity analysis
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        search_body = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        },
                        {
                            "exists": {"field": "user_id"}
                        }
                    ]
                }
            },
            "aggs": {
                "active_users": {
                    "cardinality": {
                        "field": "user_id"
                    }
                },
                "user_activity": {
                    "terms": {
                        "field": "user_id",
                        "size": 20
                    },
                    "aggs": {
                        "activity_count": {
                            "value_count": {
                                "field": "user_id"
                            }
                        }
                    }
                },
                "activity_timeline": {
                    "date_histogram": {
                        "field": "timestamp",
                        "interval": "1h"
                    },
                    "aggs": {
                        "unique_users": {
                            "cardinality": {
                                "field": "user_id"
                            }
                        }
                    }
                }
            },
            "size": 0
        }
        
        try:
            result = self.es_client.search(
                index=f"{self.index_prefix}-*",
                body=search_body
            )
            
            return {
                "active_users_count": result["aggregations"]["active_users"]["value"],
                "user_activity": result["aggregations"]["user_activity"]["buckets"],
                "activity_timeline": result["aggregations"]["activity_timeline"]["buckets"]
            }
        except Exception as e:
            print(f"Failed to analyze user activity: {e}")
            return {}
    
    def _get_index_name(self, timestamp: Optional[str] = None) -> str:
        """Get index name based on timestamp.
        
        Args:
            timestamp: Log timestamp
            
        Returns:
            Elasticsearch index name
        """
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                return f"{self.index_prefix}-{dt.strftime('%Y.%m.%d')}"
            except Exception:
                pass
        
        # Default to current date
        return f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"


class LogProcessor:
    """Process and enrich log entries before aggregation."""
    
    def __init__(self):
        """Initialize log processor."""
        self.enrichment_rules = []
    
    def add_enrichment_rule(self, rule: callable) -> None:
        """Add a log enrichment rule.
        
        Args:
            rule: Function that takes and returns a log entry dict
        """
        self.enrichment_rules.append(rule)
    
    def process_log_entry(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enrich a log entry.
        
        Args:
            log_entry: Raw log entry
            
        Returns:
            Processed and enriched log entry
        """
        processed_entry = log_entry.copy()
        
        # Apply enrichment rules
        for rule in self.enrichment_rules:
            try:
                processed_entry = rule(processed_entry)
            except Exception as e:
                print(f"Enrichment rule failed: {e}")
        
        # Add standard enrichments
        processed_entry = self._add_standard_enrichments(processed_entry)
        
        return processed_entry
    
    def _add_standard_enrichments(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Add standard enrichments to log entry.
        
        Args:
            log_entry: Log entry to enrich
            
        Returns:
            Enriched log entry
        """
        # Add service information
        log_entry["service"] = "pynomaly"
        
        # Add environment if not present
        if "environment" not in log_entry:
            log_entry["environment"] = "production"  # Default
        
        # Parse and categorize errors
        if log_entry.get("level") in ["ERROR", "CRITICAL"]:
            log_entry["error_category"] = self._categorize_error(log_entry.get("message", ""))
        
        # Extract operation type from function name
        function_name = log_entry.get("function", "")
        if function_name:
            log_entry["operation_type"] = self._extract_operation_type(function_name)
        
        return log_entry
    
    def _categorize_error(self, message: str) -> str:
        """Categorize error based on message content.
        
        Args:
            message: Error message
            
        Returns:
            Error category
        """
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ["connection", "timeout", "network"]):
            return "connectivity"
        elif any(keyword in message_lower for keyword in ["permission", "unauthorized", "forbidden"]):
            return "authorization"
        elif any(keyword in message_lower for keyword in ["validation", "invalid", "format"]):
            return "validation"
        elif any(keyword in message_lower for keyword in ["database", "sql", "query"]):
            return "database"
        elif any(keyword in message_lower for keyword in ["memory", "out of memory", "oom"]):
            return "memory"
        else:
            return "unknown"
    
    def _extract_operation_type(self, function_name: str) -> str:
        """Extract operation type from function name.
        
        Args:
            function_name: Function name
            
        Returns:
            Operation type
        """
        if any(keyword in function_name.lower() for keyword in ["detect", "predict"]):
            return "detection"
        elif any(keyword in function_name.lower() for keyword in ["train", "fit"]):
            return "training"
        elif any(keyword in function_name.lower() for keyword in ["load", "save", "store"]):
            return "data_io"
        elif any(keyword in function_name.lower() for keyword in ["auth", "login", "logout"]):
            return "authentication"
        else:
            return "general"


# Global instances
_log_aggregator: Optional[LogAggregator] = None
_log_processor: Optional[LogProcessor] = None


def get_log_aggregator(elasticsearch_url: str = "http://localhost:9200") -> LogAggregator:
    """Get global log aggregator instance.
    
    Args:
        elasticsearch_url: Elasticsearch connection URL
        
    Returns:
        Log aggregator instance
    """
    global _log_aggregator
    if _log_aggregator is None:
        _log_aggregator = LogAggregator(elasticsearch_url)
    return _log_aggregator


def get_log_processor() -> LogProcessor:
    """Get global log processor instance.
    
    Returns:
        Log processor instance
    """
    global _log_processor
    if _log_processor is None:
        _log_processor = LogProcessor()
    return _log_processor