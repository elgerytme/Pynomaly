"""Log analysis and pattern detection system."""

from __future__ import annotations

import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .log_aggregator import LogEntry


class PatternType(Enum):
    """Types of log patterns."""
    ERROR_SPIKE = "error_spike"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNUSUAL_ACTIVITY = "unusual_activity"
    SECURITY_THREAT = "security_threat"
    SYSTEM_ANOMALY = "system_anomaly"
    CUSTOM = "custom"


class Severity(Enum):
    """Severity levels for detected patterns."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogPattern:
    """Detected log pattern."""
    id: str
    pattern_type: PatternType
    severity: Severity
    title: str
    description: str
    first_occurrence: datetime
    last_occurrence: datetime
    occurrence_count: int = 1
    confidence_score: float = 0.0
    affected_loggers: list[str] = field(default_factory=list)
    sample_entries: list[LogEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "first_occurrence": self.first_occurrence.isoformat(),
            "last_occurrence": self.last_occurrence.isoformat(),
            "occurrence_count": self.occurrence_count,
            "confidence_score": self.confidence_score,
            "affected_loggers": self.affected_loggers,
            "sample_entries": [entry.to_dict() for entry in self.sample_entries],
            "metadata": self.metadata
        }


@dataclass
class PatternRule:
    """Rule for pattern detection."""
    id: str
    name: str
    pattern_type: PatternType
    conditions: dict[str, Any]
    severity: Severity = Severity.MEDIUM
    enabled: bool = True
    threshold: int = 1
    time_window: timedelta = timedelta(minutes=5)
    confidence_threshold: float = 0.5

    def matches(self, entries: list[LogEntry]) -> bool:
        """Check if pattern rule matches log entries."""
        if not self.enabled or len(entries) < self.threshold:
            return False

        # Check time window
        if self.time_window:
            now = datetime.utcnow()
            entries = [e for e in entries if now - e.timestamp <= self.time_window]
            if len(entries) < self.threshold:
                return False

        # Evaluate conditions
        return self._evaluate_conditions(entries)

    def _evaluate_conditions(self, entries: list[LogEntry]) -> bool:
        """Evaluate pattern conditions."""
        for condition_type, condition_value in self.conditions.items():
            if condition_type == "level":
                if not any(entry.level == condition_value for entry in entries):
                    return False
            elif condition_type == "levels":
                required_levels = condition_value if isinstance(condition_value, list) else [condition_value]
                if not any(entry.level in required_levels for entry in entries):
                    return False
            elif condition_type == "message_pattern":
                pattern = re.compile(condition_value, re.IGNORECASE)
                if not any(pattern.search(entry.message) for entry in entries):
                    return False
            elif condition_type == "logger_pattern":
                pattern = re.compile(condition_value, re.IGNORECASE)
                if not any(pattern.search(entry.logger_name) for entry in entries):
                    return False
            elif condition_type == "min_count":
                if len(entries) < condition_value:
                    return False
            elif condition_type == "error_rate":
                error_entries = [e for e in entries if e.level in ["ERROR", "CRITICAL"]]
                if len(entries) == 0:
                    return False
                error_rate = len(error_entries) / len(entries)
                if error_rate < condition_value:
                    return False
            elif condition_type == "unique_loggers":
                unique_loggers = {entry.logger_name for entry in entries}
                if len(unique_loggers) < condition_value:
                    return False

        return True


class LogAnalyzer:
    """Analyzes logs for patterns and anomalies."""

    def __init__(
        self,
        analysis_interval: int = 60,
        max_patterns: int = 1000,
        enable_realtime: bool = True,
        enable_background_analysis: bool = True
    ):
        """Initialize log analyzer.

        Args:
            analysis_interval: Interval for background analysis (seconds)
            max_patterns: Maximum number of patterns to keep
            enable_realtime: Whether to enable real-time analysis
            enable_background_analysis: Whether to enable background analysis
        """
        self.analysis_interval = analysis_interval
        self.max_patterns = max_patterns
        self.enable_realtime = enable_realtime
        self.enable_background_analysis = enable_background_analysis

        # Pattern detection
        self._pattern_rules: dict[str, PatternRule] = {}
        self._detected_patterns: dict[str, LogPattern] = {}
        self._lock = threading.RLock()

        # Analysis state
        self._log_buffer: list[LogEntry] = []
        self._analysis_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Statistics
        self.stats = {
            "entries_analyzed": 0,
            "patterns_detected": 0,
            "rules_count": 0,
            "last_analysis_time": None,
            "analysis_errors": 0
        }

        # Initialize default rules
        self._initialize_default_rules()

        # Start background analysis
        if self.enable_background_analysis:
            self._start_analysis_thread()

    def _initialize_default_rules(self):
        """Initialize default pattern detection rules."""
        default_rules = [
            PatternRule(
                id="error_spike",
                name="Error Spike Detection",
                pattern_type=PatternType.ERROR_SPIKE,
                conditions={"levels": ["ERROR", "CRITICAL"], "min_count": 10},
                severity=Severity.HIGH,
                threshold=10,
                time_window=timedelta(minutes=5)
            ),
            PatternRule(
                id="authentication_failures",
                name="Authentication Failure Spike",
                pattern_type=PatternType.SECURITY_THREAT,
                conditions={
                    "message_pattern": r"(authentication|login|auth).*fail",
                    "min_count": 5
                },
                severity=Severity.HIGH,
                threshold=5,
                time_window=timedelta(minutes=2)
            ),
            PatternRule(
                id="performance_degradation",
                name="Performance Degradation",
                pattern_type=PatternType.PERFORMANCE_DEGRADATION,
                conditions={
                    "message_pattern": r"(slow|timeout|performance|latency)",
                    "min_count": 3
                },
                severity=Severity.MEDIUM,
                threshold=3,
                time_window=timedelta(minutes=5)
            ),
            PatternRule(
                id="system_resource_exhaustion",
                name="System Resource Exhaustion",
                pattern_type=PatternType.SYSTEM_ANOMALY,
                conditions={
                    "message_pattern": r"(memory|disk|cpu|resource).*exhaust|full|limit",
                    "levels": ["WARNING", "ERROR", "CRITICAL"]
                },
                severity=Severity.CRITICAL,
                threshold=2,
                time_window=timedelta(minutes=10)
            ),
            PatternRule(
                id="database_connection_issues",
                name="Database Connection Issues",
                pattern_type=PatternType.SYSTEM_ANOMALY,
                conditions={
                    "message_pattern": r"database|db|connection.*error|timeout|fail",
                    "levels": ["ERROR", "CRITICAL"]
                },
                severity=Severity.HIGH,
                threshold=3,
                time_window=timedelta(minutes=5)
            ),
            PatternRule(
                id="unusual_error_rate",
                name="Unusual Error Rate",
                pattern_type=PatternType.UNUSUAL_ACTIVITY,
                conditions={"error_rate": 0.1},  # 10% error rate
                severity=Severity.MEDIUM,
                threshold=20,
                time_window=timedelta(minutes=5)
            )
        ]

        for rule in default_rules:
            self._pattern_rules[rule.id] = rule

        self.stats["rules_count"] = len(self._pattern_rules)

    def add_rule(self, rule: PatternRule):
        """Add pattern detection rule."""
        with self._lock:
            self._pattern_rules[rule.id] = rule
            self.stats["rules_count"] = len(self._pattern_rules)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove pattern detection rule."""
        with self._lock:
            if rule_id in self._pattern_rules:
                del self._pattern_rules[rule_id]
                self.stats["rules_count"] = len(self._pattern_rules)
                return True
            return False

    def get_rule(self, rule_id: str) -> PatternRule | None:
        """Get pattern detection rule."""
        with self._lock:
            return self._pattern_rules.get(rule_id)

    def list_rules(self) -> list[PatternRule]:
        """List all pattern detection rules."""
        with self._lock:
            return list(self._pattern_rules.values())

    def analyze_entry(self, entry: LogEntry):
        """Analyze single log entry (real-time analysis)."""
        if not self.enable_realtime:
            return

        with self._lock:
            self._log_buffer.append(entry)
            self.stats["entries_analyzed"] += 1

            # Keep buffer size manageable
            if len(self._log_buffer) > 10000:
                self._log_buffer = self._log_buffer[-5000:]

            # Quick real-time checks for critical patterns
            self._check_critical_patterns([entry])

    def analyze_batch(self, entries: list[LogEntry]):
        """Analyze batch of log entries."""
        with self._lock:
            self._log_buffer.extend(entries)
            self.stats["entries_analyzed"] += len(entries)

            # Keep buffer size manageable
            if len(self._log_buffer) > 10000:
                self._log_buffer = self._log_buffer[-5000:]

        # Perform comprehensive analysis
        self._perform_analysis(entries)

    def _check_critical_patterns(self, entries: list[LogEntry]):
        """Check for critical patterns that need immediate attention."""
        critical_rules = [rule for rule in self._pattern_rules.values()
                         if rule.severity == Severity.CRITICAL and rule.enabled]

        for rule in critical_rules:
            if rule.matches(entries):
                self._create_or_update_pattern(rule, entries)

    def _start_analysis_thread(self):
        """Start background analysis thread."""
        def analysis_worker():
            while not self._shutdown_event.wait(self.analysis_interval):
                try:
                    self._perform_background_analysis()
                except Exception as e:
                    self.stats["analysis_errors"] += 1
                    print(f"Error in background analysis: {e}")

        self._analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
        self._analysis_thread.start()

    def _perform_background_analysis(self):
        """Perform comprehensive background analysis."""
        with self._lock:
            if not self._log_buffer:
                return

            # Create a copy for analysis
            entries_to_analyze = self._log_buffer.copy()

        self._perform_analysis(entries_to_analyze)
        self.stats["last_analysis_time"] = datetime.utcnow()

    def _perform_analysis(self, entries: list[LogEntry]):
        """Perform comprehensive pattern analysis."""
        # Group entries by time windows for different rules
        now = datetime.utcnow()

        for rule in self._pattern_rules.values():
            if not rule.enabled:
                continue

            # Filter entries within rule's time window
            cutoff_time = now - rule.time_window
            relevant_entries = [e for e in entries if e.timestamp >= cutoff_time]

            if rule.matches(relevant_entries):
                self._create_or_update_pattern(rule, relevant_entries)

    def _create_or_update_pattern(self, rule: PatternRule, entries: list[LogEntry]):
        """Create or update detected pattern."""
        pattern_key = f"{rule.id}_{rule.pattern_type.value}"

        with self._lock:
            if pattern_key in self._detected_patterns:
                # Update existing pattern
                pattern = self._detected_patterns[pattern_key]
                pattern.occurrence_count += 1
                pattern.last_occurrence = max(entry.timestamp for entry in entries)

                # Update sample entries (keep most recent)
                pattern.sample_entries.extend(entries[:3])
                pattern.sample_entries = sorted(
                    pattern.sample_entries,
                    key=lambda x: x.timestamp,
                    reverse=True
                )[:10]

                # Update affected loggers
                new_loggers = {entry.logger_name for entry in entries}
                pattern.affected_loggers = list(set(pattern.affected_loggers) | new_loggers)
            else:
                # Create new pattern
                from uuid import uuid4

                pattern = LogPattern(
                    id=str(uuid4()),
                    pattern_type=rule.pattern_type,
                    severity=rule.severity,
                    title=rule.name,
                    description=self._generate_pattern_description(rule, entries),
                    first_occurrence=min(entry.timestamp for entry in entries),
                    last_occurrence=max(entry.timestamp for entry in entries),
                    occurrence_count=1,
                    confidence_score=self._calculate_confidence_score(rule, entries),
                    affected_loggers=list({entry.logger_name for entry in entries}),
                    sample_entries=entries[:5],
                    metadata={
                        "rule_id": rule.id,
                        "detection_time": datetime.utcnow().isoformat(),
                        "entry_count": len(entries)
                    }
                )

                self._detected_patterns[pattern_key] = pattern
                self.stats["patterns_detected"] += 1

        # Cleanup old patterns
        self._cleanup_old_patterns()

    def _generate_pattern_description(self, rule: PatternRule, entries: list[LogEntry]) -> str:
        """Generate description for detected pattern."""
        entry_count = len(entries)
        time_span = max(entry.timestamp for entry in entries) - min(entry.timestamp for entry in entries)
        unique_loggers = len({entry.logger_name for entry in entries})

        description = f"Detected {rule.name.lower()} with {entry_count} entries"
        if time_span.total_seconds() > 0:
            description += f" over {time_span.total_seconds():.1f} seconds"
        if unique_loggers > 1:
            description += f" across {unique_loggers} loggers"

        return description

    def _calculate_confidence_score(self, rule: PatternRule, entries: list[LogEntry]) -> float:
        """Calculate confidence score for pattern detection."""
        base_score = 0.5

        # Factor in entry count vs threshold
        if rule.threshold > 0:
            count_factor = min(len(entries) / rule.threshold, 2.0)
            base_score *= count_factor

        # Factor in time concentration
        if len(entries) > 1:
            time_span = max(entry.timestamp for entry in entries) - min(entry.timestamp for entry in entries)
            if time_span.total_seconds() < rule.time_window.total_seconds() / 2:
                base_score *= 1.5  # High concentration increases confidence

        # Factor in severity
        severity_multiplier = {
            Severity.LOW: 0.8,
            Severity.MEDIUM: 1.0,
            Severity.HIGH: 1.3,
            Severity.CRITICAL: 1.5
        }
        base_score *= severity_multiplier.get(rule.severity, 1.0)

        return min(base_score, 1.0)

    def _cleanup_old_patterns(self):
        """Remove old patterns to stay within limits."""
        if len(self._detected_patterns) <= self.max_patterns:
            return

        # Sort by last occurrence and keep most recent
        patterns_by_time = sorted(
            self._detected_patterns.items(),
            key=lambda x: x[1].last_occurrence,
            reverse=True
        )

        # Keep most recent patterns
        patterns_to_keep = dict(patterns_by_time[:self.max_patterns])
        self._detected_patterns = patterns_to_keep

    def get_patterns(
        self,
        severity: Severity | None = None,
        pattern_type: PatternType | None = None,
        since: datetime | None = None,
        limit: int | None = None
    ) -> list[LogPattern]:
        """Get detected patterns with optional filtering."""
        with self._lock:
            patterns = list(self._detected_patterns.values())

        # Apply filters
        if severity:
            patterns = [p for p in patterns if p.severity == severity]

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        if since:
            patterns = [p for p in patterns if p.last_occurrence >= since]

        # Sort by last occurrence (most recent first)
        patterns.sort(key=lambda x: x.last_occurrence, reverse=True)

        # Apply limit
        if limit:
            patterns = patterns[:limit]

        return patterns

    def get_pattern(self, pattern_id: str) -> LogPattern | None:
        """Get specific pattern by ID."""
        with self._lock:
            for pattern in self._detected_patterns.values():
                if pattern.id == pattern_id:
                    return pattern
            return None

    def clear_patterns(self, before: datetime | None = None):
        """Clear detected patterns."""
        with self._lock:
            if before:
                patterns_to_keep = {
                    k: v for k, v in self._detected_patterns.items()
                    if v.last_occurrence >= before
                }
                self._detected_patterns = patterns_to_keep
            else:
                self._detected_patterns.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        with self._lock:
            pattern_stats = {
                "total_patterns": len(self._detected_patterns),
                "by_severity": Counter(p.severity.value for p in self._detected_patterns.values()),
                "by_type": Counter(p.pattern_type.value for p in self._detected_patterns.values())
            }

            return {
                "analyzer_stats": self.stats,
                "pattern_stats": pattern_stats,
                "buffer_size": len(self._log_buffer),
                "rules_enabled": sum(1 for rule in self._pattern_rules.values() if rule.enabled)
            }

    def shutdown(self):
        """Shutdown analyzer."""
        self._shutdown_event.set()

        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=5)


class AnomalyDetector:
    """Statistical anomaly detection for log patterns."""

    def __init__(
        self,
        window_size: int = 100,
        sensitivity: float = 2.0,
        min_samples: int = 20
    ):
        """Initialize anomaly detector.

        Args:
            window_size: Size of sliding window for analysis
            sensitivity: Sensitivity threshold (standard deviations)
            min_samples: Minimum samples required for detection
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_samples = min_samples

        # Historical data
        self._log_counts = defaultdict(list)  # logger -> [counts]
        self._error_rates = defaultdict(list)  # logger -> [error_rates]
        self._response_times = defaultdict(list)  # logger -> [response_times]

        self._lock = threading.RLock()

    def update_metrics(self, entries: list[LogEntry]):
        """Update metrics with new log entries."""
        if not entries:
            return

        with self._lock:
            # Group by logger
            by_logger = defaultdict(list)
            for entry in entries:
                by_logger[entry.logger_name].append(entry)

            # Update counts and error rates
            for logger, logger_entries in by_logger.items():
                # Log count
                self._log_counts[logger].append(len(logger_entries))
                if len(self._log_counts[logger]) > self.window_size:
                    self._log_counts[logger] = self._log_counts[logger][-self.window_size:]

                # Error rate
                error_count = sum(1 for e in logger_entries if e.level in ["ERROR", "CRITICAL"])
                error_rate = error_count / len(logger_entries) if logger_entries else 0
                self._error_rates[logger].append(error_rate)
                if len(self._error_rates[logger]) > self.window_size:
                    self._error_rates[logger] = self._error_rates[logger][-self.window_size:]

                # Response times (if available)
                response_times = []
                for entry in logger_entries:
                    if "duration_ms" in entry.context:
                        response_times.append(entry.context["duration_ms"])

                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    self._response_times[logger].append(avg_response_time)
                    if len(self._response_times[logger]) > self.window_size:
                        self._response_times[logger] = self._response_times[logger][-self.window_size:]

    def detect_anomalies(self) -> list[dict[str, Any]]:
        """Detect statistical anomalies in log patterns."""
        anomalies = []

        with self._lock:
            # Check log count anomalies
            for logger, counts in self._log_counts.items():
                if len(counts) < self.min_samples:
                    continue

                anomaly = self._detect_statistical_anomaly(
                    counts, f"log_count_{logger}", "Log Count Anomaly"
                )
                if anomaly:
                    anomalies.append(anomaly)

            # Check error rate anomalies
            for logger, rates in self._error_rates.items():
                if len(rates) < self.min_samples:
                    continue

                anomaly = self._detect_statistical_anomaly(
                    rates, f"error_rate_{logger}", "Error Rate Anomaly"
                )
                if anomaly:
                    anomalies.append(anomaly)

            # Check response time anomalies
            for logger, times in self._response_times.items():
                if len(times) < self.min_samples:
                    continue

                anomaly = self._detect_statistical_anomaly(
                    times, f"response_time_{logger}", "Response Time Anomaly"
                )
                if anomaly:
                    anomalies.append(anomaly)

        return anomalies

    def _detect_statistical_anomaly(
        self, values: list[float], metric_id: str, metric_name: str
    ) -> dict[str, Any] | None:
        """Detect statistical anomaly in values."""
        if len(values) < self.min_samples:
            return None

        import numpy as np

        # Calculate statistics
        mean_value = np.mean(values)
        std_value = np.std(values)

        if std_value == 0:
            return None

        # Check recent values for anomalies
        recent_values = values[-5:]  # Check last 5 values

        for _i, value in enumerate(recent_values):
            z_score = abs(value - mean_value) / std_value

            if z_score > self.sensitivity:
                return {
                    "id": f"{metric_id}_{int(time.time())}",
                    "type": "statistical_anomaly",
                    "metric": metric_name,
                    "metric_id": metric_id,
                    "value": value,
                    "mean": mean_value,
                    "std": std_value,
                    "z_score": z_score,
                    "severity": "high" if z_score > 3.0 else "medium",
                    "timestamp": datetime.utcnow().isoformat(),
                    "confidence": min(z_score / 3.0, 1.0)
                }

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        with self._lock:
            return {
                "tracked_loggers": len(self._log_counts),
                "total_samples": sum(len(counts) for counts in self._log_counts.values()),
                "window_size": self.window_size,
                "sensitivity": self.sensitivity,
                "min_samples": self.min_samples
            }
