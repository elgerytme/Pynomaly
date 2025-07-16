#!/usr/bin/env python3
"""
Production Logging Setup Script for Pynomaly

This script sets up and configures the production logging infrastructure
including log directories, configuration files, and monitoring.
"""

import os
import subprocess
import sys
from pathlib import Path


def create_log_directories(base_dir: str = "/var/log/pynomaly"):
    """Create logging directory structure."""
    directories = [
        base_dir,
        f"{base_dir}/application",
        f"{base_dir}/access",
        f"{base_dir}/error",
        f"{base_dir}/audit",
        f"{base_dir}/performance",
        f"{base_dir}/security",
        f"{base_dir}/archived",
    ]

    print("Creating log directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

        # Set appropriate permissions
        os.chmod(directory, 0o755)

    print("Log directories created successfully")


def setup_log_rotation():
    """Set up log rotation configuration."""
    logrotate_config = """
# Pynomaly log rotation configuration
/var/log/pynomaly/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su pynomaly pynomaly
    postrotate
        systemctl reload pynomaly || true
    endscript
}

/var/log/pynomaly/application/*.log {
    hourly
    rotate 24
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su pynomaly pynomaly
}

/var/log/pynomaly/access/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su pynomaly pynomaly
}

/var/log/pynomaly/audit/*.log {
    daily
    rotate 365
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su pynomaly pynomaly
}
"""

    logrotate_file = "/etc/logrotate.d/pynomaly"

    print("Setting up log rotation...")
    with open(logrotate_file, "w") as f:
        f.write(logrotate_config)

    os.chmod(logrotate_file, 0o644)
    print(f"Log rotation configured: {logrotate_file}")


def setup_rsyslog_integration():
    """Set up rsyslog integration for centralized logging."""
    rsyslog_config = """
# Pynomaly logging configuration for rsyslog

# Create separate files for different log levels
:programname, isequal, "pynomaly" /var/log/pynomaly/application/pynomaly.log
:programname, isequal, "pynomaly" ~

# High priority logs
*.emerg;*.alert;*.crit;*.err /var/log/pynomaly/error/critical.log

# Security logs
authpriv.* /var/log/pynomaly/security/auth.log

# Performance logs
local0.* /var/log/pynomaly/performance/metrics.log

# Stop processing after handling pynomaly logs
& stop
"""

    rsyslog_file = "/etc/rsyslog.d/10-pynomaly.conf"

    print("Setting up rsyslog integration...")
    with open(rsyslog_file, "w") as f:
        f.write(rsyslog_config)

    os.chmod(rsyslog_file, 0o644)

    # Restart rsyslog service
    try:
        subprocess.run(["systemctl", "restart", "rsyslog"], check=True)
        print("Rsyslog configured and restarted")
    except subprocess.CalledProcessError:
        print("Warning: Could not restart rsyslog service")


def setup_fluentd_config():
    """Set up Fluentd configuration for log forwarding."""
    fluentd_config = """
# Pynomaly Fluentd configuration

<source>
  @type tail
  @id pynomaly_application_logs
  path /var/log/pynomaly/application/*.log
  pos_file /var/log/fluentd/pynomaly-application.log.pos
  tag pynomaly.application
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%LZ
</source>

<source>
  @type tail
  @id pynomaly_access_logs
  path /var/log/pynomaly/access/*.log
  pos_file /var/log/fluentd/pynomaly-access.log.pos
  tag pynomaly.access
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%LZ
</source>

<source>
  @type tail
  @id pynomaly_error_logs
  path /var/log/pynomaly/error/*.log
  pos_file /var/log/fluentd/pynomaly-error.log.pos
  tag pynomaly.error
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%LZ
</source>

<source>
  @type tail
  @id pynomaly_audit_logs
  path /var/log/pynomaly/audit/*.log
  pos_file /var/log/fluentd/pynomaly-audit.log.pos
  tag pynomaly.audit
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%LZ
</source>

<filter pynomaly.**>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    service pynomaly
    environment "#{ENV['ENVIRONMENT'] || 'production'}"
  </record>
</filter>

# Output to Elasticsearch
<match pynomaly.**>
  @type elasticsearch
  @id pynomaly_elasticsearch
  hosts elasticsearch.company.com:9200
  index_name pynomaly-logs
  type_name _doc
  include_timestamp true
  reconnect_on_error true
  reload_on_failure true
  reload_connections false
  <buffer>
    @type file
    path /var/log/fluentd/pynomaly-elasticsearch
    flush_mode interval
    flush_interval 10s
    flush_thread_count 2
    retry_type exponential_backoff
    retry_wait 1s
    retry_max_interval 60s
    retry_timeout 60m
    queued_chunks_limit_size 1024
    compress gzip
  </buffer>
</match>

# Output to CloudWatch Logs (if enabled)
<match pynomaly.**>
  @type cloudwatch_logs
  @id pynomaly_cloudwatch
  log_group_name pynomaly-production
  log_stream_name "#{Socket.gethostname}-application"
  region us-east-1
  auto_create_stream true
  <buffer>
    @type file
    path /var/log/fluentd/pynomaly-cloudwatch
    flush_mode interval
    flush_interval 10s
    retry_type exponential_backoff
    retry_wait 1s
    retry_max_interval 60s
    retry_timeout 60m
  </buffer>
</match>

# Backup to local file in case of forwarding failures
<match pynomaly.**>
  @type file
  @id pynomaly_backup
  path /var/log/pynomaly/archived/backup.%Y%m%d
  append true
  <buffer time>
    timekey 1d
    timekey_wait 10m
    timekey_use_utc true
  </buffer>
  <format>
    @type json
  </format>
</match>
"""

    fluentd_dir = "/etc/fluentd"
    Path(fluentd_dir).mkdir(parents=True, exist_ok=True)

    fluentd_file = f"{fluentd_dir}/pynomaly.conf"

    print("Setting up Fluentd configuration...")
    with open(fluentd_file, "w") as f:
        f.write(fluentd_config)

    os.chmod(fluentd_file, 0o644)
    print(f"Fluentd configuration created: {fluentd_file}")


def setup_log_monitoring():
    """Set up log monitoring and alerting."""
    monitoring_script = """#!/bin/bash
# Log monitoring script for Pynomaly

LOG_DIR="/var/log/pynomaly"
ALERT_THRESHOLD_ERRORS=10
ALERT_THRESHOLD_DISK=85
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"

# Function to send alert
send_alert() {
    local message="$1"
    local severity="$2"

    echo "$(date): $severity - $message" >> "$LOG_DIR/monitoring.log"

    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST "$SLACK_WEBHOOK" \
            -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 Pynomaly Log Alert [$severity]: $message\"}"
    fi
}

# Check error rate in last 5 minutes
check_error_rate() {
    local error_count=$(find "$LOG_DIR" -name "*.log" -mmin -5 -exec grep -c "ERROR\\|CRITICAL" {} + | awk '{sum+=$1} END {print sum+0}')

    if [ "$error_count" -gt "$ALERT_THRESHOLD_ERRORS" ]; then
        send_alert "High error rate detected: $error_count errors in last 5 minutes" "HIGH"
    fi
}

# Check disk usage
check_disk_usage() {
    local disk_usage=$(df "$LOG_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')

    if [ "$disk_usage" -gt "$ALERT_THRESHOLD_DISK" ]; then
        send_alert "High disk usage in log directory: ${disk_usage}%" "MEDIUM"
    fi
}

# Check log file growth
check_log_growth() {
    local large_files=$(find "$LOG_DIR" -name "*.log" -size +100M)

    if [ -n "$large_files" ]; then
        send_alert "Large log files detected: $large_files" "MEDIUM"
    fi
}

# Check for missing logs
check_missing_logs() {
    local expected_logs=("application/pynomaly.log" "access/access.log" "error/error.log")

    for log in "${expected_logs[@]}"; do
        if [ ! -f "$LOG_DIR/$log" ]; then
            send_alert "Expected log file missing: $LOG_DIR/$log" "HIGH"
        elif [ "$(find "$LOG_DIR/$log" -mmin +30)" ]; then
            send_alert "Log file not updated in 30 minutes: $LOG_DIR/$log" "MEDIUM"
        fi
    done
}

# Main monitoring checks
echo "$(date): Running log monitoring checks" >> "$LOG_DIR/monitoring.log"

check_error_rate
check_disk_usage
check_log_growth
check_missing_logs

echo "$(date): Log monitoring checks completed" >> "$LOG_DIR/monitoring.log"
"""

    monitoring_dir = "/opt/pynomaly/scripts/monitoring"
    Path(monitoring_dir).mkdir(parents=True, exist_ok=True)

    monitoring_file = f"{monitoring_dir}/log_monitoring.sh"

    print("Setting up log monitoring...")
    with open(monitoring_file, "w") as f:
        f.write(monitoring_script)

    os.chmod(monitoring_file, 0o755)

    # Set up cron job for monitoring
    cron_entry = f"*/5 * * * * {monitoring_file}\n"

    try:
        # Add to pynomaly user's crontab
        subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=True)
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        current_cron = result.stdout

        if monitoring_file not in current_cron:
            new_cron = current_cron + cron_entry
            subprocess.run(["crontab", "-"], input=new_cron, text=True, check=True)
            print("Log monitoring cron job added")
        else:
            print("Log monitoring cron job already exists")
    except subprocess.CalledProcessError:
        print("Warning: Could not set up cron job for log monitoring")

    print(f"Log monitoring script created: {monitoring_file}")


def setup_log_analysis_tools():
    """Set up log analysis and visualization tools."""
    analysis_script = """#!/usr/bin/env python3
'''Log analysis tool for Pynomaly logs'''

import json
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path

def analyze_error_patterns(log_dir: str, hours: int = 24):
    '''Analyze error patterns in logs'''
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    error_patterns = defaultdict(int)
    error_modules = Counter()
    error_timeline = defaultdict(int)

    log_files = Path(log_dir).glob("**/*.log")

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)

                        # Check timestamp
                        timestamp = datetime.fromisoformat(log_entry.get('timestamp', '').replace('Z', '+00:00'))
                        if timestamp < start_time:
                            continue

                        # Analyze errors
                        if log_entry.get('level') in ['ERROR', 'CRITICAL']:
                            message = log_entry.get('message', '')
                            module = log_entry.get('module', 'unknown')

                            # Extract error pattern
                            pattern = re.sub(r'\\d+', 'N', message)  # Replace numbers
                            pattern = re.sub(r'[a-f0-9]{8,}', 'HASH', pattern)  # Replace hashes

                            error_patterns[pattern] += 1
                            error_modules[module] += 1

                            # Timeline (hourly buckets)
                            hour_bucket = timestamp.replace(minute=0, second=0, microsecond=0)
                            error_timeline[hour_bucket] += 1

                    except (json.JSONDecodeError, ValueError):
                        continue  # Skip non-JSON lines

        except IOError:
            continue  # Skip files that can't be read

    # Generate report
    print(f"Error Analysis Report (Last {hours} hours)")
    print("=" * 50)

    print(f"\\nTop Error Patterns:")
    for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {count:4d}: {pattern[:80]}...")

    print(f"\\nErrors by Module:")
    for module, count in error_modules.most_common(10):
        print(f"  {count:4d}: {module}")

    print(f"\\nError Timeline (hourly):")
    for hour in sorted(error_timeline.keys())[-24:]:  # Last 24 hours
        print(f"  {hour.strftime('%Y-%m-%d %H:00')}: {error_timeline[hour]:4d} errors")

def analyze_performance_metrics(log_dir: str, hours: int = 24):
    '''Analyze performance metrics from logs'''
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    response_times = []
    endpoints = defaultdict(list)

    log_files = Path(log_dir).glob("**/*.log")

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)

                        # Check timestamp
                        timestamp = datetime.fromisoformat(log_entry.get('timestamp', '').replace('Z', '+00:00'))
                        if timestamp < start_time:
                            continue

                        # Analyze performance metrics
                        if 'execution_time' in log_entry:
                            exec_time = float(log_entry['execution_time'])
                            response_times.append(exec_time)

                            endpoint = log_entry.get('endpoint', log_entry.get('function', 'unknown'))
                            endpoints[endpoint].append(exec_time)

                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue

        except IOError:
            continue

    if response_times:
        response_times.sort()
        n = len(response_times)

        print(f"\\nPerformance Analysis Report (Last {hours} hours)")
        print("=" * 50)

        print(f"\\nResponse Time Statistics:")
        print(f"  Total requests: {n}")
        print(f"  Average: {sum(response_times)/n:.3f}s")
        print(f"  Median: {response_times[n//2]:.3f}s")
        print(f"  95th percentile: {response_times[int(n*0.95)]:.3f}s")
        print(f"  99th percentile: {response_times[int(n*0.99)]:.3f}s")
        print(f"  Max: {max(response_times):.3f}s")

        print(f"\\nSlowest Endpoints:")
        for endpoint, times in sorted(endpoints.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)[:10]:
            avg_time = sum(times) / len(times)
            print(f"  {avg_time:.3f}s avg ({len(times):4d} reqs): {endpoint}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 log_analysis.py <log_directory> [hours]")
        sys.exit(1)

    log_directory = sys.argv[1]
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    analyze_error_patterns(log_directory, hours)
    analyze_performance_metrics(log_directory, hours)
"""

    analysis_dir = "/opt/pynomaly/scripts/analysis"
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)

    analysis_file = f"{analysis_dir}/log_analysis.py"

    print("Setting up log analysis tools...")
    with open(analysis_file, "w") as f:
        f.write(analysis_script)

    os.chmod(analysis_file, 0o755)
    print(f"Log analysis script created: {analysis_file}")


def setup_log_dashboard():
    """Set up simple log dashboard."""
    dashboard_script = """#!/usr/bin/env python3
'''Simple log dashboard for Pynomaly'''

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

class LogDashboard:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.metrics = defaultdict(lambda: deque(maxlen=100))
        self.running = True

    def tail_logs(self):
        '''Tail logs and collect metrics'''
        while self.running:
            try:
                for log_file in self.log_dir.glob("**/*.log"):
                    try:
                        with open(log_file, 'r') as f:
                            # Seek to end of file
                            f.seek(0, 2)
                            while self.running:
                                line = f.readline()
                                if not line:
                                    time.sleep(0.1)
                                    continue

                                try:
                                    log_entry = json.loads(line)
                                    self.process_log_entry(log_entry)
                                except json.JSONDecodeError:
                                    continue
                    except IOError:
                        continue
            except KeyboardInterrupt:
                self.running = False

    def process_log_entry(self, log_entry: dict):
        '''Process a single log entry'''
        timestamp = datetime.now()
        level = log_entry.get('level', 'INFO')

        # Count by level
        self.metrics[f'level_{level}'].append(timestamp)

        # Count errors
        if level in ['ERROR', 'CRITICAL']:
            self.metrics['errors'].append(timestamp)

        # Track response times
        if 'execution_time' in log_entry:
            self.metrics['response_times'].append(float(log_entry['execution_time']))

    def display_dashboard(self):
        '''Display real-time dashboard'''
        while self.running:
            try:
                # Clear screen
                print('\\033[2J\\033[H')

                print("Pynomaly Log Dashboard")
                print("=" * 50)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()

                # Recent activity (last 5 minutes)
                five_min_ago = datetime.now() - timedelta(minutes=5)

                print("Activity (Last 5 minutes):")
                for level in ['INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                    count = sum(1 for ts in self.metrics[f'level_{level}'] if ts > five_min_ago)
                    print(f"  {level:8}: {count:4d}")

                print()

                # Error rate
                recent_errors = sum(1 for ts in self.metrics['errors'] if ts > five_min_ago)
                print(f"Error Rate: {recent_errors} errors in last 5 minutes")

                # Response times
                recent_response_times = [rt for rt in list(self.metrics['response_times'])[-50:]]
                if recent_response_times:
                    avg_response = sum(recent_response_times) / len(recent_response_times)
                    print(f"Avg Response Time: {avg_response:.3f}s (last 50 requests)")

                print()
                print("Press Ctrl+C to exit")

                time.sleep(5)

            except KeyboardInterrupt:
                self.running = False

if __name__ == "__main__":
    import sys
    import threading

    if len(sys.argv) < 2:
        print("Usage: python3 log_dashboard.py <log_directory>")
        sys.exit(1)

    log_directory = sys.argv[1]
    dashboard = LogDashboard(log_directory)

    # Start log tailing in background
    tail_thread = threading.Thread(target=dashboard.tail_logs)
    tail_thread.daemon = True
    tail_thread.start()

    # Display dashboard
    dashboard.display_dashboard()
"""

    dashboard_dir = "/opt/pynomaly/scripts/dashboard"
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    dashboard_file = f"{dashboard_dir}/log_dashboard.py"

    print("Setting up log dashboard...")
    with open(dashboard_file, "w") as f:
        f.write(dashboard_script)

    os.chmod(dashboard_file, 0o755)
    print(f"Log dashboard created: {dashboard_file}")


def create_logging_user():
    """Create pynomaly user for logging."""
    try:
        # Check if user exists
        result = subprocess.run(["id", "pynomaly"], capture_output=True)
        if result.returncode == 0:
            print("User 'pynomaly' already exists")
            return

        # Create user
        subprocess.run(
            [
                "useradd",
                "--system",
                "--home",
                "/opt/pynomaly",
                "--shell",
                "/bin/false",
                "pynomaly",
            ],
            check=True,
        )

        print("Created system user 'pynomaly'")

        # Set ownership of log directories
        subprocess.run(
            ["chown", "-R", "pynomaly:pynomaly", "/var/log/pynomaly"], check=True
        )
        print("Set ownership of log directories to pynomaly user")

    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not create pynomaly user: {e}")


def main():
    """Main setup function."""
    print("Setting up Pynomaly production logging infrastructure...")
    print("=" * 60)

    # Check if running as root
    if os.geteuid() != 0:
        print("Warning: Not running as root. Some operations may fail.")
        print("Consider running with sudo for full setup.")

    try:
        # Create logging user
        create_logging_user()

        # Create directory structure
        create_log_directories()

        # Set up log rotation
        setup_log_rotation()

        # Set up rsyslog integration
        setup_rsyslog_integration()

        # Set up Fluentd configuration
        setup_fluentd_config()

        # Set up monitoring
        setup_log_monitoring()

        # Set up analysis tools
        setup_log_analysis_tools()

        # Set up dashboard
        setup_log_dashboard()

        print("\n" + "=" * 60)
        print("✅ Production logging infrastructure setup completed!")
        print("\nNext steps:")
        print("1. Review and customize configuration files")
        print("2. Install and configure log forwarding agents (Fluentd, etc.)")
        print("3. Set up external log storage (Elasticsearch, CloudWatch, etc.)")
        print("4. Configure alerting and monitoring")
        print("5. Test logging setup with sample applications")

        print("\nKey files created:")
        print("- Log directories: /var/log/pynomaly/")
        print("- Log rotation: /etc/logrotate.d/pynomaly")
        print("- Rsyslog config: /etc/rsyslog.d/10-pynomaly.conf")
        print("- Fluentd config: /etc/fluentd/pynomaly.conf")
        print("- Monitoring script: /opt/pynomaly/scripts/monitoring/log_monitoring.sh")
        print("- Analysis tools: /opt/pynomaly/scripts/analysis/log_analysis.py")
        print("- Dashboard: /opt/pynomaly/scripts/dashboard/log_dashboard.py")

    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
