# Advanced Benchmarking Setup

## 1. Creating Custom Benchmark Suites

Custom benchmark suites allow you to organize and run performance tests tailored to specific components or scenarios. Here's a step-by-step guide to creating a custom suite:

### Step 1: Define the Benchmark Suite

Utilize the `BenchmarkSuite` class to create a new suite. Customize the `suite_name`, `description`, and `tags`.

```python
@dataclass
class BenchmarkSuite:
    suite_id: UUID = field(default_factory=uuid4)
    suite_name: str = ""
    description: str = ""
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    individual_results: List[PerformanceMetrics] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  # Add tags here
```

### Step 2: Add Tags to Benchmark Suites

Tags are a great way to categorize your suites.

```python
suite = BenchmarkSuite(
    suite_name="My Custom Suite",
    description="Custom benchmark for the new feature.",
    tags=["release-v1", "performance", "critical"]
)
```

## 2. Continuous Monitoring with `pynomaly-perf-server`

### Setup:
1. Ensure Prometheus and Grafana are running. Start them using Docker Compose:
   ```shell
   ./start_monitoring.sh
   ```

2. Confirm all services are up:
   ```shell
   ./verify_stack.sh
   ```

### Instrumentation:
- Ensure metrics endpoints like `/api/metrics` are set.

```python
from prometheus_client import Gauge

execution_time_gauge = Gauge('execution_time_seconds', 'Execution time of benchmark suite')
execution_time_gauge.set(execution_time)
```

## 3. Grafana Dashboard JSON Export

### Create and Export Dashboard:
1. Setup a new dashboard in Grafana.
2. Export the JSON using `Dashboard Settings` > `JSON Model`.
3. Save it to `deploy/grafana/provisioning/dashboards/`.

```json
{
  "dashboard": {
    "title": "Custom Benchmark Dashboard",
    "panels": [
      {
        "type": "graph",
        "title": "Execution Time",
        "targets": [
          {
            "expr": "execution_time_seconds",
            "format": "time_series"
          }
        ]
      }
    ]
  }
}
```

### Apply the Dashboard:
- Restart Grafana to apply the changes:
  ```shell
  docker-compose restart grafana
  ```

## Conclusion

By following these steps, you will have advanced benchmarking setups, continuous monitoring integration with Prometheus and Grafana, and a custom dashboard for insightful visualization.
