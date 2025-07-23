"""Unit tests for web functionality that can run independently."""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime


def test_chart_data_generation():
    """Test chart data generation functions."""
    
    def generate_anomaly_timeline_data(timeline_data: list) -> dict:
        """Generate Chart.js compatible data for anomaly timeline."""
        labels = [point['timestamp'] for point in timeline_data]
        detections_data = [point['detections'] for point in timeline_data]
        anomalies_data = [point['anomalies'] for point in timeline_data]
        
        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Total Detections",
                    "data": detections_data,
                    "borderColor": "rgb(59, 130, 246)",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "tension": 0.1
                },
                {
                    "label": "Anomalies Found",
                    "data": anomalies_data,
                    "borderColor": "rgb(239, 68, 68)",
                    "backgroundColor": "rgba(239, 68, 68, 0.1)",
                    "tension": 0.1
                }
            ]
        }
    
    def generate_algorithm_distribution_data(distribution_data: list) -> dict:
        """Generate Chart.js compatible data for algorithm distribution."""
        labels = [item['algorithm'] for item in distribution_data]
        data = [item['count'] for item in distribution_data]
        
        colors = [
            "#3b82f6", "#ef4444", "#10b981", "#f59e0b", 
            "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"
        ]
        
        return {
            "labels": labels,
            "datasets": [{
                "data": data,
                "backgroundColor": colors[:len(data)]
            }]
        }
    
    def generate_performance_trend_data(trend_data: list) -> dict:
        """Generate Chart.js compatible data for performance trends."""
        labels = [point['timestamp'] for point in trend_data]
        processing_times = [point['processing_time'] for point in trend_data]
        throughput = [point['throughput'] for point in trend_data]
        success_rates = [point['success_rate'] for point in trend_data]
        
        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Processing Time (ms)",
                    "data": processing_times,
                    "borderColor": "rgb(59, 130, 246)",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "yAxisID": "y"
                },
                {
                    "label": "Throughput (ops/sec)",
                    "data": throughput,
                    "borderColor": "rgb(16, 185, 129)",
                    "backgroundColor": "rgba(16, 185, 129, 0.1)",
                    "yAxisID": "y1"
                },
                {
                    "label": "Success Rate (%)",
                    "data": success_rates,
                    "borderColor": "rgb(245, 158, 11)",
                    "backgroundColor": "rgba(245, 158, 11, 0.1)",
                    "yAxisID": "y2"
                }
            ]
        }
    
    # Test anomaly timeline data generation
    timeline_input = [
        {'timestamp': '2024-01-20T10:00:00', 'detections': 15, 'anomalies': 3},
        {'timestamp': '2024-01-20T11:00:00', 'detections': 18, 'anomalies': 2},
        {'timestamp': '2024-01-20T12:00:00', 'detections': 12, 'anomalies': 4}
    ]
    
    timeline_chart = generate_anomaly_timeline_data(timeline_input)
    
    assert "labels" in timeline_chart
    assert "datasets" in timeline_chart
    assert len(timeline_chart["labels"]) == 3
    assert len(timeline_chart["datasets"]) == 2
    assert timeline_chart["datasets"][0]["label"] == "Total Detections"
    assert timeline_chart["datasets"][1]["label"] == "Anomalies Found"
    assert timeline_chart["datasets"][0]["data"] == [15, 18, 12]
    assert timeline_chart["datasets"][1]["data"] == [3, 2, 4]
    
    # Test algorithm distribution data generation
    distribution_input = [
        {'algorithm': 'isolation_forest', 'count': 75, 'percentage': 50.0},
        {'algorithm': 'lof', 'count': 45, 'percentage': 30.0},
        {'algorithm': 'one_class_svm', 'count': 30, 'percentage': 20.0}
    ]
    
    distribution_chart = generate_algorithm_distribution_data(distribution_input)
    
    assert "labels" in distribution_chart
    assert "datasets" in distribution_chart
    assert len(distribution_chart["datasets"]) == 1
    assert distribution_chart["labels"] == ['isolation_forest', 'lof', 'one_class_svm']
    assert distribution_chart["datasets"][0]["data"] == [75, 45, 30]
    assert len(distribution_chart["datasets"][0]["backgroundColor"]) == 3
    
    # Test performance trend data generation
    trend_input = [
        {'timestamp': '2024-01-20T10:00:00', 'processing_time': 1200, 'throughput': 10.5, 'success_rate': 97.0},
        {'timestamp': '2024-01-20T11:00:00', 'processing_time': 1300, 'throughput': 9.8, 'success_rate': 96.5}
    ]
    
    trend_chart = generate_performance_trend_data(trend_input)
    
    assert "labels" in trend_chart
    assert "datasets" in trend_chart
    assert len(trend_chart["datasets"]) == 3
    assert trend_chart["datasets"][0]["label"] == "Processing Time (ms)"
    assert trend_chart["datasets"][1]["label"] == "Throughput (ops/sec)"
    assert trend_chart["datasets"][2]["label"] == "Success Rate (%)"
    assert trend_chart["datasets"][0]["yAxisID"] == "y"
    assert trend_chart["datasets"][1]["yAxisID"] == "y1"
    assert trend_chart["datasets"][2]["yAxisID"] == "y2"


def test_form_validation():
    """Test form validation functions."""
    
    def validate_detection_form(form_data: dict) -> tuple:
        """Validate detection form data."""
        errors = []
        
        # Required fields
        if not form_data.get('algorithm'):
            errors.append("Algorithm is required")
        
        # Contamination validation
        contamination = form_data.get('contamination')
        if contamination is not None:
            try:
                contamination_float = float(contamination)
                if contamination_float <= 0 or contamination_float >= 1:
                    errors.append("Contamination must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("Contamination must be a valid number")
        
        # Algorithm validation
        valid_algorithms = ['isolation_forest', 'lof', 'one_class_svm', 'ensemble_majority']
        if form_data.get('algorithm') not in valid_algorithms:
            errors.append(f"Algorithm must be one of: {', '.join(valid_algorithms)}")
        
        # Sample data validation
        sample_data = form_data.get('sample_data', '')
        if sample_data.strip():
            try:
                parsed_data = json.loads(sample_data)
                if not isinstance(parsed_data, list):
                    errors.append("Sample data must be a JSON array")
                elif parsed_data and not isinstance(parsed_data[0], list):
                    errors.append("Sample data must be a 2D array")
            except json.JSONDecodeError:
                errors.append("Sample data must be valid JSON")
        
        return len(errors) == 0, errors
    
    # Test valid form data
    valid_form = {
        'algorithm': 'isolation_forest',
        'contamination': '0.1',
        'sample_data': '[[1, 2, 3], [4, 5, 6]]'
    }
    
    is_valid, errors = validate_detection_form(valid_form)
    assert is_valid
    assert len(errors) == 0
    
    # Test missing algorithm
    invalid_form = {
        'contamination': '0.1',
        'sample_data': ''
    }
    
    is_valid, errors = validate_detection_form(invalid_form)
    assert not is_valid
    assert "Algorithm is required" in errors
    
    # Test invalid contamination
    invalid_form = {
        'algorithm': 'isolation_forest',
        'contamination': '1.5',
        'sample_data': ''
    }
    
    is_valid, errors = validate_detection_form(invalid_form)
    assert not is_valid
    assert "Contamination must be between 0 and 1" in errors
    
    # Test invalid algorithm
    invalid_form = {
        'algorithm': 'invalid_algorithm',
        'contamination': '0.1',
        'sample_data': ''
    }
    
    is_valid, errors = validate_detection_form(invalid_form)
    assert not is_valid
    assert "Algorithm must be one of:" in errors[0]
    
    # Test invalid JSON data
    invalid_form = {
        'algorithm': 'isolation_forest',
        'contamination': '0.1',
        'sample_data': 'invalid json'
    }
    
    is_valid, errors = validate_detection_form(invalid_form)
    assert not is_valid
    assert "Sample data must be valid JSON" in errors


def test_response_formatting():
    """Test response formatting functions."""
    
    def format_detection_result(result_data: dict) -> dict:
        """Format detection result for display."""
        return {
            'algorithm': result_data['algorithm'].replace('_', ' ').title(),
            'total_samples': f"{result_data['total_samples']:,}",
            'anomalies_found': result_data['anomalies_found'],
            'anomaly_rate': f"{result_data['anomaly_rate']:.1f}%",
            'processing_time': f"{result_data['processing_time']:.2f}s",
            'success': result_data['success']
        }
    
    def format_system_metrics(metrics: dict) -> dict:
        """Format system metrics for display."""
        return {
            'cpu_usage': f"{metrics.get('cpu_usage', 0):.1f}%",
            'memory_usage': f"{metrics.get('memory_usage', 0):.1f}%",
            'disk_usage': f"{metrics.get('disk_usage', 0):.1f}%"
        }
    
    def format_timestamp(timestamp: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, AttributeError):
            return timestamp
    
    # Test detection result formatting
    raw_result = {
        'algorithm': 'isolation_forest',
        'total_samples': 1000,
        'anomalies_found': 45,
        'anomaly_rate': 4.5,
        'processing_time': 1.234,
        'success': True
    }
    
    formatted = format_detection_result(raw_result)
    
    assert formatted['algorithm'] == 'Isolation Forest'
    assert formatted['total_samples'] == '1,000'
    assert formatted['anomalies_found'] == 45
    assert formatted['anomaly_rate'] == '4.5%'
    assert formatted['processing_time'] == '1.23s'
    assert formatted['success'] == True
    
    # Test system metrics formatting
    raw_metrics = {
        'cpu_usage': 45.678,
        'memory_usage': 62.345,
        'disk_usage': 28.901
    }
    
    formatted_metrics = format_system_metrics(raw_metrics)
    
    assert formatted_metrics['cpu_usage'] == '45.7%'
    assert formatted_metrics['memory_usage'] == '62.3%'
    assert formatted_metrics['disk_usage'] == '28.9%'
    
    # Test timestamp formatting
    timestamp = '2024-01-20T15:30:45Z'
    formatted_time = format_timestamp(timestamp)
    assert '2024-01-20' in formatted_time
    assert '15:30:45' in formatted_time


def test_template_context_preparation():
    """Test template context preparation functions."""
    
    def prepare_dashboard_context(stats: dict, models: list) -> dict:
        """Prepare context for dashboard template."""
        return {
            'title': 'Dashboard',
            'total_models': len(models),
            'total_detections': stats.get('total_detections', 0),
            'total_anomalies': stats.get('total_anomalies', 0),
            'success_rate': f"{stats.get('success_rate', 0):.1f}%",
            'system_status': stats.get('system_status', 'unknown'),
            'recent_detections': stats.get('recent_detections', [])[:5],  # Limit to 5
            'active_algorithms': stats.get('active_algorithms', [])
        }
    
    def prepare_analytics_context(analytics_data: dict) -> dict:
        """Prepare context for analytics template."""
        return {
            'title': 'Analytics Dashboard',
            'performance_metrics': analytics_data.get('performance', {}),
            'algorithm_stats': analytics_data.get('algorithms', []),
            'data_quality': analytics_data.get('data_quality', {}),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def prepare_detection_context(algorithms: list, ensemble_methods: list) -> dict:
        """Prepare context for detection template."""
        return {
            'title': 'Run Detection',
            'algorithms': algorithms,
            'ensemble_methods': ensemble_methods,
            'default_contamination': 0.1
        }
    
    # Test dashboard context preparation
    stats = {
        'total_detections': 150,
        'total_anomalies': 23,
        'success_rate': 96.7,
        'system_status': 'healthy',
        'recent_detections': [
            {'id': 'det_001', 'algorithm': 'isolation_forest'},
            {'id': 'det_002', 'algorithm': 'lof'},
            {'id': 'det_003', 'algorithm': 'svm'},
            {'id': 'det_004', 'algorithm': 'ensemble'},
            {'id': 'det_005', 'algorithm': 'isolation_forest'},
            {'id': 'det_006', 'algorithm': 'lof'}  # Should be truncated
        ],
        'active_algorithms': ['isolation_forest', 'lof', 'svm']
    }
    
    models = [{'id': 'model_1'}, {'id': 'model_2'}]
    
    dashboard_context = prepare_dashboard_context(stats, models)
    
    assert dashboard_context['title'] == 'Dashboard'
    assert dashboard_context['total_models'] == 2
    assert dashboard_context['total_detections'] == 150
    assert dashboard_context['total_anomalies'] == 23
    assert dashboard_context['success_rate'] == '96.7%'
    assert dashboard_context['system_status'] == 'healthy'
    assert len(dashboard_context['recent_detections']) == 5  # Truncated
    assert dashboard_context['active_algorithms'] == ['isolation_forest', 'lof', 'svm']
    
    # Test analytics context preparation
    analytics_data = {
        'performance': {'throughput': 10.5, 'avg_time': 1.2},
        'algorithms': [{'name': 'isolation_forest', 'count': 100}],
        'data_quality': {'score': 95.0, 'issues': 2}
    }
    
    analytics_context = prepare_analytics_context(analytics_data)
    
    assert analytics_context['title'] == 'Analytics Dashboard'
    assert analytics_context['performance_metrics'] == {'throughput': 10.5, 'avg_time': 1.2}
    assert analytics_context['algorithm_stats'] == [{'name': 'isolation_forest', 'count': 100}]
    assert analytics_context['data_quality'] == {'score': 95.0, 'issues': 2}
    assert 'last_updated' in analytics_context
    
    # Test detection context preparation
    algorithms = [
        {'value': 'isolation_forest', 'label': 'Isolation Forest'},
        {'value': 'lof', 'label': 'Local Outlier Factor'}
    ]
    
    ensemble_methods = [
        {'value': 'majority', 'label': 'Majority Vote'},
        {'value': 'average', 'label': 'Average'}
    ]
    
    detection_context = prepare_detection_context(algorithms, ensemble_methods)
    
    assert detection_context['title'] == 'Run Detection'
    assert detection_context['algorithms'] == algorithms
    assert detection_context['ensemble_methods'] == ensemble_methods
    assert detection_context['default_contamination'] == 0.1


def test_error_handling():
    """Test error handling functions."""
    
    def handle_api_error(error: Exception) -> dict:
        """Handle API errors and return appropriate response."""
        if isinstance(error, ValueError):
            return {
                'status': 'error',
                'message': 'Invalid input provided',
                'details': str(error),
                'status_code': 400
            }
        elif isinstance(error, FileNotFoundError):
            return {
                'status': 'error',
                'message': 'Resource not found',
                'details': str(error),
                'status_code': 404
            }
        elif isinstance(error, PermissionError):
            return {
                'status': 'error',
                'message': 'Access denied',
                'details': 'Insufficient permissions',
                'status_code': 403
            }
        else:
            return {
                'status': 'error',
                'message': 'Internal server error',
                'details': 'An unexpected error occurred',
                'status_code': 500
            }
    
    def create_error_context(error_info: dict, request_path: str) -> dict:
        """Create error context for templates."""
        return {
            'title': f'Error {error_info["status_code"]}',
            'error_message': error_info['message'],
            'error_details': error_info['details'],
            'request_path': request_path,
            'status_code': error_info['status_code'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    # Test ValueError handling
    error_response = handle_api_error(ValueError("Invalid contamination value"))
    
    assert error_response['status'] == 'error'
    assert error_response['message'] == 'Invalid input provided'
    assert error_response['status_code'] == 400
    assert 'Invalid contamination value' in error_response['details']
    
    # Test FileNotFoundError handling
    error_response = handle_api_error(FileNotFoundError("Model not found"))
    
    assert error_response['status'] == 'error'
    assert error_response['message'] == 'Resource not found'
    assert error_response['status_code'] == 404
    
    # Test generic error handling
    error_response = handle_api_error(RuntimeError("Something went wrong"))
    
    assert error_response['status'] == 'error'
    assert error_response['message'] == 'Internal server error'
    assert error_response['status_code'] == 500
    
    # Test error context creation
    error_info = {
        'message': 'Page not found',
        'details': 'The requested page does not exist',
        'status_code': 404
    }
    
    error_context = create_error_context(error_info, '/nonexistent')
    
    assert error_context['title'] == 'Error 404'
    assert error_context['error_message'] == 'Page not found'
    assert error_context['error_details'] == 'The requested page does not exist'
    assert error_context['request_path'] == '/nonexistent'
    assert error_context['status_code'] == 404
    assert 'timestamp' in error_context


if __name__ == "__main__":
    # Run tests directly
    test_chart_data_generation()
    test_form_validation()
    test_response_formatting()
    test_template_context_preparation()
    test_error_handling()
    print("All web functionality tests passed!")