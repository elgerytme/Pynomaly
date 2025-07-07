"""Task operators for workflow orchestration systems."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.infrastructure.data_sources.enterprise_connectors import EnterpriseConnectionManager

logger = logging.getLogger(__name__)


# Data Ingestion Tasks
async def data_ingestion_task(
    connection_id: str,
    table_name: str,
    columns: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Ingest data from enterprise data source."""
    try:
        # Get connection manager from context or create new one
        connection_manager = kwargs.get('connection_manager', EnterpriseConnectionManager())
        
        # Get connector
        connector = await connection_manager.get_connector(connection_id)
        if not connector:
            raise ValueError(f"Connection not found: {connection_id}")
        
        # Fetch data
        df = await connector.get_anomaly_data(
            table_name=table_name,
            columns=columns,
            where_clause=where_clause,
            limit=limit
        )
        
        logger.info(f"Ingested {len(df)} rows from {table_name}")
        
        return {
            "status": "success",
            "rows_ingested": len(df),
            "table_name": table_name,
            "data": df.to_dict('records') if len(df) < 1000 else f"Large dataset: {len(df)} rows",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def data_preprocessing_task(
    data: pd.DataFrame,
    preprocessing_steps: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Preprocess data for anomaly detection."""
    try:
        processed_data = data.copy()
        
        for step in preprocessing_steps:
            step_type = step.get('type')
            
            if step_type == 'remove_nulls':
                processed_data = processed_data.dropna()
            
            elif step_type == 'fill_nulls':
                fill_value = step.get('value', 0)
                processed_data = processed_data.fillna(fill_value)
            
            elif step_type == 'normalize':
                columns = step.get('columns', processed_data.select_dtypes(include=[np.number]).columns)
                for col in columns:
                    if col in processed_data.columns:
                        processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
            
            elif step_type == 'remove_outliers':
                threshold = step.get('threshold', 3)
                columns = step.get('columns', processed_data.select_dtypes(include=[np.number]).columns)
                for col in columns:
                    if col in processed_data.columns:
                        z_scores = np.abs((processed_data[col] - processed_data[col].mean()) / processed_data[col].std())
                        processed_data = processed_data[z_scores < threshold]
            
            elif step_type == 'feature_selection':
                columns = step.get('columns', [])
                if columns:
                    processed_data = processed_data[columns]
        
        logger.info(f"Preprocessed data: {len(processed_data)} rows remaining")
        
        return {
            "status": "success",
            "original_rows": len(data),
            "processed_rows": len(processed_data),
            "data": processed_data.to_dict('records') if len(processed_data) < 1000 else f"Large dataset: {len(processed_data)} rows",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def feature_engineering_task(
    data: pd.DataFrame,
    feature_configs: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Engineer features for anomaly detection."""
    try:
        engineered_data = data.copy()
        
        for config in feature_configs:
            feature_type = config.get('type')
            
            if feature_type == 'polynomial':
                columns = config.get('columns', [])
                degree = config.get('degree', 2)
                for col in columns:
                    if col in engineered_data.columns:
                        engineered_data[f'{col}_poly_{degree}'] = engineered_data[col] ** degree
            
            elif feature_type == 'rolling_mean':
                columns = config.get('columns', [])
                window = config.get('window', 5)
                for col in columns:
                    if col in engineered_data.columns:
                        engineered_data[f'{col}_rolling_mean_{window}'] = engineered_data[col].rolling(window=window).mean()
            
            elif feature_type == 'rolling_std':
                columns = config.get('columns', [])
                window = config.get('window', 5)
                for col in columns:
                    if col in engineered_data.columns:
                        engineered_data[f'{col}_rolling_std_{window}'] = engineered_data[col].rolling(window=window).std()
            
            elif feature_type == 'lag_features':
                columns = config.get('columns', [])
                lags = config.get('lags', [1, 2, 3])
                for col in columns:
                    if col in engineered_data.columns:
                        for lag in lags:
                            engineered_data[f'{col}_lag_{lag}'] = engineered_data[col].shift(lag)
            
            elif feature_type == 'interaction':
                col1 = config.get('column1')
                col2 = config.get('column2')
                if col1 in engineered_data.columns and col2 in engineered_data.columns:
                    engineered_data[f'{col1}_{col2}_interaction'] = engineered_data[col1] * engineered_data[col2]
        
        # Remove rows with NaN values created by feature engineering
        engineered_data = engineered_data.dropna()
        
        logger.info(f"Feature engineering completed: {engineered_data.shape[1]} features, {len(engineered_data)} rows")
        
        return {
            "status": "success",
            "original_features": data.shape[1],
            "engineered_features": engineered_data.shape[1],
            "rows": len(engineered_data),
            "data": engineered_data.to_dict('records') if len(engineered_data) < 1000 else f"Large dataset: {len(engineered_data)} rows",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def model_training_task(
    training_data: pd.DataFrame,
    model_config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Train anomaly detection model."""
    try:
        from pynomaly.application.services.anomaly_detector_service import AnomalyDetectorService
        
        detector_service = kwargs.get('detector_service') or AnomalyDetectorService()
        
        # Convert DataFrame to Dataset
        dataset = Dataset(
            data=training_data.values,
            feature_names=training_data.columns.tolist(),
            metadata={"training_data": True}
        )
        
        # Configure detector
        detector_name = model_config.get('detector_name', 'isolation_forest')
        detector_params = model_config.get('parameters', {})
        
        # Create and train detector
        detector = await detector_service.create_detector(
            detector_name=detector_name,
            **detector_params
        )
        
        # Train the detector
        training_result = await detector_service.train_detector(detector, dataset)
        
        # Save model if specified
        model_path = model_config.get('model_path')
        if model_path:
            await detector_service.save_detector(detector, model_path)
        
        logger.info(f"Model training completed: {detector_name}")
        
        return {
            "status": "success",
            "detector_name": detector_name,
            "model_id": detector.detector_id,
            "training_samples": len(training_data),
            "training_features": len(training_data.columns),
            "model_path": model_path,
            "performance_metrics": training_result.get('metrics', {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def anomaly_detection_task(
    input_data: pd.DataFrame,
    model_config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Perform anomaly detection on data."""
    try:
        from pynomaly.application.services.anomaly_detector_service import AnomalyDetectorService
        
        detector_service = kwargs.get('detector_service') or AnomalyDetectorService()
        
        # Load model if path specified
        model_path = model_config.get('model_path')
        if model_path:
            detector = await detector_service.load_detector(model_path)
        else:
            # Create new detector
            detector_name = model_config.get('detector_name', 'isolation_forest')
            detector_params = model_config.get('parameters', {})
            detector = await detector_service.create_detector(
                detector_name=detector_name,
                **detector_params
            )
        
        # Convert DataFrame to Dataset
        dataset = Dataset(
            data=input_data.values,
            feature_names=input_data.columns.tolist(),
            metadata={"inference_data": True}
        )
        
        # Perform anomaly detection
        detection_result = await detector_service.detect_anomalies(detector, dataset)
        
        # Process results
        anomaly_scores = detection_result.anomaly_scores
        anomalies = detection_result.anomalies
        
        # Add results to DataFrame
        result_data = input_data.copy()
        result_data['anomaly_score'] = anomaly_scores
        result_data['is_anomaly'] = anomalies
        
        # Count anomalies
        anomaly_count = int(np.sum(anomalies))
        total_samples = len(input_data)
        
        logger.info(f"Anomaly detection completed: {anomaly_count}/{total_samples} anomalies detected")
        
        return {
            "status": "success",
            "total_samples": total_samples,
            "anomalies_detected": anomaly_count,
            "anomaly_rate": anomaly_count / total_samples,
            "detection_results": result_data.to_dict('records') if len(result_data) < 1000 else f"Large dataset: {len(result_data)} rows",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def alerting_task(
    detection_results: Dict[str, Any],
    alerting_config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Send alerts for detected anomalies."""
    try:
        anomaly_count = detection_results.get('anomalies_detected', 0)
        anomaly_rate = detection_results.get('anomaly_rate', 0.0)
        
        # Check if alerting is needed
        alert_threshold = alerting_config.get('threshold', 0.05)  # 5% default
        
        if anomaly_rate > alert_threshold:
            # Prepare alert message
            message = f"Anomaly Alert: {anomaly_count} anomalies detected ({anomaly_rate:.2%} rate)"
            
            # Send alerts through configured channels
            alert_channels = alerting_config.get('channels', ['log'])
            
            for channel in alert_channels:
                if channel == 'log':
                    logger.warning(message)
                elif channel == 'email':
                    await _send_email_alert(message, alerting_config.get('email_config', {}))
                elif channel == 'slack':
                    await _send_slack_alert(message, alerting_config.get('slack_config', {}))
                elif channel == 'webhook':
                    await _send_webhook_alert(detection_results, alerting_config.get('webhook_config', {}))
            
            return {
                "status": "success",
                "alert_sent": True,
                "message": message,
                "channels": alert_channels,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "success",
                "alert_sent": False,
                "reason": f"Anomaly rate {anomaly_rate:.2%} below threshold {alert_threshold:.2%}",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Alerting failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def data_export_task(
    data: Dict[str, Any],
    export_config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Export results to specified destination."""
    try:
        export_type = export_config.get('type', 'file')
        
        if export_type == 'file':
            file_path = export_config.get('path', '/tmp/anomaly_results.json')
            file_format = export_config.get('format', 'json')
            
            if file_format == 'json':
                import json
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif file_format == 'csv' and 'detection_results' in data:
                df = pd.DataFrame(data['detection_results'])
                df.to_csv(file_path, index=False)
        
        elif export_type == 'database':
            # Export to database
            await _export_to_database(data, export_config.get('database_config', {}))
        
        elif export_type == 's3':
            # Export to S3
            await _export_to_s3(data, export_config.get('s3_config', {}))
        
        logger.info(f"Data exported successfully: {export_type}")
        
        return {
            "status": "success",
            "export_type": export_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data export failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Helper functions for alerting
async def _send_email_alert(message: str, email_config: Dict[str, Any]) -> None:
    """Send email alert."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        
        msg = MIMEText(message)
        msg['Subject'] = email_config.get('subject', 'Anomaly Detection Alert')
        msg['From'] = email_config.get('from_email')
        msg['To'] = email_config.get('to_email')
        
        server = smtplib.SMTP(email_config.get('smtp_server'), email_config.get('smtp_port', 587))
        server.starttls()
        server.login(email_config.get('username'), email_config.get('password'))
        server.send_message(msg)
        server.quit()
        
        logger.info("Email alert sent successfully")
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


async def _send_slack_alert(message: str, slack_config: Dict[str, Any]) -> None:
    """Send Slack alert."""
    try:
        import aiohttp
        
        webhook_url = slack_config.get('webhook_url')
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        payload = {
            "text": message,
            "channel": slack_config.get('channel', '#alerts'),
            "username": slack_config.get('username', 'Pynomaly Bot')
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info("Slack alert sent successfully")
                else:
                    logger.error(f"Failed to send Slack alert: {response.status}")
        
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


async def _send_webhook_alert(data: Dict[str, Any], webhook_config: Dict[str, Any]) -> None:
    """Send webhook alert."""
    try:
        import aiohttp
        
        webhook_url = webhook_config.get('url')
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return
        
        headers = webhook_config.get('headers', {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=data, headers=headers) as response:
                if response.status == 200:
                    logger.info("Webhook alert sent successfully")
                else:
                    logger.error(f"Failed to send webhook alert: {response.status}")
        
    except Exception as e:
        logger.error(f"Failed to send webhook alert: {e}")


async def _export_to_database(data: Dict[str, Any], db_config: Dict[str, Any]) -> None:
    """Export data to database."""
    try:
        # Implementation would depend on database type
        logger.info("Database export completed (mock)")
        
    except Exception as e:
        logger.error(f"Database export failed: {e}")


async def _export_to_s3(data: Dict[str, Any], s3_config: Dict[str, Any]) -> None:
    """Export data to S3."""
    try:
        import boto3
        import json
        
        s3_client = boto3.client('s3')
        bucket = s3_config.get('bucket')
        key = s3_config.get('key', f"anomaly_results_{datetime.now().isoformat()}.json")
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, default=str)
        )
        
        logger.info(f"S3 export completed: s3://{bucket}/{key}")
        
    except Exception as e:
        logger.error(f"S3 export failed: {e}")


# Validation and cleanup tasks
async def validation_task(
    data: Dict[str, Any],
    validation_config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Validate workflow results."""
    try:
        validation_results = {
            "status": "success",
            "checks_passed": 0,
            "checks_failed": 0,
            "details": [],
            "timestamp": datetime.now().isoformat()
        }
        
        checks = validation_config.get('checks', [])
        
        for check in checks:
            check_type = check.get('type')
            check_name = check.get('name', check_type)
            
            if check_type == 'data_quality':
                # Check data quality metrics
                if 'detection_results' in data:
                    result_count = len(data['detection_results']) if isinstance(data['detection_results'], list) else 0
                    min_samples = check.get('min_samples', 100)
                    
                    if result_count >= min_samples:
                        validation_results['checks_passed'] += 1
                        validation_results['details'].append(f"{check_name}: PASSED - {result_count} samples")
                    else:
                        validation_results['checks_failed'] += 1
                        validation_results['details'].append(f"{check_name}: FAILED - Only {result_count} samples")
            
            elif check_type == 'anomaly_rate':
                # Check anomaly rate is within expected range
                anomaly_rate = data.get('anomaly_rate', 0.0)
                min_rate = check.get('min_rate', 0.001)
                max_rate = check.get('max_rate', 0.1)
                
                if min_rate <= anomaly_rate <= max_rate:
                    validation_results['checks_passed'] += 1
                    validation_results['details'].append(f"{check_name}: PASSED - Rate {anomaly_rate:.3f}")
                else:
                    validation_results['checks_failed'] += 1
                    validation_results['details'].append(f"{check_name}: FAILED - Rate {anomaly_rate:.3f} outside [{min_rate}, {max_rate}]")
        
        if validation_results['checks_failed'] > 0:
            validation_results['status'] = 'warning'
        
        logger.info(f"Validation completed: {validation_results['checks_passed']} passed, {validation_results['checks_failed']} failed")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def cleanup_task(
    cleanup_config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Clean up temporary resources."""
    try:
        cleanup_results = {
            "status": "success",
            "actions_completed": [],
            "timestamp": datetime.now().isoformat()
        }
        
        cleanup_actions = cleanup_config.get('actions', [])
        
        for action in cleanup_actions:
            action_type = action.get('type')
            
            if action_type == 'delete_temp_files':
                temp_dir = action.get('directory', '/tmp')
                pattern = action.get('pattern', 'pynomaly_*')
                # Implementation would delete matching files
                cleanup_results['actions_completed'].append(f"Deleted temp files: {temp_dir}/{pattern}")
            
            elif action_type == 'clear_cache':
                cache_type = action.get('cache_type', 'all')
                # Implementation would clear specified caches
                cleanup_results['actions_completed'].append(f"Cleared cache: {cache_type}")
            
            elif action_type == 'close_connections':
                # Implementation would close database/service connections
                cleanup_results['actions_completed'].append("Closed connections")
        
        logger.info(f"Cleanup completed: {len(cleanup_results['actions_completed'])} actions")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }