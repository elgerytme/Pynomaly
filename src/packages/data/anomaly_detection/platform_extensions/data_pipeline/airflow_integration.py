"""
Apache Airflow Integration for Pynomaly Detection
=================================================

Integration with Apache Airflow for production-grade pipeline orchestration:
- DAG generation and management
- Custom operators for anomaly detection
- Sensor integration for real-time triggers
- XCom data passing between tasks
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.operators.dummy import DummyOperator
    from airflow.sensors.filesystem import FileSensor
    from airflow.sensors.s3_key import S3KeySensor
    from airflow.hooks.postgres_hook import PostgresHook
    from airflow.hooks.S3_hook import S3Hook
    from airflow.models import Variable
    from airflow.utils.dates import days_ago
    from airflow.utils.trigger_rule import TriggerRule
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

from ...simplified_services.core_detection_service import CoreDetectionService

logger = logging.getLogger(__name__)

@dataclass
class AirflowDAGConfig:
    """Configuration for Airflow DAG."""
    dag_id: str
    description: str
    schedule_interval: Optional[str] = None
    start_date: datetime = None
    catchup: bool = False
    max_active_runs: int = 1
    default_retries: int = 3
    retry_delay: timedelta = timedelta(minutes=5)
    email_on_failure: bool = False
    email_on_retry: bool = False
    email: Optional[List[str]] = None
    tags: List[str] = None

@dataclass
class AnomalyDetectionTaskConfig:
    """Configuration for anomaly detection task."""
    task_id: str
    input_source: str  # file_path, s3_key, database_query
    output_destination: str  # file_path, s3_key, database_table
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    preprocessing_steps: List[str] = None
    validation_rules: Dict[str, Any] = None
    alert_threshold: float = 0.8
    enable_model_persistence: bool = True

class AirflowIntegration:
    """Apache Airflow integration for anomaly detection pipelines."""
    
    def __init__(self, airflow_home: Optional[str] = None):
        """Initialize Airflow integration.
        
        Args:
            airflow_home: Airflow home directory
        """
        if not AIRFLOW_AVAILABLE:
            raise ImportError("Apache Airflow is required for this integration")
        
        self.airflow_home = airflow_home or os.environ.get('AIRFLOW_HOME', '/opt/airflow')
        self.dags_folder = os.path.join(self.airflow_home, 'dags')
        self.core_service = CoreDetectionService()
        
        # DAG registry
        self.registered_dags: Dict[str, DAG] = {}
        self.dag_configs: Dict[str, AirflowDAGConfig] = {}
        
        logger.info(f"Airflow integration initialized (AIRFLOW_HOME: {self.airflow_home})")
    
    def create_anomaly_detection_dag(self, 
                                   dag_config: AirflowDAGConfig,
                                   detection_tasks: List[AnomalyDetectionTaskConfig]) -> DAG:
        """Create Airflow DAG for anomaly detection pipeline.
        
        Args:
            dag_config: DAG configuration
            detection_tasks: List of detection tasks
            
        Returns:
            Created DAG
        """
        try:
            # Set default values
            if dag_config.start_date is None:
                dag_config.start_date = days_ago(1)
            if dag_config.tags is None:
                dag_config.tags = ['pynomaly', 'anomaly_detection']
            
            # Create DAG
            dag = DAG(
                dag_id=dag_config.dag_id,
                description=dag_config.description,
                schedule_interval=dag_config.schedule_interval,
                start_date=dag_config.start_date,
                catchup=dag_config.catchup,
                max_active_runs=dag_config.max_active_runs,
                default_args={
                    'retries': dag_config.default_retries,
                    'retry_delay': dag_config.retry_delay,
                    'email_on_failure': dag_config.email_on_failure,
                    'email_on_retry': dag_config.email_on_retry,
                    'email': dag_config.email or []
                },
                tags=dag_config.tags
            )
            
            # Create start and end tasks
            start_task = DummyOperator(
                task_id='start_pipeline',
                dag=dag
            )
            
            end_task = DummyOperator(
                task_id='end_pipeline',
                dag=dag,
                trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED
            )
            
            # Create detection tasks
            detection_operators = []
            for task_config in detection_tasks:
                operators = self._create_detection_task_operators(dag, task_config)
                detection_operators.extend(operators)
            
            # Set up dependencies
            if detection_operators:
                # Chain start -> detection tasks -> end
                start_task >> detection_operators[0]
                
                for i in range(len(detection_operators) - 1):
                    detection_operators[i] >> detection_operators[i + 1]
                
                detection_operators[-1] >> end_task
            else:
                start_task >> end_task
            
            # Register DAG
            self.registered_dags[dag_config.dag_id] = dag
            self.dag_configs[dag_config.dag_id] = dag_config
            
            logger.info(f"Anomaly detection DAG created: {dag_config.dag_id}")
            return dag
            
        except Exception as e:
            logger.error(f"Failed to create DAG {dag_config.dag_id}: {e}")
            raise
    
    def create_streaming_detection_dag(self, 
                                     dag_config: AirflowDAGConfig,
                                     stream_config: Dict[str, Any]) -> DAG:
        """Create DAG for streaming anomaly detection.
        
        Args:
            dag_config: DAG configuration
            stream_config: Streaming configuration
            
        Returns:
            Created streaming DAG
        """
        try:
            # Create DAG
            dag = DAG(
                dag_id=dag_config.dag_id,
                description=dag_config.description,
                schedule_interval=dag_config.schedule_interval or timedelta(minutes=5),
                start_date=dag_config.start_date or days_ago(1),
                catchup=dag_config.catchup,
                max_active_runs=dag_config.max_active_runs,
                tags=dag_config.tags or ['pynomaly', 'streaming', 'anomaly_detection']
            )
            
            # Stream monitoring task
            monitor_stream = PythonOperator(
                task_id='monitor_stream',
                python_callable=self._monitor_stream_task,
                op_kwargs={'stream_config': stream_config},
                dag=dag
            )
            
            # Process stream data
            process_data = PythonOperator(
                task_id='process_stream_data',
                python_callable=self._process_stream_data_task,
                op_kwargs={'stream_config': stream_config},
                dag=dag
            )
            
            # Detect anomalies
            detect_anomalies = PythonOperator(
                task_id='detect_stream_anomalies',
                python_callable=self._detect_stream_anomalies_task,
                op_kwargs={'stream_config': stream_config},
                dag=dag
            )
            
            # Send alerts
            send_alerts = PythonOperator(
                task_id='send_anomaly_alerts',
                python_callable=self._send_alerts_task,
                op_kwargs={'stream_config': stream_config},
                dag=dag
            )
            
            # Set up dependencies
            monitor_stream >> process_data >> detect_anomalies >> send_alerts
            
            # Register DAG
            self.registered_dags[dag_config.dag_id] = dag
            self.dag_configs[dag_config.dag_id] = dag_config
            
            logger.info(f"Streaming detection DAG created: {dag_config.dag_id}")
            return dag
            
        except Exception as e:
            logger.error(f"Failed to create streaming DAG {dag_config.dag_id}: {e}")
            raise
    
    def create_model_training_dag(self, 
                                dag_config: AirflowDAGConfig,
                                training_config: Dict[str, Any]) -> DAG:
        """Create DAG for model training and evaluation.
        
        Args:
            dag_config: DAG configuration
            training_config: Training configuration
            
        Returns:
            Created training DAG
        """
        try:
            # Create DAG
            dag = DAG(
                dag_id=dag_config.dag_id,
                description=dag_config.description,
                schedule_interval=dag_config.schedule_interval or timedelta(days=7),
                start_date=dag_config.start_date or days_ago(1),
                catchup=dag_config.catchup,
                tags=dag_config.tags or ['pynomaly', 'model_training']
            )
            
            # Data extraction
            extract_data = PythonOperator(
                task_id='extract_training_data',
                python_callable=self._extract_training_data_task,
                op_kwargs={'training_config': training_config},
                dag=dag
            )
            
            # Data preprocessing
            preprocess_data = PythonOperator(
                task_id='preprocess_data',
                python_callable=self._preprocess_training_data_task,
                op_kwargs={'training_config': training_config},
                dag=dag
            )
            
            # Model training
            train_model = PythonOperator(
                task_id='train_anomaly_model',
                python_callable=self._train_model_task,
                op_kwargs={'training_config': training_config},
                dag=dag
            )
            
            # Model evaluation
            evaluate_model = PythonOperator(
                task_id='evaluate_model',
                python_callable=self._evaluate_model_task,
                op_kwargs={'training_config': training_config},
                dag=dag
            )
            
            # Model deployment
            deploy_model = PythonOperator(
                task_id='deploy_model',
                python_callable=self._deploy_model_task,
                op_kwargs={'training_config': training_config},
                dag=dag
            )
            
            # Set up dependencies
            extract_data >> preprocess_data >> train_model >> evaluate_model >> deploy_model
            
            # Register DAG
            self.registered_dags[dag_config.dag_id] = dag
            self.dag_configs[dag_config.dag_id] = dag_config
            
            logger.info(f"Model training DAG created: {dag_config.dag_id}")
            return dag
            
        except Exception as e:
            logger.error(f"Failed to create training DAG {dag_config.dag_id}: {e}")
            raise
    
    def save_dag_to_file(self, dag_id: str, file_path: Optional[str] = None) -> str:
        """Save DAG to Python file for Airflow.
        
        Args:
            dag_id: DAG identifier
            file_path: Optional file path (defaults to dags folder)
            
        Returns:
            Path to saved DAG file
        """
        try:
            if dag_id not in self.registered_dags:
                raise ValueError(f"DAG not found: {dag_id}")
            
            if file_path is None:
                file_path = os.path.join(self.dags_folder, f"{dag_id}.py")
            
            # Generate DAG Python code
            dag_code = self._generate_dag_code(dag_id)
            
            # Write to file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(dag_code)
            
            logger.info(f"DAG saved to file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save DAG {dag_id}: {e}")
            raise
    
    def get_dag_status(self, dag_id: str) -> Dict[str, Any]:
        """Get DAG status information.
        
        Args:
            dag_id: DAG identifier
            
        Returns:
            DAG status information
        """
        if dag_id not in self.registered_dags:
            return {'error': f'DAG not found: {dag_id}'}
        
        dag = self.registered_dags[dag_id]
        config = self.dag_configs[dag_id]
        
        return {
            'dag_id': dag_id,
            'description': config.description,
            'schedule_interval': str(config.schedule_interval),
            'start_date': config.start_date.isoformat(),
            'catchup': config.catchup,
            'max_active_runs': config.max_active_runs,
            'tags': config.tags,
            'task_count': len(dag.task_dict),
            'tasks': list(dag.task_dict.keys())
        }
    
    def _create_detection_task_operators(self, dag: DAG, task_config: AnomalyDetectionTaskConfig) -> List:
        """Create operators for anomaly detection task.
        
        Args:
            dag: Airflow DAG
            task_config: Detection task configuration
            
        Returns:
            List of created operators
        """
        operators = []
        
        # Data loading task
        load_data_task = PythonOperator(
            task_id=f'{task_config.task_id}_load_data',
            python_callable=self._load_data_task,
            op_kwargs={'task_config': task_config},
            dag=dag
        )
        operators.append(load_data_task)
        
        # Preprocessing task (if needed)
        if task_config.preprocessing_steps:
            preprocess_task = PythonOperator(
                task_id=f'{task_config.task_id}_preprocess',
                python_callable=self._preprocess_data_task,
                op_kwargs={'task_config': task_config},
                dag=dag
            )
            operators.append(preprocess_task)
        
        # Anomaly detection task
        detection_task = PythonOperator(
            task_id=f'{task_config.task_id}_detect',
            python_callable=self._anomaly_detection_task,
            op_kwargs={'task_config': task_config},
            dag=dag
        )
        operators.append(detection_task)
        
        # Validation task (if rules specified)
        if task_config.validation_rules:
            validation_task = PythonOperator(
                task_id=f'{task_config.task_id}_validate',
                python_callable=self._validation_task,
                op_kwargs={'task_config': task_config},
                dag=dag
            )
            operators.append(validation_task)
        
        # Output task
        output_task = PythonOperator(
            task_id=f'{task_config.task_id}_output',
            python_callable=self._output_results_task,
            op_kwargs={'task_config': task_config},
            dag=dag
        )
        operators.append(output_task)
        
        return operators
    
    # Airflow task functions
    def _load_data_task(self, task_config: AnomalyDetectionTaskConfig, **context):
        """Load data for anomaly detection."""
        try:
            if task_config.input_source.startswith('s3://'):
                # Load from S3
                s3_hook = S3Hook()
                bucket, key = task_config.input_source[5:].split('/', 1)
                data = s3_hook.read_key(key, bucket)
                df = pd.read_csv(io.StringIO(data))
            elif task_config.input_source.startswith('postgres://'):
                # Load from PostgreSQL
                postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
                df = postgres_hook.get_pandas_df(task_config.input_source)
            else:
                # Load from file
                df = pd.read_csv(task_config.input_source)
            
            # Store in XCom
            context['task_instance'].xcom_push(key='raw_data', value=df.to_json())
            
            logger.info(f"Loaded {len(df)} rows from {task_config.input_source}")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _preprocess_data_task(self, task_config: AnomalyDetectionTaskConfig, **context):
        """Preprocess data for anomaly detection."""
        try:
            # Get data from XCom
            raw_data_json = context['task_instance'].xcom_pull(key='raw_data')
            df = pd.read_json(raw_data_json)
            
            # Apply preprocessing steps
            for step in task_config.preprocessing_steps or []:
                if step == 'remove_nulls':
                    df = df.dropna()
                elif step == 'normalize':
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                elif step == 'remove_duplicates':
                    df = df.drop_duplicates()
            
            # Store processed data
            context['task_instance'].xcom_push(key='processed_data', value=df.to_json())
            
            logger.info(f"Preprocessed data: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _anomaly_detection_task(self, task_config: AnomalyDetectionTaskConfig, **context):
        """Perform anomaly detection."""
        try:
            # Get data from XCom
            data_key = 'processed_data' if task_config.preprocessing_steps else 'raw_data'
            data_json = context['task_instance'].xcom_pull(key=data_key)
            df = pd.read_json(data_json)
            
            # Prepare data for detection
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found for anomaly detection")
            
            X = df[numeric_columns].values
            
            # Perform anomaly detection
            result = self.core_service.detect_anomalies(
                X,
                algorithm=task_config.algorithm,
                contamination=task_config.contamination
            )
            
            # Add results to dataframe
            df['anomaly'] = result['predictions']
            df['anomaly_score'] = result.get('scores', [0] * len(df))
            
            # Store results
            context['task_instance'].xcom_push(key='detection_results', value=df.to_json())
            
            anomaly_count = (result['predictions'] == -1).sum()
            logger.info(f"Anomaly detection completed: {anomaly_count} anomalies detected")
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise
    
    def _validation_task(self, task_config: AnomalyDetectionTaskConfig, **context):
        """Validate detection results."""
        try:
            # Get results from XCom
            results_json = context['task_instance'].xcom_pull(key='detection_results')
            df = pd.read_json(results_json)
            
            # Apply validation rules
            validation_passed = True
            validation_messages = []
            
            for rule_name, rule_config in task_config.validation_rules.items():
                if rule_name == 'max_anomaly_rate':
                    anomaly_rate = (df['anomaly'] == -1).mean()
                    max_rate = rule_config.get('threshold', 0.5)
                    if anomaly_rate > max_rate:
                        validation_passed = False
                        validation_messages.append(f"Anomaly rate {anomaly_rate:.3f} exceeds threshold {max_rate}")
                
                elif rule_name == 'min_data_points':
                    min_points = rule_config.get('threshold', 100)
                    if len(df) < min_points:
                        validation_passed = False
                        validation_messages.append(f"Data points {len(df)} below minimum {min_points}")
            
            # Store validation results
            context['task_instance'].xcom_push(key='validation_passed', value=validation_passed)
            context['task_instance'].xcom_push(key='validation_messages', value=validation_messages)
            
            if not validation_passed:
                logger.warning(f"Validation failed: {validation_messages}")
            else:
                logger.info("Validation passed")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def _output_results_task(self, task_config: AnomalyDetectionTaskConfig, **context):
        """Output detection results."""
        try:
            # Get results from XCom
            results_json = context['task_instance'].xcom_pull(key='detection_results')
            df = pd.read_json(results_json)
            
            # Output results based on destination
            if task_config.output_destination.startswith('s3://'):
                # Save to S3
                s3_hook = S3Hook()
                bucket, key = task_config.output_destination[5:].split('/', 1)
                s3_hook.load_string(df.to_csv(index=False), key, bucket, replace=True)
            else:
                # Save to file
                df.to_csv(task_config.output_destination, index=False)
            
            logger.info(f"Results saved to {task_config.output_destination}")
            
        except Exception as e:
            logger.error(f"Output failed: {e}")
            raise
    
    def _monitor_stream_task(self, stream_config: Dict[str, Any], **context):
        """Monitor stream for new data."""
        # Implementation would depend on specific streaming platform
        logger.info("Stream monitoring task executed")
    
    def _process_stream_data_task(self, stream_config: Dict[str, Any], **context):
        """Process streaming data."""
        # Implementation would depend on specific streaming platform
        logger.info("Stream data processing task executed")
    
    def _detect_stream_anomalies_task(self, stream_config: Dict[str, Any], **context):
        """Detect anomalies in streaming data."""
        # Implementation would depend on specific streaming platform
        logger.info("Stream anomaly detection task executed")
    
    def _send_alerts_task(self, stream_config: Dict[str, Any], **context):
        """Send anomaly alerts."""
        # Implementation would depend on alerting configuration
        logger.info("Anomaly alerts task executed")
    
    def _extract_training_data_task(self, training_config: Dict[str, Any], **context):
        """Extract data for model training."""
        logger.info("Training data extraction task executed")
    
    def _preprocess_training_data_task(self, training_config: Dict[str, Any], **context):
        """Preprocess training data."""
        logger.info("Training data preprocessing task executed")
    
    def _train_model_task(self, training_config: Dict[str, Any], **context):
        """Train anomaly detection model."""
        logger.info("Model training task executed")
    
    def _evaluate_model_task(self, training_config: Dict[str, Any], **context):
        """Evaluate trained model."""
        logger.info("Model evaluation task executed")
    
    def _deploy_model_task(self, training_config: Dict[str, Any], **context):
        """Deploy trained model."""
        logger.info("Model deployment task executed")
    
    def _generate_dag_code(self, dag_id: str) -> str:
        """Generate Python code for DAG.
        
        Args:
            dag_id: DAG identifier
            
        Returns:
            Generated Python code
        """
        dag_config = self.dag_configs[dag_id]
        
        # Basic DAG template
        code_template = f'''
"""
Generated Pynomaly Anomaly Detection DAG: {dag_id}
Generated on: {datetime.now().isoformat()}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

# DAG configuration
default_args = {{
    'retries': {dag_config.default_retries},
    'retry_delay': timedelta(minutes={dag_config.retry_delay.total_seconds() / 60}),
    'email_on_failure': {dag_config.email_on_failure},
    'email_on_retry': {dag_config.email_on_retry},
}}

# Create DAG
dag = DAG(
    dag_id='{dag_id}',
    description='{dag_config.description}',
    schedule_interval={repr(dag_config.schedule_interval)},
    start_date=datetime({dag_config.start_date.year}, {dag_config.start_date.month}, {dag_config.start_date.day}),
    catchup={dag_config.catchup},
    max_active_runs={dag_config.max_active_runs},
    default_args=default_args,
    tags={dag_config.tags}
)

# DAG tasks would be defined here based on the specific DAG type
# This is a simplified template - actual implementation would include
# all the task definitions and dependencies

'''
        
        return code_template