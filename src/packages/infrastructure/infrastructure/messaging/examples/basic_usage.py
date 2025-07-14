"""Basic usage examples for the message queue system."""

import asyncio
from datetime import datetime, timezone

from pynomaly.infrastructure.messaging import (
    AdapterFactory,
    Message,
    MessageBroker,
    MessagePriority,
    MessagingSettings,
    QueueManager,
    Task,
    TaskProcessor,
    TaskType
)


async def example_basic_messaging():
    """Example of basic message sending and receiving."""
    print("=== Basic Messaging Example ===")
    
    # Create settings
    settings = MessagingSettings(
        queue_backend="redis",
        redis_queue_db=15,  # Use test database
        worker_concurrency=2
    )
    
    # Create adapter and queue manager
    adapter = AdapterFactory.create_adapter(settings, "redis://localhost:6379")
    queue_manager = QueueManager(adapter, settings)
    
    try:
        # Start queue manager
        await queue_manager.start()
        print("Queue manager started")
        
        # Send a message
        message_id = await queue_manager.send_message(
            "test_queue",
            {"data": "Hello, World!", "timestamp": datetime.now(timezone.utc).isoformat()},
            priority=MessagePriority.HIGH
        )
        print(f"Sent message: {message_id}")
        
        # Get queue statistics
        stats = await queue_manager.get_queue_stats("test_queue")
        print(f"Queue stats: {stats}")
        
    finally:
        await queue_manager.stop()
        print("Queue manager stopped")


async def example_task_processing():
    """Example of task submission and processing."""
    print("\n=== Task Processing Example ===")
    
    # Create settings
    settings = MessagingSettings(
        queue_backend="redis",
        redis_queue_db=15,
        worker_concurrency=1
    )
    
    # Create components
    adapter = AdapterFactory.create_adapter(settings, "redis://localhost:6379")
    queue_manager = QueueManager(adapter, settings)
    task_processor = TaskProcessor(adapter, settings)
    
    # Define a task handler
    async def handle_anomaly_detection(*args, **kwargs):
        """Example anomaly detection task handler."""
        dataset_id = kwargs.get("dataset_id")
        algorithm = kwargs.get("algorithm")
        
        print(f"Processing anomaly detection: dataset={dataset_id}, algorithm={algorithm}")
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        return {
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "anomalies_found": 5,
            "confidence": 0.95
        }
    
    # Register task handler
    task_processor.register_task_handler("anomaly_detection", handle_anomaly_detection)
    
    try:
        # Start services
        await queue_manager.start()
        print("Queue manager started")
        
        # Create and submit a task
        task = Task(
            task_type=TaskType.ANOMALY_DETECTION,
            name="Test Anomaly Detection",
            function_name="handle_anomaly_detection",
            kwargs={
                "dataset_id": "dataset_123",
                "algorithm": "isolation_forest",
                "parameters": {"contamination": 0.1}
            },
            timeout=30
        )
        
        task_id = await queue_manager.submit_task(task)
        print(f"Submitted task: {task_id}")
        
        # Check task status
        status = await queue_manager.get_task_status(task_id)
        if status:
            print(f"Task status: {status.status}")
        
        print("Task submitted successfully")
        
    finally:
        await queue_manager.stop()
        print("Queue manager stopped")


async def example_message_broker():
    """Example of using the message broker for different message types."""
    print("\n=== Message Broker Example ===")
    
    # Create settings
    settings = MessagingSettings(
        queue_backend="redis",
        redis_queue_db=15
    )
    
    # Create broker
    adapter = AdapterFactory.create_adapter(settings, "redis://localhost:6379")
    broker = MessageBroker(adapter, settings)
    
    try:
        # Start broker
        await broker.start()
        print("Message broker started")
        
        # Publish different types of messages
        
        # 1. Anomaly detection task
        task_id = await broker.create_anomaly_detection_task(
            dataset_id="dataset_456",
            algorithm="lof",
            parameters={"n_neighbors": 20},
            priority=1
        )
        print(f"Created anomaly detection task: {task_id}")
        
        # 2. Data profiling task
        task_id = await broker.create_data_profiling_task(
            dataset_id="dataset_789",
            profile_types=["basic", "advanced", "correlations"]
        )
        print(f"Created data profiling task: {task_id}")
        
        # 3. Notification
        message_id = await broker.publish_notification(
            recipient="admin@example.com",
            subject="Anomaly Detection Complete",
            content="The anomaly detection process has completed successfully.",
            notification_type="email",
            priority=MessagePriority.NORMAL
        )
        print(f"Published notification: {message_id}")
        
        # 4. Event
        message_id = await broker.publish_event(
            event_type="data_processed",
            data={
                "dataset_id": "dataset_123",
                "rows_processed": 10000,
                "processing_time": 45.2
            },
            source="data_processor",
            priority=MessagePriority.LOW
        )
        print(f"Published event: {message_id}")
        
        # Get overall statistics
        stats = await broker.get_queue_statistics()
        print(f"Queue statistics: {stats}")
        
        # Health check
        health = await broker.health_check()
        print(f"Broker health: {health}")
        
    finally:
        await broker.stop()
        print("Message broker stopped")


async def example_worker_management():
    """Example of running workers to process tasks."""
    print("\n=== Worker Management Example ===")
    
    # Create settings
    settings = MessagingSettings(
        queue_backend="redis",
        redis_queue_db=15,
        worker_concurrency=2,
        task_timeout=60
    )
    
    # Create components
    adapter = AdapterFactory.create_adapter(settings, "redis://localhost:6379")
    task_processor = TaskProcessor(adapter, settings)
    
    # Register task handlers
    async def handle_data_profiling(*args, **kwargs):
        """Example data profiling task handler."""
        dataset_id = kwargs.get("dataset_id")
        profile_types = kwargs.get("profile_types", [])
        
        print(f"Profiling dataset {dataset_id} with types: {profile_types}")
        
        # Simulate profiling work
        for profile_type in profile_types:
            print(f"  Running {profile_type} profiling...")
            await asyncio.sleep(1)
        
        return {
            "dataset_id": dataset_id,
            "profiles": {ptype: f"profile_result_{ptype}" for ptype in profile_types},
            "total_rows": 50000,
            "columns": 25
        }
    
    async def handle_report_generation(*args, **kwargs):
        """Example report generation task handler."""
        report_type = kwargs.get("report_type")
        data_source = kwargs.get("data_source")
        
        print(f"Generating {report_type} report from {data_source}")
        await asyncio.sleep(3)
        
        return {
            "report_type": report_type,
            "data_source": data_source,
            "report_url": f"/reports/{report_type}_report.pdf",
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    # Register handlers
    task_processor.register_task_handler("data_profiling", handle_data_profiling)
    task_processor.register_task_handler("report_generation", handle_report_generation)
    
    try:
        # Start task processor
        await task_processor.start(queue_names=["data_profiling", "reports"])
        print("Task processor started with workers")
        
        # Let it run for a short time
        print("Workers are running... (would run indefinitely in production)")
        await asyncio.sleep(5)
        
        # Get statistics
        stats = task_processor.get_statistics()
        print(f"Processing statistics: {stats}")
        
    finally:
        await task_processor.stop()
        print("Task processor stopped")


async def main():
    """Run all examples."""
    print("Message Queue Integration Examples")
    print("=" * 50)
    
    try:
        await example_basic_messaging()
        await example_task_processing()
        await example_message_broker()
        await example_worker_management()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())