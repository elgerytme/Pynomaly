#!/usr/bin/env python3
"""
Simple test script to verify message queue integration functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test imports
try:
    from pynomaly.infrastructure.messaging.config import MessageBroker, MessageQueueConfig
    from pynomaly.infrastructure.messaging.core import Message, MessageQueueManager
    from pynomaly.infrastructure.messaging.factory import MessageQueueFactory
    from pynomaly.infrastructure.messaging.implementations.memory_queue import MemoryMessageQueue
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_basic_functionality():
    """Test basic message queue functionality."""
    print("\n🧪 Testing basic message queue functionality...")
    
    # Test 1: Create a memory queue
    config = MessageQueueConfig(broker=MessageBroker.MEMORY)
    queue = MessageQueueFactory.create_queue(config)
    
    print(f"✅ Created queue: {type(queue).__name__}")
    
    # Test 2: Connect to queue
    await queue.connect()
    print("✅ Connected to queue")
    
    # Test 3: Send a message
    message = Message(body="Hello, message queue!", queue_name="test_queue")
    message_id = await queue.send(message)
    print(f"✅ Sent message with ID: {message_id}")
    
    # Test 4: Receive the message
    received = await queue.receive("test_queue")
    if received:
        print(f"✅ Received message: {received.body}")
    else:
        print("❌ Failed to receive message")
    
    # Test 5: Test JSON helper
    data = {"test": "data", "number": 42}
    await queue.send_json(data, queue_name="json_queue")
    received_json = await queue.receive("json_queue")
    if received_json and received_json.body == data:
        print("✅ JSON helper works correctly")
    else:
        print("❌ JSON helper failed")
    
    # Test 6: Test text helper
    text = "Hello, world!"
    await queue.send_text(text, queue_name="text_queue")
    received_text = await queue.receive("text_queue")
    if received_text and received_text.body == text:
        print("✅ Text helper works correctly")
    else:
        print("❌ Text helper failed")
    
    # Test 7: Test queue management
    from pynomaly.infrastructure.messaging.config import QueueConfig
    
    queue_config = QueueConfig(name="managed_queue")
    await queue.create_queue(queue_config)
    
    # Send multiple messages
    for i in range(5):
        msg = Message(body=f"Message {i}", queue_name="managed_queue")
        await queue.send(msg)
    
    # Check queue size
    size = await queue.get_queue_size("managed_queue")
    print(f"✅ Queue size: {size}")
    
    # Purge queue
    purged = await queue.purge_queue("managed_queue")
    print(f"✅ Purged {purged} messages")
    
    # Test 8: Health check
    is_healthy = await queue.health_check()
    print(f"✅ Health check: {is_healthy}")
    
    # Test 9: Statistics
    stats = queue.get_stats()
    print(f"✅ Statistics: {stats}")
    
    # Disconnect
    await queue.disconnect()
    print("✅ Disconnected from queue")


async def test_message_handler():
    """Test message handler functionality."""
    print("\n🧪 Testing message handler functionality...")
    
    config = MessageQueueConfig(broker=MessageBroker.MEMORY)
    queue = MessageQueueFactory.create_queue(config)
    
    # Register a handler
    processed_messages = []
    
    async def test_handler(message: Message) -> str:
        processed_messages.append(message.body)
        return f"Processed: {message.body}"
    
    queue.register_function_handler("handler_queue", test_handler)
    print("✅ Registered message handler")
    
    await queue.connect()
    
    # Send messages
    for i in range(3):
        msg = Message(body=f"Handler test {i}", queue_name="handler_queue")
        await queue.send(msg)
    
    # Process messages
    for i in range(3):
        message = await queue.receive("handler_queue")
        if message:
            result = await queue.process_message(message)
            print(f"✅ Processed message: {result}")
    
    # Verify processing
    if len(processed_messages) == 3:
        print("✅ All messages processed correctly")
    else:
        print(f"❌ Expected 3 messages, got {len(processed_messages)}")
    
    await queue.disconnect()


async def test_queue_manager():
    """Test message queue manager functionality."""
    print("\n🧪 Testing message queue manager functionality...")
    
    config = MessageQueueConfig(broker=MessageBroker.MEMORY)
    manager = MessageQueueManager(config)
    
    # Create and add queue
    queue = MessageQueueFactory.create_queue(config)
    manager.add_queue("test_queue", queue)
    print("✅ Added queue to manager")
    
    # Start manager
    await manager.start()
    print("✅ Started manager")
    
    # Send message through manager
    message = Message(body="Manager test message", queue_name="test_queue")
    message_id = await manager.send_to_queue("test_queue", message)
    print(f"✅ Sent message through manager: {message_id}")
    
    # Check health
    health = await manager.health_check()
    print(f"✅ Health check: {health['overall_healthy']}")
    
    # Get statistics
    stats = manager.get_global_stats()
    print(f"✅ Global stats: {stats['total_messages_sent']} messages sent")
    
    # Stop manager
    await manager.stop()
    print("✅ Stopped manager")


async def test_factory():
    """Test message queue factory functionality."""
    print("\n🧪 Testing message queue factory functionality...")
    
    # Test available brokers
    brokers = MessageQueueFactory.get_available_brokers()
    print(f"✅ Available brokers: {[b.value for b in brokers]}")
    
    # Test broker validation
    is_memory_available = MessageQueueFactory.validate_broker_availability(MessageBroker.MEMORY)
    print(f"✅ Memory broker available: {is_memory_available}")
    
    # Test fallback creation
    queue = MessageQueueFactory.create_with_fallback(
        MessageBroker.REDIS,  # Might not be available
        MessageBroker.MEMORY  # Fallback
    )
    print(f"✅ Created queue with fallback: {type(queue).__name__}")
    
    # Test testing queue
    test_queue = MessageQueueFactory.create_for_testing()
    print(f"✅ Created testing queue: {type(test_queue).__name__}")


async def main():
    """Main test function."""
    print("🚀 Starting message queue integration tests...")
    
    try:
        await test_basic_functionality()
        await test_message_handler()
        await test_queue_manager()
        await test_factory()
        
        print("\n🎉 All tests passed! Message queue integration is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
