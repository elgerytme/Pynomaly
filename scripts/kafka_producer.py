from kafka import KafkaProducer
import json
import time
import random
import string

# Configure Kafka producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def produce_messages(topic='test', message_count=1000):
    for _ in range(message_count):
        message = {
            'id': random.randint(1, 10000),
            'timestamp': time.time(),
            'user': generate_random_string(8),
            'action': random.choice(['login', 'purchase', 'logout'])
        }
        producer.send(topic, message)
        print(f"Produced: {message}")
        time.sleep(1 / 1000)  # 1000 messages per second

produce_messages()
