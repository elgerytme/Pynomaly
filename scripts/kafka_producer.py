#!/usr/bin/env python3
"""
Kafka Producer Script - Real-time Stream Simulation
Produces events at 1000 events per second (EPS) to simulate real-time streaming.

Usage:
    python kafka_producer.py [topic] [duration_seconds] [eps]

Requirements:
    - Apache Kafka running on localhost:9092
    - kafka-python library: pip install kafka-python

Example:
    python kafka_producer.py user_events 60 1000
"""

import sys
import json
import time
import random
import string
import threading
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError


class KafkaStreamProducer:
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='pynomaly_events'):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.running = False
        self.stats = {'sent': 0, 'failed': 0, 'start_time': None}
        
    def connect(self):
        """Initialize Kafka producer connection."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8') if k else None,
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            print(f"✅ Connected to Kafka at {self.bootstrap_servers}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Kafka: {e}")
            return False
            
    def generate_random_string(self, length=10):
        """Generate a random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        
    def generate_event(self):
        """Generate a realistic event for the stream."""
        event_types = [
            'user_login', 'user_logout', 'page_view', 'button_click',
            'purchase', 'cart_add', 'cart_remove', 'search',
            'api_call', 'error', 'warning', 'info'
        ]
        
        categories = [
            'electronics', 'clothing', 'books', 'food', 'sports',
            'automotive', 'home', 'garden', 'toys', 'beauty'
        ]
        
        return {
            'event_id': self.generate_random_string(16),
            'timestamp': datetime.now().isoformat(),
            'unix_timestamp': int(time.time() * 1000),
            'user_id': random.randint(1, 100000),
            'session_id': self.generate_random_string(12),
            'event_type': random.choice(event_types),
            'category': random.choice(categories),
            'value': round(random.uniform(0.01, 999.99), 2),
            'quantity': random.randint(1, 10),
            'status': random.choice(['success', 'failed', 'pending']),
            'user_agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            ]),
            'ip_address': f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            'location': random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU', 'BR', 'IN']),
            'device_type': random.choice(['desktop', 'mobile', 'tablet']),
            'metadata': {
                'source': 'web_app',
                'version': '1.0.0',
                'environment': 'production'
            }
        }
        
    def send_callback(self, record_metadata=None, exception=None):
        """Callback for sent messages."""
        if exception:
            self.stats['failed'] += 1
            print(f"❌ Failed to send message: {exception}")
        else:
            self.stats['sent'] += 1
            
    def produce_stream(self, duration_seconds=60, events_per_second=1000):
        """Produce events at specified rate for given duration."""
        if not self.connect():
            return False
            
        self.running = True
        self.stats['start_time'] = time.time()
        
        print(f"🚀 Starting stream production:")
        print(f"   Topic: {self.topic}")
        print(f"   Duration: {duration_seconds} seconds")
        print(f"   Rate: {events_per_second} events/second")
        print(f"   Total events: {duration_seconds * events_per_second:,}")
        print(f"   Press Ctrl+C to stop\n")
        
        events_sent = 0
        interval = 1.0 / events_per_second  # Time between events
        
        try:
            start_time = time.time()
            next_event_time = start_time
            
            while self.running and (time.time() - start_time) < duration_seconds:
                current_time = time.time()
                
                if current_time >= next_event_time:
                    event = self.generate_event()
                    
                    # Send event with partition key based on user_id for even distribution
                    key = str(event['user_id'])
                    future = self.producer.send(
                        self.topic, 
                        value=event, 
                        key=key
                    )
                    future.add_callback(self.send_callback)
                    
                    events_sent += 1
                    next_event_time += interval
                    
                    # Print progress every 1000 events
                    if events_sent % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = events_sent / elapsed if elapsed > 0 else 0
                        print(f"📊 Progress: {events_sent:,} events sent, {rate:.1f} events/sec")
                        
                # Small sleep to prevent busy waiting
                time.sleep(min(0.001, max(0, next_event_time - time.time())))
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Stream stopped by user")
        except Exception as e:
            print(f"\n❌ Error during streaming: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the producer and print final statistics."""
        self.running = False
        
        if self.producer:
            print("🔄 Flushing remaining messages...")
            self.producer.flush(timeout=10)
            self.producer.close()
            
        if self.stats['start_time']:
            duration = time.time() - self.stats['start_time']
            total_events = self.stats['sent'] + self.stats['failed']
            avg_rate = total_events / duration if duration > 0 else 0
            
            print(f"\n📈 Final Statistics:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Events sent: {self.stats['sent']:,}")
            print(f"   Events failed: {self.stats['failed']:,}")
            print(f"   Average rate: {avg_rate:.1f} events/sec")
            print(f"   Success rate: {(self.stats['sent'] / total_events * 100):.1f}%" if total_events > 0 else "   Success rate: 0%")
            
    def print_sample_event(self):
        """Print a sample event to show the data structure."""
        sample = self.generate_event()
        print(f"📋 Sample Event Structure:")
        print(json.dumps(sample, indent=2))
        

def main():
    # Parse command line arguments
    topic = sys.argv[1] if len(sys.argv) > 1 else 'pynomaly_events'
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    eps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    # Create and start producer
    producer = KafkaStreamProducer(topic=topic)
    
    # Show sample event structure
    producer.print_sample_event()
    print("\n" + "="*50 + "\n")
    
    # Start streaming
    producer.produce_stream(duration_seconds=duration, events_per_second=eps)
    

if __name__ == "__main__":
    main()
