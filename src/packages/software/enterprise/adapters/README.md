# Enterprise Adapters

Universal adapter patterns for external services and frameworks, providing consistent interfaces for databases, caches, message queues, cloud services, and machine learning frameworks.

## Features

### Database Adapters

- **SQLAlchemy**: PostgreSQL, MySQL, SQLite support with async operations
- **MongoDB**: NoSQL document database with async Motor driver
- Connection pooling, transaction management, and health checks

### Cache Adapters

- **Redis**: High-performance key-value store with advanced features
- **Memcached**: Distributed memory caching system
- **In-Memory**: Development and testing cache adapter
- Serialization support (JSON, Pickle), TTL management, and retry logic

### Message Queue Adapters

- **RabbitMQ**: Reliable message broker with AMQP protocol
- **Apache Kafka**: Distributed streaming platform
- **AWS SQS**: Cloud-native message queuing service
- Dead letter queues, message acknowledgment, and backpressure handling

### Storage Adapters

- **AWS S3**: Object storage with multipart uploads
- **Azure Blob Storage**: Microsoft cloud storage service
- **Google Cloud Storage**: Google cloud object storage
- Presigned URLs, lifecycle management, and encryption support

### ML Framework Adapters

- **Scikit-learn**: Traditional machine learning algorithms
- **PyTorch**: Deep learning framework with GPU support
- **TensorFlow**: Google's ML platform
- **JAX**: High-performance numerical computing
- Model serialization, versioning, and deployment support

## Quick Start

### Installation

```bash
# Base package
pip install enterprise-adapters

# With specific adapters
pip install enterprise-adapters[database,cache,messaging]

# All adapters
pip install enterprise-adapters[all]
```

### Database Usage

```python
from enterprise_adapters import SQLAlchemyAdapter, DatabaseConfiguration

# Configure database
config = DatabaseConfiguration(
    adapter_type="sqlalchemy",
    host="localhost",
    port=5432,
    username="user",
    password="password",
    database="myapp",
    pool_size=20,
)

# Create and use adapter
db = SQLAlchemyAdapter(config)
await db.initialize()

# Execute queries
result = await db.fetch_all(
    "SELECT * FROM users WHERE active = :active",
    {"active": True}
)

# Transaction support
transaction = await db.begin_transaction()
try:
    await db.execute_query(
        "UPDATE users SET last_login = NOW() WHERE id = :id",
        {"id": user_id},
    )
    await db.commit_transaction(transaction)
except Exception:
    await db.rollback_transaction(transaction)
```

### Cache Usage

```python
from enterprise_adapters import RedisAdapter, CacheConfiguration

# Configure cache
config = CacheConfiguration(
    adapter_type="redis",
    host="localhost",
    port=6379,
    key_prefix="myapp",
    default_ttl=3600,
    serializer="json",
)

# Create and use adapter
cache = RedisAdapter(config)
await cache.initialize()

# Cache operations
await cache.set("user:123", {"name": "John", "email": "john@example.com"})
user = await cache.get("user:123")

# Check existence and TTL
if await cache.exists("user:123"):
    ttl = await cache.get_ttl("user:123")
    print(f"User cache expires in {ttl} seconds")

# Redis-specific operations
await cache.increment("page_views")
await cache.set_expire("temp_data", 300)
```

### Message Queue Usage

```python
from enterprise_adapters import RabbitMQAdapter, MessageQueueConfiguration

# Configure message queue
config = MessageQueueConfiguration(
    adapter_type="rabbitmq",
    host="localhost",
    port=5672,
    username="guest",
    password="guest",
    queue_durability=True,
)

# Create and use adapter
mq = RabbitMQAdapter(config)
await mq.initialize()

# Send messages
await mq.send("notifications", {
    "type": "email",
    "recipient": "user@example.com",
    "subject": "Welcome!",
    "body": "Welcome to our service!"
})

# Receive messages
message = await mq.receive("notifications")
if message:
    # Process message
    print(f"Processing: {message}")
    await mq.acknowledge(message)
```

### Storage Usage

```python
from enterprise_adapters import S3Adapter, StorageConfiguration

# Configure storage
config = StorageConfiguration(
    adapter_type="s3",
    bucket="my-app-storage",
    region="us-west-2",
    access_key="AKIAEXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
)

# Create and use adapter
storage = S3Adapter(config)
await storage.initialize()

# Upload file
await storage.upload_file("documents/report.pdf", "/local/path/report.pdf")

# Download file
await storage.download_file("documents/report.pdf", "/local/path/downloaded.pdf")

# Generate presigned URL
url = await storage.generate_presigned_url("documents/report.pdf", expires_in=3600)
print(f"Download URL: {url}")

# List objects
objects = await storage.list_objects("documents/")
for obj in objects:
    print(f"File: {obj['key']}, Size: {obj['size']}")
```

### ML Framework Usage

```python
from enterprise_adapters import SklearnAdapter, MLConfiguration
import numpy as np

# Configure ML framework
config = MLConfiguration(
    adapter_type="sklearn",
    model_type="IsolationForest",
    parameters={
        "contamination": 0.1,
        "random_state": 42,
    },
)

# Create and use adapter
ml = SklearnAdapter(config)
await ml.initialize()

# Train model
X_train = np.random.randn(1000, 10)
await ml.fit(X_train)

# Make predictions
X_test = np.random.randn(100, 10)
predictions = await ml.predict(X_test)
scores = await ml.predict_scores(X_test)

# Save model
await ml.save_model("models/isolation_forest.pkl")

# Load model
await ml.load_model("models/isolation_forest.pkl")
```

## Configuration

### Environment Variables

All adapters support configuration via environment variables:

```bash
# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USERNAME=user
DATABASE_PASSWORD=password
DATABASE_NAME=myapp

# Cache
CACHE_HOST=localhost
CACHE_PORT=6379
CACHE_DEFAULT_TTL=3600

# Message Queue
MQ_HOST=localhost
MQ_PORT=5672
MQ_USERNAME=guest
MQ_PASSWORD=guest

# Storage
STORAGE_BUCKET=my-app-storage
STORAGE_REGION=us-west-2
STORAGE_ACCESS_KEY=AKIAEXAMPLE
STORAGE_SECRET_KEY=secret
```

### Configuration Files

```yaml
# config.yaml
adapters:
  database:
    adapter_type: "sqlalchemy"
    host: "localhost"
    port: 5432
    username: "user"
    password: "password"
    database: "myapp"
    pool_size: 20
    ssl_enabled: true
  
  cache:
    adapter_type: "redis"
    host: "localhost"
    port: 6379
    key_prefix: "myapp"
    default_ttl: 3600
  
  storage:
    adapter_type: "s3"
    bucket: "my-app-storage"
    region: "us-west-2"
    access_key: "${AWS_ACCESS_KEY_ID}"
    secret_key: "${AWS_SECRET_ACCESS_KEY}"
```

## Advanced Features

### Adapter Factory

```python
from enterprise_adapters import AdapterFactory, AdapterConfiguration

# Register custom adapter
@adapter("custom_db")
class CustomDatabaseAdapter(DatabaseAdapter):
    # Implementation here
    pass

# Create adapter from configuration
config = AdapterConfiguration(
    adapter_type="custom_db",
    connection_string="custom://localhost:1234/mydb"
)

db = AdapterFactory.create(config)
```

### Health Monitoring

```python
from enterprise_adapters import adapter_registry

# Register adapters
adapter_registry.register("db", db_adapter)
adapter_registry.register("cache", cache_adapter)
adapter_registry.register("mq", mq_adapter)

# Health check all adapters
health_results = await adapter_registry.health_check_all()

for name, status in health_results.items():
    print(f"{name}: {status.status} - {status.message}")
```

### Retry and Circuit Breaker

```python
from enterprise_adapters import DatabaseAdapter
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientDatabaseAdapter(DatabaseAdapter):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def execute_query(self, query: str, parameters=None):
        async with self.with_retry():
            return await super().execute_query(query, parameters)
```

### Connection Pooling

```python
# Database connection pooling
config = DatabaseConfiguration(
    adapter_type="sqlalchemy",
    pool_size=20,           # Normal pool size
    max_overflow=30,        # Additional connections
    pool_recycle=3600,      # Recycle connections hourly
    pool_pre_ping=True,     # Test connections before use
)

# Cache connection pooling
config = CacheConfiguration(
    adapter_type="redis",
    max_connections=100,    # Redis connection pool
)
```

### Custom Serialization

```python
import msgpack

class MsgPackCacheAdapter(RedisAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._serializer = msgpack.packb
        self._deserializer = msgpack.unpackb
```

## Testing

Enterprise Adapters includes testing utilities:

```python
from enterprise_adapters.testing import MockAdapter, TestConfiguration

def test_cache_operations():
    config = TestConfiguration(adapter_type="memory")
    cache = MockAdapter(config)
    
    # Test cache operations
    await cache.set("key", "value")
    assert await cache.get("key") == "value"
    
    await cache.delete("key")
    assert await cache.get("key") is None
```

## Performance Optimization

### Connection Reuse

```python
# Use connection pooling
config = DatabaseConfiguration(
    pool_size=20,
    max_overflow=10,
    pool_recycle=3600,
)

# Reuse connections across requests
async with db.with_retry():
    # Multiple operations use same connection
    await db.execute_query("INSERT INTO logs ...")
    await db.execute_query("UPDATE stats ...")
```

### Batch Operations

```python
# Batch database operations
parameters_list = [
    {"name": "John", "email": "john@example.com"},
    {"name": "Jane", "email": "jane@example.com"},
    {"name": "Bob", "email": "bob@example.com"},
]

await db.execute_many(
    "INSERT INTO users (name, email) VALUES (:name, :email)",
    parameters_list
)
```

### Caching Strategies

```python
# Cache-aside pattern
async def get_user(user_id: int):
    # Try cache first
    user = await cache.get(f"user:{user_id}")
    if user:
        return user
    
    # Fall back to database
    user = await db.fetch_one(
        "SELECT * FROM users WHERE id = :id",
        {"id": user_id}
    )
    
    if user:
        # Cache for future requests
        await cache.set(f"user:{user_id}", user, ttl=3600)
    
    return user
```

## Error Handling

```python
from enterprise_adapters import InfrastructureError

try:
    await db.execute_query("SELECT * FROM users")
except InfrastructureError as e:
    if e.error_code == "CONNECTION_FAILED":
        # Handle connection issues
        await db.reconnect()
    elif e.error_code == "QUERY_FAILED":
        # Handle query issues
        logger.error(f"Query failed: {e.details}")
    else:
        # Handle other issues
        raise
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to this package.

## License

MIT License - see [LICENSE](../../LICENSE) file for details.
