# Real-time Chat System Example

A high-performance real-time messaging system built with WebSockets and event-driven architecture, featuring presence tracking and message persistence.

## Features

- **Real-time Messaging**: WebSocket-based instant messaging
- **Presence Tracking**: Online/offline user status
- **Message History**: Persistent message storage and retrieval
- **File Sharing**: Support for image and document uploads
- **Rooms & Channels**: Group conversations and private messaging
- **Message Reactions**: Emoji reactions and message threading
- **Push Notifications**: Real-time notifications for offline users
- **Message Encryption**: End-to-end encryption for secure communication

## Architecture

Event-driven architecture with CQRS pattern:

```
src/chat_system/
├── domain/              # Core business logic
│   ├── aggregates/      # Chat, Message, User aggregates
│   ├── events/          # Domain events
│   └── value_objects/   # Message content, timestamps
├── application/         # Use cases and command handlers
│   ├── commands/        # Send message, join room commands
│   ├── queries/         # Message history, presence queries
│   └── handlers/        # Command and event handlers
├── infrastructure/     # External adapters
│   ├── websockets/     # WebSocket connection management
│   ├── storage/        # MongoDB message storage
│   ├── cache/          # Redis for presence and sessions
│   └── notifications/  # Push notification service
└── presentation/       # API and WebSocket endpoints
    ├── websocket/      # WebSocket handlers
    └── rest/           # REST API for history
```

## Quick Start

```bash
# Start the chat system
cd examples/communication-chat-system
docker-compose up -d

# Connect to WebSocket
wscat -c ws://localhost:8000/ws

# Send a message
{"type": "send_message", "room_id": "general", "content": "Hello!"}
```

## WebSocket Protocol

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onopen = () => {
    // Send authentication
    ws.send(JSON.stringify({
        type: 'authenticate',
        token: 'jwt-token'
    }));
};
```

### Message Types

#### Send Message
```json
{
    "type": "send_message",
    "room_id": "room-123",
    "content": "Hello everyone!",
    "message_type": "text"
}
```

#### Join Room
```json
{
    "type": "join_room",
    "room_id": "room-123"
}
```

#### Presence Update
```json
{
    "type": "presence_update",
    "status": "online"
}
```

#### File Share
```json
{
    "type": "share_file",
    "room_id": "room-123",
    "file_url": "https://example.com/file.pdf",
    "file_name": "document.pdf",
    "file_size": 1024000
}
```

## REST API

### Message History
```http
GET /api/rooms/{room_id}/messages?limit=50&before=2024-01-01T00:00:00Z
```

### Room Management
```http
POST /api/rooms
GET /api/rooms
PUT /api/rooms/{room_id}
DELETE /api/rooms/{room_id}
```

### User Presence
```http
GET /api/presence/users/{user_id}
GET /api/presence/room/{room_id}
```

## Technology Stack

- **Framework**: FastAPI with WebSockets
- **Database**: MongoDB for message storage
- **Cache**: Redis for presence and session management
- **Message Queue**: RabbitMQ for event processing
- **File Storage**: AWS S3 for file uploads
- **Search**: Elasticsearch for message search
- **Monitoring**: Prometheus + Grafana

## Real-time Features

### Connection Management
- Automatic reconnection with exponential backoff
- Connection heartbeat and health monitoring
- Graceful degradation for connection issues
- Horizontal scaling with Redis pub/sub

### Message Delivery
- At-least-once delivery guarantee
- Message acknowledgments and retry logic
- Offline message queuing
- Message ordering within rooms

### Presence System
- Real-time online/offline status
- Last seen timestamps
- Typing indicators
- User activity tracking

## Scalability

### Horizontal Scaling
- Stateless WebSocket handlers
- Redis-based session clustering
- Message queue for async processing
- Load balancing with sticky sessions

### Performance Optimizations
- Connection pooling
- Message batching
- Lazy loading of message history
- CDN for file sharing

### Monitoring
- Connection metrics (active, total, errors)
- Message throughput and latency
- Room activity and user engagement
- Resource utilization tracking

## Security

### Authentication
- JWT token-based authentication
- WebSocket authentication handshake
- Token refresh for long-lived connections
- Rate limiting per connection

### Message Security
- Input validation and sanitization
- XSS protection for message content
- File upload validation
- Message size limits

### Privacy
- Room-based access control
- Message encryption at rest
- Audit logging for compliance
- GDPR-compliant data handling

## Configuration

```yaml
# WebSocket settings
websocket:
  max_connections_per_user: 5
  heartbeat_interval: 30
  connection_timeout: 60

# Message settings
messages:
  max_message_length: 4096
  max_file_size: 10485760  # 10MB
  retention_days: 90

# Rate limiting
rate_limiting:
  messages_per_minute: 60
  files_per_hour: 10
  rooms_per_day: 5

# Presence
presence:
  offline_threshold_minutes: 5
  cleanup_interval_minutes: 15
```

## Deployment

### Docker Compose
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Environment Variables
```bash
# Database
MONGODB_URL=mongodb://localhost:27017/chat_db
REDIS_URL=redis://localhost:6379/0

# Message Queue
RABBITMQ_URL=amqp://localhost:5672

# File Storage
AWS_S3_BUCKET=chat-files
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# Security
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# Features
ENABLE_FILE_SHARING=true
ENABLE_MESSAGE_REACTIONS=true
ENABLE_PUSH_NOTIFICATIONS=true
```

## Extensions

This chat system can be extended with:
- **Video/Voice Calling**: WebRTC integration
- **Screen Sharing**: Real-time screen sharing
- **Bots & Integrations**: Chatbot framework
- **Advanced Search**: Full-text message search
- **Message Translation**: Multi-language support
- **Moderation Tools**: Content filtering and admin controls