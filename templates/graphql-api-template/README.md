# GraphQL API Template

A comprehensive template for building modern GraphQL APIs with schema-first development, advanced resolvers, subscriptions, and production-ready features.

## 🎯 Features

### Core GraphQL Capabilities
- **Schema-First Development**: SDL-driven API design
- **Advanced Resolvers**: Efficient data fetching and transformation
- **Real-time Subscriptions**: WebSocket-based live updates
- **DataLoader Integration**: N+1 query problem elimination
- **Query Complexity Analysis**: DoS protection and resource management
- **Schema Stitching**: Microservice integration
- **Introspection Control**: Production security features

### Performance & Optimization
- **Query Caching**: Redis-based response caching
- **Persisted Queries**: Automatic query optimization
- **DataLoader Batching**: Efficient database access
- **Query Depth Limiting**: Recursive query protection
- **Field-Level Caching**: Granular cache control
- **Connection Pooling**: Database optimization
- **CDN Integration**: Global content delivery

### Security & Validation
- **Authentication**: JWT and OAuth2 integration
- **Authorization**: Field-level access control
- **Input Validation**: Comprehensive data validation
- **Rate Limiting**: API usage protection
- **CORS Configuration**: Cross-origin security
- **Query Whitelisting**: Production query control
- **Audit Logging**: Complete request tracking

### Developer Experience
- **GraphQL Playground**: Interactive API explorer
- **Schema Documentation**: Auto-generated docs
- **Type Generation**: Automatic TypeScript types
- **Mock Server**: Development and testing
- **Error Handling**: Structured error responses
- **Metrics & Monitoring**: Performance insights
- **Federation Support**: Distributed schema management

## 🏗️ Architecture

```
graphql-api/
├── schema/                     # GraphQL schema definitions
│   ├── types/                 # Object type definitions
│   ├── inputs/                # Input type definitions
│   ├── scalars/               # Custom scalar types
│   ├── enums/                 # Enumeration types
│   ├── interfaces/            # Interface definitions
│   └── unions/                # Union type definitions
├── resolvers/                 # GraphQL resolvers
│   ├── queries/              # Query resolvers
│   ├── mutations/            # Mutation resolvers
│   ├── subscriptions/        # Subscription resolvers
│   └── fields/               # Field resolvers
├── dataloaders/              # DataLoader implementations
│   ├── user_loader.py        # User data batching
│   ├── post_loader.py        # Post data batching
│   └── comment_loader.py     # Comment data batching
├── middleware/               # GraphQL middleware
│   ├── auth.py              # Authentication middleware
│   ├── validation.py        # Input validation
│   ├── caching.py           # Response caching
│   └── metrics.py           # Performance monitoring
├── subscriptions/            # Real-time subscriptions
│   ├── handlers/            # Subscription handlers
│   ├── filters/             # Event filtering
│   └── pubsub/              # Pub/Sub integration
└── federation/               # Schema federation
    ├── gateway/             # Federation gateway
    ├── services/            # Federated services
    └── directives/          # Federation directives
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone template
cp -r templates/graphql-api-template/ my-graphql-api
cd my-graphql-api

# Install dependencies
pip install -e ".[dev,subscriptions,federation]"

# Start services
docker-compose up -d redis postgresql

# Run GraphQL server
python -m graphql_api.main
```

### 2. GraphQL Development

```bash
# Generate schema
python -m graphql_api.cli schema generate

# Add new type
python -m graphql_api.cli type create User --fields "id:ID! name:String! email:String!"

# Add resolver
python -m graphql_api.cli resolver create users --type Query --return-type "[User!]!"

# Start playground
open http://localhost:8000/graphql
```

### 3. Example Queries

```graphql
# Query users with posts
query GetUsersWithPosts {
  users {
    id
    name
    email
    posts {
      id
      title
      content
      comments {
        id
        content
        author {
          name
        }
      }
    }
  }
}

# Create user mutation
mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    id
    name
    email
    createdAt
  }
}

# Subscribe to new posts
subscription NewPosts {
  postCreated {
    id
    title
    author {
      name
    }
  }
}
```

## 📊 Core Components

### Schema Definition
- **Type System**: Complete GraphQL type definitions
- **Schema Composition**: Modular schema organization
- **Custom Scalars**: Specialized data types
- **Directives**: Schema behavior modification
- **Documentation**: Inline schema documentation
- **Validation**: Schema consistency checking

### Resolver System
- **Query Resolvers**: Data fetching operations
- **Mutation Resolvers**: Data modification operations
- **Subscription Resolvers**: Real-time event handling
- **Field Resolvers**: Computed field logic
- **Context Management**: Request context handling
- **Error Handling**: Structured error responses

### DataLoader Integration
- **Batch Loading**: Efficient data fetching
- **Caching**: Per-request result caching
- **N+1 Prevention**: Query optimization
- **Custom Loaders**: Domain-specific batching
- **Cache Control**: Cache invalidation strategies
- **Performance Monitoring**: Load tracking

### Subscription Engine
- **WebSocket Support**: Real-time connections
- **Event Filtering**: Selective subscription delivery
- **Authentication**: Secure subscription channels
- **Rate Limiting**: Subscription abuse prevention
- **Connection Management**: Client connection handling
- **Scalability**: Multi-server subscription support

## 🛠️ Technology Stack

### GraphQL Framework
- **Strawberry GraphQL**: Modern Python GraphQL library
- **FastAPI**: Async web framework integration
- **Pydantic**: Data validation and serialization
- **WebSockets**: Real-time subscription support

### Data Layer
- **SQLAlchemy**: ORM with async support
- **Alembic**: Database migrations
- **PostgreSQL**: Primary database
- **Redis**: Caching and pub/sub
- **Elasticsearch**: Search capabilities (optional)

### Performance & Monitoring
- **DataLoader**: Query optimization
- **Redis Cache**: Response caching
- **Prometheus**: Metrics collection
- **Jaeger**: Distributed tracing
- **New Relic**: APM monitoring (optional)

### Development Tools
- **GraphQL Playground**: Interactive explorer
- **Schema Inspector**: Schema analysis
- **Query Analyzer**: Performance profiling
- **Mock Server**: Development testing
- **Type Generator**: TypeScript integration

## 📁 Project Structure

```
my-graphql-api/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── src/graphql_api/
│   ├── schema/              # Schema definitions
│   │   ├── __init__.py     # Schema composition
│   │   ├── user.py         # User type definition
│   │   ├── post.py         # Post type definition
│   │   └── scalars.py      # Custom scalars
│   ├── resolvers/          # GraphQL resolvers
│   │   ├── __init__.py     # Resolver composition
│   │   ├── user.py         # User resolvers
│   │   ├── post.py         # Post resolvers
│   │   └── subscription.py # Subscription resolvers
│   ├── dataloaders/        # DataLoader implementations
│   │   ├── __init__.py     # Loader factory
│   │   ├── user.py         # User data loader
│   │   └── post.py         # Post data loader
│   ├── middleware/         # GraphQL middleware
│   │   ├── auth.py         # Authentication
│   │   ├── caching.py      # Response caching
│   │   └── validation.py   # Input validation
│   ├── services/           # Business logic
│   │   ├── user_service.py # User operations
│   │   └── post_service.py # Post operations
│   ├── database/           # Database layer
│   │   ├── models.py       # SQLAlchemy models
│   │   └── connection.py   # Database connection
│   └── main.py             # Application entry point
├── tests/                  # Test suite
├── configs/               # Configuration files
├── scripts/              # Utility scripts
└── docs/                 # Documentation
```

## 🔧 Configuration

### GraphQL Configuration
```yaml
# graphql.yml
server:
  host: "0.0.0.0"
  port: 8000
  debug: false
  
schema:
  introspection: true  # Disable in production
  playground: true     # Disable in production
  max_depth: 10
  max_complexity: 1000
  
subscriptions:
  enabled: true
  keepalive_interval: 15
  connection_init_timeout: 60
```

### Performance Configuration
```yaml
# performance.yml
dataloaders:
  enabled: true
  cache_size: 1000
  batch_size: 100
  
caching:
  redis_url: "redis://localhost:6379"
  default_ttl: 300
  cache_control: true
  
query_analysis:
  enabled: true
  complexity_limit: 1000
  depth_limit: 10
  timeout: 30
```

### Security Configuration
```yaml
# security.yml
authentication:
  jwt_secret: "${JWT_SECRET}"
  jwt_algorithm: "HS256"
  token_expire: 3600
  
authorization:
  field_level: true
  default_deny: false
  
rate_limiting:
  enabled: true
  requests_per_minute: 100
  burst_size: 10
```

## 🔄 Schema Development

### Type Definition
```python
import strawberry
from typing import List, Optional

@strawberry.type
class User:
    id: strawberry.ID
    name: str
    email: str
    posts: List["Post"] = strawberry.field(resolver=resolve_user_posts)
    
@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    content: str
    author: User = strawberry.field(resolver=resolve_post_author)
```

### Resolver Implementation
```python
@strawberry.field
async def users(info: Info) -> List[User]:
    """Get all users with DataLoader optimization."""
    user_service = info.context["user_service"]
    return await user_service.get_all_users()

@strawberry.mutation
async def create_user(info: Info, input: CreateUserInput) -> User:
    """Create a new user."""
    user_service = info.context["user_service"]
    return await user_service.create_user(input)
```

### Subscription Setup
```python
@strawberry.subscription
async def post_created(info: Info) -> AsyncIterator[Post]:
    """Subscribe to new post events."""
    async for event in info.context["pubsub"].subscribe("post_created"):
        yield event
```

## 🧪 Testing Strategy

### Schema Testing
```bash
# Test schema validity
pytest tests/schema/ --cov=graphql_api.schema

# Test introspection
pytest tests/introspection/
```

### Resolver Testing
```bash
# Test query resolvers
pytest tests/resolvers/queries/ --cov=graphql_api.resolvers

# Test mutation resolvers
pytest tests/resolvers/mutations/

# Test subscription resolvers
pytest tests/resolvers/subscriptions/
```

### Integration Testing
```bash
# Test full GraphQL operations
pytest tests/integration/ --graphql

# Test performance
pytest tests/performance/ --benchmark
```

### Load Testing
```bash
# GraphQL load testing
artillery run tests/load/graphql-load-test.yml

# Subscription load testing
artillery run tests/load/subscription-load-test.yml
```

## 📈 Monitoring & Observability

### Query Metrics
- **Query Count**: Total queries executed
- **Query Duration**: Execution time distribution
- **Query Complexity**: Complexity score tracking
- **Field Resolution Time**: Per-field performance
- **Error Rate**: Query failure percentage

### DataLoader Metrics
- **Batch Efficiency**: Batching effectiveness
- **Cache Hit Rate**: Cache performance
- **Load Time**: Data loading performance
- **N+1 Prevention**: Query optimization success

### Subscription Metrics
- **Active Connections**: Real-time connection count
- **Event Throughput**: Events per second
- **Subscription Duration**: Connection lifetime
- **Memory Usage**: Subscription memory consumption

### Business Metrics
- **API Usage**: Feature adoption tracking
- **User Engagement**: Query pattern analysis
- **Performance Trends**: Long-term performance tracking
- **Error Analysis**: Failure pattern identification

## 🚀 Deployment Options

### Local Development
```bash
# Start with hot reload
python -m graphql_api.main --reload

# Start with playground
python -m graphql_api.main --playground
```

### Docker Deployment
```bash
# Build image
docker build -t my-graphql-api .

# Run container
docker run -p 8000:8000 my-graphql-api
```

### Kubernetes Deployment
```bash
# Deploy to cluster
kubectl apply -f k8s/

# Enable federation gateway
kubectl apply -f k8s/federation/
```

### Serverless Deployment
```bash
# Deploy to AWS Lambda
serverless deploy --stage production

# Deploy to Google Cloud Functions
gcloud functions deploy graphql-api --runtime python39
```

## 🔐 Security

### Authentication & Authorization
- **JWT Tokens**: Stateless authentication
- **OAuth2 Integration**: Third-party auth
- **Field-Level Security**: Granular access control
- **Role-Based Access**: Permission management

### Query Security
- **Query Complexity Analysis**: DoS prevention
- **Depth Limiting**: Recursive query protection
- **Query Whitelisting**: Production query control
- **Rate Limiting**: Abuse prevention

### Data Security
- **Input Sanitization**: XSS prevention
- **SQL Injection Protection**: Parameterized queries
- **Data Masking**: Sensitive data protection
- **Audit Logging**: Complete request tracking

## 🔄 CI/CD Pipeline

### Schema Pipeline
1. **Schema Validation**: SDL syntax checking
2. **Breaking Change Detection**: Schema evolution analysis
3. **Performance Testing**: Query performance validation
4. **Security Scanning**: Schema security analysis

### Deployment Pipeline
1. **Unit Tests**: Resolver and service testing
2. **Integration Tests**: Full GraphQL testing
3. **Performance Tests**: Load and stress testing
4. **Security Tests**: Vulnerability scanning
5. **Production Deployment**: Blue-green deployment

## 📚 Documentation

### API Documentation
- **Schema Explorer**: Interactive schema browser
- **Query Examples**: Complete query samples
- **Subscription Guide**: Real-time feature docs
- **Performance Guide**: Optimization recommendations

### Developer Documentation
- **Resolver Patterns**: Best practices
- **DataLoader Usage**: Optimization techniques
- **Security Guide**: Authentication and authorization
- **Deployment Guide**: Production setup

## 🤝 Contributing

1. **Schema Design**: Follow GraphQL best practices
2. **Resolver Implementation**: Efficient data fetching
3. **Testing**: Comprehensive test coverage
4. **Documentation**: Schema and resolver docs
5. **Performance**: DataLoader optimization

## 📄 License

MIT License - see LICENSE file for details.

---

**⚡ Ready for modern GraphQL APIs!**
**🚀 From REST to GraphQL in minutes!**