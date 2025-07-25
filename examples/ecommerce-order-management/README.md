# E-commerce Order Management Example

This example demonstrates a complete self-contained package for order management in an e-commerce system.

## Architecture

Uses Clean Architecture with clear separation of concerns:

```
src/order_management/
├── domain/           # Business entities and rules
├── application/      # Use cases and business logic
├── infrastructure/   # External adapters (DB, APIs)
└── presentation/     # API endpoints and controllers
```

## Features

- **Order Creation**: Create orders with multiple items
- **Order Tracking**: Track order status and history
- **Payment Integration**: Handle payment processing
- **Inventory Validation**: Check product availability
- **Event Publishing**: Publish domain events for other services
- **Monitoring**: Comprehensive metrics and health checks

## Quick Start

```bash
# Start local development environment
cd examples/ecommerce-order-management
docker-compose up -d

# Run tests
make test

# Start the application
make dev
```

## API Endpoints

### Create Order
```http
POST /orders
Content-Type: application/json

{
  "customer_id": "customer-123",
  "items": [
    {
      "product_id": "product-456",
      "quantity": 2,
      "unit_price": 29.99
    }
  ]
}
```

### Get Order
```http
GET /orders/{order_id}
```

### Update Order Status
```http
PATCH /orders/{order_id}/status
Content-Type: application/json

{
  "status": "confirmed"
}
```

## Domain Model

### Order
- **ID**: Unique identifier
- **Customer ID**: Reference to customer
- **Items**: List of order items
- **Status**: Current order status
- **Total Amount**: Calculated total
- **Timestamps**: Created/updated dates

### Order Item
- **Product ID**: Reference to product
- **Quantity**: Number of items
- **Unit Price**: Price per item
- **Total Price**: Calculated line total

## Business Rules

1. **Order Total**: Automatically calculated from items
2. **Status Transitions**: Valid status progression rules
3. **Inventory Check**: Validate product availability
4. **Payment Validation**: Ensure payment before confirmation

## Technology Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with async SQLAlchemy
- **Caching**: Redis for session and caching
- **Message Queue**: RabbitMQ for event publishing
- **Monitoring**: Prometheus + Grafana
- **Logging**: Structured logging with correlation IDs

## Testing Strategy

### Unit Tests
- Domain model validation
- Business rule enforcement
- Use case logic

### Integration Tests
- API endpoint testing
- Database operations
- External service integration

### End-to-End Tests
- Complete user workflows
- Cross-service communication
- Performance testing

## Deployment

### Local Development
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Monitoring
- **Metrics**: Custom business metrics
- **Health Checks**: Database and service health
- **Alerts**: Order processing SLA violations
- **Dashboards**: Order volume and performance

## Security

- **Input Validation**: Comprehensive request validation
- **Authentication**: JWT token validation
- **Authorization**: Role-based access control
- **Audit Logging**: All order operations logged

## Performance

- **Database Indexing**: Optimized queries
- **Caching Strategy**: Redis for frequently accessed data
- **Connection Pooling**: Efficient database connections
- **Async Processing**: Non-blocking I/O operations

## Extensions

This example can be extended with:
- **Order Fulfillment**: Integration with warehouse systems
- **Shipping**: Integration with shipping providers
- **Returns**: Order return and refund processing
- **Analytics**: Order analytics and reporting
- **Multi-tenant**: Support for multiple merchants