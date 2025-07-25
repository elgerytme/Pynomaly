# Payment Gateway Example

A secure, PCI-compliant payment processing system supporting multiple payment methods and currencies, built with hexagonal architecture for maximum flexibility.

## Features

- **Multi-Payment Methods**: Credit/debit cards, digital wallets, bank transfers
- **Currency Support**: 50+ currencies with real-time exchange rates
- **Fraud Detection**: ML-powered transaction analysis
- **PCI Compliance**: Secure card data handling
- **Webhook System**: Real-time payment notifications
- **Reconciliation**: Automated settlement reconciliation
- **Subscription Billing**: Recurring payment management
- **Marketplace Support**: Split payments and escrow

## Architecture

Hexagonal architecture with clear separation of concerns:

```
src/payment_gateway/
├── domain/                    # Core business logic
│   ├── entities/             # Payment, Transaction entities
│   ├── value_objects/        # Money, Currency, CardDetails
│   ├── aggregates/           # Payment aggregate root
│   └── services/             # Domain services
├── application/              # Use cases
│   ├── use_cases/           # Process payment, refund
│   ├── ports/               # Input/output ports
│   └── dto/                 # Data transfer objects
├── infrastructure/          # External adapters
│   ├── payment_providers/   # Stripe, PayPal adapters
│   ├── fraud_detection/     # ML fraud detection
│   ├── databases/           # PostgreSQL repositories
│   └── messaging/           # Event publishing
└── presentation/            # API layer
    ├── rest/                # REST API endpoints
    └── webhooks/            # Webhook handlers
```

## Quick Start

```bash
# Start the payment gateway
cd examples/fintech-payment-gateway
docker-compose up -d

# Process a test payment
curl -X POST http://localhost:8000/api/payments \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "amount": 1000,
    "currency": "USD",
    "payment_method": {
      "type": "card",
      "card_number": "4242424242424242",
      "exp_month": 12,
      "exp_year": 2025,
      "cvc": "123"
    },
    "customer_id": "cust_123"
  }'
```

## API Endpoints

### Payments
- `POST /api/payments` - Process payment
- `GET /api/payments/{payment_id}` - Get payment details
- `POST /api/payments/{payment_id}/capture` - Capture authorized payment
- `POST /api/payments/{payment_id}/refund` - Refund payment

### Customers
- `POST /api/customers` - Create customer
- `GET /api/customers/{customer_id}` - Get customer
- `POST /api/customers/{customer_id}/payment_methods` - Add payment method

### Subscriptions
- `POST /api/subscriptions` - Create subscription
- `GET /api/subscriptions/{subscription_id}` - Get subscription
- `PUT /api/subscriptions/{subscription_id}` - Update subscription
- `DELETE /api/subscriptions/{subscription_id}` - Cancel subscription

### Webhooks
- `POST /webhooks/stripe` - Stripe webhook handler
- `POST /webhooks/paypal` - PayPal webhook handler

## Payment Processing Flow

### Credit Card Payment
```python
# 1. Validate card details
payment_request = PaymentRequest(
    amount=Money(amount=1000, currency="USD"),
    payment_method=CardPaymentMethod(
        card_number="4242424242424242",
        exp_month=12,
        exp_year=2025,
        cvc="123"
    ),
    customer_id="cust_123"
)

# 2. Fraud detection
fraud_score = fraud_detector.analyze(payment_request)
if fraud_score > FRAUD_THRESHOLD:
    return PaymentResult.rejected("High fraud risk")

# 3. Process with payment provider
result = payment_processor.process(payment_request)

# 4. Store transaction
transaction_repo.save(result.transaction)

# 5. Send notifications
event_publisher.publish(PaymentProcessedEvent(result))
```

### Digital Wallet Payment
```python
# PayPal, Apple Pay, Google Pay integration
wallet_payment = WalletPaymentRequest(
    amount=Money(amount=2500, currency="EUR"),
    wallet_type="paypal",
    wallet_token="EC-123456789",
    customer_id="cust_456"
)

result = wallet_processor.process(wallet_payment)
```

## Supported Payment Methods

### Credit/Debit Cards
- Visa, Mastercard, American Express, Discover
- 3D Secure authentication
- Tokenization for repeat customers
- Installment payments

### Digital Wallets
- PayPal, Apple Pay, Google Pay
- Alipay, WeChat Pay (for Chinese market)
- Local payment methods by region

### Bank Transfers
- ACH (US), SEPA (EU), BACS (UK)
- Open Banking integration
- Real-time payment verification

### Cryptocurrencies
- Bitcoin, Ethereum, Litecoin
- Stablecoin support (USDC, USDT)
- Automatic conversion to fiat

## Security & Compliance

### PCI DSS Compliance
- Level 1 PCI DSS certification
- Tokenization of sensitive card data
- Encrypted data transmission
- Secure key management

### Fraud Prevention
- Machine learning risk scoring
- Velocity checks and limits
- Device fingerprinting
- Geolocation analysis

### Data Protection
- GDPR compliance for EU customers
- PII encryption at rest
- Audit logging
- Data retention policies

## Multi-Currency Support

### Exchange Rates
- Real-time rates from multiple providers
- Automatic rate updates
- Historical rate tracking
- Rate markup configuration

### Settlement
- Multi-currency merchant accounts
- Automatic currency conversion
- FX hedging options
- Settlement in local currencies

## Configuration

```yaml
# Payment providers
providers:
  stripe:
    enabled: true
    secret_key: sk_test_...
    webhook_secret: whsec_...
    
  paypal:
    enabled: true
    client_id: your_client_id
    client_secret: your_client_secret
    environment: sandbox

# Fraud detection
fraud_detection:
  enabled: true
  threshold: 0.7
  rules:
    - velocity_check
    - geolocation_check
    - device_fingerprint

# Currencies
currencies:
  default: USD
  supported:
    - USD
    - EUR
    - GBP
    - CAD
    - AUD
    - JPY

# Limits
limits:
  max_transaction_amount:
    USD: 100000
    EUR: 90000
  daily_volume_limit: 1000000
  monthly_volume_limit: 10000000
```

## Monitoring & Analytics

### Key Metrics
- Transaction success rates
- Average processing time
- Fraud detection accuracy
- Revenue by payment method
- Chargeback rates

### Dashboards
- Real-time transaction monitoring
- Financial reconciliation reports
- Fraud detection analytics
- Performance metrics

### Alerts
- High decline rates
- Unusual transaction patterns
- System performance issues
- Compliance violations

## Testing

### Test Cards
```
# Successful payments
4242424242424242  # Visa
5555555555554444  # Mastercard
378282246310005   # American Express

# Declined payments
4000000000000002  # Generic decline
4000000000009995  # Insufficient funds
4000000000009987  # Lost card
```

### Test Environment
```bash
# Run integration tests
pytest tests/integration/ -v

# Test fraud detection
pytest tests/fraud/ -v

# Load testing
locust -f tests/load/locustfile.py
```

## Deployment

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/payments_db

# Payment providers
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
PAYPAL_CLIENT_ID=your_client_id
PAYPAL_CLIENT_SECRET=your_client_secret

# Security
ENCRYPTION_KEY=your_32_byte_key
JWT_SECRET_KEY=your_jwt_secret

# Features
FRAUD_DETECTION_ENABLED=true
WEBHOOK_RETRY_ATTEMPTS=3
TRANSACTION_TIMEOUT_SECONDS=30
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

## Extensions

This payment gateway can be extended with:
- **Buy Now Pay Later**: Installment payment options
- **Marketplace Features**: Multi-vendor split payments
- **Advanced Fraud**: Behavioral biometrics
- **Open Banking**: Account-to-account transfers
- **Cryptocurrency**: DeFi protocol integration
- **Analytics**: Advanced financial reporting