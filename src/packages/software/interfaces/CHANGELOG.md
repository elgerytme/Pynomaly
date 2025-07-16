# Changelog - Interfaces Package

All notable changes to the Monorepo interfaces package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced WebSocket real-time streaming capabilities
- Progressive Web App (PWA) offline support
- Advanced CLI interactive wizards and autocomplete
- GraphQL subscriptions for real-time updates
- Mobile-responsive dashboard improvements

### Changed
- Improved API response caching and compression
- Enhanced CLI performance with lazy loading
- Better error handling and user feedback
- Optimized web bundle size and loading times

### Fixed
- WebSocket connection stability under high load
- CLI command parsing edge cases
- Mobile browser compatibility issues
- API rate limiting accuracy

## [1.0.0] - 2025-07-14

### Added
- **REST API**: Comprehensive FastAPI-based async API
  - 65+ endpoints for complete functionality coverage
  - OpenAPI 3.0 documentation with interactive Swagger UI
  - Async request handling for high performance
  - Comprehensive input validation and error handling
  - JWT-based authentication and authorization
- **Command Line Interface**: Rich CLI with Typer framework
  - Interactive commands with progress bars and spinners
  - Batch processing capabilities for large datasets
  - Configuration management and profiles
  - Export functionality to multiple formats
  - Comprehensive help system and autocomplete
- **Web Application**: Progressive Web App with modern stack
  - HTMX and Tailwind CSS for reactive UI
  - Real-time dashboard with WebSocket updates
  - Responsive design for mobile and desktop
  - Dark/light theme support with system preference detection
  - Offline capabilities with service worker
- **Python SDK**: Client library for programmatic access
  - Async and sync client implementations
  - Type-safe request/response models
  - Comprehensive error handling and retries
  - Built-in caching and request optimization
  - Session management and authentication
- **JavaScript SDK**: Browser and Node.js client library
  - TypeScript definitions for type safety
  - Promise-based async operations
  - Automatic request retries and error handling
  - WebSocket support for real-time features
  - Bundle optimization for web applications

### API Endpoints
- **Detection**: `/api/v1/detection/*` - Anomaly detection operations
- **Datasets**: `/api/v1/datasets/*` - Dataset management
- **Models**: `/api/v1/models/*` - Model operations and training
- **Results**: `/api/v1/results/*` - Detection result management
- **Users**: `/api/v1/users/*` - User management and authentication
- **System**: `/api/v1/system/*` - System health and metrics

### CLI Commands
- `pynomaly detect` - Run anomaly detection on datasets
- `pynomaly train` - Train and save detection models
- `pynomaly evaluate` - Evaluate model performance
- `pynomaly export` - Export results to various formats
- `pynomaly config` - Manage configuration profiles
- `pynomaly server` - Start API server and web application

### Web Features
- **Dashboard**: Real-time anomaly detection monitoring
- **Dataset Manager**: Upload, view, and manage datasets
- **Model Studio**: Train, evaluate, and compare models
- **Results Viewer**: Interactive result visualization
- **Settings**: Configuration and user preferences
- **Reports**: Generate and export detection reports

### Real-time Features
- **WebSocket Streaming**: Live anomaly detection results
- **Real-time Dashboard**: Live metrics and visualizations
- **Notifications**: Real-time alerts and notifications
- **Collaborative Features**: Multi-user real-time collaboration
- **Live Model Training**: Progress tracking for long-running operations

## [0.9.0] - 2025-06-01

### Added
- Initial REST API implementation
- Basic CLI commands for core operations
- Foundation web application structure
- Initial client SDK development

### Changed
- Refined API endpoint structure
- Improved CLI command organization
- Enhanced error handling across interfaces

### Fixed
- Initial performance optimizations
- API response consistency improvements
- CLI output formatting enhancements

## [0.1.0] - 2025-01-15

### Added
- Project structure for user interfaces
- Basic API endpoint definitions
- CLI framework setup

---

## Interface Support Matrix

| Interface | Technology | Status | Real-time | Mobile Support |
|-----------|------------|--------|-----------|----------------|
| REST API | FastAPI | ✅ Stable | ✅ WebSocket | ✅ Responsive |
| CLI | Typer + Rich | ✅ Stable | ❌ | ❌ |
| Web App | HTMX + Tailwind | ✅ Stable | ✅ WebSocket | ✅ PWA |
| Python SDK | aiohttp | ✅ Stable | ✅ WebSocket | N/A |
| JavaScript SDK | TypeScript | ✅ Stable | ✅ WebSocket | ✅ |
| GraphQL | Strawberry | 🚧 Beta | ✅ Subscriptions | ✅ |

## Performance Benchmarks

### API Performance
- **Throughput**: 10,000 requests/sec with proper caching
- **Latency**: < 50ms median response time
- **WebSocket**: 1,000 concurrent connections supported
- **Memory Usage**: < 512MB for typical workloads

### CLI Performance
- **Startup Time**: < 500ms for most commands
- **Large Datasets**: Progress tracking for 1M+ samples
- **Export Speed**: 100MB/sec for common formats
- **Memory Efficient**: Streaming processing for large files

### Web Application
- **Initial Load**: < 2 seconds on 3G networks
- **Bundle Size**: < 500KB gzipped JavaScript
- **PWA Score**: 95+ Lighthouse performance score
- **Real-time Latency**: < 100ms WebSocket round-trip

## API Reference

### Authentication
```python
# Python SDK
from pynomaly.interfaces.sdk.python import PynomalyClient

client = PynomalyClient(
    base_url="https://api.pynomaly.com",
    api_key="your-api-key"
)

# JavaScript SDK
import { PynomalyClient } from '@pynomaly/sdk';

const client = new PynomalyClient({
    baseUrl: 'https://api.pynomaly.com',
    apiKey: 'your-api-key'
});
```

### Detection Operations
```python
# Async detection
result = await client.detect_anomalies(
    dataset_id="dataset_001",
    algorithm="isolation_forest",
    parameters={"contamination": 0.1}
)

# Real-time streaming
async for update in client.stream_detection(dataset_id):
    print(f"Progress: {update.progress}%")
    if update.completed:
        print(f"Found {len(update.anomalies)} anomalies")
```

### CLI Usage Examples
```bash
# Interactive detection wizard
pynomaly detect --interactive

# Batch processing
pynomaly detect \
    --input data.csv \
    --algorithm isolation_forest \
    --contamination 0.1 \
    --output results.json

# Start development server
pynomaly server --host 0.0.0.0 --port 8000 --reload
```

## Configuration Examples

### API Server Configuration
```python
from pynomaly.interfaces.config import APISettings

settings = APISettings(
    host="0.0.0.0",
    port=8000,
    workers=4,
    enable_cors=True,
    cors_origins=["http://localhost:3000"],
    enable_docs=True,
    rate_limit="100/minute",
    jwt_secret="your-secret-key"
)
```

### Web Application Configuration
```python
from pynomaly.interfaces.config import WebSettings

settings = WebSettings(
    theme="auto",  # auto, light, dark
    enable_pwa=True,
    enable_offline=True,
    websocket_url="ws://localhost:8000/ws",
    api_base_url="http://localhost:8000/api/v1",
    update_interval=5000  # milliseconds
)
```

## Migration Guide

### Upgrading to 1.0.0

```python
# Before (0.9.x)
from pynomaly.api import create_app
app = create_app()

# After (1.0.0)
from pynomaly.interfaces.api import create_app
from pynomaly.interfaces.config import APISettings

settings = APISettings(host="0.0.0.0", port=8000)
app = create_app(settings)
```

## Security Features

1. **API Security**
   - JWT-based authentication with refresh tokens
   - Role-based access control (RBAC)
   - Rate limiting and DDoS protection
   - CORS configuration for web security
   - Input validation and sanitization

2. **Web Security**
   - Content Security Policy (CSP) headers
   - HTTPS enforcement in production
   - Secure cookie configuration
   - XSS and CSRF protection
   - Secure WebSocket connections

3. **CLI Security**
   - Secure credential storage
   - API key management
   - Audit logging for sensitive operations

## Accessibility

- **WCAG 2.1 AA Compliance**: Full accessibility standard compliance
- **Keyboard Navigation**: Complete keyboard-only navigation support
- **Screen Reader Support**: ARIA labels and semantic HTML
- **High Contrast**: Dark mode and high contrast theme options
- **Responsive Design**: Mobile and tablet optimized interfaces

## Dependencies

### Runtime Dependencies
- `fastapi>=0.103.0`: Modern async web framework
- `typer>=0.9.0`: CLI framework with rich features
- `aiohttp>=3.8.0`: Async HTTP client
- `websockets>=11.0.0`: WebSocket support
- `pydantic>=2.0.0`: Data validation

### Frontend Dependencies
- `htmx>=1.9.0`: Reactive web interactions
- `tailwindcss>=3.3.0`: Utility-first CSS framework
- `alpinejs>=3.13.0`: Lightweight JavaScript framework

### Development Dependencies
- `pytest-asyncio>=0.21.0`: Async testing support
- `httpx>=0.24.0`: Modern HTTP client for testing
- `playwright>=1.37.0`: End-to-end testing

## Contributing

When contributing interface components:

1. **User Experience**: Prioritize intuitive and accessible design
2. **Performance**: Optimize for speed and responsiveness
3. **Security**: Follow security best practices
4. **Testing**: Add comprehensive E2E tests
5. **Documentation**: Update user guides and API docs

For detailed contribution guidelines, see [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## Support

- **Package Documentation**: [docs/](docs/)
- **API Documentation**: [Interactive API Docs](http://localhost:8000/docs)
- **User Guide**: [docs/user_guide.md](docs/user_guide.md)
- **CLI Reference**: [docs/cli_reference.md](docs/cli_reference.md)
- **Issues**: [GitHub Issues](../../../issues)

[Unreleased]: https://github.com/elgerytme/Pynomaly/compare/interfaces-v1.0.0...HEAD
[1.0.0]: https://github.com/elgerytme/Pynomaly/releases/tag/interfaces-v1.0.0
[0.9.0]: https://github.com/elgerytme/Pynomaly/releases/tag/interfaces-v0.9.0
[0.1.0]: https://github.com/elgerytme/Pynomaly/releases/tag/interfaces-v0.1.0