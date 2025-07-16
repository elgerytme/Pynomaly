# API Versioning Strategy

## Overview

Pynomaly API follows semantic versioning and provides multiple versioning strategies to ensure backward compatibility and smooth upgrades.

## Versioning Scheme

### URL Path Versioning (Current)
- **Current**: `/api/v1/`
- **Future**: `/api/v2/`, `/api/v3/`

### Header Versioning (Alternative)
```
Accept: application/vnd.pynomaly.v1+json
```

## Version Lifecycle

### Version 1.0 (Current)
- **Status**: Stable
- **Support**: Full support
- **Deprecation**: TBD
- **Features**: 
  - JWT Authentication
  - MFA Support
  - Anomaly Detection
  - Admin Operations

### Version 2.0 (Planned)
- **Status**: In Development
- **Features**:
  - GraphQL API
  - Real-time Streaming
  - Enhanced ML Models
  - Advanced Analytics

## Backward Compatibility

### Breaking Changes Policy
- **Major Version**: Breaking changes allowed
- **Minor Version**: Backward compatible features
- **Patch Version**: Bug fixes only

### Deprecation Process
1. **Announce**: 90 days notice
2. **Mark**: Add deprecation headers
3. **Document**: Update documentation
4. **Remove**: In next major version

### Migration Support
- **Migration Guide**: Step-by-step instructions
- **Dual Support**: 12 months overlap
- **Automated Tools**: Migration scripts

## Version Detection

### Client Implementation
```python
class VersionedClient:
    def __init__(self, base_url: str, version: str = "v1"):
        self.base_url = base_url
        self.version = version
    
    def get_endpoint(self, path: str) -> str:
        return f"{self.base_url}/api/{self.version}{path}"
    
    def get_headers(self) -> dict:
        return {
            "Accept": f"application/vnd.pynomaly.{self.version}+json",
            "User-Agent": "Pynomaly-Client/1.0"
        }
```

### Server Response Headers
```
API-Version: 1.0
Supported-Versions: 1.0, 1.1
Deprecated-Versions: 0.9
```

## Feature Flags

### Implementation
```python
class FeatureManager:
    def __init__(self, version: str):
        self.version = version
        self.features = self._load_features()
    
    def is_enabled(self, feature: str) -> bool:
        return self.features.get(feature, False)
    
    def _load_features(self) -> dict:
        return {
            "v1": {
                "mfa": True,
                "streaming": False,
                "graphql": False
            },
            "v2": {
                "mfa": True,
                "streaming": True,
                "graphql": True
            }
        }.get(self.version, {})
```

## Testing Strategy

### Cross-Version Testing
```python
import pytest

@pytest.mark.parametrize("version", ["v1", "v2"])
def test_health_endpoint(version):
    client = VersionedClient("http://localhost:8000", version)
    response = client.get("/health")
    assert response.status_code == 200
```

### Compatibility Matrix
| Feature | v1.0 | v1.1 | v2.0 |
|---------|------|------|------|
| JWT Auth | ✓ | ✓ | ✓ |
| MFA | ✓ | ✓ | ✓ |
| Streaming | ✗ | ✗ | ✓ |
| GraphQL | ✗ | ✗ | ✓ |
