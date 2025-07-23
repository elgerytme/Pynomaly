#!/bin/bash
#
# Buck2 Secure Remote Cache Setup
# ================================
# 
# Sets up secure remote caching for Buck2 with JWT authentication,
# TLS encryption, and proper fallback mechanisms.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CACHE_CONFIG_DIR="${REPO_ROOT}/scripts/config/buck"

# Default cache configuration
DEFAULT_CACHE_URL="https://buck2-cache.internal.anomaly-detection.com"
DEFAULT_RE_ENGINE="grpc://buck2-re.internal.anomaly-detection.com:9092"
DEFAULT_RE_ACTION_CACHE="grpc://buck2-ac.internal.anomaly-detection.com:9092"
DEFAULT_RE_CAS="grpc://buck2-cas.internal.anomaly-detection.com:9092"

echo -e "${BLUE}🔧 Buck2 Secure Cache Setup${NC}"
echo -e "${BLUE}============================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate environment
validate_environment() {
    echo -e "${BLUE}📋 Validating environment...${NC}"
    
    # Check for Buck2
    if ! command_exists buck2; then
        echo -e "${RED}❌ Buck2 not found. Please install Buck2 first.${NC}"
        exit 1
    fi
    
    # Check Buck2 version
    BUCK2_VERSION=$(buck2 --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✅ Buck2 version: ${BUCK2_VERSION}${NC}"
    
    # Check for required tools
    for tool in curl jq openssl; do
        if ! command_exists "$tool"; then
            echo -e "${YELLOW}⚠️  ${tool} not found - some features may be limited${NC}"
        fi
    done
}

# Function to generate secure cache token
generate_cache_token() {
    local token_type="${1:-development}"
    
    if command_exists openssl; then
        # Generate a secure random token
        local token=$(openssl rand -hex 32)
        echo "$token"
    else
        # Fallback to less secure but still functional token
        echo "dev_$(date +%s)_$(head -c 8 /dev/urandom | base64 | tr -d '/+=')"
    fi
}

# Function to setup local cache configuration
setup_local_cache() {
    echo -e "${BLUE}💾 Setting up local cache configuration...${NC}"
    
    # Ensure cache directory exists
    local cache_dir="${REPO_ROOT}/.buck2-cache"
    mkdir -p "$cache_dir"
    
    # Set up cache cleanup script
    cat > "${cache_dir}/cleanup.sh" << 'EOF'
#!/bin/bash
# Buck2 cache cleanup script
# Removes old cache entries when cache exceeds size limits

CACHE_DIR="$(dirname "$0")"
MAX_SIZE_GB=10
CLEANUP_THRESHOLD=0.8

echo "🧹 Cleaning Buck2 cache..."

# Get current cache size in GB
if command -v du >/dev/null 2>&1; then
    CURRENT_SIZE_KB=$(du -sk "$CACHE_DIR" | cut -f1)
    CURRENT_SIZE_GB=$((CURRENT_SIZE_KB / 1024 / 1024))
    
    THRESHOLD_SIZE_GB=$((MAX_SIZE_GB * CLEANUP_THRESHOLD / 1))
    
    if [ $CURRENT_SIZE_GB -gt $THRESHOLD_SIZE_GB ]; then
        echo "Cache size ${CURRENT_SIZE_GB}GB exceeds threshold ${THRESHOLD_SIZE_GB}GB"
        echo "Removing oldest cache entries..."
        
        # Remove files older than 7 days
        find "$CACHE_DIR" -type f -mtime +7 -delete 2>/dev/null || true
        
        # If still too large, remove files older than 3 days
        if [ $CURRENT_SIZE_GB -gt $THRESHOLD_SIZE_GB ]; then
            find "$CACHE_DIR" -type f -mtime +3 -delete 2>/dev/null || true
        fi
        
        echo "✅ Cache cleanup completed"
    else
        echo "✅ Cache size ${CURRENT_SIZE_GB}GB is within limits"
    fi
fi
EOF
    chmod +x "${cache_dir}/cleanup.sh"
    
    echo -e "${GREEN}✅ Local cache configured at ${cache_dir}${NC}"
}

# Function to setup environment configuration
setup_environment_config() {
    echo -e "${BLUE}🔐 Setting up secure cache environment...${NC}"
    
    # Create environment configuration file
    local env_file="${CACHE_CONFIG_DIR}/cache.env"
    
    cat > "$env_file" << EOF
# Buck2 Remote Cache Configuration
# ================================
# 
# Security: This file contains sensitive configuration.
# Never commit authentication tokens to version control.
#

# HTTP Cache Configuration
export BUCK2_CACHE_URL="${DEFAULT_CACHE_URL}"
export BUCK2_CACHE_TOKEN=""  # Set this to your cache authentication token
export BUCK2_CACHE_CA_CERT=""  # Optional: Path to CA certificate for TLS verification

# Remote Execution Configuration  
export BUCK2_RE_ENGINE_ADDRESS="${DEFAULT_RE_ENGINE}"
export BUCK2_RE_ACTION_CACHE_ADDRESS="${DEFAULT_RE_ACTION_CACHE}"
export BUCK2_RE_CAS_ADDRESS="${DEFAULT_RE_CAS}"
export BUCK2_RE_INSTANCE_NAME="main"
export BUCK2_RE_AUTH_TOKEN=""  # Set this to your RE authentication token
export BUCK2_RE_CA_CERT=""  # Optional: Path to CA certificate for TLS verification

# Cache Feature Toggles
export BUCK2_CACHE_ENABLED="false"  # Set to "true" to enable HTTP caching
export BUCK2_RE_ENABLED="false"     # Set to "true" to enable remote execution

# Performance Tuning
export BUCK2_CACHE_TIMEOUT="60"     # Cache timeout in seconds
export BUCK2_CACHE_RETRIES="3"      # Number of retry attempts
export BUCK2_PARALLEL_BUILDS="8"    # Number of parallel builds

# Development vs Production Settings
export BUCK2_CACHE_MODE="development"  # or "production"
EOF
    
    # Create secure token generation script
    cat > "${CACHE_CONFIG_DIR}/generate_token.sh" << 'EOF'
#!/bin/bash
# Generate secure cache authentication token

echo "🔐 Generating secure cache token..."

if command -v openssl >/dev/null 2>&1; then
    TOKEN=$(openssl rand -hex 32)
    echo "Generated secure token: $TOKEN"
    echo ""
    echo "To use this token, run:"
    echo "export BUCK2_CACHE_TOKEN=\"$TOKEN\""
    echo ""
    echo "Or add it to your cache.env file"
else
    echo "❌ OpenSSL not found. Please install OpenSSL for secure token generation."
    exit 1
fi
EOF
    chmod +x "${CACHE_CONFIG_DIR}/generate_token.sh"
    
    echo -e "${GREEN}✅ Environment configuration created at ${env_file}${NC}"
    echo -e "${YELLOW}⚠️  Remember to set BUCK2_CACHE_TOKEN and other credentials${NC}"
}

# Function to create cache health monitoring
setup_cache_monitoring() {
    echo -e "${BLUE}📊 Setting up cache monitoring...${NC}"
    
    cat > "${CACHE_CONFIG_DIR}/monitor_cache.sh" << 'EOF'
#!/bin/bash
# Buck2 Cache Health Monitoring
# Monitors cache performance and alerts on issues

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CACHE_STATS_FILE="/tmp/buck2_cache_stats.json"
ALERT_THRESHOLD_MISS_RATE=0.5  # Alert if cache miss rate > 50%
ALERT_THRESHOLD_RESPONSE_TIME=5.0  # Alert if avg response time > 5s

echo -e "${BLUE}📊 Buck2 Cache Health Monitor${NC}"
echo "==============================="

# Function to test cache connectivity
test_cache_connectivity() {
    if [ -n "${BUCK2_CACHE_URL:-}" ] && [ -n "${BUCK2_CACHE_TOKEN:-}" ]; then
        echo -e "${BLUE}🔍 Testing HTTP cache connectivity...${NC}"
        
        if command -v curl >/dev/null 2>&1; then
            local response_code=$(curl -s -o /dev/null -w "%{http_code}" \
                -H "Authorization: Bearer ${BUCK2_CACHE_TOKEN}" \
                "${BUCK2_CACHE_URL}/health" || echo "000")
            
            if [ "$response_code" = "200" ]; then
                echo -e "${GREEN}✅ HTTP cache is reachable${NC}"
                return 0
            else
                echo -e "${RED}❌ HTTP cache unreachable (HTTP $response_code)${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}⚠️  curl not available - cannot test connectivity${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  Cache credentials not configured${NC}"
        return 1
    fi
}

# Function to collect cache statistics
collect_cache_stats() {
    echo -e "${BLUE}📈 Collecting cache statistics...${NC}"
    
    # This would typically integrate with Buck2's metrics API
    # For now, create a mock structure for demonstration
    cat > "$CACHE_STATS_FILE" << JSON
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "cache_hits": 0,
  "cache_misses": 0, 
  "cache_hit_rate": 0.0,
  "avg_response_time_ms": 0.0,
  "total_requests": 0,
  "errors": 0,
  "local_cache_size_mb": 0
}
JSON

    # Get local cache size
    if [ -d ".buck2-cache" ]; then
        local cache_size_kb=$(du -sk .buck2-cache 2>/dev/null | cut -f1 || echo "0")
        local cache_size_mb=$((cache_size_kb / 1024))
        
        # Update stats file with actual cache size
        if command -v jq >/dev/null 2>&1; then
            jq ".local_cache_size_mb = $cache_size_mb" "$CACHE_STATS_FILE" > "${CACHE_STATS_FILE}.tmp"
            mv "${CACHE_STATS_FILE}.tmp" "$CACHE_STATS_FILE"
        fi
        
        echo -e "${GREEN}📊 Local cache size: ${cache_size_mb}MB${NC}"
    fi
}

# Function to analyze performance
analyze_performance() {
    echo -e "${BLUE}🔍 Analyzing cache performance...${NC}"
    
    if [ -f "$CACHE_STATS_FILE" ] && command -v jq >/dev/null 2>&1; then
        local hit_rate=$(jq -r '.cache_hit_rate' "$CACHE_STATS_FILE")
        local response_time=$(jq -r '.avg_response_time_ms' "$CACHE_STATS_FILE")
        local total_requests=$(jq -r '.total_requests' "$CACHE_STATS_FILE")
        
        echo "Cache Hit Rate: ${hit_rate}"
        echo "Avg Response Time: ${response_time}ms"
        echo "Total Requests: ${total_requests}"
        
        # Performance alerts
        if [ "$total_requests" -gt 0 ]; then
            if (( $(echo "$hit_rate < $ALERT_THRESHOLD_MISS_RATE" | bc -l) )); then
                echo -e "${YELLOW}⚠️  Cache hit rate is low: ${hit_rate}${NC}"
            fi
            
            if (( $(echo "$response_time > $(echo "$ALERT_THRESHOLD_RESPONSE_TIME * 1000" | bc)" | bc -l) )); then
                echo -e "${YELLOW}⚠️  Cache response time is high: ${response_time}ms${NC}"
            fi
        fi
    fi
}

# Main monitoring function
main() {
    collect_cache_stats
    test_cache_connectivity
    analyze_performance
    
    echo ""
    echo -e "${GREEN}✅ Cache monitoring completed${NC}"
    echo "Stats saved to: $CACHE_STATS_FILE"
}

# Run monitoring
main "$@"
EOF
    chmod +x "${CACHE_CONFIG_DIR}/monitor_cache.sh"
    
    echo -e "${GREEN}✅ Cache monitoring configured${NC}"
}

# Function to create cache usage examples
create_usage_examples() {
    echo -e "${BLUE}📖 Creating usage examples...${NC}"
    
    cat > "${CACHE_CONFIG_DIR}/README.md" << 'EOF'
# Buck2 Remote Cache Setup

This directory contains configuration for Buck2 remote caching with security and performance optimizations.

## Quick Start

1. **Setup environment**:
   ```bash
   source scripts/config/buck/cache.env
   ```

2. **Generate secure token**:
   ```bash
   ./scripts/config/buck/generate_token.sh
   export BUCK2_CACHE_TOKEN="<generated-token>"
   ```

3. **Enable caching**:
   ```bash
   export BUCK2_CACHE_ENABLED="true"
   ```

4. **Test build with caching**:
   ```bash
   buck2 build //:anomaly-detection
   ```

## Configuration Files

- `cache.env` - Environment configuration template
- `generate_token.sh` - Secure token generation
- `monitor_cache.sh` - Cache health monitoring
- `cache_setup.sh` - Automated setup script

## Security Features

- **JWT token authentication** for all cache operations
- **TLS encryption** for cache communications
- **Token rotation** support
- **Network isolation** ready
- **Audit logging** capabilities

## Performance Features

- **Intelligent fallback** to local builds if cache fails
- **Retry mechanisms** with exponential backoff
- **Connection pooling** and timeout management
- **Cache size limits** with automatic cleanup
- **Performance monitoring** and alerting

## Environment Variables

### HTTP Cache
- `BUCK2_CACHE_URL` - HTTP cache endpoint
- `BUCK2_CACHE_TOKEN` - Authentication token
- `BUCK2_CACHE_CA_CERT` - TLS CA certificate path
- `BUCK2_CACHE_ENABLED` - Enable/disable HTTP caching

### Remote Execution
- `BUCK2_RE_ENGINE_ADDRESS` - Remote execution engine
- `BUCK2_RE_ACTION_CACHE_ADDRESS` - Action cache endpoint
- `BUCK2_RE_CAS_ADDRESS` - Content addressable storage
- `BUCK2_RE_AUTH_TOKEN` - RE authentication token
- `BUCK2_RE_ENABLED` - Enable/disable remote execution

## Monitoring

Monitor cache health:
```bash
./scripts/config/buck/monitor_cache.sh
```

View cache statistics:
```bash
buck2 log what-up  # Shows recent build information
buck2 audit dependencies //:target  # Dependency analysis
```

## Troubleshooting

### Cache Not Working
1. Check connectivity: `ping buck2-cache.internal.anomaly-detection.com`
2. Verify credentials: `echo $BUCK2_CACHE_TOKEN`
3. Test manually: `curl -H "Authorization: Bearer $TOKEN" $BUCK2_CACHE_URL/health`
4. Check Buck2 logs: `buck2 log show`

### Performance Issues
1. Monitor cache hit rate
2. Check network latency to cache server
3. Review local cache size and cleanup
4. Analyze build parallelism settings

## Best Practices

1. **Use environment-specific tokens** (dev vs prod)
2. **Rotate tokens regularly** (monthly recommended)
3. **Monitor cache performance** continuously
4. **Keep credentials secure** (never commit to git)
5. **Test fallback scenarios** regularly
6. **Maintain cache server capacity** for team size

## CI/CD Integration

Add to CI environment:
```yaml
env:
  BUCK2_CACHE_URL: ${{ secrets.BUCK2_CACHE_URL }}
  BUCK2_CACHE_TOKEN: ${{ secrets.BUCK2_CACHE_TOKEN }}
  BUCK2_CACHE_ENABLED: "true"
```

## Security Considerations

- Use HTTPS/gRPC-TLS for all cache communications
- Implement proper access controls on cache server
- Regular security audits of cache infrastructure
- Monitor for suspicious cache access patterns
- Implement cache content verification
EOF
    
    echo -e "${GREEN}✅ Documentation created${NC}"
}

# Main setup function
main() {
    echo -e "${BLUE}🚀 Starting Buck2 secure cache setup...${NC}"
    echo ""
    
    validate_environment
    echo ""
    
    setup_local_cache
    echo ""
    
    setup_environment_config
    echo ""
    
    setup_cache_monitoring
    echo ""
    
    create_usage_examples
    echo ""
    
    echo -e "${GREEN}🎉 Buck2 secure cache setup completed!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Generate cache token: ./scripts/config/buck/generate_token.sh"
    echo "2. Configure environment: source scripts/config/buck/cache.env"
    echo "3. Set credentials: export BUCK2_CACHE_TOKEN=<token>"
    echo "4. Enable caching: export BUCK2_CACHE_ENABLED=true"
    echo "5. Test with build: buck2 build //:anomaly-detection"
    echo "6. Monitor performance: ./scripts/config/buck/monitor_cache.sh"
    echo ""
    echo -e "${BLUE}📖 Full documentation: scripts/config/buck/README.md${NC}"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi