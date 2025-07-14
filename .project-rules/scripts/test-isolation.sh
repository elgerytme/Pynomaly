#!/bin/bash

# Test script for Pynomaly isolation framework
# Validates that isolation system works correctly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🧪 Testing Pynomaly Isolation Framework"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

test_passed() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

test_failed() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

test_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Test 1: Check if isolation configuration exists
echo "📋 Test 1: Configuration file exists"
if [ -f "$PROJECT_ROOT/.project-rules/isolation-config.yaml" ]; then
    test_passed "Configuration file found"
else
    test_failed "Configuration file missing"
fi

# Test 2: Check if main isolation script exists and is executable
echo "📋 Test 2: Main isolation script"
if [ -x "$PROJECT_ROOT/.project-rules/scripts/isolate.sh" ]; then
    test_passed "Isolation script is executable"
else
    test_failed "Isolation script missing or not executable"
fi

# Test 3: Check if templates exist
echo "📋 Test 3: Docker templates"
if [ -f "$PROJECT_ROOT/.project-rules/templates/Dockerfile.isolation" ]; then
    test_passed "Dockerfile template found"
else
    test_failed "Dockerfile template missing"
fi

if [ -f "$PROJECT_ROOT/.project-rules/templates/docker-compose.isolation.yml" ]; then
    test_passed "Docker Compose template found"
else
    test_failed "Docker Compose template missing"
fi

# Test 4: Check if helper scripts exist
echo "📋 Test 4: Helper scripts"
HELPER_SCRIPTS=("help.sh" "start-dev.sh" "test.sh")
for script in "${HELPER_SCRIPTS[@]}"; do
    if [ -x "$PROJECT_ROOT/.project-rules/scripts/$script" ]; then
        test_passed "Helper script $script is executable"
    else
        test_failed "Helper script $script missing or not executable"
    fi
done

# Test 5: Check if automation scripts exist
echo "📋 Test 5: Automation scripts"
if [ -x "$PROJECT_ROOT/.project-rules/automation/install-hooks.sh" ]; then
    test_passed "Hook installer script is executable"
else
    test_failed "Hook installer script missing or not executable"
fi

# Test 6: Test isolation script help command
echo "📋 Test 6: Isolation script help"
if "$PROJECT_ROOT/.project-rules/scripts/isolate.sh" help >/dev/null 2>&1; then
    test_passed "Isolation script help command works"
else
    test_failed "Isolation script help command failed"
fi

# Test 7: Test isolation script status command
echo "📋 Test 7: Isolation script status"
if "$PROJECT_ROOT/.project-rules/scripts/isolate.sh" status >/dev/null 2>&1; then
    test_passed "Isolation script status command works"
else
    test_failed "Isolation script status command failed"
fi

# Test 8: Check Docker availability (if enabled)
echo "📋 Test 8: Docker availability"
if command -v docker >/dev/null 2>&1; then
    if docker ps >/dev/null 2>&1; then
        test_passed "Docker is available and running"
    else
        test_warning "Docker is installed but not running"
    fi
else
    test_warning "Docker not installed (required for container isolation)"
fi

# Test 9: Check if isolation is currently disabled (safety)
echo "📋 Test 9: Safety check - isolation disabled"
if grep -q "enabled: false" "$PROJECT_ROOT/.project-rules/isolation-config.yaml" 2>/dev/null; then
    test_passed "Isolation is safely disabled for testing"
else
    test_warning "Isolation may be enabled - check configuration"
fi

# Test 10: Validate YAML configuration syntax
echo "📋 Test 10: Configuration syntax"
if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import yaml; yaml.safe_load(open('$PROJECT_ROOT/.project-rules/isolation-config.yaml'))" 2>/dev/null; then
        test_passed "Configuration YAML syntax is valid"
    else
        test_failed "Configuration YAML syntax is invalid"
    fi
else
    test_warning "Python3 not available to validate YAML syntax"
fi

# Test 11: Check if isolation work directory structure is correct
echo "📋 Test 11: Directory structure"
REQUIRED_DIRS=(
    ".project-rules"
    ".project-rules/scripts"
    ".project-rules/templates"
    ".project-rules/hooks"
    ".project-rules/automation"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$PROJECT_ROOT/$dir" ]; then
        test_passed "Directory $dir exists"
    else
        test_failed "Directory $dir missing"
    fi
done

# Test 12: Dry run test of folder isolation (safest test)
echo "📋 Test 12: Dry run folder isolation creation"
if grep -q "enabled: true" "$PROJECT_ROOT/.project-rules/isolation-config.yaml" 2>/dev/null; then
    test_warning "Isolation is enabled - skipping creation test"
else
    # Since isolation is disabled, we can't actually test creation
    # But we can test the command parsing
    if "$PROJECT_ROOT/.project-rules/scripts/isolate.sh" create folder 2>&1 | grep -q "Isolation is currently disabled"; then
        test_passed "Isolation correctly reports disabled status"
    else
        test_failed "Isolation script doesn't properly check enabled status"
    fi
fi

echo ""
echo "🎯 Test Summary"
echo "=============="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Isolation framework is ready.${NC}"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Enable isolation by setting 'enabled: true' in isolation-config.yaml"
    echo "2. Install Git hooks with: .project-rules/automation/install-hooks.sh"
    echo "3. Test with a real isolation: .project-rules/scripts/isolate.sh create folder"
    echo "4. Gradually test more complex isolation strategies"
    exit 0
else
    echo -e "${RED}💥 Some tests failed. Please fix issues before using isolation.${NC}"
    echo ""
    echo "📋 Common fixes:"
    echo "1. Make scripts executable: chmod +x .project-rules/scripts/*.sh"
    echo "2. Install Docker if using container isolation"
    echo "3. Check file paths and permissions"
    exit 1
fi
