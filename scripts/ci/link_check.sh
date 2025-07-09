#!/bin/bash

# Link Checker for CI/CD Pipeline
# Validates documentation links and references after cleanup

set -e

echo "Running Documentation Link Check"
echo "=================================="

# Check if Python link checker exists
if [ ! -f "scripts/analysis/quick_link_check.py" ]; then
    echo "ERROR: Link checker script not found"
    exit 1
fi

# Run Python link checker
echo "Running Python link checker..."
python scripts/analysis/quick_link_check.py

LINK_CHECK_EXIT_CODE=$?

# Check MkDocs configuration
echo ""
echo "Validating MkDocs configuration..."
if [ -f "config/docs/mkdocs.yml" ]; then
    echo "MkDocs configuration found"
else
    echo "ERROR: MkDocs configuration not found"
    exit 1
fi

# Summary
echo ""
echo "Link Check Summary"
echo "=================="

if [ $LINK_CHECK_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: All critical links are valid"
else
    echo "ERROR: Found $LINK_CHECK_EXIT_CODE link issues"
fi

# Exit with error code if critical issues found
if [ $LINK_CHECK_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Link check failed - please fix broken links"
    exit $LINK_CHECK_EXIT_CODE
fi

echo ""
echo "Documentation link check completed successfully"
exit 0
