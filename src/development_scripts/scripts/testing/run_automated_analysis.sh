#!/bin/bash

# Run Automated Test Coverage Analysis and Issue Creation
# This script demonstrates the complete automated workflow

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

print_status "Starting Automated Test Coverage Analysis and Issue Creation"
print_status "Project Root: $PROJECT_ROOT"
print_status "Script Directory: $SCRIPT_DIR"

# Check dependencies
print_status "Checking dependencies..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Check if required Python packages are available
python3 -c "import click, json, pathlib" 2>/dev/null || {
    print_warning "Installing required Python packages..."
    pip install click
}

print_success "Dependencies checked"

# Step 1: Run automated test coverage analysis
print_status "Running automated test coverage analysis..."

cd "$PROJECT_ROOT"

# Run quick analysis (no actual test execution for speed)
if python3 "$SCRIPT_DIR/automated_test_coverage_analysis.py" --project-root . --output-format both; then
    print_success "Test coverage analysis completed"
else
    print_error "Test coverage analysis failed"
    exit 1
fi

# Step 2: Create GitHub issues
print_status "Creating GitHub issues for identified gaps..."

if python3 "$SCRIPT_DIR/create_github_issues.py" --project-root .; then
    print_success "GitHub issues created"
else
    print_error "GitHub issue creation failed"
    exit 1
fi

# Step 3: Display results summary
print_status "Generating results summary..."

# Find the latest report
LATEST_REPORT=$(find reports -name "test_coverage_summary_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -f "$LATEST_REPORT" ]; then
    print_status "Analysis Results Summary:"
    echo "=========================="
    
    # Extract key metrics using Python (since jq might not be available)
    python3 << EOF
import json
import sys

try:
    with open('$LATEST_REPORT', 'r') as f:
        data = json.load(f)
    
    structure = data.get('structure', {})
    ratios = structure.get('coverage_ratios', {})
    gaps = data.get('gaps', [])
    
    print(f"📊 Coverage Metrics:")
    print(f"   Overall Coverage: {ratios.get('overall', 0):.1f}%")
    print(f"   CLI Coverage: {ratios.get('area_cli', 0):.1f}%")
    print(f"   Infrastructure Coverage: {ratios.get('layer_infrastructure', 0):.1f}%")
    print(f"   Domain Coverage: {ratios.get('layer_domain', 0):.1f}%")
    
    print(f"\n🚨 Gap Analysis:")
    critical_gaps = [g for g in gaps if g.get('priority') == 'critical']
    high_gaps = [g for g in gaps if g.get('priority') == 'high']
    
    print(f"   Critical Gaps: {len(critical_gaps)}")
    print(f"   High Priority Gaps: {len(high_gaps)}")
    print(f"   Total Gaps: {len(gaps)}")
    
    if critical_gaps:
        print(f"\n❌ Critical Issues:")
        for gap in critical_gaps[:3]:  # Show first 3
            print(f"   - {gap.get('category', 'Unknown').title()}: {gap.get('current_coverage', 0):.1f}% (Target: {gap.get('target_coverage', 0)}%)")
    
except Exception as e:
    print(f"Error reading report: {e}")
    sys.exit(1)
EOF

    print_success "Report analysis completed"
else
    print_warning "No analysis report found"
fi

# Step 4: Display next steps
print_status "Next Steps:"
echo "============"
echo "1. Review generated reports in: reports/"
echo "2. Review generated GitHub issues in: issues/"
echo "3. Create GitHub issues using the GitHub CLI or web interface"
echo "4. Assign issues to team members"
echo "5. Set up milestones for implementation phases"
echo "6. Begin implementation according to the improvement plan"

# Step 5: Display helpful commands
print_status "Helpful Commands:"
echo "=================="
echo "# View latest analysis report:"
echo "cat $LATEST_REPORT | python3 -m json.tool"
echo ""
echo "# Run full analysis with actual test execution:"
echo "python3 $SCRIPT_DIR/automated_test_coverage_analysis.py --project-root . --run-tests"
echo ""
echo "# Create GitHub issues using GitHub CLI (if available):"
echo "cd issues/critical_gaps && for file in *.md; do echo \"Processing \$file...\"; done"

# Check if we're in a CI environment
if [ "$CI" = "true" ] || [ "$GITHUB_ACTIONS" = "true" ]; then
    print_status "Running in CI environment - additional outputs for GitHub Actions"
    
    # Set GitHub Actions outputs if environment variables are available
    if [ -n "$GITHUB_OUTPUT" ] && [ -f "$LATEST_REPORT" ]; then
        python3 << EOF
import json
import os

try:
    with open('$LATEST_REPORT', 'r') as f:
        data = json.load(f)
    
    ratios = data.get('structure', {}).get('coverage_ratios', {})
    gaps = data.get('gaps', [])
    critical_gaps = len([g for g in gaps if g.get('priority') == 'critical'])
    
    with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
        f.write(f"overall_coverage={ratios.get('overall', 0):.1f}\n")
        f.write(f"critical_gaps={critical_gaps}\n")
        f.write(f"cli_coverage={ratios.get('area_cli', 0):.1f}\n")
        f.write(f"infrastructure_coverage={ratios.get('layer_infrastructure', 0):.1f}\n")
    
    print("GitHub Actions outputs set")
except Exception as e:
    print(f"Error setting GitHub outputs: {e}")
EOF
    fi
fi

print_success "Automated analysis and issue creation completed!"
print_status "Check the reports/ and issues/ directories for detailed results"

# Return to original directory
cd "$SCRIPT_DIR"

exit 0