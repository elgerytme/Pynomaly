#!/bin/bash

# Documentation verification script
# This script verifies that the documentation can be built and served properly

set -e

echo "🔍 Verifying documentation setup..."

# Check if hatch is available
if ! command -v hatch &> /dev/null; then
    echo "❌ Hatch is not installed or not in PATH"
    exit 1
fi

echo "✅ Hatch is available"

# Check if mkdocs.yml exists
if [ ! -f "mkdocs.yml" ]; then
    echo "⚠️  mkdocs.yml not found in root, checking config/docs/mkdocs.yml..."
    if [ -f "config/docs/mkdocs.yml" ]; then
        echo "📋 Copying mkdocs.yml from config/docs/"
        cp config/docs/mkdocs.yml mkdocs.yml
    else
        echo "❌ mkdocs.yml not found in either location"
        exit 1
    fi
fi

echo "✅ mkdocs.yml is available"

# Verify docs environment exists
echo "🔧 Checking docs environment..."
if ! hatch env find docs &> /dev/null; then
    echo "📦 Creating docs environment..."
    hatch env create docs
fi

echo "✅ Docs environment is ready"

# Test documentation build
echo "🔨 Testing documentation build..."
if hatch run docs:build; then
    echo "✅ Documentation build successful"
else
    echo "❌ Documentation build failed"
    exit 1
fi

# Check if site directory exists and contains files
if [ -d "site" ] && [ "$(ls -A site)" ]; then
    echo "✅ Site directory created with content"
    echo "📊 Generated files:"
    find site -name "*.html" | head -10
else
    echo "❌ Site directory is empty or doesn't exist"
    exit 1
fi

# Check for key files
required_files=("site/index.html" "site/sitemap.xml")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

echo ""
echo "🎉 Documentation verification complete!"
echo ""
echo "📖 To serve the documentation locally, run:"
echo "   hatch run docs:serve"
echo ""
echo "🚀 The GitHub Pages deployment will automatically trigger on push to main branch"
echo "   when changes are made to docs/, mkdocs.yml, or .github/workflows/deploy-docs.yml"
