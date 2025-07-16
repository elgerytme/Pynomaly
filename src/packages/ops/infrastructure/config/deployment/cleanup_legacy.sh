#!/bin/bash
# Legacy Configuration Cleanup Script
# Run this after verifying the new configuration structure works properly

set -e

echo "🧹 Legacy Configuration Cleanup Script"
echo "======================================="
echo ""

LEGACY_DIR="/mnt/c/Users/andre/Pynomaly/deployment/config_files/config"
BACKUP_DIR="/mnt/c/Users/andre/Pynomaly/config/backup/legacy_configs"

if [ ! -d "$LEGACY_DIR" ]; then
    echo "❌ Legacy directory not found: $LEGACY_DIR"
    exit 1
fi

echo "📋 This script will:"
echo "  1. Create backup of legacy configurations"
echo "  2. Remove the legacy config directory"
echo "  3. Verify new configuration structure"
echo ""

read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

echo ""
echo "🔄 Creating backup of legacy configurations..."
mkdir -p "$BACKUP_DIR"
cp -r "$LEGACY_DIR"/* "$BACKUP_DIR/" 2>/dev/null || true

echo "✅ Backup created at: $BACKUP_DIR"

echo ""
echo "🔄 Verifying new configuration structure..."

# Check that the new structure exists
NEW_CONFIG_DIR="/mnt/c/Users/andre/Pynomaly/config/deployment"
if [ ! -d "$NEW_CONFIG_DIR" ]; then
    echo "❌ New configuration directory not found: $NEW_CONFIG_DIR"
    echo "❌ Cleanup aborted - new structure not ready"
    exit 1
fi

# Count files in new structure
NEW_FILE_COUNT=$(find "$NEW_CONFIG_DIR" -type f | wc -l)
echo "✅ New configuration structure found with $NEW_FILE_COUNT files"

echo ""
echo "🔄 Removing legacy configuration directory..."
rm -rf "$LEGACY_DIR"

echo "✅ Legacy directory removed: $LEGACY_DIR"

echo ""
echo "🎉 Cleanup completed successfully!"
echo ""
echo "📋 Summary:"
echo "  • Legacy configs backed up to: $BACKUP_DIR"
echo "  • Legacy directory removed: $LEGACY_DIR"
echo "  • New structure active with $NEW_FILE_COUNT files"
echo ""
echo "💡 To rollback (if needed):"
echo "  mkdir -p $LEGACY_DIR"
echo "  cp -r $BACKUP_DIR/* $LEGACY_DIR/"
echo ""
echo "🚀 You can now use the new configuration structure!"
echo "   See: $NEW_CONFIG_DIR/README.deployment.md"