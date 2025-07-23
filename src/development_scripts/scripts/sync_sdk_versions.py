#!/usr/bin/env python3
"""
SDK Version Synchronization Script

This script ensures all SDKs (Python, TypeScript, Java) maintain synchronized
version numbers for consistent releases.

Usage:
    python3 sync_sdk_versions.py --version 1.2.3
    python3 sync_sdk_versions.py --check
    python3 sync_sdk_versions.py --sync-from-git
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import xml.etree.ElementTree as ET


class SDKVersionManager:
    """Manages version synchronization across all SDKs."""
    
    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).parent.parent.parent.parent.resolve()
        self.sdk_paths = {
            'python': self.repo_root / 'src' / 'packages' / 'data' / 'anomaly_detection',
            'typescript': self.repo_root / 'src' / 'templates' / 'client_sdks' / 'typescript',
            'java': self.repo_root / 'src' / 'templates' / 'client_sdks' / 'java'
        }
        
        # Version files
        self.version_files = {
            'python': self.sdk_paths['python'] / 'pyproject.toml',
            'typescript': self.sdk_paths['typescript'] / 'package.json',
            'java': self.sdk_paths['java'] / 'pom.xml'
        }
    
    def get_git_version(self) -> Optional[str]:
        """Get the latest git tag version."""
        try:
            result = subprocess.run(
                ['git', 'describe', '--tags', '--abbrev=0'],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                # Remove 'v' prefix if present
                if version.startswith('v'):
                    version = version[1:]
                return version
            return None
        except Exception as e:
            print(f"❌ Error getting git version: {e}")
            return None
    
    def get_python_version(self) -> Optional[str]:
        """Get current Python package version."""
        pyproject_file = self.version_files['python']
        
        if not pyproject_file.exists():
            print(f"❌ Python pyproject.toml not found: {pyproject_file}")
            return None
        
        try:
            with open(pyproject_file, 'r') as f:
                content = f.read()
            
            # Look for version in [project] section
            version_match = re.search(r'version\\s*=\\s*["\']([^"\']+)["\']', content)
            if version_match:
                return version_match.group(1)
            
            print("❌ Version not found in pyproject.toml")
            return None
        except Exception as e:
            print(f"❌ Error reading Python version: {e}")
            return None
    
    def get_typescript_version(self) -> Optional[str]:
        """Get current TypeScript package version."""
        package_json = self.version_files['typescript']
        
        if not package_json.exists():
            print(f"❌ TypeScript package.json not found: {package_json}")
            return None
        
        try:
            with open(package_json, 'r') as f:
                data = json.load(f)
            
            return data.get('version')
        except Exception as e:
            print(f"❌ Error reading TypeScript version: {e}")
            return None
    
    def get_java_version(self) -> Optional[str]:
        """Get current Java package version."""
        pom_xml = self.version_files['java']
        
        if not pom_xml.exists():
            print(f"❌ Java pom.xml not found: {pom_xml}")
            return None
        
        try:
            tree = ET.parse(pom_xml)
            root = tree.getroot()
            
            # Handle namespace
            ns = {'maven': 'http://maven.apache.org/POM/4.0.0'}
            version_elem = root.find('.//maven:version', ns) or root.find('.//version')
            
            if version_elem is not None:
                return version_elem.text
            
            print("❌ Version not found in pom.xml")
            return None
        except Exception as e:
            print(f"❌ Error reading Java version: {e}")
            return None
    
    def get_all_versions(self) -> Dict[str, Optional[str]]:
        """Get versions from all SDKs."""
        return {
            'git': self.get_git_version(),
            'python': self.get_python_version(),
            'typescript': self.get_typescript_version(),
            'java': self.get_java_version()
        }
    
    def check_version_sync(self) -> Tuple[bool, Dict[str, Optional[str]]]:
        """Check if all SDK versions are synchronized."""
        versions = self.get_all_versions()
        
        # Get non-None versions
        active_versions = {k: v for k, v in versions.items() if v is not None}
        
        if not active_versions:
            print("❌ No versions found in any SDK")
            return False, versions
        
        # Check if all versions match
        unique_versions = set(active_versions.values())
        is_synced = len(unique_versions) == 1
        
        return is_synced, versions
    
    def set_python_version(self, version: str) -> bool:
        """Set Python package version."""
        pyproject_file = self.version_files['python']
        
        if not pyproject_file.exists():
            print(f"❌ Python pyproject.toml not found: {pyproject_file}")
            return False
        
        try:
            with open(pyproject_file, 'r') as f:
                content = f.read()
            
            # Replace version in [project] section
            new_content = re.sub(
                r'(version\\s*=\\s*["\'])[^"\']+(["\'])',
                f'\\g<1>{version}\\g<2>',
                content
            )
            
            if new_content == content:
                print("❌ Could not find version field to update in pyproject.toml")
                return False
            
            with open(pyproject_file, 'w') as f:
                f.write(new_content)
            
            print(f"✅ Updated Python version to {version}")
            return True
        except Exception as e:
            print(f"❌ Error setting Python version: {e}")
            return False
    
    def set_typescript_version(self, version: str) -> bool:
        """Set TypeScript package version."""
        package_json = self.version_files['typescript']
        
        if not package_json.exists():
            print(f"❌ TypeScript package.json not found: {package_json}")
            return False
        
        try:
            with open(package_json, 'r') as f:
                data = json.load(f)
            
            data['version'] = version
            
            with open(package_json, 'w') as f:
                json.dump(data, f, indent=2)
                f.write('\\n')  # Add trailing newline
            
            print(f"✅ Updated TypeScript version to {version}")
            return True
        except Exception as e:
            print(f"❌ Error setting TypeScript version: {e}")
            return False
    
    def set_java_version(self, version: str) -> bool:
        """Set Java package version."""
        pom_xml = self.version_files['java']
        
        if not pom_xml.exists():
            print(f"❌ Java pom.xml not found: {pom_xml}")
            return False
        
        try:
            # Use Maven versions plugin for reliable version update
            result = subprocess.run([
                'mvn', 'versions:set', f'-DnewVersion={version}', 
                '-DgenerateBackupPoms=false'
            ], cwd=self.sdk_paths['java'], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Updated Java version to {version}")
                return True
            else:
                print(f"❌ Maven versions:set failed: {result.stderr}")
                
                # Fallback to manual XML editing
                return self._set_java_version_xml(version)
        except FileNotFoundError:
            print("❌ Maven not found, falling back to XML editing")
            return self._set_java_version_xml(version)
        except Exception as e:
            print(f"❌ Error setting Java version: {e}")
            return False
    
    def _set_java_version_xml(self, version: str) -> bool:
        """Fallback method to set Java version by editing XML directly."""
        pom_xml = self.version_files['java']
        
        try:
            tree = ET.parse(pom_xml)
            root = tree.getroot()
            
            # Handle namespace
            ns = {'maven': 'http://maven.apache.org/POM/4.0.0'}
            version_elem = root.find('.//maven:version', ns) or root.find('.//version')
            
            if version_elem is not None:
                version_elem.text = version
                tree.write(pom_xml, encoding='utf-8', xml_declaration=True)
                print(f"✅ Updated Java version to {version} (XML fallback)")
                return True
            else:
                print("❌ Could not find version element in pom.xml")
                return False
        except Exception as e:
            print(f"❌ Error in XML fallback: {e}")
            return False
    
    def sync_versions(self, target_version: str) -> bool:
        """Synchronize all SDK versions to the target version."""
        print(f"🔄 Synchronizing all SDKs to version {target_version}\\n")
        
        success = True
        
        # Update each SDK
        if not self.set_python_version(target_version):
            success = False
        
        if not self.set_typescript_version(target_version):
            success = False
        
        if not self.set_java_version(target_version):
            success = False
        
        if success:
            print(f"\\n✅ All SDKs synchronized to version {target_version}")
        else:
            print(f"\\n❌ Failed to synchronize some SDKs to version {target_version}")
        
        return success
    
    def display_version_status(self):
        """Display current version status of all SDKs."""
        print("📦 SDK Version Status\\n")
        
        is_synced, versions = self.check_version_sync()
        
        # Display versions
        for sdk, version in versions.items():
            if version:
                status = "✅" if is_synced else "⚠️ "
                print(f"  {status} {sdk.capitalize():<12} {version}")
            else:
                print(f"  ❌ {sdk.capitalize():<12} Not found")
        
        print()
        
        if is_synced and any(versions.values()):
            print("✅ All SDKs are synchronized!")
        elif not any(versions.values()):
            print("❌ No versions found!")
        else:
            print("⚠️  SDKs are NOT synchronized!")
            
            # Show recommended action
            sdk_versions = [v for k, v in versions.items() if k != 'git' and v]
            if sdk_versions:
                most_recent = max(sdk_versions)  # Simple string comparison
                print(f"💡 Recommended: Sync all to {most_recent}")
    
    def validate_version_format(self, version: str) -> bool:
        """Validate version follows semantic versioning."""
        # Semantic versioning pattern
        pattern = r'^\\d+\\.\\d+\\.\\d+(?:-[a-zA-Z0-9]+(?:\\.[a-zA-Z0-9]+)*)?(?:\\+[a-zA-Z0-9]+(?:\\.[a-zA-Z0-9]+)*)?$'
        
        if re.match(pattern, version):
            return True
        else:
            print(f"❌ Invalid version format: {version}")
            print("Expected format: X.Y.Z (e.g., 1.2.3, 1.0.0-beta, 1.0.0+build.1)")
            return False


def main():
    parser = argparse.ArgumentParser(description='Synchronize SDK versions')
    parser.add_argument('--version', type=str, 
                       help='Set all SDKs to this version')
    parser.add_argument('--check', action='store_true', 
                       help='Check current version status')
    parser.add_argument('--sync-from-git', action='store_true', 
                       help='Sync all SDKs to the latest git tag version')
    parser.add_argument('--repo-root', type=Path, 
                       help='Path to repository root (auto-detected by default)')
    
    args = parser.parse_args()
    
    manager = SDKVersionManager(repo_root=args.repo_root)
    
    if args.check:
        manager.display_version_status()
    
    elif args.sync_from_git:
        git_version = manager.get_git_version()
        if git_version:
            if manager.validate_version_format(git_version):
                manager.sync_versions(git_version)
            else:
                sys.exit(1)
        else:
            print("❌ No git tag version found")
            sys.exit(1)
    
    elif args.version:
        if manager.validate_version_format(args.version):
            manager.sync_versions(args.version)
        else:
            sys.exit(1)
    
    else:
        # Default: show status
        manager.display_version_status()
        
        print("\\n🚀 Usage:")
        print("  python3 sync_sdk_versions.py --check                  # Check status")
        print("  python3 sync_sdk_versions.py --version 1.2.3          # Set version")
        print("  python3 sync_sdk_versions.py --sync-from-git          # Sync to git tag")


if __name__ == '__main__':
    main()