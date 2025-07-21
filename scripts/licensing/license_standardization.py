#!/usr/bin/env python3
"""
License Standardization and Compliance Framework

Implements automated license standardization, compliance checking, and 
legal documentation management across all packages.

Issue: #826 - Standardize Licensing Across Packages
"""

import os
import sys
import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import yaml
import toml
from packaging import version


@dataclass
class LicenseInfo:
    """License information structure"""
    name: str
    spdx_id: str
    text: str
    compatible: bool
    type: str  # permissive, copyleft, proprietary
    requires_attribution: bool = True
    allows_commercial: bool = True
    allows_modification: bool = True
    allows_distribution: bool = True


@dataclass
class PackageLicenseStatus:
    """Package license status"""
    package_path: str
    current_license: Optional[str] = None
    license_file_exists: bool = False
    license_in_pyproject: bool = False
    dependencies_checked: bool = False
    compliant: bool = False
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class LicenseStandardizer:
    """Main license standardization framework"""
    
    def __init__(self):
        self.standard_license = self._get_standard_license()
        self.compatible_licenses = self._get_compatible_licenses()
        self.package_statuses: List[PackageLicenseStatus] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_standard_license(self) -> LicenseInfo:
        """Get the standard license for the project"""
        return LicenseInfo(
            name="MIT License",
            spdx_id="MIT",
            text="""MIT License

Copyright (c) 2024 Anomaly Detection Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""",
            compatible=True,
            type="permissive"
        )
    
    def _get_compatible_licenses(self) -> Dict[str, LicenseInfo]:
        """Get dictionary of compatible licenses"""
        return {
            "MIT": LicenseInfo(
                name="MIT License",
                spdx_id="MIT",
                text="",
                compatible=True,
                type="permissive"
            ),
            "Apache-2.0": LicenseInfo(
                name="Apache License 2.0",
                spdx_id="Apache-2.0",
                text="",
                compatible=True,
                type="permissive"
            ),
            "BSD-3-Clause": LicenseInfo(
                name="BSD 3-Clause License",
                spdx_id="BSD-3-Clause",
                text="",
                compatible=True,
                type="permissive"
            ),
            "BSD-2-Clause": LicenseInfo(
                name="BSD 2-Clause License",
                spdx_id="BSD-2-Clause",
                text="",
                compatible=True,
                type="permissive"
            ),
            "GPL-3.0": LicenseInfo(
                name="GNU General Public License v3.0",
                spdx_id="GPL-3.0",
                text="",
                compatible=False,
                type="copyleft"
            ),
            "AGPL-3.0": LicenseInfo(
                name="GNU Affero General Public License v3.0",
                spdx_id="AGPL-3.0",
                text="",
                compatible=False,
                type="copyleft"
            )
        }
    
    def find_packages(self, root_path: str = "src/packages") -> List[str]:
        """Find all packages in the repository"""
        packages = []
        root = Path(root_path)
        
        if not root.exists():
            self.logger.warning(f"Package root {root_path} does not exist")
            return packages
        
        # Find all directories with pyproject.toml
        for pyproject_file in root.rglob("pyproject.toml"):
            package_dir = pyproject_file.parent
            packages.append(str(package_dir))
        
        # Also check root directory
        if Path("pyproject.toml").exists():
            packages.append(".")
        
        return packages
    
    def check_package_license(self, package_path: str) -> PackageLicenseStatus:
        """Check license status of a package"""
        status = PackageLicenseStatus(package_path=package_path)
        package_dir = Path(package_path)
        
        # Check for license file
        license_files = [
            package_dir / "LICENSE",
            package_dir / "LICENSE.txt",
            package_dir / "LICENSE.md",
            package_dir / "COPYING"
        ]
        
        for license_file in license_files:
            if license_file.exists():
                status.license_file_exists = True
                status.current_license = self._detect_license_from_file(license_file)
                break
        
        # Check pyproject.toml
        pyproject_file = package_dir / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r') as f:
                    pyproject_data = toml.load(f)
                
                license_info = pyproject_data.get("project", {}).get("license", {})
                if license_info:
                    status.license_in_pyproject = True
                    if "text" in license_info:
                        status.current_license = license_info["text"]
                    elif "file" in license_info:
                        license_file_path = package_dir / license_info["file"]
                        if license_file_path.exists():
                            status.current_license = self._detect_license_from_file(license_file_path)
            except Exception as e:
                status.issues.append(f"Error reading pyproject.toml: {e}")
        
        # Check compliance
        if status.current_license:
            if status.current_license in self.compatible_licenses:
                license_info = self.compatible_licenses[status.current_license]
                status.compliant = license_info.compatible
                if not license_info.compatible:
                    status.issues.append(f"License {status.current_license} is not compatible")
            else:
                status.issues.append(f"Unknown license: {status.current_license}")
        else:
            status.issues.append("No license found")
        
        return status
    
    def _detect_license_from_file(self, license_file: Path) -> str:
        """Detect license type from file content"""
        try:
            with open(license_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            # Simple license detection
            if "mit license" in content or "mit" in content:
                return "MIT"
            elif "apache license" in content or "apache" in content:
                return "Apache-2.0"
            elif "bsd" in content:
                if "3-clause" in content:
                    return "BSD-3-Clause"
                else:
                    return "BSD-2-Clause"
            elif "gpl" in content and "v3" in content:
                return "GPL-3.0"
            elif "agpl" in content:
                return "AGPL-3.0"
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
    def standardize_package_license(self, package_path: str) -> bool:
        """Standardize license for a package"""
        try:
            package_dir = Path(package_path)
            
            # Create/update LICENSE file
            license_file = package_dir / "LICENSE"
            with open(license_file, 'w') as f:
                f.write(self.standard_license.text)
            
            # Update pyproject.toml
            pyproject_file = package_dir / "pyproject.toml"
            if pyproject_file.exists():
                try:
                    with open(pyproject_file, 'r') as f:
                        pyproject_data = toml.load(f)
                    
                    # Update license information
                    if "project" not in pyproject_data:
                        pyproject_data["project"] = {}
                    
                    pyproject_data["project"]["license"] = {"text": self.standard_license.spdx_id}
                    
                    # Add license classifier
                    if "classifiers" not in pyproject_data["project"]:
                        pyproject_data["project"]["classifiers"] = []
                    
                    # Remove existing license classifiers
                    pyproject_data["project"]["classifiers"] = [
                        c for c in pyproject_data["project"]["classifiers"]
                        if not c.startswith("License ::")
                    ]
                    
                    # Add standard license classifier
                    pyproject_data["project"]["classifiers"].append(
                        "License :: OSI Approved :: MIT License"
                    )
                    
                    with open(pyproject_file, 'w') as f:
                        toml.dump(pyproject_data, f)
                
                except Exception as e:
                    self.logger.error(f"Error updating pyproject.toml in {package_path}: {e}")
                    return False
            
            self.logger.info(f"✅ Standardized license for {package_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error standardizing license for {package_path}: {e}")
            return False
    
    def check_dependency_licenses(self, package_path: str) -> Dict[str, Any]:
        """Check licenses of dependencies"""
        dependency_licenses = {}
        package_dir = Path(package_path)
        
        # Check pyproject.toml dependencies
        pyproject_file = package_dir / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r') as f:
                    pyproject_data = toml.load(f)
                
                dependencies = pyproject_data.get("project", {}).get("dependencies", [])
                
                for dep in dependencies:
                    # Extract package name
                    dep_name = re.split(r'[>=<~!]', dep)[0].strip()
                    
                    # Mock license check (in real implementation, would query PyPI)
                    if dep_name in ["pydantic", "fastapi", "requests"]:
                        dependency_licenses[dep_name] = "MIT"
                    elif dep_name in ["numpy", "pandas", "scikit-learn"]:
                        dependency_licenses[dep_name] = "BSD-3-Clause"
                    else:
                        dependency_licenses[dep_name] = "Unknown"
                
            except Exception as e:
                self.logger.error(f"Error checking dependencies in {package_path}: {e}")
        
        return dependency_licenses
    
    def generate_license_report(self, output_file: str = "license_report.html"):
        """Generate comprehensive license report"""
        
        # Calculate summary statistics
        total_packages = len(self.package_statuses)
        compliant_packages = sum(1 for status in self.package_statuses if status.compliant)
        non_compliant_packages = total_packages - compliant_packages
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>License Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .summary-item {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 5px; }}
                .compliant {{ background: #28a745; color: white; }}
                .non-compliant {{ background: #dc3545; color: white; }}
                .package-status {{ margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid; }}
                .package-status.compliant {{ background: #d1ecf1; border-left-color: #28a745; }}
                .package-status.non-compliant {{ background: #f8d7da; border-left-color: #dc3545; }}
                .issue {{ margin: 5px 0; padding: 5px; background: #fff3cd; border-radius: 3px; }}
                .dependency-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .dependency-table th, .dependency-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .dependency-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>License Compliance Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="summary-item">
                    <h3>{total_packages}</h3>
                    <p>Total Packages</p>
                </div>
                <div class="summary-item compliant">
                    <h3>{compliant_packages}</h3>
                    <p>Compliant</p>
                </div>
                <div class="summary-item non-compliant">
                    <h3>{non_compliant_packages}</h3>
                    <p>Non-Compliant</p>
                </div>
            </div>
            
            <h2>Standard License</h2>
            <div class="package-status compliant">
                <h3>{self.standard_license.name} ({self.standard_license.spdx_id})</h3>
                <p><strong>Type:</strong> {self.standard_license.type}</p>
                <p><strong>Compatible:</strong> {self.standard_license.compatible}</p>
                <pre>{self.standard_license.text[:200]}...</pre>
            </div>
            
            <h2>Package License Status</h2>
        """
        
        for status in self.package_statuses:
            status_class = "compliant" if status.compliant else "non-compliant"
            status_text = "✅ COMPLIANT" if status.compliant else "❌ NON-COMPLIANT"
            
            html_content += f"""
            <div class="package-status {status_class}">
                <h3>{status.package_path} - {status_text}</h3>
                <p><strong>Current License:</strong> {status.current_license or 'None'}</p>
                <p><strong>License File:</strong> {'✅' if status.license_file_exists else '❌'}</p>
                <p><strong>PyProject License:</strong> {'✅' if status.license_in_pyproject else '❌'}</p>
                
                {f'<h4>Issues:</h4>' if status.issues else ''}
                {''.join(f'<div class="issue">{issue}</div>' for issue in status.issues)}
            </div>
            """
        
        html_content += """
            <h2>Compatible Licenses</h2>
            <table class="dependency-table">
                <thead>
                    <tr>
                        <th>License</th>
                        <th>SPDX ID</th>
                        <th>Type</th>
                        <th>Compatible</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for license_id, license_info in self.compatible_licenses.items():
            compatible_text = "✅ Yes" if license_info.compatible else "❌ No"
            html_content += f"""
                <tr>
                    <td>{license_info.name}</td>
                    <td>{license_info.spdx_id}</td>
                    <td>{license_info.type}</td>
                    <td>{compatible_text}</td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"License report generated: {output_file}")
    
    def run_standardization(self, packages: List[str] = None) -> bool:
        """Run complete license standardization process"""
        if packages is None:
            packages = self.find_packages()
        
        self.logger.info(f"Running license standardization on {len(packages)} packages")
        
        success = True
        
        # Check current license status
        for package_path in packages:
            status = self.check_package_license(package_path)
            self.package_statuses.append(status)
            
            if status.compliant:
                self.logger.info(f"✅ {package_path} - License compliant")
            else:
                self.logger.warning(f"❌ {package_path} - License non-compliant: {status.issues}")
        
        # Standardize licenses
        for package_path in packages:
            if not self.standardize_package_license(package_path):
                success = False
        
        # Re-check after standardization
        self.package_statuses = []
        for package_path in packages:
            status = self.check_package_license(package_path)
            self.package_statuses.append(status)
        
        # Generate report
        self.generate_license_report()
        
        # Check final compliance
        non_compliant = [s for s in self.package_statuses if not s.compliant]
        if non_compliant:
            self.logger.error(f"License standardization failed for {len(non_compliant)} packages")
            success = False
        else:
            self.logger.info("✅ All packages are now license compliant")
        
        return success
    
    def run_ci_license_check(self, packages: List[str] = None) -> bool:
        """Run license compliance check in CI environment"""
        if packages is None:
            packages = self.find_packages()
        
        self.logger.info("Running CI license compliance check")
        
        # Check all packages
        for package_path in packages:
            status = self.check_package_license(package_path)
            self.package_statuses.append(status)
        
        # Generate report
        self.generate_license_report()
        
        # Check compliance
        non_compliant = [s for s in self.package_statuses if not s.compliant]
        if non_compliant:
            self.logger.error("License compliance check failed:")
            for status in non_compliant:
                self.logger.error(f"  - {status.package_path}: {status.issues}")
            return False
        
        self.logger.info("License compliance check passed")
        return True


def main():
    """Main entry point for license standardization"""
    if len(sys.argv) > 1 and sys.argv[1] == "ci":
        # Run CI license check
        standardizer = LicenseStandardizer()
        success = standardizer.run_ci_license_check()
        sys.exit(0 if success else 1)
    else:
        # Run full standardization
        standardizer = LicenseStandardizer()
        success = standardizer.run_standardization()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()