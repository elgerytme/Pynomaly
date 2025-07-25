#!/usr/bin/env python3
"""
Package Registry and Discovery System

Central registry for discovering, cataloging, and managing domain packages
within the monorepo ecosystem.
"""

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import subprocess
import requests
import yaml
import hashlib
import semantic_version
from enum import Enum

class PackageStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"

class PackageType(Enum):
    DOMAIN = "domain"
    SHARED = "shared"
    INFRASTRUCTURE = "infrastructure"
    INTEGRATION = "integration"

@dataclass
class PackageMetadata:
    """Complete metadata for a domain package."""
    name: str
    version: str
    description: str
    package_type: PackageType
    status: PackageStatus
    domain: str
    bounded_context: str
    
    # Technical metadata
    path: str
    languages: List[str]
    frameworks: List[str]
    dependencies: List[str]
    api_endpoints: List[str]
    events_published: List[str]
    events_consumed: List[str]
    
    # Quality metrics
    test_coverage: float
    complexity_score: int
    security_score: int
    performance_score: int
    independence_score: int
    
    # Documentation
    readme_path: Optional[str] = None
    documentation_url: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    
    # Maintainership
    maintainers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_commit_hash: Optional[str] = None
    
    # Usage statistics
    download_count: int = 0
    usage_count: int = 0
    rating: float = 0.0
    
    # Compatibility
    min_platform_version: Optional[str] = None
    max_platform_version: Optional[str] = None
    compatible_packages: List[str] = field(default_factory=list)
    conflicting_packages: List[str] = field(default_factory=list)

@dataclass
class PackageSearchCriteria:
    """Search criteria for package discovery."""
    query: Optional[str] = None
    domain: Optional[str] = None
    package_type: Optional[PackageType] = None
    status: Optional[PackageStatus] = None
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    min_coverage: Optional[float] = None
    min_security_score: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    sort_by: str = "popularity"  # popularity, updated, created, rating
    limit: int = 50

@dataclass
class PackageRating:
    """Package rating and review."""
    package_name: str
    user_id: str
    rating: int  # 1-5
    review: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

class PackageAnalyzer:
    """Analyzes packages to extract metadata."""
    
    def __init__(self, monorepo_root: str):
        self.monorepo_root = Path(monorepo_root)
    
    def analyze_package(self, package_path: str) -> PackageMetadata:
        """Analyze a package and extract comprehensive metadata."""
        package_path = Path(package_path)
        package_name = package_path.name
        
        # Read package configuration
        config = self._read_package_config(package_path)
        
        # Extract basic metadata
        metadata = PackageMetadata(
            name=package_name,
            version=config.get('version', '0.1.0'),
            description=config.get('description', ''),
            package_type=PackageType(config.get('type', 'domain')),
            status=PackageStatus(config.get('status', 'active')),
            domain=config.get('domain', 'unknown'),
            bounded_context=config.get('bounded_context', package_name),
            path=str(package_path),
            languages=self._detect_languages(package_path),
            frameworks=self._detect_frameworks(package_path),
            dependencies=self._extract_dependencies(package_path),
            api_endpoints=self._extract_api_endpoints(package_path),
            events_published=config.get('events_published', []),
            events_consumed=config.get('events_consumed', []),
            maintainers=config.get('maintainers', []),
            last_commit_hash=self._get_last_commit_hash(package_path)
        )
        
        # Calculate quality metrics
        metadata.test_coverage = self._calculate_test_coverage(package_path)
        metadata.complexity_score = self._calculate_complexity(package_path)
        metadata.security_score = self._calculate_security_score(package_path)
        metadata.performance_score = self._calculate_performance_score(package_path)
        metadata.independence_score = self._calculate_independence_score(package_path)
        
        # Extract documentation
        readme_path = package_path / "README.md"
        if readme_path.exists():
            metadata.readme_path = str(readme_path)
        
        metadata.documentation_url = config.get('documentation_url')
        metadata.examples = self._find_examples(package_path)
        
        return metadata
    
    def _read_package_config(self, package_path: Path) -> Dict[str, Any]:
        """Read package configuration file."""
        config_files = [
            package_path / "package.json",
            package_path / "pyproject.toml",
            package_path / "Cargo.toml",
            package_path / "package.yml",
            package_path / "package.yaml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                if config_file.suffix == '.json':
                    with open(config_file) as f:
                        return json.load(f)
                elif config_file.suffix in ['.yml', '.yaml']:
                    with open(config_file) as f:
                        return yaml.safe_load(f)
                elif config_file.suffix == '.toml':
                    try:
                        import toml
                        with open(config_file) as f:
                            return toml.load(f)
                    except ImportError:
                        pass
        
        return {}
    
    def _detect_languages(self, package_path: Path) -> List[str]:
        """Detect programming languages used in the package."""
        languages = set()
        
        file_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.kt': 'kotlin',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php'
        }
        
        for ext, lang in file_extensions.items():
            if list(package_path.rglob(f"*{ext}")):
                languages.add(lang)
        
        return list(languages)
    
    def _detect_frameworks(self, package_path: Path) -> List[str]:
        """Detect frameworks used in the package."""
        frameworks = set()
        
        # Check Python frameworks
        if (package_path / "requirements.txt").exists():
            with open(package_path / "requirements.txt") as f:
                deps = f.read().lower()
                if 'fastapi' in deps:
                    frameworks.add('fastapi')
                if 'django' in deps:
                    frameworks.add('django')
                if 'flask' in deps:
                    frameworks.add('flask')
                if 'sqlalchemy' in deps:
                    frameworks.add('sqlalchemy')
        
        # Check JavaScript frameworks
        if (package_path / "package.json").exists():
            with open(package_path / "package.json") as f:
                package_json = json.load(f)
                deps = {**package_json.get('dependencies', {}), **package_json.get('devDependencies', {})}
                
                if 'react' in deps:
                    frameworks.add('react')
                if 'vue' in deps:
                    frameworks.add('vue')
                if 'angular' in deps:
                    frameworks.add('angular')
                if 'express' in deps:
                    frameworks.add('express')
                if 'nestjs' in deps:
                    frameworks.add('nestjs')
        
        return list(frameworks)
    
    def _extract_dependencies(self, package_path: Path) -> List[str]:
        """Extract package dependencies."""
        dependencies = []
        
        # Python dependencies
        requirements_files = [
            package_path / "requirements.txt",
            package_path / "pyproject.toml"
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                if req_file.name == "requirements.txt":
                    with open(req_file) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                dep = line.split('==')[0].split('>=')[0].split('<=')[0]
                                dependencies.append(dep.strip())
        
        # JavaScript dependencies
        package_json = package_path / "package.json"
        if package_json.exists():
            with open(package_json) as f:
                data = json.load(f)
                deps = data.get('dependencies', {})
                dependencies.extend(deps.keys())
        
        return dependencies
    
    def _extract_api_endpoints(self, package_path: Path) -> List[str]:
        """Extract API endpoints from the package."""
        endpoints = []
        
        # Look for FastAPI endpoints
        for py_file in package_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple regex to find FastAPI routes
                import re
                patterns = [
                    r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                    r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for method, path in matches:
                        endpoints.append(f"{method.upper()} {path}")
                        
            except (UnicodeDecodeError, IOError):
                continue
        
        return endpoints
    
    def _calculate_test_coverage(self, package_path: Path) -> float:
        """Calculate test coverage percentage."""
        try:
            # Count test files vs source files
            src_files = len(list(package_path.rglob("src/**/*.py")))
            test_files = len(list(package_path.rglob("tests/**/*.py")))
            
            if src_files == 0:
                return 0.0
            
            # Simple approximation - in practice, would run coverage tools
            return min(100.0, (test_files / src_files) * 80)
            
        except Exception:
            return 0.0
    
    def _calculate_complexity(self, package_path: Path) -> int:
        """Calculate complexity score (lower is better)."""
        try:
            # Simple line count-based complexity
            total_lines = 0
            for py_file in package_path.rglob("src/**/*.py"):
                try:
                    with open(py_file) as f:
                        total_lines += len(f.readlines())
                except Exception:
                    continue
            
            # Normalize to 1-100 scale (inverse)
            return max(1, min(100, 100 - (total_lines // 100)))
            
        except Exception:
            return 50
    
    def _calculate_security_score(self, package_path: Path) -> int:
        """Calculate security score."""
        score = 100
        
        # Check for common security issues
        security_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_?key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']'
        ]
        
        for py_file in package_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                import re
                for pattern in security_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        score -= 20  # Deduct points for security issues
                        
            except Exception:
                continue
        
        return max(0, score)
    
    def _calculate_performance_score(self, package_path: Path) -> int:
        """Calculate performance score."""
        # Placeholder - would integrate with actual performance testing
        return 85
    
    def _calculate_independence_score(self, package_path: Path) -> int:
        """Calculate package independence score."""
        score = 100
        
        # Check for cross-package imports
        for py_file in package_path.rglob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for imports from other packages
                import re
                cross_package_imports = re.findall(
                    r'from\s+(?!\.|\w+\.application|\w+\.domain|\w+\.infrastructure)(\w+)',
                    content
                )
                
                # Deduct points for each cross-package dependency
                score -= len(cross_package_imports) * 10
                
            except Exception:
                continue
        
        return max(0, score)
    
    def _get_last_commit_hash(self, package_path: Path) -> Optional[str]:
        """Get last commit hash for the package."""
        try:
            result = subprocess.run([
                'git', 'log', '-1', '--format=%H', '--', str(package_path)
            ], capture_output=True, text=True, cwd=self.monorepo_root)
            
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return None
    
    def _find_examples(self, package_path: Path) -> List[str]:
        """Find example files in the package."""
        examples = []
        
        example_dirs = [
            package_path / "examples",
            package_path / "samples",
            package_path / "demos"
        ]
        
        for example_dir in example_dirs:
            if example_dir.exists():
                for example_file in example_dir.rglob("*"):
                    if example_file.is_file():
                        examples.append(str(example_file.relative_to(package_path)))
        
        return examples

class PackageRegistry:
    """Central registry for package discovery and management."""
    
    def __init__(self, db_path: str = "package_registry.db"):
        self.db_path = db_path
        self.analyzer = None
        self._init_database()
    
    def _init_database(self):
        """Initialize the package registry database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Packages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS packages (
                name TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                description TEXT,
                package_type TEXT,
                status TEXT,
                domain TEXT,
                bounded_context TEXT,
                path TEXT,
                languages TEXT,
                frameworks TEXT,
                dependencies TEXT,
                api_endpoints TEXT,
                events_published TEXT,
                events_consumed TEXT,
                test_coverage REAL,
                complexity_score INTEGER,
                security_score INTEGER,
                performance_score INTEGER,
                independence_score INTEGER,
                readme_path TEXT,
                documentation_url TEXT,
                examples TEXT,
                maintainers TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                last_commit_hash TEXT,
                download_count INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                min_platform_version TEXT,
                max_platform_version TEXT,
                compatible_packages TEXT,
                conflicting_packages TEXT
            )
        ''')
        
        # Package ratings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS package_ratings (
                id TEXT PRIMARY KEY,
                package_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                review TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (package_name) REFERENCES packages (name)
            )
        ''')
        
        # Package dependencies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS package_dependencies (
                id TEXT PRIMARY KEY,
                package_name TEXT NOT NULL,
                dependency_name TEXT NOT NULL,
                dependency_type TEXT,
                version_constraint TEXT,
                FOREIGN KEY (package_name) REFERENCES packages (name)
            )
        ''')
        
        # Usage analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS package_usage (
                id TEXT PRIMARY KEY,
                package_name TEXT NOT NULL,
                user_id TEXT,
                action TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (package_name) REFERENCES packages (name)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_domain ON packages (domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_type ON packages (package_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_status ON packages (status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_rating ON packages (rating)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_updated ON packages (updated_at)')
        
        conn.commit()
        conn.close()
    
    def register_package(self, package_metadata: PackageMetadata) -> bool:
        """Register a new package or update existing one."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert metadata to database format
            data = self._metadata_to_db_format(package_metadata)
            
            # Insert or update package
            cursor.execute('''
                INSERT OR REPLACE INTO packages (
                    name, version, description, package_type, status, domain,
                    bounded_context, path, languages, frameworks, dependencies,
                    api_endpoints, events_published, events_consumed,
                    test_coverage, complexity_score, security_score,
                    performance_score, independence_score, readme_path,
                    documentation_url, examples, maintainers, created_at,
                    updated_at, last_commit_hash, download_count, usage_count,
                    rating, min_platform_version, max_platform_version,
                    compatible_packages, conflicting_packages
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error registering package {package_metadata.name}: {e}")
            return False
    
    def search_packages(self, criteria: PackageSearchCriteria) -> List[PackageMetadata]:
        """Search packages based on criteria."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM packages WHERE 1=1"
        params = []
        
        if criteria.query:
            query += " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{criteria.query}%", f"%{criteria.query}%"])
        
        if criteria.domain:
            query += " AND domain = ?"
            params.append(criteria.domain)
        
        if criteria.package_type:
            query += " AND package_type = ?"
            params.append(criteria.package_type.value)
        
        if criteria.status:
            query += " AND status = ?"
            params.append(criteria.status.value)
        
        if criteria.min_coverage:
            query += " AND test_coverage >= ?"
            params.append(criteria.min_coverage)
        
        if criteria.min_security_score:
            query += " AND security_score >= ?"
            params.append(criteria.min_security_score)
        
        # Add sorting
        sort_columns = {
            'popularity': 'usage_count DESC, rating DESC',
            'updated': 'updated_at DESC',
            'created': 'created_at DESC',
            'rating': 'rating DESC',
            'name': 'name ASC'
        }
        
        order_by = sort_columns.get(criteria.sort_by, 'rating DESC')
        query += f" ORDER BY {order_by}"
        
        if criteria.limit:
            query += " LIMIT ?"
            params.append(criteria.limit)
        
        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to metadata objects
        packages = []
        for row in rows:
            metadata = self._db_row_to_metadata(row)
            packages.append(metadata)
        
        return packages
    
    def get_package(self, name: str) -> Optional[PackageMetadata]:
        """Get package by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM packages WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._db_row_to_metadata(row)
        return None
    
    def rate_package(self, rating: PackageRating) -> bool:
        """Add or update package rating."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert rating
            rating_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT OR REPLACE INTO package_ratings 
                (id, package_name, user_id, rating, review, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (rating_id, rating.package_name, rating.user_id, 
                  rating.rating, rating.review, rating.created_at))
            
            # Update package average rating
            cursor.execute('''
                UPDATE packages 
                SET rating = (
                    SELECT AVG(rating) FROM package_ratings 
                    WHERE package_name = ?
                )
                WHERE name = ?
            ''', (rating.package_name, rating.package_name))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error rating package: {e}")
            return False
    
    def record_usage(self, package_name: str, user_id: str, action: str, metadata: Dict[str, Any] = None):
        """Record package usage analytics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            usage_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO package_usage (id, package_name, user_id, action, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (usage_id, package_name, user_id, action, json.dumps(metadata or {})))
            
            # Update usage count
            if action in ['download', 'install', 'clone']:
                cursor.execute('''
                    UPDATE packages 
                    SET usage_count = usage_count + 1,
                        download_count = download_count + 1
                    WHERE name = ?
                ''', (package_name,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error recording usage: {e}")
    
    def get_package_statistics(self, package_name: str) -> Dict[str, Any]:
        """Get comprehensive package statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute('''
            SELECT download_count, usage_count, rating,
                   (SELECT COUNT(*) FROM package_ratings WHERE package_name = ?) as review_count
            FROM packages WHERE name = ?
        ''', (package_name, package_name))
        
        row = cursor.fetchone()
        if not row:
            return {}
        
        stats = {
            'download_count': row[0],
            'usage_count': row[1],
            'rating': row[2],
            'review_count': row[3]
        }
        
        # Usage analytics
        cursor.execute('''
            SELECT action, COUNT(*) as count 
            FROM package_usage 
            WHERE package_name = ? 
            GROUP BY action
        ''', (package_name,))
        
        usage_by_action = dict(cursor.fetchall())
        stats['usage_by_action'] = usage_by_action
        
        # Trending data (last 30 days)
        cursor.execute('''
            SELECT DATE(timestamp) as day, COUNT(*) as count
            FROM package_usage 
            WHERE package_name = ? 
              AND timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY day
        ''', (package_name,))
        
        trend_data = cursor.fetchall()
        stats['trend_data'] = [{'date': row[0], 'count': row[1]} for row in trend_data]
        
        conn.close()
        return stats
    
    def discover_packages(self, monorepo_root: str) -> int:
        """Discover and register all packages in the monorepo."""
        if not self.analyzer:
            self.analyzer = PackageAnalyzer(monorepo_root)
        
        packages_dir = Path(monorepo_root) / "src" / "packages"
        if not packages_dir.exists():
            return 0
        
        discovered_count = 0
        
        for package_dir in packages_dir.iterdir():
            if package_dir.is_dir() and (package_dir / "src").exists():
                try:
                    metadata = self.analyzer.analyze_package(str(package_dir))
                    if self.register_package(metadata):
                        discovered_count += 1
                        print(f"Registered package: {metadata.name}")
                except Exception as e:
                    print(f"Error analyzing package {package_dir.name}: {e}")
        
        return discovered_count
    
    def _metadata_to_db_format(self, metadata: PackageMetadata) -> tuple:
        """Convert PackageMetadata to database format."""
        return (
            metadata.name,
            metadata.version,
            metadata.description,
            metadata.package_type.value,
            metadata.status.value,
            metadata.domain,
            metadata.bounded_context,
            metadata.path,
            json.dumps(metadata.languages),
            json.dumps(metadata.frameworks),
            json.dumps(metadata.dependencies),
            json.dumps(metadata.api_endpoints),
            json.dumps(metadata.events_published),
            json.dumps(metadata.events_consumed),
            metadata.test_coverage,
            metadata.complexity_score,
            metadata.security_score,
            metadata.performance_score,
            metadata.independence_score,
            metadata.readme_path,
            metadata.documentation_url,
            json.dumps(metadata.examples),
            json.dumps(metadata.maintainers),
            metadata.created_at,
            metadata.updated_at,
            metadata.last_commit_hash,
            metadata.download_count,
            metadata.usage_count,
            metadata.rating,
            metadata.min_platform_version,
            metadata.max_platform_version,
            json.dumps(metadata.compatible_packages),
            json.dumps(metadata.conflicting_packages)
        )
    
    def _db_row_to_metadata(self, row) -> PackageMetadata:
        """Convert database row to PackageMetadata."""
        return PackageMetadata(
            name=row[0],
            version=row[1],
            description=row[2] or "",
            package_type=PackageType(row[3]),
            status=PackageStatus(row[4]),
            domain=row[5] or "",
            bounded_context=row[6] or "",
            path=row[7] or "",
            languages=json.loads(row[8] or "[]"),
            frameworks=json.loads(row[9] or "[]"),
            dependencies=json.loads(row[10] or "[]"),
            api_endpoints=json.loads(row[11] or "[]"),
            events_published=json.loads(row[12] or "[]"),
            events_consumed=json.loads(row[13] or "[]"),
            test_coverage=row[14] or 0.0,
            complexity_score=row[15] or 0,
            security_score=row[16] or 0,
            performance_score=row[17] or 0,
            independence_score=row[18] or 0,
            readme_path=row[19],
            documentation_url=row[20],
            examples=json.loads(row[21] or "[]"),
            maintainers=json.loads(row[22] or "[]"),
            created_at=datetime.fromisoformat(row[23]) if row[23] else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row[24]) if row[24] else datetime.utcnow(),
            last_commit_hash=row[25],
            download_count=row[26] or 0,
            usage_count=row[27] or 0,
            rating=row[28] or 0.0,
            min_platform_version=row[29],
            max_platform_version=row[30],
            compatible_packages=json.loads(row[31] or "[]"),
            conflicting_packages=json.loads(row[32] or "[]")
        )

def main():
    """Command line interface for package registry."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Package Registry Management")
    parser.add_argument("--discover", help="Discover packages in monorepo")
    parser.add_argument("--search", help="Search packages")
    parser.add_argument("--stats", help="Get package statistics")
    parser.add_argument("--db", default="package_registry.db", help="Database path")
    
    args = parser.parse_args()
    
    registry = PackageRegistry(args.db)
    
    if args.discover:
        count = registry.discover_packages(args.discover)
        print(f"Discovered {count} packages")
    
    elif args.search:
        criteria = PackageSearchCriteria(query=args.search)
        packages = registry.search_packages(criteria)
        
        print(f"Found {len(packages)} packages:")
        for pkg in packages:
            print(f"- {pkg.name} ({pkg.version}): {pkg.description}")
            print(f"  Domain: {pkg.domain}, Type: {pkg.package_type.value}")
            print(f"  Rating: {pkg.rating:.1f}, Coverage: {pkg.test_coverage:.1f}%")
            print()
    
    elif args.stats:
        stats = registry.get_package_statistics(args.stats)
        if stats:
            print(f"Statistics for {args.stats}:")
            print(json.dumps(stats, indent=2, default=str))
        else:
            print(f"Package {args.stats} not found")

if __name__ == "__main__":
    main()