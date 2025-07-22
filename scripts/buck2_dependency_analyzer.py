#!/usr/bin/env python3
"""
Buck2 Dependency Analysis and Package Consolidation Tool
Analyzes package dependencies and recommends consolidation opportunities
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

class Buck2DependencyAnalyzer:
    """Analyzes Buck2 target dependencies and suggests optimizations"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.dependency_graph = defaultdict(set)
        self.reverse_deps = defaultdict(set)
        self.target_metadata = {}
        
    def log(self, message: str) -> None:
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[DEPS] {message}")
    
    def get_all_targets(self) -> List[str]:
        """Get all Buck2 targets in the repository"""
        self.log("Discovering all Buck2 targets...")
        
        try:
            result = subprocess.run(
                ['buck2', 'targets', '//...'],
                capture_output=True, text=True, check=True
            )
            targets = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            self.log(f"Found {len(targets)} targets")
            return targets
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to get targets: {e}")
            return []
        except FileNotFoundError:
            self.log("Buck2 not found - using mock data for analysis")
            return self._get_mock_targets()
    
    def _get_mock_targets(self) -> List[str]:
        """Generate mock targets based on BUCK file structure"""
        return [
            "//:ai-anomaly-detection",
            "//:ai-machine-learning", 
            "//:ai-mlops",
            "//:ai-all",
            "//:data-analytics",
            "//:data-engineering",
            "//:data-quality",
            "//:data-observability", 
            "//:data-profiling",
            "//:data-all",
            "//:enterprise-auth",
            "//:enterprise-governance",
            "//:enterprise-scalability",
            "//:enterprise-all",
            "//:pynomaly"
        ]
    
    def analyze_dependencies(self, targets: List[str]) -> None:
        """Analyze dependencies between targets"""
        self.log("Analyzing target dependencies...")
        
        for target in targets:
            self.log(f"Analyzing {target}")
            deps = self._get_target_dependencies(target)
            
            self.dependency_graph[target] = set(deps)
            
            # Build reverse dependency graph
            for dep in deps:
                self.reverse_deps[dep].add(target)
            
            # Get target metadata
            self.target_metadata[target] = self._get_target_metadata(target)
    
    def _get_target_dependencies(self, target: str) -> List[str]:
        """Get direct dependencies for a target"""
        try:
            result = subprocess.run(
                ['buck2', 'cquery', f'deps("{target}", 1)'],
                capture_output=True, text=True, check=True
            )
            deps = [line.strip() for line in result.stdout.split('\n') 
                    if line.strip() and line.strip() != target]
            return deps
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Use heuristic dependency analysis based on target names
            return self._infer_dependencies(target)
    
    def _infer_dependencies(self, target: str) -> List[str]:
        """Infer dependencies based on target naming patterns"""
        deps = []
        
        # Domain-level dependencies
        if target == "//:ai-all":
            deps = ["//:ai-anomaly-detection", "//:ai-machine-learning", "//:ai-mlops"]
        elif target == "//:data-all":
            deps = ["//:data-analytics", "//:data-engineering", "//:data-quality", 
                   "//:data-observability", "//:data-profiling"]
        elif target == "//:enterprise-all":
            deps = ["//:enterprise-auth", "//:enterprise-governance", "//:enterprise-scalability"]
        elif target == "//:pynomaly":
            deps = ["//:ai-all", "//:data-all", "//:enterprise-all"]
        
        # Cross-domain dependencies (inferred)
        if "machine-learning" in target:
            deps.append("//:ai-anomaly-detection")
        if "mlops" in target:
            deps.extend(["//:ai-anomaly-detection", "//:ai-machine-learning"])
        if "data-quality" in target:
            deps.append("//:data-engineering")
        if "data-observability" in target:
            deps.append("//:data-quality")
        if "enterprise-governance" in target:
            deps.append("//:enterprise-auth")
        if "enterprise-scalability" in target:
            deps.append("//:enterprise-governance")
        
        return deps
    
    def _get_target_metadata(self, target: str) -> Dict:
        """Get metadata for a target"""
        try:
            result = subprocess.run(
                ['buck2', 'cquery', f'"{target}"', '--output-attribute', 'buck.type'],
                capture_output=True, text=True, check=True
            )
            
            return {
                'type': result.stdout.strip() if result.stdout.strip() else 'python_library',
                'domain': self._extract_domain(target),
                'complexity': self._estimate_complexity(target)
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                'type': 'python_library',
                'domain': self._extract_domain(target),
                'complexity': self._estimate_complexity(target)
            }
    
    def _extract_domain(self, target: str) -> str:
        """Extract domain from target name"""
        if 'ai-' in target:
            return 'ai'
        elif 'data-' in target:
            return 'data'
        elif 'enterprise-' in target:
            return 'enterprise'
        else:
            return 'core'
    
    def _estimate_complexity(self, target: str) -> str:
        """Estimate target complexity based on naming"""
        if target.endswith('-all') or target == '//:pynomaly':
            return 'high'
        elif any(keyword in target for keyword in ['mlops', 'scalability', 'governance']):
            return 'high'
        elif any(keyword in target for keyword in ['engineering', 'analytics', 'observability']):
            return 'medium'
        else:
            return 'low'
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies using DFS"""
        self.log("Searching for circular dependencies...")
        
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(target: str, path: List[str]) -> bool:
            if target in rec_stack:
                # Found cycle
                cycle_start = path.index(target)
                cycle = path[cycle_start:] + [target]
                cycles.append(cycle)
                return True
            
            if target in visited:
                return False
            
            visited.add(target)
            rec_stack.add(target)
            
            for dep in self.dependency_graph[target]:
                if dfs(dep, path + [target]):
                    return True
            
            rec_stack.remove(target)
            return False
        
        for target in self.dependency_graph:
            if target not in visited:
                dfs(target, [])
        
        self.log(f"Found {len(cycles)} circular dependencies")
        return cycles
    
    def find_consolidation_opportunities(self) -> Dict:
        """Find package consolidation opportunities"""
        self.log("Analyzing consolidation opportunities...")
        
        opportunities = {
            'duplicate_dependencies': [],
            'small_packages': [],
            'highly_coupled': [],
            'domain_violations': []
        }
        
        # Find packages with identical dependencies
        dep_groups = defaultdict(list)
        for target, deps in self.dependency_graph.items():
            if deps:  # Only consider targets with dependencies
                dep_key = tuple(sorted(deps))
                dep_groups[dep_key].append(target)
        
        for dep_key, targets in dep_groups.items():
            if len(targets) > 1:
                opportunities['duplicate_dependencies'].append({
                    'targets': targets,
                    'shared_dependencies': list(dep_key),
                    'consolidation_potential': 'high'
                })
        
        # Find packages with few dependencies (potential candidates for merging)
        for target, deps in self.dependency_graph.items():
            if len(deps) <= 2 and not target.endswith('-all'):
                opportunities['small_packages'].append({
                    'target': target,
                    'dependency_count': len(deps),
                    'dependencies': list(deps)
                })
        
        # Find highly coupled packages
        coupling_scores = {}
        for target in self.dependency_graph:
            deps = self.dependency_graph[target]
            rdeps = self.reverse_deps[target]
            coupling_score = len(deps) * len(rdeps)
            coupling_scores[target] = coupling_score
        
        # Sort by coupling score
        highly_coupled = sorted(coupling_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        opportunities['highly_coupled'] = [
            {
                'target': target,
                'coupling_score': score,
                'dependencies': len(self.dependency_graph[target]),
                'dependents': len(self.reverse_deps[target])
            }
            for target, score in highly_coupled if score > 0
        ]
        
        # Find cross-domain dependencies (potential violations)
        for target, deps in self.dependency_graph.items():
            target_domain = self.target_metadata.get(target, {}).get('domain', 'unknown')
            
            for dep in deps:
                dep_domain = self.target_metadata.get(dep, {}).get('domain', 'unknown')
                
                if target_domain != dep_domain and target_domain != 'core' and dep_domain != 'core':
                    opportunities['domain_violations'].append({
                        'source': target,
                        'source_domain': target_domain,
                        'dependency': dep,
                        'dependency_domain': dep_domain
                    })
        
        return opportunities
    
    def generate_consolidation_plan(self, opportunities: Dict) -> Dict:
        """Generate a consolidation plan based on opportunities"""
        self.log("Generating consolidation plan...")
        
        plan = {
            'priority_actions': [],
            'domain_reorganization': [],
            'merge_candidates': [],
            'cleanup_actions': []
        }
        
        # High priority: Fix circular dependencies
        cycles = self.find_circular_dependencies()
        for cycle in cycles:
            plan['priority_actions'].append({
                'action': 'break_circular_dependency',
                'targets': cycle,
                'priority': 'critical',
                'description': f"Break circular dependency: {' -> '.join(cycle)}"
            })
        
        # Medium priority: Consolidate duplicate dependencies
        for dup in opportunities['duplicate_dependencies']:
            if len(dup['targets']) <= 3:  # Only small groups
                plan['merge_candidates'].append({
                    'action': 'merge_packages',
                    'targets': dup['targets'],
                    'priority': 'medium',
                    'description': f"Merge packages with identical dependencies: {', '.join(dup['targets'])}"
                })
        
        # Domain reorganization
        domain_violations = opportunities['domain_violations']
        if domain_violations:
            plan['domain_reorganization'].append({
                'action': 'review_cross_domain_dependencies',
                'violations': domain_violations,
                'priority': 'low',
                'description': f"Review {len(domain_violations)} cross-domain dependencies"
            })
        
        # Cleanup small packages
        small_packages = opportunities['small_packages']
        if len(small_packages) > 5:
            plan['cleanup_actions'].append({
                'action': 'consolidate_small_packages',
                'packages': small_packages[:5],  # Top 5 candidates
                'priority': 'low',
                'description': "Consider consolidating packages with minimal dependencies"
            })
        
        return plan
    
    def generate_report(self, opportunities: Dict, plan: Dict) -> str:
        """Generate dependency analysis report"""
        
        total_targets = len(self.dependency_graph)
        total_deps = sum(len(deps) for deps in self.dependency_graph.values())
        avg_deps = total_deps / total_targets if total_targets > 0 else 0
        
        report = f"""# Buck2 Dependency Analysis Report

## Summary
- Total targets analyzed: {total_targets}
- Total dependencies: {total_deps}
- Average dependencies per target: {avg_deps:.1f}

## Consolidation Opportunities

### Duplicate Dependencies
Found {len(opportunities['duplicate_dependencies'])} groups of packages with identical dependencies:
"""
        
        for dup in opportunities['duplicate_dependencies'][:3]:  # Show top 3
            report += f"- {', '.join(dup['targets'])}\n"
            report += f"  Shared dependencies: {len(dup['shared_dependencies'])}\n"
        
        report += f"""
### Small Packages
Found {len(opportunities['small_packages'])} packages with minimal dependencies:
"""
        
        for small in opportunities['small_packages'][:5]:  # Show top 5
            report += f"- {small['target']}: {small['dependency_count']} dependencies\n"
        
        report += f"""
### Highly Coupled Packages
Top packages by coupling score:
"""
        
        for coupled in opportunities['highly_coupled'][:3]:
            report += f"- {coupled['target']}: score {coupled['coupling_score']} ({coupled['dependencies']} deps, {coupled['dependents']} dependents)\n"
        
        report += f"""
### Cross-Domain Dependencies
Found {len(opportunities['domain_violations'])} cross-domain dependencies:
"""
        
        for violation in opportunities['domain_violations'][:5]:
            report += f"- {violation['source_domain']} -> {violation['dependency_domain']}: {violation['source']} -> {violation['dependency']}\n"
        
        report += f"""
## Consolidation Plan

### Priority Actions ({len(plan['priority_actions'])})
"""
        for action in plan['priority_actions']:
            report += f"- **{action['priority'].upper()}**: {action['description']}\n"
        
        report += f"""
### Merge Candidates ({len(plan['merge_candidates'])})
"""
        for merge in plan['merge_candidates']:
            report += f"- {merge['description']}\n"
        
        report += f"""
### Domain Reorganization ({len(plan['domain_reorganization'])})
"""
        for reorg in plan['domain_reorganization']:
            report += f"- {reorg['description']}\n"
        
        report += """
## Recommendations

1. **Address circular dependencies first** - These can cause build failures
2. **Consider merging packages with identical dependencies** - Reduces maintenance overhead
3. **Review cross-domain dependencies** - May indicate architectural issues
4. **Consolidate small packages** - Reduces build graph complexity

## Next Steps

1. Run `buck2 cquery` commands to validate dependency analysis
2. Create merge plan for packages with identical dependencies
3. Review cross-domain dependencies with architecture team
4. Implement consolidation changes incrementally with testing
"""
        
        return report

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Buck2 dependencies and suggest package consolidation"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help="Specific targets to analyze (default: all targets)"
    )
    parser.add_argument(
        "--report",
        help="Output report to file"
    )
    parser.add_argument(
        "--json",
        help="Output analysis as JSON to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    analyzer = Buck2DependencyAnalyzer(verbose=args.verbose)
    
    print("🔍 Buck2 Dependency Analysis Tool")
    
    # Get targets to analyze
    if args.targets:
        targets = args.targets
        print(f"Analyzing {len(targets)} specified targets...")
    else:
        targets = analyzer.get_all_targets()
        print(f"Analyzing all {len(targets)} targets...")
    
    if not targets:
        print("❌ No targets found to analyze")
        sys.exit(1)
    
    # Analyze dependencies
    analyzer.analyze_dependencies(targets)
    
    # Find consolidation opportunities
    opportunities = analyzer.find_consolidation_opportunities()
    
    # Generate consolidation plan
    plan = analyzer.generate_consolidation_plan(opportunities)
    
    # Output results
    if args.json:
        result = {
            'targets': targets,
            'dependency_graph': {k: list(v) for k, v in analyzer.dependency_graph.items()},
            'opportunities': opportunities,
            'plan': plan
        }
        
        with open(args.json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"📊 JSON analysis saved to {args.json}")
    
    if args.report:
        report = analyzer.generate_report(opportunities, plan)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"📋 Report saved to {args.report}")
    
    # Print summary
    print(f"\n📈 Analysis Summary:")
    print(f"  - Duplicate dependency groups: {len(opportunities['duplicate_dependencies'])}")
    print(f"  - Small packages: {len(opportunities['small_packages'])}")
    print(f"  - Highly coupled packages: {len(opportunities['highly_coupled'])}")
    print(f"  - Cross-domain dependencies: {len(opportunities['domain_violations'])}")
    print(f"  - Priority actions: {len(plan['priority_actions'])}")
    print(f"  - Merge candidates: {len(plan['merge_candidates'])}")

if __name__ == "__main__":
    main()