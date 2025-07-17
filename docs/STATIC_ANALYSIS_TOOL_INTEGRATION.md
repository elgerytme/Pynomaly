# Static Analysis Tool Integration Strategy

## Tool Selection Matrix

### Primary Analysis Tools

| Tool | Category | Capabilities | Priority | Integration Complexity |
|------|----------|--------------|----------|----------------------|
| **MyPy** | Type Checking | Static type checking, generics, protocols | High | Medium |
| **Pyright** | Type Checking | Advanced inference, performance | High | Medium |
| **Ruff** | Code Quality | Fast linting, formatting, import sorting | High | Low |
| **Black** | Formatting | Code formatting, style consistency | High | Low |
| **Bandit** | Security | Security vulnerability scanning | High | Low |
| **Safety** | Security | Dependency vulnerability checking | High | Low |
| **Vulture** | Dead Code | Unused code detection | Medium | Low |
| **Pyflakes** | Logic Errors | Import errors, undefined names | Medium | Low |
| **isort** | Import Management | Import sorting and organization | Medium | Low |
| **Semgrep** | Pattern Analysis | Custom pattern matching | Medium | Medium |
| **Pydocstyle** | Documentation | Docstring style checking | Medium | Low |
| **Autoflake** | Code Cleanup | Remove unused imports and variables | Low | Low |

### Tool Compatibility Matrix

| Tool Combination | Compatibility | Conflicts | Resolution |
|------------------|---------------|-----------|------------|
| MyPy + Pyright | High | Type annotation preferences | Use MyPy as primary |
| Ruff + Black | High | Formatting rules | Ruff can replace Black |
| Ruff + isort | High | Import sorting | Ruff includes isort functionality |
| Bandit + Semgrep | High | Security rule overlap | Merge results, prioritize Bandit |
| MyPy + Pyflakes | Medium | Error reporting overlap | Prefer MyPy errors |

## Tool-Specific Integration

### MyPy Integration

#### Configuration Generation
```python
class MyPyAdapter(ToolAdapter):
    def generate_config(self) -> str:
        config = f"""
[mypy]
python_version = {self.config.python_version}
strict = {self.config.strict_mode}
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = {self.config.require_type_annotations}
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
"""
        
        # Add plugin configurations
        for plugin in self.config.mypy_plugins:
            config += f"plugins = {plugin}\n"
        
        return config
```

#### Output Processing
```python
def parse_mypy_output(self, output: str) -> List[Issue]:
    issues = []
    
    for line in output.strip().split('\n'):
        if not line:
            continue
            
        try:
            data = json.loads(line)
            issue = Issue(
                file=Path(data['file']),
                line=data['line'],
                column=data['column'],
                severity=self._map_severity(data['severity']),
                message=data['message'],
                rule=data.get('error_code', 'mypy'),
                tool='mypy',
                fixable=self._is_fixable(data)
            )
            issues.append(issue)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse MyPy output: {line}")
    
    return issues
```

### Pyright Integration

#### LSP Communication
```python
class PyrightAdapter(ToolAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lsp_client = LSPClient()
    
    async def analyze(self, files: List[Path]) -> List[Issue]:
        # Start Pyright language server
        await self.lsp_client.start_server('pyright-langserver', '--stdio')
        
        issues = []
        for file in files:
            # Send file content to LSP
            await self.lsp_client.send_notification('textDocument/didOpen', {
                'textDocument': {
                    'uri': file.as_uri(),
                    'languageId': 'python',
                    'version': 1,
                    'text': file.read_text()
                }
            })
            
            # Get diagnostics
            diagnostics = await self.lsp_client.get_diagnostics(file.as_uri())
            issues.extend(self._convert_diagnostics(diagnostics))
        
        await self.lsp_client.stop_server()
        return issues
```

### Ruff Integration

#### Command Line Interface
```python
class RuffAdapter(ToolAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ruff_config = self._create_ruff_config()
    
    async def analyze(self, files: List[Path]) -> List[Issue]:
        cmd = [
            'ruff', 'check',
            '--config', self.ruff_config,
            '--output-format', 'json',
            '--select', ','.join(self.config.ruff_rules),
            *[str(f) for f in files]
        ]
        
        result = await self._run_command(cmd)
        return self._parse_ruff_output(result.stdout)
    
    def _create_ruff_config(self) -> Path:
        config = {
            'line-length': self.config.line_length,
            'select': self.config.ruff_rules,
            'ignore': self.config.ruff_ignore,
            'exclude': self.config.exclude_patterns,
            'per-file-ignores': self.config.per_file_ignores
        }
        
        config_path = Path.cwd() / '.ruff_temp.toml'
        with open(config_path, 'w') as f:
            toml.dump({'tool': {'ruff': config}}, f)
        
        return config_path
```

### Bandit Integration

#### Security Rule Configuration
```python
class BanditAdapter(ToolAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.security_level = config.get('security_level', 'medium')
        self.custom_rules = config.get('custom_security_rules', [])
    
    async def analyze(self, files: List[Path]) -> List[Issue]:
        cmd = [
            'bandit',
            '-r', '.',
            '-f', 'json',
            '-ll',  # Low confidence, low severity
            '--exclude', ','.join(self.config.exclude_patterns),
            *[str(f) for f in files]
        ]
        
        # Add custom rules
        if self.custom_rules:
            cmd.extend(['--skip', ','.join(self.custom_rules)])
        
        result = await self._run_command(cmd)
        return self._parse_bandit_output(result.stdout)
    
    def _parse_bandit_output(self, output: str) -> List[Issue]:
        try:
            data = json.loads(output)
            issues = []
            
            for result in data.get('results', []):
                issue = Issue(
                    file=Path(result['filename']),
                    line=result['line_number'],
                    column=result.get('col_offset', 0),
                    severity=self._map_severity(result['issue_severity']),
                    message=result['issue_text'],
                    rule=result['test_id'],
                    tool='bandit',
                    fixable=False,
                    category='security'
                )
                issues.append(issue)
            
            return issues
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Bandit output: {output}")
            return []
```

### Safety Integration

#### Dependency Vulnerability Scanning
```python
class SafetyAdapter(ToolAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ignore_vulnerabilities = config.get('ignore_vulnerabilities', [])
    
    async def analyze(self, files: List[Path]) -> List[Issue]:
        # Safety works on requirements files, not individual Python files
        requirements_files = [f for f in files if f.name in ('requirements.txt', 'Pipfile', 'poetry.lock')]
        
        if not requirements_files:
            return []
        
        cmd = [
            'safety', 'check',
            '--json',
            '--ignore', ','.join(self.ignore_vulnerabilities)
        ]
        
        result = await self._run_command(cmd)
        return self._parse_safety_output(result.stdout)
    
    def _parse_safety_output(self, output: str) -> List[Issue]:
        try:
            vulnerabilities = json.loads(output)
            issues = []
            
            for vuln in vulnerabilities:
                issue = Issue(
                    file=Path('requirements.txt'),  # Generic file
                    line=0,
                    column=0,
                    severity='high' if vuln['severity'] == 'high' else 'medium',
                    message=f"Vulnerability in {vuln['package_name']}: {vuln['advisory']}",
                    rule=vuln['vulnerability_id'],
                    tool='safety',
                    fixable=True,
                    category='security',
                    suggestion=f"Upgrade to {vuln['package_name']}>={vuln['analyzed_version']}"
                )
                issues.append(issue)
            
            return issues
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Safety output: {output}")
            return []
```

## Advanced Tool Integration

### Semgrep Integration

#### Custom Pattern Analysis
```python
class SemgrepAdapter(ToolAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_rules = config.get('semgrep_rules', [])
        self.rule_directories = config.get('rule_directories', [])
    
    async def analyze(self, files: List[Path]) -> List[Issue]:
        cmd = [
            'semgrep',
            '--json',
            '--config', 'auto',  # Use default rules
            *[str(f) for f in files]
        ]
        
        # Add custom rules
        for rule_dir in self.rule_directories:
            cmd.extend(['--config', str(rule_dir)])
        
        result = await self._run_command(cmd)
        return self._parse_semgrep_output(result.stdout)
    
    def create_custom_rule(self, rule_config: Dict[str, Any]) -> str:
        """Create a custom Semgrep rule from configuration."""
        rule = {
            'id': rule_config['id'],
            'patterns': rule_config['patterns'],
            'message': rule_config['message'],
            'severity': rule_config['severity'],
            'languages': ['python']
        }
        
        return yaml.dump({'rules': [rule]})
```

### Vulture Integration

#### Dead Code Detection
```python
class VultureAdapter(ToolAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.confidence_threshold = config.get('vulture_confidence', 80)
        self.whitelist_files = config.get('vulture_whitelist', [])
    
    async def analyze(self, files: List[Path]) -> List[Issue]:
        cmd = [
            'vulture',
            '--min-confidence', str(self.confidence_threshold),
            *[str(f) for f in files]
        ]
        
        # Add whitelist files
        cmd.extend(self.whitelist_files)
        
        result = await self._run_command(cmd)
        return self._parse_vulture_output(result.stdout)
    
    def _parse_vulture_output(self, output: str) -> List[Issue]:
        issues = []
        
        for line in output.strip().split('\n'):
            if not line:
                continue
            
            # Parse vulture output format
            match = re.match(r'(.+):(\d+): (.+) \(confidence: (\d+)%\)', line)
            if match:
                file_path, line_no, message, confidence = match.groups()
                
                issue = Issue(
                    file=Path(file_path),
                    line=int(line_no),
                    column=0,
                    severity='info',
                    message=message,
                    rule='vulture-unused',
                    tool='vulture',
                    fixable=True,
                    category='dead-code',
                    metadata={'confidence': int(confidence)}
                )
                issues.append(issue)
        
        return issues
```

## Tool Orchestration

### Parallel Execution Strategy
```python
class ToolOrchestrator:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.tools = self._initialize_tools()
        self.executor = ParallelExecutor(max_workers=config.max_workers)
    
    def _initialize_tools(self) -> List[ToolAdapter]:
        tools = []
        
        # Type checking tools
        if self.config.enable_mypy:
            tools.append(MyPyAdapter(self.config.mypy_config))
        if self.config.enable_pyright:
            tools.append(PyrightAdapter(self.config.pyright_config))
        
        # Code quality tools
        if self.config.enable_ruff:
            tools.append(RuffAdapter(self.config.ruff_config))
        if self.config.enable_black:
            tools.append(BlackAdapter(self.config.black_config))
        
        # Security tools
        if self.config.enable_bandit:
            tools.append(BanditAdapter(self.config.bandit_config))
        if self.config.enable_safety:
            tools.append(SafetyAdapter(self.config.safety_config))
        
        # Analysis tools
        if self.config.enable_vulture:
            tools.append(VultureAdapter(self.config.vulture_config))
        if self.config.enable_semgrep:
            tools.append(SemgrepAdapter(self.config.semgrep_config))
        
        return tools
    
    async def run_analysis(self, files: List[Path]) -> List[AnalysisResult]:
        # Group files by tool compatibility
        tool_file_groups = self._group_files_by_tools(files)
        
        # Run tools in parallel
        results = await self.executor.execute_tools(tool_file_groups)
        
        return results
```

### Result Merging and Deduplication
```python
class ResultMerger:
    def __init__(self):
        self.similarity_threshold = 0.8
    
    def merge_results(self, results: List[AnalysisResult]) -> AnalysisResult:
        all_issues = []
        
        # Collect all issues
        for result in results:
            all_issues.extend(result.issues)
        
        # Deduplicate similar issues
        unique_issues = self._deduplicate_issues(all_issues)
        
        # Merge metadata
        merged_metadata = self._merge_metadata(results)
        
        return AnalysisResult(
            issues=unique_issues,
            metadata=merged_metadata
        )
    
    def _deduplicate_issues(self, issues: List[Issue]) -> List[Issue]:
        unique_issues = []
        seen_signatures = set()
        
        for issue in issues:
            signature = self._create_issue_signature(issue)
            
            if signature not in seen_signatures:
                unique_issues.append(issue)
                seen_signatures.add(signature)
            else:
                # Merge with existing issue
                existing_issue = self._find_existing_issue(unique_issues, signature)
                if existing_issue:
                    existing_issue.merge_with(issue)
        
        return unique_issues
    
    def _create_issue_signature(self, issue: Issue) -> str:
        """Create a unique signature for an issue to enable deduplication."""
        return f"{issue.file}:{issue.line}:{issue.rule}:{hash(issue.message)}"
```

## Performance Optimization

### Caching Strategy
```python
class ToolResultCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cached_result(self, tool: str, file: Path) -> Optional[AnalysisResult]:
        # Check if file has changed
        file_hash = self._calculate_file_hash(file)
        cache_key = f"{tool}:{file.name}:{file_hash}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return AnalysisResult.from_dict(data)
            except Exception:
                # Cache corrupted, remove it
                cache_file.unlink()
        
        return None
    
    def store_result(self, tool: str, file: Path, result: AnalysisResult) -> None:
        file_hash = self._calculate_file_hash(file)
        cache_key = f"{tool}:{file.name}:{file_hash}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(result.to_dict(), f)
```

### Incremental Analysis
```python
class IncrementalAnalyzer:
    def __init__(self, cache: ToolResultCache):
        self.cache = cache
        self.dependency_graph = DependencyGraph()
    
    def get_files_needing_analysis(self, files: List[Path], changed_files: List[Path]) -> List[Path]:
        """Determine which files need analysis based on changes."""
        files_to_analyze = set(changed_files)
        
        # Add files that depend on changed files
        for changed_file in changed_files:
            dependents = self.dependency_graph.get_dependents(changed_file)
            files_to_analyze.update(dependents)
        
        # Filter to only include files in our analysis set
        return [f for f in files if f in files_to_analyze]
```

This comprehensive tool integration strategy ensures that all analysis tools work together effectively, providing accurate, fast, and actionable results while minimizing conflicts and maximizing coverage.