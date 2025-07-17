"""Main orchestrator for comprehensive static analysis."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from .config.manager import AnalysisConfig
from .tools.adapter_base import ToolAdapter, AnalysisResult
from .tools.mypy_adapter import MyPyAdapter
from .tools.ruff_adapter import RuffAdapter
from .tools.bandit_adapter import BanditAdapter
from .tools.safety_adapter import SafetyAdapter
from .tools.vulture_adapter import VultureAdapter
from .tools.pyright_adapter import PyrightAdapter
from .utils.file_discovery import FileDiscovery
from .utils.parallel_executor import ParallelExecutor

logger = logging.getLogger(__name__)


class ComprehensiveAnalysisResult:
    """Result of comprehensive analysis."""
    
    def __init__(self, files_analyzed: int, results: List[AnalysisResult], 
                 execution_time: float, success: bool = True, 
                 error_message: str = "", metadata: Optional[Dict[str, Any]] = None):
        self.files_analyzed = files_analyzed
        self.results = results
        self.execution_time = execution_time
        self.success = success
        self.error_message = error_message
        self.metadata = metadata or {}
    
    def get_all_issues(self) -> List:
        """Get all issues from all tool results."""
        all_issues = []
        for result in self.results:
            all_issues.extend(result.issues)
        return all_issues
    
    def get_issues_by_severity(self) -> Dict[str, List]:
        """Get issues grouped by severity."""
        issues_by_severity = {"error": [], "warning": [], "info": []}
        
        for issue in self.get_all_issues():
            if issue.severity in issues_by_severity:
                issues_by_severity[issue.severity].append(issue)
        
        return issues_by_severity
    
    def get_issues_by_file(self) -> Dict[Path, List]:
        """Get issues grouped by file."""
        issues_by_file = {}
        
        for issue in self.get_all_issues():
            if issue.file not in issues_by_file:
                issues_by_file[issue.file] = []
            issues_by_file[issue.file].append(issue)
        
        return issues_by_file
    
    def get_issues_by_tool(self) -> Dict[str, List]:
        """Get issues grouped by tool."""
        issues_by_tool = {}
        
        for issue in self.get_all_issues():
            if issue.tool not in issues_by_tool:
                issues_by_tool[issue.tool] = []
            issues_by_tool[issue.tool].append(issue)
        
        return issues_by_tool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "files_analyzed": self.files_analyzed,
            "results": [result.to_dict() for result in self.results],
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class AnalysisOrchestrator:
    """Main orchestrator for comprehensive static analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.file_discovery = FileDiscovery(
            include_patterns=config.include_patterns,
            exclude_patterns=config.exclude_patterns
        )
        self.parallel_executor = ParallelExecutor(max_workers=config.max_workers)
        self.tools = self._initialize_tools()
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.config.show_context else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_tools(self) -> List[ToolAdapter]:
        """Initialize analysis tools based on configuration."""
        tools = []
        
        # Type checking tools
        if self.config.enable_type_checking:
            mypy_config = self.config.tool_configs.get("mypy", {})
            if MyPyAdapter({**mypy_config, "python_version": self.config.python_version}).is_available():
                tools.append(MyPyAdapter({**mypy_config, "python_version": self.config.python_version}))
                logger.info("Initialized MyPy adapter")
            
            # Enhanced type checking with Pyright
            pyright_config = self.config.tool_configs.get("pyright", {})
            if PyrightAdapter(pyright_config).is_available():
                tools.append(PyrightAdapter(pyright_config))
                logger.info("Initialized Pyright adapter")
        
        # Code quality tools
        ruff_config = self.config.tool_configs.get("ruff", {})
        if RuffAdapter(ruff_config).is_available():
            tools.append(RuffAdapter(ruff_config))
            logger.info("Initialized Ruff adapter")
        
        # Dead code detection
        if self.config.enable_dead_code_detection:
            vulture_config = self.config.tool_configs.get("vulture", {})
            if VultureAdapter(vulture_config).is_available():
                tools.append(VultureAdapter(vulture_config))
                logger.info("Initialized Vulture adapter")
        
        # Security analysis
        if self.config.enable_security_analysis:
            bandit_config = self.config.tool_configs.get("bandit", {})
            if BanditAdapter(bandit_config).is_available():
                tools.append(BanditAdapter(bandit_config))
                logger.info("Initialized Bandit adapter")
            
            safety_config = self.config.tool_configs.get("safety", {})
            if SafetyAdapter(safety_config).is_available():
                tools.append(SafetyAdapter(safety_config))
                logger.info("Initialized Safety adapter")
        
        if not tools:
            logger.warning("No analysis tools available or enabled")
        
        return tools
    
    async def analyze(self, paths: List[Path]) -> ComprehensiveAnalysisResult:
        """Run comprehensive analysis on given paths."""
        logger.info(f"Starting comprehensive analysis on {len(paths)} paths")
        start_time = time.time()
        
        try:
            # Discover files
            files = self.file_discovery.discover(paths)
            if not files:
                logger.warning("No files found for analysis")
                return ComprehensiveAnalysisResult(
                    files_analyzed=0,
                    results=[],
                    execution_time=0.0,
                    success=True,
                    metadata={"message": "No files found for analysis"}
                )
            
            # Show progress if enabled
            if self.config.show_progress:
                file_stats = self.file_discovery.get_file_stats(files)
                logger.info(f"Discovered {file_stats['total_files']} files in {file_stats['directories']} directories")
                logger.info(f"File extensions: {file_stats['extensions']}")
            
            # Run analysis tools
            results = await self._run_analysis_tools(files)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create comprehensive result
            comprehensive_result = ComprehensiveAnalysisResult(
                files_analyzed=len(files),
                results=results,
                execution_time=execution_time,
                success=True,
                metadata={
                    "tools_used": [tool.name for tool in self.tools],
                    "file_stats": self.file_discovery.get_file_stats(files),
                    "config_profile": self.config.profile,
                }
            )
            
            logger.info(f"Analysis completed in {execution_time:.2f}s")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return ComprehensiveAnalysisResult(
                files_analyzed=0,
                results=[],
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def _run_analysis_tools(self, files: List[Path]) -> List[AnalysisResult]:
        """Run analysis tools on discovered files."""
        if not self.tools:
            logger.warning("No analysis tools available")
            return []
        
        logger.info(f"Running {len(self.tools)} analysis tools on {len(files)} files")
        
        # Create tasks for each tool
        tool_tasks = []
        for tool in self.tools:
            # Filter files to only those supported by this tool
            supported_files = tool._filter_files(files)
            if supported_files:
                task_info = {
                    "tool": tool,
                    "files": supported_files
                }
                tool_tasks.append(task_info)
                logger.debug(f"Tool {tool.name} will analyze {len(supported_files)} files")
        
        # Execute tools in parallel
        results = await self.parallel_executor.execute_tool_analysis(tool_tasks)
        
        # Filter successful results
        successful_results = [r for r in results if r and r.success]
        failed_results = [r for r in results if r and not r.success]
        
        if failed_results:
            logger.warning(f"{len(failed_results)} tools failed to complete analysis")
            for result in failed_results:
                logger.error(f"Tool {result.tool} failed: {result.error_message}")
        
        logger.info(f"Completed analysis with {len(successful_results)} successful tool results")
        return successful_results
    
    def get_analysis_summary(self, result: ComprehensiveAnalysisResult) -> Dict[str, Any]:
        """Get a summary of analysis results."""
        if not result.success:
            return {
                "status": "failed",
                "error": result.error_message,
                "execution_time": result.execution_time,
            }
        
        # Count issues by severity
        issue_counts = {"error": 0, "warning": 0, "info": 0}
        total_issues = 0
        tools_used = []
        
        for tool_result in result.results:
            tools_used.append(tool_result.tool)
            
            for issue in tool_result.issues:
                issue_counts[issue.severity] += 1
                total_issues += 1
        
        # Calculate file coverage
        files_with_issues = set()
        for tool_result in result.results:
            for issue in tool_result.issues:
                files_with_issues.add(issue.file)
        
        return {
            "status": "completed",
            "files_analyzed": result.files_analyzed,
            "files_with_issues": len(files_with_issues),
            "total_issues": total_issues,
            "issue_counts": issue_counts,
            "tools_used": tools_used,
            "execution_time": result.execution_time,
            "performance": {
                "files_per_second": result.files_analyzed / result.execution_time if result.execution_time > 0 else 0,
                "issues_per_file": total_issues / result.files_analyzed if result.files_analyzed > 0 else 0,
            }
        }
    
    async def shutdown(self):
        """Shutdown the orchestrator and clean up resources."""
        logger.info("Shutting down AnalysisOrchestrator")
        await self.parallel_executor.shutdown()