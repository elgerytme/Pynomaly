"""Import tracer to debug circular import issues."""

import importlib.util
import logging
import sys
from types import ModuleType
from typing import Any, Dict, Optional


class ImportTracer:
    """Traces module imports to help identify circular import patterns."""
    
    def __init__(self, enabled: bool = True):
        """Initialize the import tracer.
        
        Args:
            enabled: Whether to enable tracing
        """
        self.enabled = enabled
        self.import_stack: list[str] = []
        self.import_count: Dict[str, int] = {}
        self.circular_imports: list[tuple[str, ...]] = []
        self.logger = logging.getLogger("pynomaly.imports")
        
        if self.enabled:
            self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup import logging."""
        # Set up import logger
        import_logger = logging.getLogger("importlib")
        import_logger.setLevel(logging.DEBUG)
        
        # Create handler if not exists
        if not import_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            import_logger.addHandler(handler)
        
        # Hook into import machinery
        self._original_import = __builtins__.__import__
        __builtins__.__import__ = self._traced_import
    
    def _traced_import(
        self, 
        name: str, 
        globals: Optional[Dict[str, Any]] = None, 
        locals: Optional[Dict[str, Any]] = None, 
        fromlist: tuple = (), 
        level: int = 0
    ) -> ModuleType:
        """Traced import function."""
        # Check for circular imports
        if name in self.import_stack:
            circular_path = tuple(self.import_stack[self.import_stack.index(name):] + [name])
            if circular_path not in self.circular_imports:
                self.circular_imports.append(circular_path)
                self.logger.warning(f"Circular import detected: {' -> '.join(circular_path)}")
        
        # Track import
        self.import_stack.append(name)
        self.import_count[name] = self.import_count.get(name, 0) + 1
        
        try:
            # Log import attempt
            if name.startswith('pynomaly'):
                caller_module = globals.get('__name__', 'unknown') if globals else 'unknown'
                self.logger.debug(f"Importing {name} from {caller_module}")
                
                # Show import stack for pynomaly modules
                if len(self.import_stack) > 1:
                    stack_str = ' -> '.join(self.import_stack[-5:])  # Last 5 imports
                    self.logger.debug(f"Import stack: {stack_str}")
            
            # Perform actual import
            result = self._original_import(name, globals, locals, fromlist, level)
            return result
            
        finally:
            # Remove from stack
            if self.import_stack and self.import_stack[-1] == name:
                self.import_stack.pop()
    
    def report_circular_imports(self) -> None:
        """Report detected circular imports."""
        if self.circular_imports:
            self.logger.error(f"Found {len(self.circular_imports)} circular import patterns:")
            for i, circular_path in enumerate(self.circular_imports, 1):
                self.logger.error(f"  {i}. {' -> '.join(circular_path)}")
        else:
            self.logger.info("No circular imports detected")
    
    def report_import_stats(self) -> None:
        """Report import statistics."""
        pynomaly_imports = {k: v for k, v in self.import_count.items() if k.startswith('pynomaly')}
        if pynomaly_imports:
            self.logger.info("Pynomaly module import counts:")
            for module, count in sorted(pynomaly_imports.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {module}: {count}")
    
    def cleanup(self) -> None:
        """Cleanup tracer and restore original import."""
        if self.enabled and hasattr(self, '_original_import'):
            __builtins__.__import__ = self._original_import


# Global tracer instance
_tracer: Optional[ImportTracer] = None


def enable_import_tracing() -> ImportTracer:
    """Enable import tracing globally.
    
    Returns:
        The import tracer instance
    """
    global _tracer
    if _tracer is None:
        _tracer = ImportTracer(enabled=True)
    return _tracer


def disable_import_tracing() -> None:
    """Disable import tracing globally."""
    global _tracer
    if _tracer:
        _tracer.cleanup()
        _tracer = None


def get_import_tracer() -> Optional[ImportTracer]:
    """Get the current import tracer instance.
    
    Returns:
        The tracer instance if enabled, None otherwise
    """
    return _tracer
