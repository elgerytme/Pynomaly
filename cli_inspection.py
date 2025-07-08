#!/usr/bin/env python3
"""CLI Surface Enumeration Script for Pynomaly.

This script statically analyzes the Typer CLI application to extract
all commands and their options without requiring runtime imports.
"""

import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable objects safely."""
    
    def default(self, obj):
        if obj is ...:
            return "<ellipsis>"
        elif hasattr(obj, '__dict__'):
            return f"<object: {obj.__class__.__name__}>"
        else:
            return f"<non-serializable: {type(obj).__name__}>"
        return super().default(obj)


class TyperCommandExtractor(ast.NodeVisitor):
    """Extract Typer commands and options from AST."""
    
    def __init__(self):
        self.commands = []
        self.subapps = {}
        self.app_name = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to find Typer commands."""
        # Check if function has @app.command() decorator
        command_info = self._extract_command_info(node)
        if command_info:
            self.commands.append(command_info)
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit call expressions to find app.add_typer() calls."""
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == 'add_typer' and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'app'):
            
            # Extract subapp information
            subapp_info = self._extract_subapp_info(node)
            if subapp_info:
                self.subapps[subapp_info['name']] = subapp_info
        
        self.generic_visit(node)
    
    def _extract_command_info(self, node: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Extract command information from function node."""
        for decorator in node.decorator_list:
            if self._is_command_decorator(decorator):
                command_name = self._get_command_name(decorator, node.name)
                options = self._extract_function_parameters(node)
                docstring = ast.get_docstring(node) or ""
                
                return {
                    "command": command_name,
                    "function": node.name,
                    "docstring": docstring.split('\n')[0] if docstring else "",
                    "options": options
                }
        return None
    
    def _is_command_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is a Typer command decorator."""
        if isinstance(decorator, ast.Call):
            if (isinstance(decorator.func, ast.Attribute) and
                decorator.func.attr == 'command'):
                return True
        elif isinstance(decorator, ast.Attribute):
            if decorator.attr == 'command':
                return True
        return False
    
    def _get_command_name(self, decorator: ast.expr, func_name: str) -> str:
        """Extract command name from decorator or use function name."""
        if isinstance(decorator, ast.Call) and decorator.args:
            if isinstance(decorator.args[0], ast.Constant):
                return decorator.args[0].value
        return func_name.replace('_', '-')
    
    def _extract_function_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters and their Typer option information."""
        options = []
        
        # Map argument positions to default values
        num_args = len(node.args.args)
        num_defaults = len(node.args.defaults)
        default_offset = num_args - num_defaults
        
        for i, arg in enumerate(node.args.args):
            if arg.arg == 'self':
                continue
                
            param_info = {
                "name": arg.arg,
                "type": self._get_type_annotation(arg),
                "default": None,
                "help": "",
                "option_names": [],
                "is_argument": False,
                "is_option": False,
                "required": True
            }
            
            # Extract default value and Typer information
            if i >= default_offset:
                default_index = i - default_offset
                if default_index < len(node.args.defaults):
                    default_value = node.args.defaults[default_index]
                    typer_info = self._parse_typer_default(default_value)
                    if typer_info:
                        param_info.update(typer_info)
                    else:
                        param_info["default"] = self._ast_to_value(default_value)
                        param_info["required"] = False
            
            options.append(param_info)
        
        return options
    
    def _get_type_annotation(self, arg: ast.arg) -> str:
        """Get type annotation as string."""
        if arg.annotation:
            return ast.unparse(arg.annotation)
        return "Any"
    
    def _parse_typer_default(self, default_node: ast.expr) -> Optional[Dict[str, Any]]:
        """Parse typer.Option() or typer.Argument() calls."""
        if not isinstance(default_node, ast.Call):
            return None
        
        # Check if this is a typer.Option or typer.Argument call
        func_name = self._get_call_name(default_node)
        if func_name not in ['typer.Option', 'typer.Argument', 'Option', 'Argument']:
            return None
        
        info = {
            "is_option": func_name in ['typer.Option', 'Option'],
            "is_argument": func_name in ['typer.Argument', 'Argument'],
            "required": True,
            "option_names": [],
            "help": "",
            "default": None
        }
        
        # Parse positional arguments (default value, option names)
        if default_node.args:
            if func_name in ['typer.Option', 'Option']:
                # First arg might be default value or option names
                first_arg = default_node.args[0]
                if isinstance(first_arg, ast.Constant) and not isinstance(first_arg.value, str):
                    info["default"] = first_arg.value
                    info["required"] = False
                elif isinstance(first_arg, ast.Constant) and first_arg.value.startswith('-'):
                    info["option_names"].append(first_arg.value)
                
                # Additional args are usually option names
                for arg in default_node.args[1:]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        info["option_names"].append(arg.value)
            else:  # Argument
                # First arg is usually the default or ellipsis
                first_arg = default_node.args[0]
                if isinstance(first_arg, ast.Constant):
                    if first_arg.value is ...:
                        info["default"] = "<ellipsis>"
                        info["required"] = True
                    else:
                        info["default"] = first_arg.value
                        info["required"] = False
                elif isinstance(first_arg, ast.Ellipsis):
                    info["default"] = "<ellipsis>"
                    info["required"] = True
        
        # Parse keyword arguments
        for keyword in default_node.keywords:
            if keyword.arg == "help" and isinstance(keyword.value, ast.Constant):
                info["help"] = keyword.value.value
            elif keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                info["default"] = keyword.value.value
                info["required"] = False
        
        return info
    
    def _get_call_name(self, call_node: ast.Call) -> str:
        """Get the name of a function call."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name):
                return f"{call_node.func.value.id}.{call_node.func.attr}"
            else:
                return call_node.func.attr
        return "<unknown>"
    
    def _ast_to_value(self, node: ast.expr) -> Any:
        """Convert simple AST nodes to Python values."""
        if isinstance(node, ast.Constant):
            # Handle ellipsis specially for JSON serialization
            if node.value is ...:
                return "<ellipsis>"
            return node.value
        elif isinstance(node, ast.Ellipsis):
            return "<ellipsis>"
        elif isinstance(node, ast.NameConstant):  # For older Python versions
            return node.value
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        elif isinstance(node, ast.Str):  # For older Python versions
            return node.s
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {self._ast_to_value(k): self._ast_to_value(v) for k, v in zip(node.keys, node.values)}
        else:
            return f"<complex: {ast.unparse(node)}>"
    
    def _extract_default_info(self, node: ast.FunctionDef, param_name: str) -> Optional[Dict[str, Any]]:
        """Extract default value and Typer option information."""
        # This method is now replaced by _parse_typer_default
        # Keeping for compatibility
        defaults = node.args.defaults
        if defaults:
            return {
                "default": "<complex>",
                "is_option": True
            }
        return None
    
    def _extract_subapp_info(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """Extract subapp information from app.add_typer() call."""
        if len(node.args) < 1:
            return None
        
        # Extract the subapp module/object
        subapp_ref = None
        if isinstance(node.args[0], ast.Attribute):
            subapp_ref = ast.unparse(node.args[0])
        
        # Extract name from keyword arguments
        name = None
        help_text = ""
        
        for keyword in node.keywords:
            if keyword.arg == 'name' and isinstance(keyword.value, ast.Constant):
                name = keyword.value.value
            elif keyword.arg == 'help' and isinstance(keyword.value, ast.Constant):
                help_text = keyword.value.value
        
        if name and subapp_ref:
            return {
                "name": name,
                "module": subapp_ref,
                "help": help_text,
                "commands": []  # Will be populated by analyzing the submodule
            }
        
        return None


def analyze_cli_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a single CLI Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        extractor = TyperCommandExtractor()
        extractor.visit(tree)
        
        return {
            "file": str(file_path.relative_to(Path.cwd())),
            "commands": extractor.commands,
            "subapps": extractor.subapps
        }
    except Exception as e:
        return {
            "file": str(file_path),
            "error": str(e),
            "commands": [],
            "subapps": {}
        }


def find_cli_files(root_path: Path) -> List[Path]:
    """Find all CLI-related Python files."""
    cli_dir = root_path / "src" / "pynomaly" / "presentation" / "cli"
    if not cli_dir.exists():
        raise FileNotFoundError(f"CLI directory not found: {cli_dir}")
    
    cli_files = []
    for file_path in cli_dir.rglob("*.py"):
        if file_path.name != "__init__.py":
            cli_files.append(file_path)
    
    return sorted(cli_files)


def main():
    """Main function to enumerate CLI surface."""
    print("🔍 Enumerating Pynomaly CLI surface...")
    
    try:
        root_path = Path.cwd()
        cli_files = find_cli_files(root_path)
        
        print(f"Found {len(cli_files)} CLI files to analyze")
        
        cli_surface = {
            "metadata": {
                "tool": "pynomaly-cli-inspector",
                "version": "1.0.0",
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "total_files": len(cli_files)
            },
            "files": [],
            "summary": {
                "total_commands": 0,
                "total_subapps": 0,
                "total_options": 0
            }
        }
        
        total_commands = 0
        total_subapps = 0
        total_options = 0
        
        for cli_file in cli_files:
            print(f"Analyzing: {cli_file.name}")
            file_analysis = analyze_cli_file(cli_file)
            cli_surface["files"].append(file_analysis)
            
            # Update totals
            total_commands += len(file_analysis.get("commands", []))
            total_subapps += len(file_analysis.get("subapps", {}))
            
            for command in file_analysis.get("commands", []):
                total_options += len(command.get("options", []))
        
        # Update summary
        cli_surface["summary"].update({
            "total_commands": total_commands,
            "total_subapps": total_subapps,
            "total_options": total_options
        })
        
        # Write to JSON file
        output_file = "cli_surface.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cli_surface, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
        
        print(f"\n✅ CLI surface analysis complete!")
        print(f"📊 Summary:")
        print(f"   - Total commands: {total_commands}")
        print(f"   - Total subapps: {total_subapps}")
        print(f"   - Total options: {total_options}")
        print(f"   - Output file: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()

