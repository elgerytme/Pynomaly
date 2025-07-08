"""
Compatibility layer for Typer internal attribute access.

This module provides a thin compatibility layer to handle changes in how Typer
exposes command information internally, particularly the transition from
'registered_commands' to 'commands' attribute in different versions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import typer


def list_commands(group: typer.Typer) -> Optional[Dict[str, Any]]:
    """
    Get commands from a Typer group with version compatibility.
    
    This function handles the transition between different Typer versions
    where command storage might use different attribute names and structures.
    
    Args:
        group: The Typer group to extract commands from
        
    Returns:
        Dictionary of commands if available, None otherwise
        
    Examples:
        >>> app = typer.Typer()
        >>> @app.command()
        ... def hello():
        ...     pass
        >>> commands = list_commands(app)
        >>> commands is not None
        True
    """
    # Try newer attribute first (typer >= 0.15.1)
    registered_commands = getattr(group, "registered_commands", None)
    
    if registered_commands is not None:
        # In Typer >= 0.15.1, registered_commands is a list of CommandInfo objects
        if isinstance(registered_commands, list):
            # Convert list to dictionary using command names as keys
            commands_dict = {}
            for cmd_info in registered_commands:
                if hasattr(cmd_info, 'name') and cmd_info.name:
                    commands_dict[cmd_info.name] = cmd_info
            return commands_dict
        elif isinstance(registered_commands, dict):
            # If it's already a dict, return as is
            return registered_commands
        else:
            # Return the raw structure if it's neither list nor dict
            return registered_commands
    
    # Fall back to older attribute if newer one doesn't exist
    commands = getattr(group, "commands", {})
    return commands if commands is not None else {}


def get_command_names(group: typer.Typer) -> list[str]:
    """
    Get list of command names from a Typer group.
    
    Args:
        group: The Typer group to extract command names from
        
    Returns:
        List of command names
        
    Examples:
        >>> app = typer.Typer()
        >>> @app.command("hello")
        ... def hello_cmd():
        ...     pass
        >>> names = get_command_names(app)
        >>> "hello" in names
        True
    """
    commands = list_commands(group)
    if commands is None:
        return []
    
    if isinstance(commands, dict):
        return list(commands.keys())
    elif hasattr(commands, 'keys'):
        return list(commands.keys())
    else:
        # Handle case where commands might be a list or other structure
        return []


def get_command(group: typer.Typer, name: str) -> Any:
    """
    Get a specific command from a Typer group by name.
    
    Args:
        group: The Typer group to search in
        name: Name of the command to retrieve
        
    Returns:
        The command object if found, None otherwise
        
    Examples:
        >>> app = typer.Typer()
        >>> @app.command("hello")
        ... def hello_cmd():
        ...     pass
        >>> cmd = get_command(app, "hello")
        >>> cmd is not None
        True
    """
    commands = list_commands(group)
    if commands is None:
        return None
    
    if isinstance(commands, dict):
        return commands.get(name)
    elif hasattr(commands, 'get'):
        return commands.get(name)
    else:
        return None


def has_commands(group: typer.Typer) -> bool:
    """
    Check if a Typer group has any registered commands.
    
    Args:
        group: The Typer group to check
        
    Returns:
        True if the group has commands, False otherwise
        
    Examples:
        >>> app = typer.Typer()
        >>> has_commands(app)
        False
        >>> @app.command()
        ... def hello():
        ...     pass
        >>> has_commands(app)
        True
    """
    commands = list_commands(group)
    if commands is None:
        return False
    
    if isinstance(commands, dict):
        return len(commands) > 0
    elif hasattr(commands, '__len__'):
        return len(commands) > 0
    else:
        return False


def count_commands(group: typer.Typer) -> int:
    """
    Count the number of registered commands in a Typer group.
    
    Args:
        group: The Typer group to count commands for
        
    Returns:
        Number of registered commands
        
    Examples:
        >>> app = typer.Typer()
        >>> count_commands(app)
        0
        >>> @app.command()
        ... def hello():
        ...     pass
        >>> count_commands(app)
        1
    """
    commands = list_commands(group)
    if commands is None:
        return 0
    
    if isinstance(commands, dict):
        return len(commands)
    elif hasattr(commands, '__len__'):
        return len(commands)
    else:
        return 0
