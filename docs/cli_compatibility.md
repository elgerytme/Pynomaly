# CLI Compatibility Layer

## Overview

The Pynomaly CLI includes a compatibility layer (`src/pynomaly/presentation/cli/_compat.py`) that provides stable access to Typer command information across different versions of the Typer library. This ensures that the CLI continues to work correctly even when Typer's internal structure changes.

## Problem Statement

Typer's internal command storage mechanism has evolved across versions:
- **Typer >= 0.15.1**: Commands are stored in `registered_commands` as a list of `CommandInfo` objects
- **Older versions**: Commands might be stored in `commands` as a dictionary
- **Future versions**: The internal structure might change again

Direct access to these internal attributes would create brittle coupling and version incompatibility issues.

## Solution

The compatibility layer provides a consistent API that abstracts away version-specific differences:

```python
from pynomaly.presentation.cli._compat import (
    list_commands,
    get_command_names, 
    get_command,
    has_commands,
    count_commands
)
```

## API Reference

### `list_commands(group: typer.Typer) -> Optional[Dict[str, Any]]`

Returns a dictionary of commands from a Typer group, handling different internal structures:

```python
app = typer.Typer()

@app.command("hello")
def hello_cmd():
    pass

commands = list_commands(app)
# Returns: {"hello": CommandInfo(...)}
```

### `get_command_names(group: typer.Typer) -> list[str]`

Returns a list of command names:

```python
names = get_command_names(app)
# Returns: ["hello"]
```

### `get_command(group: typer.Typer, name: str) -> Any`

Gets a specific command by name:

```python
hello_cmd = get_command(app, "hello")
# Returns: CommandInfo object or None
```

### `has_commands(group: typer.Typer) -> bool`

Checks if a group has any registered commands:

```python
has_cmds = has_commands(app)
# Returns: True or False
```

### `count_commands(group: typer.Typer) -> int`

Counts the number of registered commands:

```python
count = count_commands(app)
# Returns: 1
```

## Version Compatibility

The compatibility layer handles these scenarios:

### Typer >= 0.15.1 (Current)
- `registered_commands` is a list of `CommandInfo` objects
- Converted to dictionary format: `{command.name: command}`

### Older Typer Versions
- `commands` attribute as dictionary
- Used directly when `registered_commands` is not available

### Future Versions
- Graceful fallback to empty dictionary when attributes are missing
- Extensible design allows easy addition of new version support

## Implementation Details

The core function `list_commands()` implements the version detection:

```python
def list_commands(group: typer.Typer) -> Optional[Dict[str, Any]]:
    # Try newer attribute first (typer >= 0.15.1)
    registered_commands = getattr(group, "registered_commands", None)
    
    if registered_commands is not None:
        if isinstance(registered_commands, list):
            # Convert list to dictionary using command names as keys
            commands_dict = {}
            for cmd_info in registered_commands:
                if hasattr(cmd_info, 'name') and cmd_info.name:
                    commands_dict[cmd_info.name] = cmd_info
            return commands_dict
        # ... handle other structures
    
    # Fall back to older attribute
    commands = getattr(group, "commands", {})
    return commands if commands is not None else {}
```

## Testing

The compatibility layer includes comprehensive unit tests covering:

- Different Typer versions and structures
- Edge cases (missing attributes, None values)
- Real Typer app integration
- Mock scenarios for version simulation

Run tests with:
```bash
python -m pytest tests/presentation/cli/test_compat.py -v
```

## Demo

A demonstration script is available at `examples/cli_compat_demo.py`:

```bash
python examples/cli_compat_demo.py
```

## Usage Guidelines

### DO ✅
- Use compatibility layer functions instead of direct attribute access
- Import from `pynomaly.presentation.cli._compat` or `pynomaly.presentation.cli`
- Extend the compatibility layer for new Typer versions as needed

### DON'T ❌
- Access `app.registered_commands` or `app.commands` directly
- Assume any specific internal structure of Typer objects
- Skip version compatibility testing when upgrading Typer

## Dependency Management

The project pins Typer to a compatible version range:

```toml
# pyproject.toml
cli = ["typer[all]>=0.15.1", "rich>=13.9.4"]
```

This ensures:
- Known compatible version baseline
- Protection against breaking changes
- Consistent behavior across environments

## Maintenance

When upgrading Typer:

1. **Test compatibility**: Run existing tests with new version
2. **Check internals**: Verify command storage structure hasn't changed
3. **Update layer**: Add support for new structures if needed
4. **Add tests**: Include test cases for new version behavior
5. **Update docs**: Document any new compatibility considerations

## Future Considerations

The compatibility layer is designed to be:
- **Extensible**: Easy to add support for new Typer versions
- **Backward compatible**: Won't break existing code
- **Performance conscious**: Minimal overhead over direct access
- **Type safe**: Provides proper type hints and validation

This design ensures the Pynomaly CLI remains stable and reliable across Typer version updates.
