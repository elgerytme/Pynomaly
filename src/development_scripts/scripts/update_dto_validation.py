"""Script to update all DTOs with strict validation (extra='forbid')."""

import re
from pathlib import Path


def update_dto_files():
    """Update all DTO files to enable strict validation."""
    # Find all DTO files
    dto_dir = Path("src/pynomaly/application/dto")
    dto_files = list(dto_dir.glob("*.py"))

    updated_files = []

    for file_path in dto_files:
        if file_path.name == "__init__.py":
            continue

        print(f"Processing {file_path}")

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Pattern 1: ConfigDict with from_attributes but no extra
        pattern1 = r"model_config = ConfigDict\(from_attributes=True\)"
        replacement1 = 'model_config = ConfigDict(from_attributes=True, extra="forbid")'
        content = re.sub(pattern1, replacement1, content)

        # Pattern 2: ConfigDict with from_attributes=True, and other parameters
        pattern2 = r"model_config = ConfigDict\(\s*from_attributes=True,([^}]+)\)"

        def replacement2(match):
            other_params = match.group(1)
            if "extra=" not in other_params:
                return f'model_config = ConfigDict(from_attributes=True, extra="forbid",{other_params})'
            return match.group(0)

        content = re.sub(pattern2, replacement2, content)

        # Pattern 3: ConfigDict with other parameters but no extra
        pattern3 = r"model_config = ConfigDict\(([^)]+)\)"

        def replacement3(match):
            params = match.group(1)
            if "extra=" not in params and "from_attributes" not in params:
                return f'model_config = ConfigDict(extra="forbid", {params})'
            return match.group(0)

        content = re.sub(pattern3, replacement3, content)

        # Pattern 4: Add model_config to classes that don't have it
        class_pattern = (
            r'class (\w+DTO)\(BaseModel\):\s*\n\s*"""([^"]+)"""\s*\n\s*([^m])'
        )

        def add_model_config(match):
            class_name = match.group(1)
            docstring = match.group(2)
            next_line = match.group(3)
            return f'class {class_name}(BaseModel):\n    """{docstring}"""\n    \n    model_config = ConfigDict(extra="forbid")\n    {next_line}'

        content = re.sub(class_pattern, add_model_config, content)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            updated_files.append(file_path)
            print(f"  ✓ Updated {file_path}")
        else:
            print(f"  - No changes needed for {file_path}")

    print(f"\nUpdated {len(updated_files)} files:")
    for file_path in updated_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    update_dto_files()
