import os

CONFIG_FILES = [
    ".pre-commit-config.yaml",
    "config/pytest.ini",
    "config/tox.ini",
    "config/README.md",
    "config/mutmut.toml",
]


DUPLICATES = {
    "config/pytest.ini": ["pytest.ini", "tests/pytest.ini", "tests/integration/pytest.ini"],
    ".pre-commit-config.yaml": ["config/.pre-commit-config.yaml", "config/git/.pre-commit-config.yaml"],
    "config/tox.ini": ["tox.ini"],
    "config/mutmut.toml": [".mutmut.toml"]
}


def assert_no_duplicates():
    messages = []
    for file, duplicates in DUPLICATES.items():
        for duplicate in duplicates:
            if os.path.exists(duplicate):
                messages.append(f"Duplicate found: {duplicate} should not exist when {file} is present.")

    if messages:
        for message in messages:
            print(message)
        raise AssertionError("Duplicate configuration files found!")
    else:
        print("No duplicate configuration files detected.")


if __name__ == "__main__":
    assert_no_duplicates()

