"""Configuration for mutation testing with mutmut."""


def pre_mutation(context):
    """Run before each mutation."""
    pass


def post_mutation(context, mutation_outcome):
    """Run after each mutation."""
    pass


# Mutmut configuration
paths_to_mutate = [
    "src/pynomaly/domain/",
    "src/pynomaly/application/",
]

# Exclude test files and external libraries
paths_to_exclude = [
    "tests/",
    "src/pynomaly/infrastructure/adapters/",  # External library wrappers
    "src/pynomaly/presentation/",  # UI layer mutations less critical
]

# Test command to run after each mutation
test_command = "python -m pytest tests/mutation/ -x -q"

# Minimum mutation score threshold
minimum_mutation_score = 70

# Focus on critical business logic
focus_modules = [
    "src/pynomaly/domain/value_objects/",
    "src/pynomaly/domain/entities/",
    "src/pynomaly/domain/services/",
    "src/pynomaly/application/use_cases/",
    "src/pynomaly/application/services/",
]
