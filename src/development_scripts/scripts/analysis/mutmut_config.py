"""Configuration for mutation testing with mutmut."""


def pre_mutation(context):
    """Run before each mutation."""
    pass


def post_mutation(context, mutation_outcome):
    """Run after each mutation."""
    pass


# Mutmut configuration
paths_to_mutate = [
    "src/anomaly_detection/domain/",
    "src/anomaly_detection/application/",
]

# Exclude test files and external libraries
paths_to_exclude = [
    "tests/",
    "src/anomaly_detection/infrastructure/adapters/",  # External library wrappers
    "src/anomaly_detection/presentation/",  # UI layer mutations less critical
]

# Test command to run after each mutation
test_command = "python -m pytest tests/mutation/ -x -q"

# Minimum mutation score threshold
minimum_mutation_score = 70

# Focus on critical business logic
focus_modules = [
    "src/anomaly_detection/domain/value_objects/",
    "src/anomaly_detection/domain/entities/",
    "src/anomaly_detection/domain/services/",
    "src/anomaly_detection/application/use_cases/",
    "src/anomaly_detection/application/services/",
]
