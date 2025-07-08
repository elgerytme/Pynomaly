import json
from importlib import import_module

# Modules to test
modules_to_test = [
    'pynomaly.infrastructure.adapters.tensorflow_adapter',
    'pynomaly.infrastructure.adaptors.deep_learning.pytorch_adapter',
    'pynomaly.infrastructure.explainers.shap_explainer',
    'pynomaly.infrastructure.streaming.stream_processor',
    'pynomaly.infrastructure.automl.advanced_optimizer',
]

results = {}

for module_name in modules_to_test:
    try:
        module = import_module(module_name)
        # Check for a trivial method or attribute
        if hasattr(module, 'list_available_algorithms'):
            module.list_available_algorithms()
            results[module_name] = "implemented"
        else:
            # If there is no trivial method, mark as implemented but untested
            results[module_name] = "implemented (untested)"
    except ImportError as e:
        if "not available" in str(e):
            results[module_name] = "missing"
        else:
            results[module_name] = "stub"
    except NotImplementedError:
        results[module_name] = "stub"
    except Exception as e:
        results[module_name] = f"error: {str(e)}"

# Output results to JSON
with open('reports/feature_matrix.json', 'w') as f:
    json.dump(results, f, indent=4)

