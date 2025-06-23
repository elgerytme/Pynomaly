"""Infrastructure explainers package."""

# Optional explainer imports
try:
    from .shap_explainer import SHAPExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAPExplainer = None
    SHAP_AVAILABLE = False

try:
    from .lime_explainer import LIMEExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIMEExplainer = None
    LIME_AVAILABLE = False

__all__ = [
    "SHAPExplainer",
    "LIMEExplainer", 
    "SHAP_AVAILABLE",
    "LIME_AVAILABLE",
]