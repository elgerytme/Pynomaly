#!/usr/bin/env python3
"""
Script to check which algorithms in ALGORITHM_MAPPING actually work in PyOD 2.0.5
"""

import importlib
import sys
import traceback

# Copy the mapping from the adapter
ALGORITHM_MAPPING = {
    # Linear models
    "PCA": ("pyod.models.pca", "PCA"),
    "MCD": ("pyod.models.mcd", "MCD"),
    "OCSVM": ("pyod.models.ocsvm", "OCSVM"),
    "LMDD": ("pyod.models.lmdd", "LMDD"),
    # Proximity-based
    "LOF": ("pyod.models.lof", "LOF"),
    "COF": ("pyod.models.cof", "COF"),
    "CBLOF": ("pyod.models.cblof", "CBLOF"),
    "LOCI": ("pyod.models.loci", "LOCI"),
    "HBOS": ("pyod.models.hbos", "HBOS"),
    "KNN": ("pyod.models.knn", "KNN"),
    "AvgKNN": ("pyod.models.knn", "KNN"),
    "MedKNN": ("pyod.models.knn", "KNN"),
    "SOD": ("pyod.models.sod", "SOD"),
    "ROD": ("pyod.models.rod", "ROD"),
    # Probabilistic
    "ABOD": ("pyod.models.abod", "ABOD"),
    "FastABOD": ("pyod.models.abod", "FastABOD"),
    "COPOD": ("pyod.models.copod", "COPOD"),
    "MAD": ("pyod.models.mad", "MAD"),
    "SOS": ("pyod.models.sos", "SOS"),
    # Ensemble
    "IsolationForest": ("pyod.models.iforest", "IForest"),
    "IForest": ("pyod.models.iforest", "IForest"),
    "FeatureBagging": ("pyod.models.feature_bagging", "FeatureBagging"),
    "LSCP": ("pyod.models.lscp", "LSCP"),
    "XGBOD": ("pyod.models.xgbod", "XGBOD"),
    "LODA": ("pyod.models.loda", "LODA"),
    "SUOD": ("pyod.models.suod", "SUOD"),
    # Neural networks
    "AutoEncoder": ("pyod.models.auto_encoder", "AutoEncoder"),
    "VAE": ("pyod.models.vae", "VAE"),
    "Beta-VAE": ("pyod.models.vae", "BetaVAE"),
    "SO_GAAL": ("pyod.models.so_gaal", "SO_GAAL"),
    "MO_GAAL": ("pyod.models.mo_gaal", "MO_GAAL"),
    "DeepSVDD": ("pyod.models.deep_svdd", "DeepSVDD"),
    # Graph-based
    "R-Graph": ("pyod.models.rgraph", "RGraph"),
    "LUNAR": ("pyod.models.lunar", "LUNAR"),
    # Deep Learning (Additional)
    "ALAD": ("pyod.models.alad", "ALAD"),
    "AnoGAN": ("pyod.models.anogan", "AnoGAN"),
    "DIF": ("pyod.models.dif", "DIF"),
    # Statistical/Other
    "CLF": ("pyod.models.clf", "CLF"),
    "KPCA": ("pyod.models.kpca", "KPCA"),
    "PCA-MAD": ("pyod.models.pca", "PCA"),  # With MAD option
    "QMCD": ("pyod.models.qmcd", "QMCD"),
    # Other
    "INNE": ("pyod.models.inne", "INNE"),
    "ECOD": ("pyod.models.ecod", "ECOD"),
    "CD": ("pyod.models.cd", "CD"),
    "KDE": ("pyod.models.kde", "KDE"),
    "Sampling": ("pyod.models.sampling", "Sampling"),
    "GMM": ("pyod.models.gmm", "GMM"),
}

def check_algorithm(algorithm_name, module_path, class_name):
    """Check if an algorithm can be imported and instantiated."""
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Try to instantiate with minimal parameters
        try:
            # Most PyOD models accept contamination parameter
            model = model_class(contamination=0.1)
            return True, "OK"
        except Exception as e:
            # If instantiation fails, it might still be importable
            return True, f"Import OK, but instantiation failed: {str(e)}"
    except ImportError as e:
        return False, f"ImportError: {str(e)}"
    except AttributeError as e:
        return False, f"AttributeError: {str(e)}"
    except Exception as e:
        return False, f"Other error: {str(e)}"

def main():
    """Check all algorithms in the mapping."""
    print("Checking PyOD algorithms in ALGORITHM_MAPPING...")
    print("=" * 60)
    
    working_algorithms = []
    failed_algorithms = []
    
    for algorithm_name, (module_path, class_name) in ALGORITHM_MAPPING.items():
        print(f"Testing {algorithm_name:15} ({module_path}.{class_name})")
        success, message = check_algorithm(algorithm_name, module_path, class_name)
        
        if success:
            print(f"  ✓ {message}")
            working_algorithms.append(algorithm_name)
        else:
            print(f"  ✗ {message}")
            failed_algorithms.append((algorithm_name, module_path, class_name, message))
    
    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"Working algorithms: {len(working_algorithms)}")
    print(f"Failed algorithms: {len(failed_algorithms)}")
    
    if failed_algorithms:
        print("\nFAILED ALGORITHMS:")
        for algo, module, cls, error in failed_algorithms:
            print(f"  - {algo} ({module}.{cls}): {error}")
    
    if working_algorithms:
        print(f"\nWORKING ALGORITHMS:")
        for algo in working_algorithms:
            print(f"  - {algo}")
    
    return failed_algorithms

if __name__ == "__main__":
    failed = main()
    sys.exit(1 if failed else 0)
