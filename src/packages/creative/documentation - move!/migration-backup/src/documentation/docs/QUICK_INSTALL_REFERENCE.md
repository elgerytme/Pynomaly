# Pynomaly Quick Install Reference

## Most Common Installations

### 🎯 CLI User
```bash
pip install pynomaly[cli,minimal]
```

### 🧪 Data Scientist
```bash
pip install pynomaly[cli,ml,explainability]
```

### 🚀 Production Server
```bash
pip install pynomaly[production]
```

### 👨‍💻 Developer
```bash
pip install -e .[dev,test,lint]
```

## Fix Common Test Errors

| Error Message | Solution |
|---------------|----------|
| `please install torch first` | `pip install pynomaly[torch]` |
| `SHAP not available` | `pip install pynomaly[explainability]` |
| `LIME not available` | `pip install pynomaly[explainability]` |
| `Optuna not available` | `pip install pynomaly[automl]` |
| `JAX is required` | `pip install pynomaly[jax]` |

## Feature Extras

| Feature | Command |
|---------|---------|
| Deep Learning (All) | `pip install pynomaly[ml-all]` |
| PyTorch Only | `pip install pynomaly[torch]` |
| TensorFlow Only | `pip install pynomaly[tensorflow]` |
| JAX Only | `pip install pynomaly[jax]` |
| AutoML | `pip install pynomaly[automl]` |
| Explainable AI | `pip install pynomaly[explainability]` |
| Graph Detection | `pip install pynomaly[graph]` |
| All Features | `pip install pynomaly[all]` |

## Quick Verification

```bash
# Test basic functionality
python -c "import pynomaly; print('✓ Pynomaly installed')"

# Test CLI
pynomaly --help

# Test features
python -c "
features = []
try: import torch; features.append('PyTorch')
except: pass
try: import shap; features.append('SHAP')  
except: pass
try: import lime; features.append('LIME')
except: pass
try: import optuna; features.append('Optuna')
except: pass
print('Available features:', ', '.join(features) if features else 'None')
"
```

## Troubleshooting

### Fresh Install
```bash
python -m venv fresh-env
source fresh-env/bin/activate  # Windows: fresh-env\Scripts\activate
pip install pynomaly[cli,ml]
```

### Memory Issues
```bash
pip install --no-cache-dir pynomaly[minimal]
```

### Permission Issues
```bash
pip install --user pynomaly[cli,ml]
```