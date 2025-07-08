# Optimization Recommendations

## Algorithm Selection Matrix

### Selection by Use Case
- **Speed**: Use HBOS, ECOD, or EllipticEnvelope for operations requiring O(n) or O(n log n) complexity.
- **Accuracy**: Prefer Ensemble or AutoEncoder for best performance.
- **Scalability**: IsolationForest, HBOS, and ECOD handle large datasets well.

### Data Characteristics
- **Tabular Data**: 
  - Small: LOF or EllipticEnvelope.
  - Large: IsolationForest or HBOS.
- **Time Series**: LSTM AutoEncoder is ideal for most anomalies.

### Performance Matrix
- IsolationForest is ideal for high-dimensional and large datasets.
- LOF is suitable for small tabular datasets requiring interpretability.
- AutoEncoders are powerful for high-dimensional data.

### Decision Framework
Evaluate priorities such as execution speed, memory usage, interpretability, and fit with infrastructure to select the appropriate algorithm.


## Hyper-Parameter Tuning Tips

### General Guidelines
- Perform initial runs using default parameters.
- Apply grid or random search to fine-tune parameters progressively.

### Strategies
- **Grid Search**: Start with conservative search ranges and expand based on performance feedback.
- **Bayesian Optimization**: Use libraries like `Optuna` to efficiently search the parameter space with fewer evaluations.

### Tuning Examples
- **Isolation Forest**: Focus on tuning `n_estimators`, `contamination`, and `max_samples`.
- **AutoEncoder**: Adjust neural architecture and training parameters such as `epochs` and `learning_rate`.


## System-Level Tweaks

### CPU Pinning and NUMA Awareness
- Use tools like `taskset` and `numactl` to bind processes to specific CPUs and NUMA nodes for reduced latency and improved cache usage.

### Environmental Settings
- Set environment variables for parallel processing with tools such as OpenMP, MKL, and OpenBLAS:
  ```bash
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  export OPENBLAS_NUM_THREADS=8
  ```

### Memory Management
- Implement adaptive memory management techniques that align with the application's memory footprint, monitoring through adaptive memory managers as needed.


## Related Resources
- **[Performance Section](https://pynomaly.readthedocs.io/performance)**
- **[Algorithm Comparison](docs/reference/algorithms/algorithm-comparison.md)**
- **[Core Algorithms](docs/reference/algorithms/core-algorithms.md)**
- **[Advanced Python Performance](https://realpython.com/python-performance/)**
