# Quantum Clustering using VQE and Affinity Propagation

A hybrid quantum-classical machine learning pipeline that performs clustering using Variational Quantum Eigensolver (VQE) to compute similarity matrices, followed by Affinity Propagation clustering.

## Overview

This project implements a novel approach to clustering by:
1. Computing pairwise similarities using quantum circuits
2. Optimizing quantum parameters with hybrid classical-quantum algorithms
3. Using the quantum-computed similarity matrix for classical clustering
4. Leveraging parallel processing for scalability

## Features

- **Quantum Similarity Computation**: Uses parameterized quantum circuits to compute pairwise similarities
- **Hybrid Optimization**: Combines SPSA and COBYLA optimizers with Optuna hyperparameter tuning
- **Parallel Processing**: Multi-core execution for faster VQE evaluations
- **Robust Pipeline**: Error handling and fallback mechanisms
- **Visualization**: PCA-based 2D visualization of clustering results

## Requirements

```
numpy
matplotlib
scikit-learn
qiskit
scipy
optuna
multiprocessing (built-in)
```

## Installation

```bash
pip install numpy matplotlib scikit-learn qiskit scipy optuna
```

## Usage

### Basic Usage

```python
python quantum_clustering.py
```

The script will automatically:
1. Load and preprocess the Iris dataset
2. Build quantum Hamiltonians for pairwise similarities
3. Run VQE optimization in parallel
4. Perform Affinity Propagation clustering
5. Display results with PCA visualization

### Key Components

#### 1. Data Preprocessing
```python
X_scaled = load_and_scale_data()
```
- Loads Iris dataset
- Applies StandardScaler normalization

#### 2. Quantum Circuit Design
```python
circuit, theta = create_full_ansatz()
```
- Creates a 2-qubit parameterized quantum circuit
- Uses RX, RY, RZ rotation gates + CNOT entanglement
- 12 trainable parameters total

#### 3. VQE Optimization
```python
energy = run_vqe_with_optuna_cached(hamiltonian, circuit, theta, n_trials=5)
```
- Optuna-based hyperparameter optimization
- Choice between SPSA and COBYLA optimizers
- Automatic parameter tuning for each optimizer

#### 4. Parallel Processing
```python
with Pool(processes=min(cpu_count(), 8)) as pool:
    results = pool.map(worker, hamiltonians)
```
- Distributes VQE computations across CPU cores
- Processes multiple qubit pairs simultaneously

## Algorithm Details

### Quantum Ansatz
The quantum circuit uses a layered ansatz with:
- Initial rotation layer: RX(θ₀), RY(θ₁), RZ(θ₂) on each qubit
- Entanglement: CNOT gate between qubits
- Final rotation layer: RX(θ₆), RY(θ₇), RZ(θ₈) on each qubit

### Similarity Computation
1. Classical similarity: `S_ij = -||x_i - x_j||²`
2. Quantum Hamiltonian: `H_ij = -S_ij * ZZ`
3. VQE energy: `E_ij = ⟨ψ(θ)|H_ij|ψ(θ)⟩`

### Optimization Strategies
- **SPSA**: Simultaneous Perturbation Stochastic Approximation
  - Good for noisy optimization landscapes
  - Parameters: learning rates `a`, `c`, max iterations
- **COBYLA**: Constrained Optimization BY Linear Approximation
  - Gradient-free method
  - Good for smooth landscapes

## Configuration Options

### Dataset Size
```python
X_scaled = X_scaled[:30]  # Reduce for faster demo
```
Adjust the slice to control dataset size vs. computation time.

### VQE Trials
```python
energy = run_vqe_with_optuna_cached(hamiltonian, circuit, theta, n_trials=5)
```
Increase `n_trials` for better optimization (slower execution).

### Parallel Workers
```python
with Pool(processes=min(cpu_count(), 8)) as pool:
```
Adjust the maximum number of processes based on your system.

## Performance Notes

- **Computation Time**: O(n²) pairwise evaluations where n is dataset size
- **Memory Usage**: Stores full n×n similarity matrices
- **Scalability**: Parallel processing scales with available CPU cores
- **Quantum Simulation**: Uses Qiskit's statevector simulator (classical simulation)

## Output

The script produces:
1. **Console Output**: Progress updates and timing information
2. **Cluster Labels**: Array of cluster assignments for each data point
3. **Visualization**: 2D scatter plot with cluster colors
4. **Statistics**: Number of clusters found

Example output:
```
[INFO] Data loaded and standardized.
[INFO] Similarity matrix computed.
[INFO] 435 pairwise Hamiltonians created.
[INFO] Using 8 CPU cores with multiprocessing...
[INFO] VQE evaluation completed in 45.23 seconds.
[INFO] Running Affinity Propagation clustering...
Cluster labels: [0 0 0 1 1 1 2 2 2 ...]
Number of clusters: 3
```

## Customization

### Different Datasets
Replace the data loading section:
```python
def load_and_scale_data():
    # Load your custom dataset here
    X = your_data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
```

### Modified Quantum Circuit
Customize the ansatz in `create_full_ansatz()`:
```python
def create_full_ansatz():
    theta = ParameterVector("θ", num_params)
    qc = QuantumCircuit(num_qubits)
    # Add your custom gates here
    return qc, theta
```

### Alternative Clustering
Replace Affinity Propagation with other algorithms:
```python
from sklearn.cluster import KMeans, DBSCAN
clusterer = KMeans(n_clusters=3)
labels = clusterer.fit_predict(similarity_for_affinity)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce dataset size or increase system RAM
2. **Slow Execution**: Decrease `n_trials` or dataset size
3. **Import Errors**: Ensure all dependencies are installed
4. **Convergence Issues**: Try different optimizer parameters

### Performance Optimization

- Use GPU-accelerated Qiskit backends for larger problems
- Implement quantum circuit compilation for real quantum hardware
- Cache intermediate results to avoid recomputation
- Use approximate similarity computations for very large datasets

## References

- Variational Quantum Eigensolver (VQE): Peruzzo et al., Nature Communications 5, 4213 (2014)
- Affinity Propagation: Frey & Dueck, "Clustering by passing messages between data points," Science 315, 972-976 (2007)
- SPSA Optimization: Spall, IEEE Transactions on Automatic Control 37, 332-341 (1992)

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to fork, modify, and submit pull requests. Areas for improvement:
- Real quantum hardware integration
- Advanced quantum ansatz designs
- Alternative clustering algorithms
- Performance optimizations
- Extended benchmarking studies
