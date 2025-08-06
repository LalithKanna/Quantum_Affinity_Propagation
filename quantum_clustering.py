import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
import optuna
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# === 1. LOAD & SCALE DATA ===
def load_and_scale_data():
    data = load_iris()
    X = data.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[INFO] Data loaded and standardized.")
    return X_scaled

# === 2. SIMILARITY MATRIX ===
def build_similarity_matrix(X_scaled):
    n = len(X_scaled)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = X_scaled[i] - X_scaled[j]
            similarity_matrix[i, j] = -np.dot(diff, diff)
    print("[INFO] Similarity matrix computed.")
    return similarity_matrix

# === 3. BUILD PAIRWISE HAMILTONIANS ===
def build_pairwise_hamiltonians(similarity_matrix):
    hamiltonians = []
    n = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):  # upper triangle only
            sim = similarity_matrix[i, j]
            flipped_sim = -sim
            h = SparsePauliOp.from_list([("ZZ", flipped_sim)])
            hamiltonians.append(((i, j), h))
    print(f"[INFO] {len(hamiltonians)} pairwise Hamiltonians created.")
    return hamiltonians

# === 4. ANSATZ AND EXPECTATION ===
def create_full_ansatz():
    theta = ParameterVector("θ", 12)
    qc = QuantumCircuit(2)
    for i in range(2):
        qc.rx(theta[i*3], i)
        qc.ry(theta[i*3+1], i)
        qc.rz(theta[i*3+2], i)
    qc.cx(0, 1)
    for i in range(2):
        qc.rx(theta[6+i*3], i)
        qc.ry(theta[6+i*3+1], i)
        qc.rz(theta[6+i*3+2], i)
    print("[INFO] Ansatz Circuit:")
    print(qc.draw(output="text"))
    return qc, theta

def expectation_value(params, circuit, theta, hamiltonian):
    bound = circuit.assign_parameters({theta[i]: params[i] for i in range(len(params))})
    state = Statevector.from_instruction(bound)
    return np.real(state.expectation_value(hamiltonian))

# === 5. CUSTOM SPSA OPTIMIZER ===
def custom_spsa(cost_fn, x0, a=0.1, c=0.1, maxiter=200, tolerance=1e-6):
    n = len(x0)
    x = x0.copy()
    best_loss = cost_fn(x)
    for k in range(maxiter):
        ak = a / (k + 1)**0.602
        ck = c / (k + 1)**0.101
        delta = 2 * np.random.randint(0, 2, n) - 1
        x_plus = x + ck * delta
        x_minus = x - ck * delta
        loss_plus = cost_fn(x_plus)
        loss_minus = cost_fn(x_minus)
        grad = (loss_plus - loss_minus) / (2.0 * ck) * delta
        x -= ak * grad
        x = np.clip(x, -2 * np.pi, 2 * np.pi)
        loss = cost_fn(x)
        if abs(loss_plus - loss_minus) < tolerance:
            break
    return x, best_loss

# === 6. OPTUNA-BASED VQE RUN (CACHED ANSATZ) ===
def run_vqe_with_optuna_cached(hamiltonian, circuit, theta, n_trials=5):
    def objective(trial):
        optimizer_name = trial.suggest_categorical("optimizer", ["SPSA", "COBYLA"])
        init_params = np.array([
            trial.suggest_uniform(f"init_{i}", -np.pi, np.pi)
            for i in range(12)
        ])
        cost_fn = lambda p: expectation_value(p, circuit, theta, hamiltonian)

        if optimizer_name == "SPSA":
            a = trial.suggest_float("spsa_a", 0.01, 0.3)
            c = trial.suggest_float("spsa_c", 0.01, 0.3)
            maxiter = trial.suggest_int("spsa_maxiter", 100, 300)
            _, energy = custom_spsa(cost_fn, init_params, a=a, c=c, maxiter=maxiter)
        else:
            maxiter = trial.suggest_int("cobyla_maxiter", 100, 500)
            result = minimize(cost_fn, x0=init_params, method='COBYLA', options={'maxiter': maxiter})
            energy = result.fun

        return energy

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_value

# === 7. PARALLEL WORKER FUNCTION ===
def process_pair(pair_hamiltonian, circuit, theta):
    (i, j), hamiltonian = pair_hamiltonian
    try:
        energy = run_vqe_with_optuna_cached(hamiltonian, circuit, theta, n_trials=5)
    except Exception as e:
        print(f"[WARN] Failed at pair ({i},{j}):", str(e))
        energy = 0.0
    return (i, j, energy)

# === 8. MAIN PIPELINE ===
def full_pipeline():
    X_scaled = load_and_scale_data()
    X_scaled = X_scaled[:30]  # TEMP: reduce dataset size for faster demo
    similarity_matrix = build_similarity_matrix(X_scaled)
    hamiltonians = build_pairwise_hamiltonians(similarity_matrix)

    n = X_scaled.shape[0]
    vqe_energy_matrix = np.zeros((n, n))

    circuit, theta = create_full_ansatz()
    print(f"\n[INFO] Using {cpu_count()} CPU cores with multiprocessing...")

    start_time = time.time()
    with Pool(processes=min(cpu_count(), 8)) as pool:
        worker = partial(process_pair, circuit=circuit, theta=theta)
        results = pool.map(worker, hamiltonians)

    for i, j, energy in results:
        vqe_energy_matrix[i, j] = energy
        vqe_energy_matrix[j, i] = energy

    end_time = time.time()
    print(f"[INFO] VQE evaluation completed in {end_time - start_time:.2f} seconds.")

    # Convert energy matrix to similarity
    similarity_for_affinity = -vqe_energy_matrix
    np.fill_diagonal(similarity_for_affinity, similarity_for_affinity.max())

    # Clustering
    print("[INFO] Running Affinity Propagation clustering...")
    affinity = AffinityPropagation(affinity='precomputed', random_state=42)
    affinity.fit(similarity_for_affinity)
    labels = affinity.labels_
    print("Cluster labels:", labels)
    print("Number of clusters:", len(np.unique(labels)))

    # Visualization
    X_2d = PCA(n_components=2).fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=40)
    plt.title("Quantum Clustering using VQE + Affinity Propagation")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === EXECUTE ===
if __name__ == "__main__":
    full_pipeline()
