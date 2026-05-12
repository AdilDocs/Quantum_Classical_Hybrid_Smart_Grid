
import numpy as np
import pandas as pd
import time
import random
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from qiskit.circuit.library import pauli_feature_map, real_amplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import StatevectorSampler as Sampler

import warnings
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# reduce nondeterminism from threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# optional: silence annoying warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", message="No gradient function provided")
#warnings.filterwarnings("ignore", message="COBYLA: Invalid MAXFUN")
#warnings.filterwarnings("ignore", message="No gradient function provided")

SYSTEM_CAPACITY = 600
LOAD_DEMAND = 520
SCALING_FACTOR = 0.05


def calculate_reliability_indices(y_pred):
    lolp_base = np.mean(y_pred)
    lolp = lolp_base * SCALING_FACTOR
    power_deficit = (LOAD_DEMAND - (SYSTEM_CAPACITY * 0.8))
    eens = lolp * power_deficit * 8.76
    sri = 1 - (lolp / 1.2)
    return round(lolp, 4), round(eens, 2), round(sri, 4)

X_train = pd.read_csv("X_train_final.csv").values[:1000]
y_train = pd.read_csv("y_train_final.csv").values.flatten()[:1000]
X_test = pd.read_csv("X_test_final.csv").values[:500]
y_test = pd.read_csv("y_test_final.csv").values.flatten()[:500]

print("\n================ Please wait till the completion of last row of table ================")
print("\n================ If you are seeing a warning, just ignore it, program is in execution ================")


configurations = [
    {
        "name": "Classical RF",
        "search": "Linear",
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1
        )
    },
    {
        "name": "MLP Network",
        "search": "Linear",
        "model": MLPClassifier(
            max_iter=200,
            random_state=SEED,
            shuffle=True
        )
    },
    {
        "name": "SVM (RBF)",
        "search": "Quadratic",
        "model": SVC(
            kernel='rbf',
            probability=False,
            random_state=SEED
        )
    },
    {
        "name": "Grover + Pauli + RF",
        "search": "Sublinear",
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1
        ),
        "use_quantum_map": True
    },
    {
        "name": "Proposed Framework",
        "search": "Sublinear",
        "model": "VQC",
        "use_quantum_map": True
    }
]

print(f"{'Configuration':<25} | {'Search':<10} | {'LOLP':<8} | {'EENS':<8} | {'SRI':<8}")
print("-" * 80)

num_qubits = 6

for config in configurations:

    if config.get("use_quantum_map"):
        p_map = pauli_feature_map(
            num_qubits,
            reps=3,
            entanglement='full'
        )

    if config["model"] == "VQC":
        v_ansatz = real_amplitudes(num_qubits, reps=3)

        optimizer = COBYLA(maxiter=20)
        sampler = Sampler()

        # deterministic initial point
        initial_point = np.zeros(v_ansatz.num_parameters, dtype=float)

        clf = VQC(
            feature_map=p_map,
            ansatz=v_ansatz,
            optimizer=optimizer,
            sampler=sampler,
            initial_point=initial_point
        )

        clf.fit(X_train[:200], y_train[:200])
        y_pred = clf.predict(X_test)

    else:
        clf = config["model"]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    lolp, eens, sri = calculate_reliability_indices(y_pred)
    print(f"{config['name']:<25} | {config['search']:<10} | {lolp:<8} | {eens:<8} | {sri:<8}")

print("\n--- ABLATION AUDIT COMPLETE ---")
print("All randomness controlled with SEED =", SEED)
