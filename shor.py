import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


class ShorAlgorithm:
    """
    Generic implementation of Shor's Algorithm using matrix-based
    modular multiplication for pedagogical and testing purposes.
    """

    def __init__(self, base_a, modulus_n, precision_qubits=8):
        self.a = base_a
        self.n = modulus_n
        self.precision_qubits = precision_qubits
        self.target_qubits = int(np.ceil(np.log2(modulus_n)))
        self.simulator = AerSimulator()

    def _create_permutation_matrix(self, current_a):
        """Generates the unitary matrix for (current_a * i) % N"""
        dim = 2**self.target_qubits
        matrix = np.zeros((dim, dim))

        for i in range(dim):
            if i < self.n:
                target = (current_a * i) % self.n
            else:
                target = i
            matrix[target, i] = 1
        return matrix

    def _get_mod_exp_gate(self, current_a):
        """Wraps the modular multiplication into a controlled quantum gate"""
        qc = QuantumCircuit(self.target_qubits)
        matrix = self._create_permutation_matrix(current_a)
        qc.unitary(matrix, range(self.target_qubits))
        return qc.to_gate(label=f" {current_a}x mod {self.n} ")

    def build_circuit(self):
        """Constructs the complete quantum circuit for Shor's algorithm"""
        total_qubits = self.precision_qubits + self.target_qubits
        qc = QuantumCircuit(total_qubits, self.precision_qubits)

        for q in range(self.precision_qubits):
            qc.h(q)

        qc.x(self.precision_qubits)

        for q in range(self.precision_qubits):
            exponent = 2**q
            a_powered = pow(self.a, exponent, self.n)
            mod_gate = self._get_mod_exp_gate(a_powered)
            qc.append(
                mod_gate.control(),
                [q] + list(range(self.precision_qubits, total_qubits)),
            )

        qc.append(QFT(self.precision_qubits).inverse(), range(self.precision_qubits))
        qc.measure(range(self.precision_qubits), range(self.precision_qubits))
        return qc

    def run(self, shots=1024):
        """Transpiles and executes the circuit on the Aer simulator"""
        circuit = self.build_circuit()
        compiled_circuit = transpile(circuit, self.simulator)
        result = self.simulator.run(compiled_circuit, shots=shots).result()
        return result.get_counts()

    def plot_results(self, counts):
        """Plots a histogram of the measurement results."""
        # We convert the keys to decimal for the histogram labels
        decimal_counts = {int(k, 2): v for k, v in counts.items()}
        plot_histogram(
            decimal_counts, title=f"Shor's Results (Decimal) for N={self.n}, a={self.a}"
        )
        plt.show()


# --- Execution Block ---
if __name__ == "__main__":
    N = 15
    base = 7
    shor = ShorAlgorithm(base_a=base, modulus_n=N)

    print("--- Starting Shor's Algorithm ---")
    print(f"Modulus (N): {shor.n}")
    print(f"Base (a): {shor.a}")
    print(f"Precision Qubits: {shor.precision_qubits}")

    counts = shor.run()

    print(f"\n{'Binary':<12} | {'Decimal':<8} | {'Phase (k/r)':<12} | {'Frequency'}")
    print("-" * 50)

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    for bitstring, frequency in sorted_counts:
        decimal_val = int(bitstring, 2)
        phase = decimal_val / (2**shor.precision_qubits)
        print(f"{bitstring:<12} | {decimal_val:<8} | {phase:<12.3f} | {frequency}")

    shor.plot_results(counts)
