# Shor's Algorithm Quantum Simulator

Implementation of Shor's algorithm for integer factorization using Qiskit.

## Description

This project implements the quantum Shor's algorithm to factorize composite numbers into their prime factors. The main example demonstrates the factorization of the number 15 into its prime factors (3 and 5).

## Project Structure

- `shor.py`: Main implementation of Shor's algorithm with Qiskit

## Requirements

```
qiskit
qiskit-aer
matplotlib
numpy
```

## Installation

```bash
uv add qiskit qiskit-aer matplotlib numpy
```

## Usage

```python
from shor import ShorAlgorithm

N = 15
base = 7
shor = ShorAlgorithm(base_a=base, modulus_n=N)

counts = shor.run()

for bitstring, frequency in sorted(counts.items(), key=lambda item: item[1], reverse=True):
    decimal_val = int(bitstring, 2)
    phase = decimal_val / (2**shor.precision_qubits)
    print(f"{bitstring:<12} | {decimal_val:<8} | {phase:<12.3f} | {frequency}")

shor.plot_results(counts)
```

## Shor's Algorithm

Shor's algorithm consists of the following stages:

1. **Base selection**: Find a number `a` coprime with `N`
2. **Period finding**: Find the period `r` where `a^r mod N = 1`
3. **Factor calculation**: Use the period to calculate `gcd(a^(r/2) ± 1, N)`
4. **Verification**: Validate that factors are prime

## Example: Factorization of 15

```bash
python shor.py
```

Expected output:
```
--- Starting Shor's Algorithm ---
Modulus (N): 15
Base (a): 7
Precision Qubits: 8

Binary       | Decimal  | Phase (k/r)  | Frequency
--------------------------------------------------
00000000     | 0        | 0.000        | 272
01000000     | 64       | 0.250        | 264
11000000     | 192      | 0.750        | 259
10000000     | 128      | 0.500        | 229
```
