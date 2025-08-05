## Weak-SINDy Compression Project

This project implements Streaming Weak-SINDy Compression for high-dimensional time-series data. It is designed to support the compression of scientific simulation data using Proper Orthogonal Decomposition (POD) and sparse identification of nonlinear dynamics (SINDy) in a streaming (memory-efficient) fashion.

⸻

# Project Structure

Weak_sindy_compression/
├── src/
│   └── Reduction_with_POD/
│       └── Sample_Data.py      # Core logic for loading data and performing POD
├── sample_data.csv            # Input dataset (n variables × m time steps)
└── README.md                  # Project documentation


⸻

# Modules

Sample_Data.py

This module defines a class SampleData which:
	•	Loads and parses time-series data from sample_data.csv
	•	Computes the matrix AA^T for SVD analysis
	•	Calculates eigenvalues and eigenvectors
	•	Performs Singular Value Decomposition (SVD)

⸻

# Mathematical Foundation

Dimensionality Reduction via POD

Given snapshot data matrix A \in \mathbb{R}^{n \times m}:
	•	Compute AA^T
	•	Perform SVD: A = U \Sigma V^T
	•	Truncate top r modes: A \approx U_r \Sigma_r V_r^T

Weak Form SINDy
	•	Transforms \frac{d\mathbf{a}}{dt} = f(\mathbf{a}) into an integral formulation
	•	Avoids explicit differentiation using test functions \psi(t)
	•	Learns f via sparse regression

⸻

Goals
	•	✔️ Build robust and reusable dimensionality reduction code
	•	✔️ Support weak form integration via symbolic time functions
	•	✔️ Enable streaming model updates for compression efficiency

⸻

# Getting Started

Prerequisites
	•	Python 3.8+
	•	NumPy

Run the Analysis

python3 src/Reduction_with_POD/Sample_Data.py

Ensure your sample_data.csv is formatted with rows as variables and columns as time steps.

⸻

# Roadmap
	•	Integrate symbolic test functions \psi(t)
	•	Construct feature library \Theta(\mathbf{a})
	•	Implement streaming regression update logic
	•	Reconstruct original system state from compressed form

⸻

# References
	•	Russo et al., Streaming Compression of Scientific Data via Weak-SINDy, arXiv:2308.14962

⸻

Catherine Earl

MIT-style license © 2025