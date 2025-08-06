## Streaming Weak-SINDy Compression Architecture (Narrative + Math)

⸻

# Dimensionality Reduction with POD (Proper Orthogonal Decomposition)

In scientific computing, we often deal with high-dimensional data that varies over time. To make this data manageable, we begin by reducing its dimensionality using Proper Orthogonal Decomposition (POD) — a physics-informed form of PCA.

# 1. Collect Snapshot Data

Let \mathbf{x}(t) \in \mathbb{R}^n denote the state of the system at time t. Collect m such states:

	\mathbf{X} = \begin{bmatrix} \mathbf{x}(t_1) & \mathbf{x}(t_2) & \dots & \mathbf{x}(t_m) \end{bmatrix} \in \mathbb{R}^{n \times m}
	•	n: number of spatial variables
	•	m: number of time steps (parameter, to be tuned later)

# 2. Compute the SVD

Perform Singular Value Decomposition (SVD):

	\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top
	•	\mathbf{U} \in \mathbb{R}^{n \times n}: left singular vectors (spatial modes)
	•	\mathbf{\Sigma} \in \mathbb{R}^{n \times m}: diagonal matrix of singular values
	•	\mathbf{V} \in \mathbb{R}^{m \times m}: right singular vectors (temporal modes)

# 3. Truncate to Dominant Modes

We retain the top r \ll n modes:

	\mathbf{U}_r = \begin{bmatrix} \mathbf{u}_1 & \dots & \mathbf{u}_r \end{bmatrix} \in \mathbb{R}^{n \times r}

This becomes our POD basis.

# 4. Project into POD Space

Each state \mathbf{x}(t) can now be approximated:

	\mathbf{x}(t) \approx \mathbf{U}_r \mathbf{a}(t)

Where:

	\mathbf{a}(t) = \mathbf{U}_r^\top \mathbf{x}(t) \in \mathbb{R}^r
	•	\mathbf{a}(t): temporal coefficients capturing evolution in reduced space

⸻

## Weak Formulation of Dynamics

We now model how the reduced coefficients \mathbf{a}(t) evolve over time, using the Weak Formulation of Sparse Identification of Nonlinear Dynamics (Weak-SINDy).

1. 🚀 Problem Setup

Assume a dynamical system:

	\frac{d\mathbf{a}}{dt} = \mathbf{f}(\mathbf{a}(t))

Instead of differentiating noisy data, multiply both sides by a smooth test function \psi(t) and integrate:

	\int_0^T \psi(t) \frac{da_j(t)}{dt} dt = \int_0^T \psi(t) f_j(\mathbf{a}(t)) dt

Apply integration by parts:

	•	\int_0^T \psi’(t) a_j(t) dt = \int_0^T \psi(t) f_j(\mathbf{a}(t)) dt

This avoids derivative estimation and improves robustness.

⸻

📚 Sparse Regression Formulation

	To approximate f_j(\mathbf{a}), we build a library of nonlinear features:

	\Theta(\mathbf{a}) = [1, a_1, \dots, a_r, a_1^2, \dots, a_r^2, \dots]

The regression becomes:

	b_j = G_j \mathbf{c}_j

	Where:
	•	G_j = \int_0^T \psi(t) \Theta(\mathbf{a}(t)) dt
	•	b_j = - \int_0^T \psi’(t) a_j(t) dt
	•	\mathbf{c}_j: sparse coefficient vector

Solve via Lasso:

	\mathbf{c}_j = \arg\min \| G_j \mathbf{c}_j - b_j \|^2 + \lambda \| \mathbf{c}_j \|_1

⸻

## Streaming Updates (Online Compression)

Instead of storing full data, we update G_j and b_j incrementally:

	\begin{aligned}
	G_j &\leftarrow G_j + \Delta t \cdot \psi(t_i) \Theta(\mathbf{a}(t_i)) \\
	b_j &\leftarrow b_j - \Delta t \cdot \psi’(t_i) a_j(t_i)
	\end{aligned}

This enables memory-efficient streaming without storing snapshots.

⸻

## Model Reconstruction (Offline)

Once streaming ends, we solve:

	\mathbf{c}_j = (G_j^T G_j)^{-1} G_j^T b_j

Or use a sparse solver. Then reconstruct:

	\frac{d\mathbf{a}}{dt} = \Theta(\mathbf{a}) C

Finally, reconstruct original states:

	\mathbf{x}(t) \approx \mathbf{U}_r \mathbf{a}(t)

⸻

Step	Description
POD	Dimensionality reduction via SVD
Weak-SINDy	Avoids differentiation by weak form
Regression	Learns governing dynamics in compressed space
Streaming	Compresses online using updated G, b
Reconstruction	Rebuilds full state from reduced model


⸻

# Reference

Russo et al., Streaming Compression of Scientific Data via Weak-SINDy, arXiv:2308.14962v2 (2024)
🔗 https://arxiv.org/abs/2308.14962
