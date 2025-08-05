# Streaming Weak-SINDy: Compression Architecture (Math + Flow)

## Overview

This method provides **lossless compression** for scientific data streams by combining:

- **Proper Orthogonal Decomposition (POD)** for dimensionality reduction
- **Weak Sparse Identification of Nonlinear Dynamics (Weak-SINDy)** to learn governing equations

---

## Architecture Components

### 1. Dimensionality Reduction (POD)

Project high-dimensional data \( \mathbf{x}(t) \in \mathbb{R}^n \) into reduced space:

\[
\mathbf{x}(t) \approx \mathbf{U}_r \mathbf{a}(t)
\]

- \( \mathbf{U}_r \in \mathbb{R}^{n \times r} \): POD basis (from SVD)- left singualr vectors from SVD of \( \mathbf{X} \)
- \( \mathbf{a}(t) \in \mathbb{R}^r \): temporal coefficients to use time varying reduced co-ordinates
- \( r \ll n \): reduced dimensionality

---

### 2. Weak Formulation

Avoid noisy derivative estimates by integrating against test functions \( \psi_k(t) \):

\[
\int_0^T \psi_k(t) \frac{da_j(t)}{dt} dt = \int_0^T \psi_k(t) f_j(\mathbf{a}(t)) dt
\]

Integration by parts yields:

\[
- \int_0^T \psi_k'(t) a_j(t) dt = \int_0^T \psi_k(t) f_j(\mathbf{a}(t)) dt
\]

---

### 3. Sparse Regression

Define nonlinear features:

\[
\Theta(\mathbf{a}(t)) = \left[ \theta_1(\mathbf{a}), \theta_2(\mathbf{a}), \ldots, \theta_K(\mathbf{a}) \right]
\]

Then build the regression problem:

\[
b_j = G_j \mathbf{c}_j
\]

Where:

- \( b_j = -\int \psi_k'(t) a_j(t) dt \)
- \( G_j = \int \psi_k(t) \Theta(\mathbf{a}(t)) dt \)

Solve via Lasso:

\[
\mathbf{c}_j = \arg\min \|G_j \mathbf{c}_j - b_j\|^2 + \lambda \|\mathbf{c}_j\|_1
\]

---

### 4. Streaming Update

As each snapshot \( \mathbf{x}_i \) arrives:

\[
\begin{aligned}
G_j &\leftarrow G_j + \Delta t \cdot \psi_k(t_i) \Theta(\mathbf{a}(t_i)) \\
b_j &\leftarrow b_j - \Delta t \cdot \psi_k'(t_i) a_j(t_i)
\end{aligned}
\]

- Only low-memory quantities are updated
- No full snapshot storage required

---

### 5. Model Reconstruction (Offline)

Solve:

\[
\mathbf{c}_j = (G_j^T G_j)^{-1} G_j^T b_j
\]

or using a sparse solver. Then reconstruct:

\[
\mathbf{x}(t) \approx \mathbf{U}_r \cdot \mathbf{a}(t)
\]

where \( \frac{d\mathbf{a}}{dt} = \Theta(\mathbf{a}) \cdot C \)

---

## Full Algorithm Flow

```plaintext
          ┌────────────────────────────┐
          │  Streaming data snapshots  │
          └────────────┬───────────────┘
                       │
             +---------▼----------+
             |  Streaming POD     |
             |  (dim. reduction)  |
             +---------┬----------+
                       │
             +---------▼----------+
             | Project to POD     |         (Temporal modes a(t))
             +---------┬----------+
                       │
             +---------▼----------+
             | Streaming Update   |
             | of G and b via     |
             | test functions ψ(t)|
             +---------┬----------+
                       │
             ┌────────▼───────────┐
             │   G_j a_k matrix   │
             │   b_j vector       │
             └────────┬───────────┘
                      ▼
           [ Offline Sparse Regression ]
                      ▼
         ┌────────────┴─────────────┐
         │ Model coefficients c_j   │
         └────────────┬─────────────┘
                      ▼
         ┌────────────▼─────────────┐
         │  Reconstruct a(t) using  │
         │  learned dynamics model  │
         └────────────┬─────────────┘
                      ▼
         ┌────────────▼─────────────┐
         │  Rebuild x(t) ≈ U_r a(t) │
         └──────────────────────────┘