# 🧠 Dimensionality Reduction with POD (Proper Orthogonal Decomposition)

In scientific and high-performance computing scenarios, data is typically generated as a **high-dimensional time-dependent signal** — for example, from the solution of a PDE or large-scale simulation.

To compress such data, we first perform **dimensionality reduction** using **Proper Orthogonal Decomposition (POD)**, which is a physics-inspired form of PCA.

---

## 🗂️ Step 1: Collect Snapshot Data

Let \( \mathbf{x}(t) \in \mathbb{R}^n \) denote the state of the system at time \( t \). We collect **snapshots** of the system at \( m \) different time steps \( \{ t_1, t_2, \dots, t_m \} \). These snapshots are stacked into a matrix:

\[
\mathbf{X} = 
\begin{bmatrix}
\mathbf{x}(t_1) & \mathbf{x}(t_2) & \dots & \mathbf{x}(t_m)
\end{bmatrix}
\in \mathbb{R}^{n \times m}
\]

Here:
- \( n \): number of spatial/field variables (e.g., grid points, dimensions)
- \( m \): number of time steps (parameter, to be confirmed later)

---

## 🔍 Step 2: Compute the Singular Value Decomposition (SVD)

To extract the principal spatial patterns (i.e., modes), we compute the **SVD** of \( \mathbf{X} \):

\[
\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top
\]

Where:
- \( \mathbf{U} \in \mathbb{R}^{n \times n} \): spatial modes (orthonormal)
- \( \mathbf{\Sigma} \in \mathbb{R}^{n \times m} \): diagonal matrix of singular values
- \( \mathbf{V} \in \mathbb{R}^{m \times m} \): temporal modes (orthonormal)

The columns of \( \mathbf{U} \) capture dominant **spatial patterns**, while rows of \( \mathbf{V}^\top \) capture **temporal coefficients**.

---

## ✂️ Step 3: Truncate to \( r \) Dominant Modes

To reduce dimensionality, we retain only the top \( r \ll n \) dominant singular values and their associated vectors. Define:

\[
\mathbf{U}_r = 
\begin{bmatrix}
\mathbf{u}_1 & \mathbf{u}_2 & \dots & \mathbf{u}_r
\end{bmatrix}
\in \mathbb{R}^{n \times r}
\]

This defines the **reduced POD basis**, capturing the most energetic structures in the data.

---

## 📉 Step 4: Project Data into POD Basis

For each time \( t \), we approximate the high-dimensional signal \( \mathbf{x}(t) \) as a linear combination of the \( r \) basis vectors:

\[
\mathbf{x}(t) \approx \mathbf{U}_r \mathbf{a}(t)
\]

Where:
- \( \mathbf{a}(t) \in \mathbb{R}^r \) are the **temporal coefficients** (or modal amplitudes)

These coefficients are obtained by **orthogonal projection**:

\[
\mathbf{a}(t) = \mathbf{U}_r^\top \mathbf{x}(t)
\]

---

## ✅ Summary of Variables

| Symbol            | Meaning                               |
|------------------|----------------------------------------|
| \( \mathbf{x}(t) \) | High-dimensional state at time \( t \) |
| \( \mathbf{X} \)     | Matrix of snapshots \( \in \mathbb{R}^{n \times m} \) |
| \( \mathbf{U}_r \)   | Reduced POD basis \( \in \mathbb{R}^{n \times r} \) |
| \( \mathbf{a}(t) \) | Temporal POD coefficients \( \in \mathbb{R}^r \) |

---

## 🔄 Optional: Streaming Implementation

If storing \( \mathbf{X} \) is infeasible (e.g., \( m \) is very large or unknown in advance), one can use **streaming SVD/POD** methods that:
- Build \( \mathbf{U}_r \) incrementally
- Update \( \mathbf{a}(t) \) in real time
- Allow adaptive truncation (e.g. based on spectral energy)

---

## ⏭️ Next Step

With dimensionality reduced via POD:
- We now work in a much smaller space \( \mathbb{R}^r \)
- Instead of compressing \( \mathbf{x}(t) \), we compress \( \mathbf{a}(t) \)
- This sets the stage for applying the **weak-SINDy surrogate modeling and compression**

→ [Continue to Weak-SINDy Formulation →](#)