import numpy as np
import csv

class SampleData:
    def __init__(self):
        self.data = self.load_sample_data()
        self.time = self.extract_time()
        self.variables = self.extract_variables()
    
    def load_sample_data(self):
        """
        Parses data from a sample file or generates synthetic data.
        Returns:
            data: numpy array of shape (n_snapshotvariables, n_time_steps)
        """
        with open("sample_data.csv", "r") as f:
            reader = csv.reader(f)
            data = np.array(list(reader)).astype(float)
        # return data
        return data

    def AA_tMatrix(self):
        """
        Compute the Singular Value Decomposition (SVD) of the data.
        
        Returns:
            U: Left singular vectors (spatial modes)
            S: Singular values
            Vt: Right singular vectors (temporal modes)
        """
        #debug the data if it has loaded correctly 
        if self.data is None:
            raise ValueError("Data is not loaded correctly.")
        if self.data.shape[0] == 0 or self.data.shape[1] == 0:
            raise ValueError("Data is empty.")
        # Perform Singular Value Decomposition (SVD)
        # calculate the AA^T matrix where A^T is the transpose of Matrix A
        """
        where we are looking at the matrix A as a collection of snapshots,
        each snapshot is a row in the matrix A.
        if we say snapshot i is the vector of A [i, :]
        we can then say that i_1 + i_2 + i_3 + ... + i_n = AA^T
        where i_1, i_2, i_3, ..., i_n are the snapshots.
        where we bring in time stamps we denote the snapshots as A[i, t],
        where i is the snapshot index and t is the time index.
        this transfers to 
        
        {/sigma_[i=1]^n}[AA^T]_ii = tr(A A^T)

        where tr is the trace of the matrix.
        where sigma_i is the i-th singular value.
        by breacking this down even further we can denote that with l as
        the timestamp.
        
        = {/sigma_[i=1]^m}{/sigma_[k=1]^n [A]_ik . [A^t]_ki
        = {/sigma)[k=1]^m}{/sigma_[i=1]^n [A^t]_ki . [A]_ik}
        = {/sigma_[k=1]^m}[A^tA]_kk
        = tr(A^tA)
        """
        A = self.data
        AA_T = np.dot(A, A.T)
        return AA_T
    
    def _calculate_eignvalues_and_vectors(self, AA_T):
        """
        Av=λv 
        where A is the matrix, v is the eigenvector, and λ is the eigenvalue.
        1. use the equation of |(A - λI)| = 0 to find the eigenvalues
        2. name the eigenvalues as λ_i, lambda_i, or λ_i
        3. find the vector X associated with each eigenvalue λ_i
        AV_r = λV_r - right eigenvectors
        V_lA = V_lλ - left eigenvectors

        find the eigenvector of 2 x 2 by det(A - λI) = 0
        then subsitute the eigenvalue λ into the equation (A - λI)v = 0
        """
        eigenvalues, eigenvectors = np.linalg.eig(AA_T)
        return eigenvalues, eigenvectors
    
    def compute_svd(self):
        """
        compute the Singular Value Decomposition (SVD) of the data matrix.
        Returns:
            U: Left singular vectors (spatial modes)
            S: Singular values
            Vt: Right singular vectors (temporal modes)
        """
    # compute the simular value decomposition
        AA_T = self.AA_tMatrix()
        U, S, Vt = np.linalg.svd(self.data, full_matrices=False)
        return U, S, Vt
    #   •	\mathbf{U} \in \mathbb{R}^{n \times n}: left singular vectors (spatial modes)
    #   •	\mathbf{\Sigma} \in \mathbb{R}^{n \times m}: diagonal matrix of singular values
	#   •	\mathbf{V} \in \mathbb{R}^{m \times m}: right singular vectors (temporal modes)