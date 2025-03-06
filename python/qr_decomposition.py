import numpy as np
from triangular_solver import triangular_solver_eff



def qr_simple(A):
    """Compute the QR decomposition of matrix A using Gram-Schmidt process."""
    m, n = A.shape  # Get the shape of A
    Q = np.zeros((m, n))  # Initialize Q matrix
    R = np.zeros((n, n))  # Initialize R matrix

    for i in range(n):
        # Start with column i of A
        v = A[:, i]

        # Subtract projections onto previous q vectors
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])  # Compute dot product
            v = v - R[j, i] * Q[:, j]  # Subtract projection

        # Normalize the resulting vector to get the ith column of Q
        R[i, i] = np.linalg.norm(v)  # Compute norm
        Q[:, i] = v / R[i, i]  # Normalize

    return Q, R




def qr_efficient(A):
    """
    Performs a QR decomposition of A (m x n) via Cholesky factorization of (A^T A).
    Returns Q and R such that A = Q R.
    
    Requirements:
      - A must have full column rank (rank n <= m).
      - A^T A must be positive definite (no rank deficiency).
    """
    # Step 1: Form A^T A
    B = A.T @ A  # shape (n, n)
    
    # Step 2: Cholesky factorization: B = L L^T
    # np.linalg.cholesky returns the lower triangular matrix L
    L = np.linalg.cholesky(B)  # shape (n, n), lower triangular
    
    # The R we want is the upper triangular factor
    R = L.T  # shape (n, n), upper triangular
    
    # Step 3: Q = A * R^{-1}
    # We can do this by explicitly inverting R or by solving a triangular system.
    # For demonstration, we'll invert R directly (though solve_triangular is usually preferred).
    R_inv = np.linalg.inv(R)   # shape (n, n)
    Q = A @ R_inv              # shape (m, n)
    
    return Q, R




if __name__ == "__main__":

    np.random.seed(0)
    # A = np.array([[12, -51, 4],
    #             [6, 167, -68],
    #             [-4, 24, -41]], dtype=float)
    A = np.random.randn(3, 3)

    # Perform QR decomposition
    Q_np, R_np = np.linalg.qr(A)
    A_np = Q_np @ R_np

    Q_simple, R_simple = qr_simple(A)
    Q_simple = -Q_simple
    R_simple = -R_simple
    A_simple = Q_simple @ R_simple

    Q_eff, R_eff = qr_efficient(A)
    Q_eff = -Q_eff
    R_eff = -R_eff
    A_eff = Q_eff @ R_eff

    print("Error norm of simple method for Q.QT-I:", np.linalg.norm(Q_simple.T @ Q_simple - np.eye(Q_simple.shape[1])))
    print("Error norm of efficient method for Q.QT-I:", np.linalg.norm(Q_eff.T @ Q_eff - np.eye(Q_eff.shape[1])))

    print("Error norm of simple method for Q:", np.linalg.norm(Q_simple - Q_np))
    print("Error norm of simple method for R:", np.linalg.norm(R_simple - R_np))
    print("Error norm of simple method for A:", np.linalg.norm(A_simple - A_np))
    print("Error norm of efficient method for Q:", np.linalg.norm(Q_eff - Q_np))
    print("Error norm of efficient method for R:", np.linalg.norm(R_eff - R_np))
    print("Error norm of efficient method for A:", np.linalg.norm(A_eff - A_np))
    # print("Error norm:", np.linalg.norm(A_np - A))



