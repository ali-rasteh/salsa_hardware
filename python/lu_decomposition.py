import numpy as np
from scipy.linalg import lu



def lu_decomposition_simple(A):
    """
    Performs LU decomposition of a square matrix A using Doolittle's method.
    A = L * U where:
    - L is a lower triangular matrix with ones on the diagonal
    - U is an upper triangular matrix
    
    Parameters:
    A (numpy.ndarray): Square matrix to decompose

    Returns:
    L (numpy.ndarray): Lower triangular matrix
    U (numpy.ndarray): Upper triangular matrix
    """
    n = A.shape[0]
    L = np.eye(n)  # Initialize L as identity matrix
    U = np.zeros((n, n))  # Initialize U as zero matrix

    for i in range(n):
        # Upper triangular matrix U
        for k in range(i, n):
            U[i, k] = A[i, k] - np.sum(L[i, :i] * U[:i, k])
        
        # Lower triangular matrix L
        for k in range(i+1, n):
            L[k, i] = (A[k, i] - np.sum(L[k, :i] * U[:i, i])) / U[i, i]
    
    return L, U



def naive_lu_with_pivoting(A_in):
    """
    Perform LU decomposition on a square matrix A_in using
    Doolittle's algorithm with partial pivoting.
    
    Returns:
        P: Permutation matrix
        L: Lower triangular matrix with unit diagonal
        U: Upper triangular matrix
    """
    A = A_in.copy().astype(float)
    n = A.shape[0]
    
    # Permutation matrix, start as the identity
    P = np.eye(n)
    
    # Initialize L and U
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for k in range(n):
        # ============== Pivot selection (partial) ==============
        # Find pivot row (where the absolute value in column k is max, from row k down)
        pivot_row = np.argmax(np.abs(A[k:, k])) + k
        
        if A[pivot_row, k] == 0:
            raise ValueError("Matrix is singular or nearly singular!")
        
        # ============== Row swap in A, P, L ==============
        if pivot_row != k:
            A[[k, pivot_row], :] = A[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            # For L, swap the rows below the diagonal (columns up to k only) 
            # to keep already-computed factors consistent.
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        # ============== Elimination to form L and U ==============
        # The pivot is now A[k, k]. That becomes part of U.
        U[k, k:] = A[k, k:]
        
        # For each row i below pivot row k
        for i in range(k+1, n):
            # L[i,k] = factor that zeroes out A[i,k]
            L[i, k] = A[i, k] / A[k, k]
            
            # Subtract multiple of pivot row from row i
            A[i, k:] = A[i, k:] - L[i, k] * A[k, k:]
    
    return P, L, U

# ==================== Example usage ====================

if __name__ == "__main__":
    # Example:
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)
    
    # Construct a random matrix
    A = np.random.randn(3, 3)
    
    # LU factorization
    P_scp, L_scp, U_scp = lu(A)
    L_simple, U_simple = lu_decomposition_simple(A)
    P_eff, L_eff, U_eff = naive_lu_with_pivoting(A)
    
    A_scp = P_scp @ L_scp @ U_scp
    A_simple = L_simple @ U_simple
    A_eff = P_eff @ L_eff @ U_eff
    
    print(A)
    print(A_scp)
    print(A_simple)
    print(A_eff)
    print("Error norm:", np.linalg.norm(A_scp - A))
    print("Error norm:", np.linalg.norm(A_simple - A))
    print("Error norm:", np.linalg.norm(A_eff - A))

