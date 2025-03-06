import numpy as np



def cholesky_simple(A):
    """
    Performs Cholesky decomposition on a given symmetric, positive-definite matrix A.
    Returns the lower triangular matrix L such that A = L * L.T
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            
            if i == j:  # Diagonal elements
                L[i][j] = np.sqrt(A[i][i] - sum_k)
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]
    
    return L


def cholesky_efficient(A):
    """
    Efficient Cholesky decomposition as per the given algorithm.
    Decomposes A into L such that A = L * L.T.
    """
    n = A.shape[0]
    L = np.zeros_like(A)

    for k in range(n):
        inv = 1.0 / A[k, k]
        invsqr = 1.0 / np.sqrt(A[k, k])

        for j in range(k, n):
            L[j, k] = A[k, j] * invsqr

        for j in range(k + 1, n):
            for i in range(j, n):
                A[j,i] -= A[k, i] * A[k, j] * inv

    return L



if __name__ == "__main__":

    # Example usage
    # A = np.array([[4, 12, -16],
    #             [12, 37, -43],
    #             [-16, -43, 98]], dtype=float)
    
    # Generate a symmetric positive definite matrix
    A = np.random.rand(3, 3)
    A = np.dot(A, A.T)


    L_np = np.linalg.cholesky(A)
    # A_np = L_np @ L_np.T
    A_np = np.dot(L_np, L_np.T)

    L_simple = cholesky_simple(A)
    # A_simple = L_simple @ L_simple.T
    A_simple = np.dot(L_simple, L_simple.T)
    
    L_eff = cholesky_efficient(A)
    # A_eff = L_eff @ L_eff.T
    A_eff = np.dot(L_eff, L_eff.T)


    # print(A)
    # print(A_np)
    # print(A_simple)
    # print(A_eff)

    # print("Error norm:", np.linalg.norm(A_np - A))
    print("Error norm of simple method:", np.linalg.norm(A_simple - A_np))
    print("Error norm of efficient method:", np.linalg.norm(A_eff - A_np))


