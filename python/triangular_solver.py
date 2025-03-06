import numpy as np




def triangular_solver_simple(A, b, lower=True):
    """
    Solves Ax = b for x where A is a triangular matrix.
    
    Parameters:
    A (numpy.ndarray): Coefficient matrix (must be triangular)
    b (numpy.ndarray): Right-hand side vector
    lower (bool): If True, A is lower triangular; otherwise, A is upper triangular
    
    Returns:
    numpy.ndarray: Solution vector x
    """
    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
    
    if lower:
        # Forward substitution for lower triangular matrix
        for i in range(n):
            if A[i, i] == 0:
                raise ValueError("Matrix is singular")
            x[i] = (b[i] - np.dot(A[i, :i], x[:i])) / A[i, i]
    else:
        # Backward substitution for upper triangular matrix
        for i in range(n-1, -1, -1):
            if A[i, i] == 0:
                raise ValueError("Matrix is singular")
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x



# This doesn't seem to be any different than the simple function
def triangular_solver_eff(a, b, lower=True):
    n = len(b)
    # Ensure inputs are NumPy arrays
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    # for j in range(n):
    #     b[j] = b[j] / a[j, j]
    #     for i in range(j + 1, n):
    #         b[i] -= b[j] * a[j, i]

    # return b

    if lower:
        # Solving for a lower triangular matrix using forward substitution
        x = np.zeros_like(b)
        for i in range(n):
            x[i] = b[i]
            for j in range(i):
                x[i] -= a[i, j] * x[j]
            x[i] /= a[i, i]
    else:
        # Solving for an upper triangular matrix using back substitution
        x = np.zeros_like(b)
        for i in range(n - 1, -1, -1):
            x[i] = b[i]
            for j in range(i + 1, n):
                x[i] -= a[i, j] * x[j]
            x[i] /= a[i, i]

    return x





# Example Usage
if __name__ == "__main__":

    A_lower = np.array([[2, 0, 0], [1, 3, 0], [4, -2, 1]], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)

    x_np = np.linalg.solve(A_lower, b)  # Works only if A is triangular and non-singular
    x_simple = triangular_solver_simple(A_lower, b, lower=True)
    x_eff = triangular_solver_eff(A_lower, b, lower=True)

    print("Error norm of simple method:", np.linalg.norm(x_simple - x_np))
    print("Error norm of efficient method:", np.linalg.norm(x_eff - x_np))




    A_upper = np.array([[2, -1, 3], [0, 3, 2], [0, 0, 1]], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)

    x_np = np.linalg.solve(A_upper, b)  # Works only if A is triangular and non-singular
    x_simple = triangular_solver_simple(A_upper, b, lower=False)
    x_eff = triangular_solver_eff(A_upper, b, lower=False)

    print("Error norm of simple method:", np.linalg.norm(x_simple - x_np))
    print("Error norm of efficient method:", np.linalg.norm(x_eff - x_np))



