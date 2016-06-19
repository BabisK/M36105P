from numpy import matmul
from scipy.linalg import cho_factor, cho_solve

def normal_equations(A, b):
    ATA = matmul(A.T, A)
    ATb = matmul(A.T, b)
    L, lower = cho_factor(ATA, lower=True)
    x = cho_solve((L, lower), ATb)
    return x