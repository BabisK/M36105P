from numpy import matmul
from scipy.linalg import cho_factor, cho_solve

def normal_equations(A, b):
    '''Computes the x of the least squares problem using the normal equations (Cholesky factorization)'''

    #Calculate the product of A transpose with A
    ATA = matmul(A.T, A)
    #Calculate the product of A transpose with b
    ATb = matmul(A.T, b)

    #Calculate the L lower triagonal using cholesky factorization
    L, lower = cho_factor(ATA, lower=True)

    #Solve the Lx=c problem where L is a lower triagonal
    x = cho_solve((L, lower), ATb)

    return x