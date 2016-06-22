from numpy import matmul
from numpy.linalg import qr
from scipy.linalg import solve_triangular

def householder(A, b):
    #Calculate the reduced Q and R using the householder transformations
    Q, R = qr(A, mode='reduced')
    #Calculate the product of Q transpose with b
    c = matmul(Q.T, b)
    #Solve the system Rx=c where R is triangular using replacement
    x = solve_triangular(R, c)
    return x