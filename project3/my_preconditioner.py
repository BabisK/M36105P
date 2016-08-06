from numpy import zeros
from scipy.linalg import toeplitz

def my_preconditioner(n):
    column = zeros(n)
    column[0] = 2
    column[1] = -1
    matrix = toeplitz(column)
    return matrix