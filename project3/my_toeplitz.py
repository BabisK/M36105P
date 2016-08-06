from numpy import array, pi
from scipy.linalg import toeplitz

def my_toeplitz(n):
    column = array([(pi**2)/3 if k == 0 else ((-1)**k)*(2/(k**2)) for k in range(n)])
    matrix = toeplitz(column)
    return matrix