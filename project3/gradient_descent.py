from numpy import pi, array
from scipy.linalg import toeplitz

def c_toeplitz(n):
    column = array([((-1)**k)*(2/(k**2)) if k > 0 else (pi**2)/3 for k in range(n)])
    toeplitz_matrix = toeplitz(column)
    print('Done')

def gradient_descent(A, x0,  b, max_iterations):
    pass

if __name__=='__main__':
    c_toeplitz(2**9)