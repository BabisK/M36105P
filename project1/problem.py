from numpy import empty, exp, sin, power, matmul
from householder import householder
from normal_equations import normal_equations
from gram_schmidt import gram_schmidt
from modified_gram_schmidt import modified_gram_schmidt
from svd import svd

def problem():
    #Initialize parameters of the problem
    m = 15
    n = 100
    c = 2006.787453080206

    #Initilize matrix A and vector b
    b = empty(n)
    A = empty((n,m))

    #Calculate A and b according to the parameters
    i = 0
    while i < n:
        A[i,0] = 1
        A[i,1] = i/(n-1)
        j = 2
        while j < m:
            A[i,j] = power(A[i,1], j)
            j = j + 1
        b[i] = (1/c)*exp(sin(4*A[i,1]))
        i = i +1

    print('Matrix A:')
    print(A)

    print('\nVector b:')
    print(b)

    #Calculate x by solving the least squares problem using the normal equations
    #NOTE: This won't execute because A is very ill conditioned (det ~= 1E-90)
    #ne_x = normal_equations(A, b)
    #print(ne_x)

    #Calculate x by solving the least squares problem using the Householder tranformations
    h_x = householder(A, b)
    print('\nValue of x using Householder:')
    print(h_x)

    #Calculate x by solving the least squares problem using the Gram-Schmidt
    gs_x = gram_schmidt(A, b)
    print('\nValue of x using Gram-Schmidt:')
    print(gs_x)

    #Calculate x by solving the least squares problem using the modified Gram-Schmidt
    mgs_x = modified_gram_schmidt(A, b)
    print('\nValue of x using modified Gram-Schmidt:')
    print(mgs_x)

    #Calculate x by solving the least squares problem using the SVD decomposition (pseudoinverse)
    svd_x = svd(A, b)
    print('\nValue of x using the SVD decomposition (pseudoinverse):')
    print(svd_x)

if __name__=='__main__':
    problem()