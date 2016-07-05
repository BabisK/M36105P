from numpy import zeros_like, zeros, dot, matmul
from numpy.linalg import norm
from scipy.linalg import solve_triangular

def modified_gram_schmidt(A, b):
    '''Computes the x of the least squares problem using the QR decomposition by applying modified Gram-Schmidt method'''

    #Initialize Q and R
    Q = zeros_like(A)
    R = zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    #For each column of A
    k = 0
    while k <= A.shape[1]-1:
        #Calculate the diagoanal elementy of R
        R[k,k] = norm(A[:,k], ord=2)
        #Calculate the column of Q
        Q[:,k] = A[:,k]/R[k,k]
        j = k + 1
        #For the elements of R above the diagonal
        while j < A.shape[1]-1:
            #Calculate the element of R and update the column of A
            R[k,j] = dot(Q[:,k].T, A[:,j])
            A[:,j] = A[:,j] - R[k,j]*Q[:,k]
            j = j+1
        k = k + 1

    #Calculate the product of Q transpose with b
    c = matmul(Q.T, b)
    #Solve the system Rx=c where R is triangular using replacement
    x = solve_triangular(R, c)

    return x