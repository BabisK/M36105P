from numpy import zeros_like, zeros, dot, matmul
from numpy.linalg import norm
from scipy.linalg import solve_triangular

def gram_schmidt(A, b):
    '''Computes the x of the least squares problem using the QR decomposition by applying Gram-Schmidt method'''

    #Initialize Q and R
    Q = zeros_like(A)
    R = zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    #Iterate over the columns of A
    for index, column in enumerate(A.T[:,]):
        #Initialize the column pf Q
        Q[:,index] = column
        j = 0
        #For matrix values of R above the diagonal
        while(j<=index-1):
            #Calculate those values and update the Q column
            R[j,index] = dot(Q[:,j], column)
            Q[:,index] = Q[:,index] - R[j,index]*Q[:,index]
            j = j+1
        #Calculate the diagonal of R
        R[index,index] = norm(Q[:,index], ord=2)
        #Update the Q column according to R
        Q[:,index] = Q[:,index]/R[index,index]

    #Calculate the product of Q transpose with b
    c = matmul(Q.T, b)
    #Solve the system Rx=c where R is triangular using replacement
    x = solve_triangular(R, c)

    return x