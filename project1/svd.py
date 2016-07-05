from numpy import matmul
from scipy.linalg import pinv2

def svd(A, b):
    #Cacluate the pseudoinverse of A using the SVD and keep all the singular values
    Apinv = pinv2(A, cond=1E-100)
    #Calculate x by multiplying the pseudoinverse of A with b
    x = matmul(Apinv, b)
    return x