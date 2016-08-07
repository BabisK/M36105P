from project3.my_toeplitz import my_toeplitz
from project3.my_preconditioner import my_preconditioner
from project3.preconditioned_conjugate_gradient import preconditioned_conjugate_gradient
from numpy.linalg import cholesky,solve
from scipy.linalg import cho_solve, cho_factor
from numpy import ones, zeros
from time import process_time
import logging

def problem5():
    logging.basicConfig(filename='problem2.log', level=logging.DEBUG)

    logging.info('Starting problem 2')

    lmin = 13
    lmax = 13

    matrix = [my_toeplitz(2**l) for l in range(lmin, lmax + 1)]
    b = [ones(2**l) for l in range(lmin, lmax + 1)]
    x0 = [zeros(2**l) for l in range(lmin, lmax + 1)]
    preconditioner = [my_preconditioner(2 ** l) for l in range(lmin, lmax + 1)]
    max_iterations = 1000

    chotime=process_time()
    L = cho_factor(matrix[0], lower= True)
    x = cho_solve(L, b[0])
    print('Cho time: {}'.format(process_time()-chotime))

    pcg_time = process_time()
    result_preconditioned_conjugate_gradient = (preconditioned_conjugate_gradient(matrix[0], x0[0], b[0], preconditioner[0], max_iterations))
    pcg_time = process_time() - pcg_time

    print('Dimension {}: PCG {} iterations in {}s'.format(2 ** 13, result_preconditioned_conjugate_gradient[1], pcg_time))

if __name__=='__main__':
    problem5()