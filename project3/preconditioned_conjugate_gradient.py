from project3.fftmul import precompute_g, fftmul
from numpy.linalg.linalg import inv, dot
from numpy.linalg import norm
from scipy.linalg import solve
from numpy import inner
import logging

def preconditioned_conjugate_gradient(A, x0, b, preconditioner, max_iterations):
    # Calculate the FFT of the first column on the circular of A
    gA = precompute_g(A)

    # Store initial solution
    x = x0

    # Calculate initial residuals
    residual = b - fftmul(gA, x0)

    # Norm of the initial residual
    norm_residual0 = norm(residual, ord=2)

    # Calculate M based on the preconditioner and get its inverse
    Minv = inv(dot(preconditioner, preconditioner.T))
    #M = dot(preconditioner, preconditioner.T)

    # Initial z vector
    z = dot(Minv, residual)

    # z^T*z
    zr = inner(z, residual)

    # Calculate initial direction
    direction = z

    for  i in range(max_iterations):
        if i % 10000 == 0:
            logging.info('Preconditioned Conjugate Gradient iteration {}, dimenstion {}'.format(i, A.shape[0]))

        # If residuals are too small, terminate the algorithm
        if norm(residual, ord=2)/norm_residual0 < 0.0000001:
            logging.info('Preconditioned Conjugate Gradient converged after {} iterations'.format(i))
            break

        # Store previous residuals and direction
        old_residual = residual
        old_direction = direction
        old_z  = z
        old_zr = zr

        # Caclulate new update factor
        Ar = fftmul(gA, old_direction)
        a = old_zr / inner(old_direction, Ar)

        # Update solution
        x = x + a*old_direction

        # Update residuals
        residual = old_residual - a*Ar

        z = dot(Minv, residual)

        # Update direction
        zr = inner(z, residual)
        b = zr / old_zr
        direction = z + b*old_direction

    return x, i

def test():
    from time import process_time
    from numpy import zeros
    from numpy.random import rand, seed
    from numpy.linalg import solve
    from scipy.linalg import toeplitz

    seed(1)

    dimension = 10

    A = toeplitz([dimension-i for i in range(dimension)])
    b = rand(dimension)
    x0 = rand(dimension)

    column = zeros(dimension)
    column[0] = 2
    column[1] = -1
    preconditioner = toeplitz(column)
    time = process_time()
    preconditioned_conjugate_gradient_solution, iterations = preconditioned_conjugate_gradient(A, x0, b, preconditioner, 1000000)
    time = process_time() - time
    exact_solution = solve(A, b)

    print('PCG solution:\n{}'.format(preconditioned_conjugate_gradient_solution))
    print('PCG iterations: {} in {}s'.format(iterations, time))
    print('Exact solution:\n{}'.format(exact_solution))


if __name__=='__main__':
    test()