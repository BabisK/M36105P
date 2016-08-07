from project3.fftmul import precompute_g, fftmul
from numpy.linalg import norm
from numpy import inner
import logging

def conjugate_gradient(A, x0, b, max_iterations):
    # Calculate the FFT of the first column on the circular of A
    g = precompute_g(A)

    # Initial solution
    x = x0

    # Initial residual and direction vector
    residual = b - fftmul(g, x)
    direction = residual

    # Norm of the initial residual
    norm_residual0 = norm(residual, ord=2)

    # r^T*r
    rr = inner(residual,residual)

    for  i in range(max_iterations):
        if i%10000 == 0:
            logging.info('Conjugate Gradient iteration {}, dimenstion {}'.format(i, A.shape[0]))

            # If norm of the residual to norm of the initial residual is less that 10^{-7} terminate
        if norm(residual, ord=2)/norm_residual0 < 0.0000001:
            logging.info('Conjugate Gradient converged after {} iterations'.format(i))
            break

        # Store previous residuals and direction
        old_residual = residual
        old_direction = direction
        old_rr = rr

        # Calculate A*residual using the FFT
        Ap = fftmul(g, old_direction)

        # Calculate the learning rate
        a = old_rr / inner(old_direction, Ap)

        # Update solution
        x = x + a*old_direction

        # Update residuals
        residual = old_residual - a*Ap

        # New r^T*r
        rr = inner(residual, residual)

        # Update direction
        b = rr / old_rr
        direction = residual + b*old_direction

    return x, i

def test():
    from time import process_time
    from numpy.random import rand, seed
    from numpy.linalg import solve
    from scipy.linalg import toeplitz

    seed(1)

    dimension = 1000

    A = toeplitz([dimension-i for i in range(dimension)])
    b = rand(dimension)
    x0 = rand(dimension)
    time = process_time()
    conjugate_gradient_solution, iterations = conjugate_gradient(A, x0, b, 1000000)
    time = process_time() - time
    exact_solution = solve(A, b)

    print('CG solution:\n{}'.format(conjugate_gradient_solution))
    print('CG iterations: {} in {}s'.format(iterations, time))
    print('Exact solution:\n{}'.format(exact_solution))

if __name__=='__main__':
    test()