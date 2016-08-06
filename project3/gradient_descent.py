from project3.fftmul import precompute_g, fftmul
from numpy.linalg import norm
from numpy import inner
import logging

def gradient_descent(A, x0, b, max_iterations):
    # Calculate the FFT of the first column on the circular of A
    g = precompute_g(A)

    # Initial solution
    x = x0

    # Initial residual
    residual = b - fftmul(g, x)

    # Norm of the initial residual
    norm_residual0 = norm(residual, ord = 2)

    for i in range(max_iterations):
        if i%10000 == 0:
            logging.info('Gradient Descent iteration {}, dimenstion {}'.format(i, A.shape[0]))

        # If norm of the residual to norm of the initial residual is less that 10^{-7} terminate
        if norm(residual, ord=2)/norm_residual0 < 0.0000001:
            logging.info('Gradient Descent converged after {} iterations'.format(i))
            break

        # Calculate A*residual using the FFT
        Ares = fftmul(g, residual)

        # Calculate the learning rate (steepest descent)
        a = inner(residual, residual)/inner(residual,Ares)

        # Update x
        x = x + a*residual

        # Update residual
        residual = residual - a*Ares

    return x, i

def test():
    from numpy.random import rand, seed
    from numpy.linalg import solve
    from scipy.linalg import toeplitz
    from time import process_time

    seed(1)

    dimension = 500

    A = toeplitz([dimension-i for i in range(dimension)])
    b = rand(dimension)
    x0 = rand(dimension)
    time = process_time()
    conjugate_gradient_solution, iterations = gradient_descent(A, x0, b, 1000000)
    time = process_time() - time
    #exact_solution = solve(A, b)

    print('GD solution:\n{}'.format(conjugate_gradient_solution))
    print('GD iterations: {} in {}s'.format(iterations, time))
    print('Exact solution:\n{}'.format(exact_solution))

if __name__=='__main__':
    test()