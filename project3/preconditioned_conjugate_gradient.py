from numpy.linalg.linalg import dot, inv
from numpy import vdot

def preconditioned_conjugate_gradient(A, x0, b, preconditioner, max_iterations):
    # Calculate initial residuals
    residual = b - dot(A, x0)

    # Calculate M based on the preconditioner and get its inverse
    Minv = inv(dot(preconditioner, preconditioner.T))
    z = dot(Minv, residual)

    # Calculate initial direction
    direction = z

    # Store initial solution
    x = x0

    for  i in range(max_iterations):
        # Store previous residuals and direction
        old_residual = residual
        old_direction = direction
        old_z  = z

        # Caclulate new update factor
        a = vdot(old_residual, old_z) / vdot(old_direction, dot(A, old_direction))

        # Update solution
        x = x + a*old_direction

        # Update residuals
        residual = old_residual - a*dot(A, old_direction)

        z = dot(Minv, residual)

        # Update direction
        b = vdot(z, residual) / vdot(old_z, old_residual)
        direction = z + b*old_direction

        # If residuals are too small, terminate the algorithm
        if (abs(residual) < 0.000001).all():
            break

    return x, i

def test():
    from numpy import array, zeros, pi
    from scipy.linalg import toeplitz
    column = array([((-1)**k)*(2/(k**2)) if k > 0 else (pi**2)/3 for k in range(100)])
    toeplitz_matrix = toeplitz(column)

    column = zeros(100)
    column[0] = 2
    column[1] = -1
    preconditioner = toeplitz(column)

    solution, iterations = preconditioned_conjugate_gradient(toeplitz_matrix, list(range(10000,10100)), list(range(100)), preconditioner, 10000)
    print(solution)
    print(iterations)

if __name__=='__main__':
    test()