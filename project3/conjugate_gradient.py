from numpy.linalg.linalg import dot
from numpy import vdot

def conjugate_gradient(A, x0,  b, max_iterations):
    # Calculate initial residuals and direction vector
    residual = b - dot(A, x0)
    direction = residual

    # Store initial solution
    x = x0

    for  i in range(max_iterations):
        # Store previous residuals and direction
        old_residual = residual
        old_direction = direction

        # Caclulate new update factor
        a = vdot(old_residual, old_residual) / vdot(old_direction, dot(A, old_direction))

        # Update solution
        x = x + a*old_direction

        # Update residuals
        residual = old_residual - a*dot(A, old_direction)

        # Update direction
        b = vdot(residual, residual) / vdot(old_residual, old_residual)
        direction = residual + b*old_direction

        # If residuals are too small, terminate the algorithm
        if (abs(residual) < 0.000001).all():
            break

    return x, i

if __name__=='__main__':
    from numpy import pi, array
    from scipy.linalg import toeplitz
    column = array([((-1)**k)*(2/(k**2)) if k > 0 else (pi**2)/3 for k in range(100)])
    toeplitz_matrix = toeplitz(column)
    solution, iterations = conjugate_gradient(toeplitz_matrix, list(range(10000,10100)), list(range(100)),150)
    print(solution)
    print(iterations)