from numpy import array, squeeze
from numpy.linalg import solve

def mutlivariate_newton(x, y, z, d):
    # The data from the 4 satelites
    ABCD = array([[15600, 7540, 20140, 0.07074], [18760, 2750, 18610, 0.07220], [17610, 14630, 13480, 0.07690], [19170, 610, 18390, 0.07242]])
    A = array([15600, 7540, 20140, 0.07074])
    B = array([18760, 2750, 18610, 0.07220])
    C = array([17610, 14630, 13480, 0.07690])
    T = array([19170, 610, 18390, 0.07242])

    # The speed of light (km/s)
    c = 299792.458

    initial = array([x, y, z, d])

    F = (x - A)**2 + (y - B)**2 + (z - C)**2 - (c*(T - d))**2

    dF = squeeze(array([[2*(x - A)], [2*(y - B)], [2*(z - C)], [2*c**2*(T - d)]]))

    s = solve(dF, -F)
    f = initial + s

    return s

print(mutlivariate_newton(0,0,6370,0))