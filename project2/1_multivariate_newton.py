from numpy import array, squeeze, inf
from numpy.linalg import solve

# The data from the 4 satelites
ABCD = array([[15600, 7540, 20140, 0.07074], [18760, 2750, 18610, 0.07220], [17610, 14630, 13480, 0.07690],
              [19170, 610, 18390, 0.07242]])

# The speed of light (km/s)
c = 299792.458


def f(x, y, z, d):
    return (x - ABCD[:, 0]) ** 2 + (y - ABCD[:, 1]) ** 2 + (z - ABCD[:, 2]) ** 2 - (c * (ABCD[:, 3] - d)) ** 2


def dF(x, y, z, d):
    return squeeze(array(
        [[2 * (x - ABCD[:, 0])], [2 * (y - ABCD[:, 1])], [2 * (z - ABCD[:, 2])], [2 * (c ** 2) * (ABCD[:, 3] - d)]]).T)


def mutlivariate_newton(x, y, z, d):
    initial = array([x, y, z, d])
    i = 0
    fw = initial
    s = array([inf, inf, inf, inf])
    while (abs(s) > 0.000000001).any():
        s = solve(dF(fw[0], fw[1], fw[2], fw[3]), -f(fw[0], fw[1], fw[2], fw[3]))
        fw = fw + s
        i += 1

    return fw, i

print(mutlivariate_newton(0, 0, 6370, 0))
