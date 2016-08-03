from numpy import array, squeeze, inf, sqrt
from numpy.linalg import solve

# The data from the 4 satelites
ABCD = array([[15600, 7540, 20140, 0.07074], [18760, 2750, 18610, 0.07220], [17610, 14630, 13480, 0.07690],
              [19170, 610, 18390, 0.07242]])

# The speed of light (km/s)
c = 299792.458


def f(x, y, z, d, abcd):
    return (x - abcd[:, 0]) ** 2 + (y - abcd[:, 1]) ** 2 + (z - abcd[:, 2]) ** 2 - (c * (abcd[:, 3] - d)) ** 2


def dF(x, y, z, d, abcd):
    return squeeze(array(
        [[2 * (x - abcd[:, 0])], [2 * (y - abcd[:, 1])], [2 * (z - abcd[:, 2])], [2 * (c ** 2) * (abcd[:, 3] - d)]]).T)


def mutlivariate_newton(x, y, z, d, abcd = ABCD):
    initial = array([x, y, z, d])
    i = 0
    fw = initial
    s = array([inf, inf, inf, inf])
    while (abs(s) > 0.000000001).any():
        s = solve(dF(fw[0], fw[1], fw[2], fw[3], abcd), -f(fw[0], fw[1], fw[2], fw[3], abcd))
        fw = fw + s
        i += 1

    return fw[0], fw[1], fw[2], fw[3], sqrt(fw[0]**2 + fw[1]**2 + fw[2]**2) - 6370, i

def main():
    (x, y, z, d, height, iterations) = mutlivariate_newton(0, 0, 6370, 0)

    print('x: {}\ny: {}\nz: {}\nd: {}\nheight: {}'.format(x, y, z, d, height))

if __name__=='__main__':
    main()