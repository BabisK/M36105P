from numpy import array, cos, sin, sqrt, pi, inf, nan
from numpy.linalg import norm
from project2.multivariate_newton import mutlivariate_newton

# The speed of light (km/s)
cc = 299792.458

phi = array([0, pi/6, pi/3, pi/2])
theta = array([0, pi/2, pi, 3*pi/2])
r = 26570

A = r*cos(phi)*cos(theta)
B = r*cos(phi)*sin(theta)
C = r*sin(phi)

def calc_emf(x,y,z,d, a = A, b = B, c = C):
    R = sqrt((a-x)**2 + (b-y)**2 + (c-z)**2)
    D = d + R/cc
    ABCD = array([a, b, c, D]).T

    true_data = mutlivariate_newton(0,0,6370,0.0001, ABCD)
    print('True data: {}'.format(true_data))

    error_time = 0.00000001
    results = []
    for t1mod in range(-1,2):
        for t2mod in range(-1,2):
            for t3mod in range(-1, 2):
                for t4mod in range(-1, 2):
                    result = {'dt1': t1mod * error_time, 'dt2': t2mod * error_time, 'dt3': t3mod * error_time, 'dt4': t4mod * error_time}
                    Derr = array([D[0] + result['dt1'], D[1] + result['dt2'], D[2] + result['dt3'], D[3] + result['dt4']])
                    ABCD = array([a, b, c, Derr]).T
                    false_data = mutlivariate_newton(0,0,6370,0.0001,ABCD)
                    ddist = array([ true_data[i] - false_data[i] for i in range(3)])
                    result['dpos'] = norm(ddist, inf)
                    result['distance'] = sqrt(sum(ddist**2))
                    dtime = array([D[i] - Derr[i] for i in range(4)])
                    if dtime.any() != 0:
                        result['emf'] = result['dpos']/(cc*norm(dtime, inf))
                    else:
                        result['emf'] = nan
                    results.append(result)

    [print('dt: {} {} {} {}, dpos: {}, emf: {}, distance: {}'.format(i['dt1'], i['dt2'], i['dt3'], i['dt4'], i['dpos'], i['emf'], i['distance'])) for i in results]

def main():
    calc_emf(0,0, 6370, 0.0001)

if __name__=='__main__':
    main()