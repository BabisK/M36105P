from numpy import array, cos, sin
from project2.emf import calc_emf

# The speed of light (km/s)
cc = 299792.458

phi = array([0.96, 0.98, 1, 1.02])
theta = array([1.5, 1.54, 1.58, 1.62])
r = 26570

A = r*cos(phi)*cos(theta)
B = r*cos(phi)*sin(theta)
C = r*sin(phi)

def main():
    calc_emf(0,0, 6370, 0.0001, A, B, C)

if __name__=='__main__':
    main()