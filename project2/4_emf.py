from multivariate_newton import multivariate_newton
from math import pi, cos, sin, sqrt

# The speed of light (km/s)
c = 299792.458

phi = [0, pi/6, pi/3, pi/2]
theta = [0, pi/2, pi, 3*pi/2]
r = 26570

A = r*cos(phi)*cos(theta)
B = r*cos(phi)*sin(theta)
C = r*sin(phi)

def calc_emf(x,y,z,d):
    R = sqrt((A-x)**2 + (B-y)**2 + (C-z)**2)
    D = d + R/c
