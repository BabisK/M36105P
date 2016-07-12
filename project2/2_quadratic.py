from numpy import array, column_stack, sum, sqrt
from numpy.linalg import det

# The data from the 4 satelites
ABCD = array([[15600, 7540, 20140, 0.07074], [18760, 2750, 18610, 0.07220], [17610, 14630, 13480, 0.07690],
              [19170, 610, 18390, 0.07242]])

# The speed of light (km/s)
c = 299792.458

U = -column_stack((-2*(ABCD[1:,0:3]-ABCD[0,0:3]), 2*(c**2)*(ABCD[1:,3]-ABCD[0,3])))
W = (ABCD[0,0]**2) + (ABCD[0,1]**2) + (ABCD[0,2]**2) -sum(ABCD[1:,0:3]**2, axis=1) - (c**2)*(ABCD[0,3]**2) + (c**2)*(ABCD[1:,3]**2)

k = det(U[:,0:3])

xd = -det(column_stack((U[:,1], U[:,2], U[:,3])))
xw = -det(column_stack((U[:,1], U[:,2], W)))

yd = -det(column_stack((U[:,0], U[:,2], U[:,3])))
yw = -det(column_stack((U[:,0], U[:,2], W)))

zd = -det(column_stack((U[:,0], U[:,1], U[:,3])))
zw = -det(column_stack((U[:,0], U[:,1], W)))

a = (xd/k)**2 + (yd/k)**2 + (zd/k)**2 - (c**2)
b = 2*((xd/k)*(xw/k) - (xd/k)*ABCD[0,0] + (yd/k)*(yw/k) - (yd/k)*ABCD[0,1] + (zd/k)*(zw/k) - (zd/k)*ABCD[0,2] + (c**2)*ABCD[0,3])
ct = (xw/k)**2 - 2*ABCD[0,0]*(xw/k) + ABCD[0,0]**2 + (yw/k)**2 - 2*ABCD[0,1]*(yw/k) + ABCD[0,1]**2 + (zw/k)**2 - 2*ABCD[0,2]*(zw/k) + ABCD[0,2]**2 - (c**2)*(ABCD[0,3]**2)

r1 = (-b + sqrt(b**2 - 4*a*ct))/(2*a)
r2 = (-b - sqrt(b**2 - 4*a*ct))/(2*a)

s1 = array([(xd/k)*r1 + xw/k, (yd/k)*r1 + yw/k, (zd/k)*r1 + zw/k])
s2 = array([(xd/k)*r2 + xw/k, (yd/k)*r2 + yw/k, (zd/k)*r2 + zw/k])

print("Solution 1: {}", s1)
print("Solution 2: {}", s2)