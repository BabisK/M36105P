from numpy import ones, inner

from project4.dangling_nodes import dangling_nodes_vector

def power_method(P, a, t):
    P_size = P.shape[0]
    x = ones(P_size)/P_size
    dangling = dangling_nodes_vector(P)
    iteration = 0
    while(True):
        iteration += 1
        x_old = x
        x = a*P.transpose().dot(x) + (a * inner(x, dangling) + (1 - a))*(ones(P_size)/P_size)
        if (abs(x-x_old) < t).all():
            break
    return x, iteration