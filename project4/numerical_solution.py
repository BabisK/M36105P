from project4.dangling_nodes import dangling_nodes_vector
from project4.load_data import loaddata
from scipy.sparse import dok_matrix, eye
from scipy.sparse.linalg import spsolve
from numpy import count_nonzero, ones, hstack
from numpy.linalg import norm


def numerical_solution(P, a):
    P_size = P.shape[0]
    dangling = dangling_nodes_vector(P)
    permutation = eye(P.shape[0])
    permutation = dok_matrix(permutation)

    lastelement = len(dangling) - 1
    for index in range(len(dangling)):
        if(dangling[index] == 1):
            for j in range(lastelement, 0, -1):
                if(dangling[j] == 0):
                    lastelement = j
                    break
            permutation[index, lastelement] = 1
            permutation[lastelement, index] = 1
            permutation[index, index] = 0
            permutation[lastelement, lastelement] = 0

    PT = P.transpose().dot(permutation.transpose())
    P = PT.transpose()
    P = P.dot(permutation)

    dangling_number = count_nonzero(dangling)

    P11 = P[:dangling.shape[0] - dangling_number, :dangling.shape[0] - dangling_number]
    P12 = P[:dangling.shape[0] - dangling_number, dangling.shape[0] - dangling_number:]

    A11 = (eye(P11.shape[0]) - a*P11).transpose()
    b11 = ones(P11.shape[0])/P.shape[0]
    x11 = spsolve(A11, b11)

    x12 = a * x11.dot(P12) + ones(P12.shape[1])/P.shape[0]

    x = hstack((x11, x12))

    x = x/norm(x, ord=1)

    print(x[:100])

if __name__=='__main__':
    P = loaddata()
    numerical_solution(P, 0.85)