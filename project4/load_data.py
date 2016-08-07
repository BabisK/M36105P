from numpy import loadtxt, save, load, unique
from scipy.sparse import csc_matrix
import os

def calculate_P(A):
    counts = unique(A[:,0], return_counts=True, return_inverse=True)
    weights = []
    for i in counts[1]:
        weights.append(1/counts[2][i])
    return csc_matrix((weights, (A[:,0], A[:,1])))

def loaddata():
    a = None
    if os.path.isfile('./out.web-BerkStan.data.npy') and os.path.isfile('./out.web-BerkStan.indices.npy') and os.path.isfile('./out.web-BerkStan.indptr.npy'):
        data = load('./out.web-BerkStan.data.npy')
        indices = load('./out.web-BerkStan.indices.npy')
        indptr = load('./out.web-BerkStan.indptr.npy')
        a = csc_matrix((data, indices, indptr))
    elif os.path.isfile('./out.web-BerkStan.npy'):
        a = load('./out.web-BerkStan.npy')
        a = calculate_P(a)
        save('./out.web-BerkStan.data.npy', a.data)
        save('./out.web-BerkStan.indices.npy', a.indices)
        save('./out.web-BerkStan.indptr.npy', a.indptr)
    else:
        a = loadtxt('./out.web-BerkStan', comments='%')
        save('./out.web-BerkStan.npy', a)
        a = calculate_P(a)
        save('./out.web-BerkStan.data.npy', a.data)
        save('./out.web-BerkStan.indices.npy', a.indices)
        save('./out.web-BerkStan.indptr.npy', a.indptr)
    return a