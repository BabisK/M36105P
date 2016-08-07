from numpy import ones, unique

def dangling_nodes_vector(A):
    nonzero_indices = unique(A.nonzero()[0])
    dangling = ones(A.shape[0])
    for i in nonzero_indices:
        dangling[i-1] = 0
    return dangling