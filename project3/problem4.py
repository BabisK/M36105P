from project3.my_preconditioner import my_preconditioner
from project3.my_toeplitz import my_toeplitz
from numpy.linalg import inv, eigvals
from numpy.linalg.linalg import dot
from numpy import log10
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

def problem4():
    logging.basicConfig(filename='problem4.log', level=logging.DEBUG)

    logging.info('Starting problem 3')

    lmin = 10
    lmax = 13

    matrix = [my_toeplitz(2 ** l) for l in range(lmin, lmax + 1)]
    preconditioner = [my_preconditioner(2 ** l) for l in range(lmin, lmax + 1)]
    precond_matrix = [dot(inv(preconditioner[i]), matrix[i]) for i in range(len(matrix))]

    matrix_eig = [eigvals(m) for m in matrix]
    precond_matrix_eig = [eigvals(m) for m in precond_matrix]

    matrix_cond = [max(matrix_eig[i])/min(matrix_eig[i]) for i in range(len(matrix_eig))]
    precond_matrix_cond = [max(precond_matrix_eig[i]) / min(precond_matrix_eig[i]) for i in range(len(precond_matrix_eig))]

    mateig = plt.plot([2 ** i for i in range(lmin, lmax + 1)], matrix_cond, label = 'CondT')
    pmateig = plt.plot([2 ** i for i in range(lmin, lmax + 1)], precond_matrix_cond, label='CondPT')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Condition')
    mateig_line = mpatches.Patch(color='blue', label='$T_n$')
    pmateig_line = mpatches.Patch(color='green', label='$P_n^{-1}T_n$')
    plt.legend(handles = [mateig_line, pmateig_line], loc = 0)
    plt.savefig('4condition')

    plt.close()

    mateig = plt.plot([2 ** i for i in range(lmin, lmax + 1)], log10(matrix_cond), label = 'CondT')
    pmateig = plt.plot([2 ** i for i in range(lmin, lmax + 1)], log10(precond_matrix_cond), label='CondPT')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('$Log_{10}(Condition)$')
    mateig_line = mpatches.Patch(color='blue', label='$T_n$')
    pmateig_line = mpatches.Patch(color='green', label='$P_n^{-1}T_n$')
    plt.legend(handles = [mateig_line, pmateig_line], loc = 0)
    plt.savefig('4logcondition')

if __name__=='__main__':
    problem4()