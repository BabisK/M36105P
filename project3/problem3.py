from project3.my_toeplitz import  my_toeplitz
from project3.my_preconditioner import my_preconditioner
from project3.preconditioned_conjugate_gradient import preconditioned_conjugate_gradient
from numpy import log10, ones, zeros
from time import process_time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

def problem3():
    logging.basicConfig(filename='problem3.log', level=logging.DEBUG)

    logging.info('Starting problem 3')

    lmin = 6
    lmax = 10

    matrix = [my_toeplitz(2**l) for l in range(lmin, lmax + 1)]
    b = [ones(2**l) for l in range(lmin, lmax + 1)]
    x0 = [zeros(2**l) for l in range(lmin, lmax + 1)]
    preconditioner = [my_preconditioner(2**l) for l in range(lmin, lmax + 1)]

    max_iterations = 1000

    pcg_time = []

    result_preconditioned_conjugate_gradient = []

    for i in range(len(matrix)):
        logging.info('Starting caclulation for matrix of size {}'.format(i))

        pcg_time.append(process_time())
        result_preconditioned_conjugate_gradient.append(preconditioned_conjugate_gradient(matrix[i], x0[i], b[i], preconditioner[i], max_iterations))
        pcg_time[-1] = process_time() - pcg_time[-1]

        logging.info('Ending calculation for matrix of size {}'.format(i))

    logging.info('Ending problem 3')

    for i in range(lmin, lmax+1):
        print('Dimension {}: PCG {} iterations in {}s'.format(2**i, result_preconditioned_conjugate_gradient[i-lmin][1], pcg_time[i-lmin]))

    pcg_plot = plt.plot([2 ** i for i in range(lmin, lmax + 1)], log10(pcg_time), label = 'Preconditioned Conjugate Gradient', color='red')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('$Log_{10}(time (s))$')
    pcg_line = mpatches.Patch(color='red', label='Preconditioned Conjugate Gradient')
    plt.legend(handles = [pcg_line], loc = 0)
    plt.savefig('3logtime')

    plt.close()

    pcg_plot = plt.plot([2 ** i for i in range(lmin, lmax + 1)], pcg_time, label = 'Preconditioned Conjugate Gradient', color='red')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Time (s)')
    pcg_line = mpatches.Patch(color='red', label='Preconditioned Conjugate Gradient')
    plt.legend(handles = [pcg_line], loc = 0)
    plt.savefig('3time')

    plt.close()

    pcg_plot = plt.plot([2 ** i for i in range(lmin, lmax + 1)], log10([result_preconditioned_conjugate_gradient[i][1] for i in range(len(result_preconditioned_conjugate_gradient))]), label = 'Preconditioned Conjugate Gradient', color='red')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('$Log_{10}(iterations)$')
    pcg_line = mpatches.Patch(color='red', label='Preconditioned Conjugate Gradient')
    plt.legend(handles = [pcg_line], loc = 0)
    plt.savefig('3logiterations')

    plt.close()

    pcg_plot = plt.plot([2 ** i for i in range(lmin, lmax + 1)], [result_preconditioned_conjugate_gradient[i][1] for i in range(len(result_preconditioned_conjugate_gradient))], label = 'Conjugate Gradient', color='red')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Iterations')
    pcg_line = mpatches.Patch(color='red', label='Preconditioned Conjugate Gradient')
    plt.legend(handles = [pcg_line], loc = 0)
    plt.savefig('3iterations')


if __name__=='__main__':
    problem3()