from project3.my_toeplitz import  my_toeplitz
from project3.gradient_descent import gradient_descent
from project3.conjugate_gradient import conjugate_gradient
from numpy import log10, zeros, ones
from time import process_time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import logging

def problem2():
    logging.basicConfig(filename='problem2.log', level=logging.DEBUG)

    logging.info('Starting problem 2')

    lmin = 9
    lmax = 13

    matrix = [my_toeplitz(2**l) for l in range(lmin, lmax + 1)]
    b = [ones(2**l) for l in range(lmin, lmax + 1)]
    x0 = [zeros(2**l) for l in range(lmin, lmax + 1)]
    max_iterations = 1000

    gd_time = []
    cg_time = []

    result_conjugate_gradient = []
    result_gradient_descent = []

    for i in range(len(matrix)):
        logging.info('Starting caclulation for matrix of size {}'.format(i))

        cg_time.append(process_time())
        result_conjugate_gradient.append(conjugate_gradient(matrix[i], x0[i], b[i], max_iterations))
        cg_time[-1] = process_time() - cg_time[-1]

        gd_time.append(process_time())
        result_gradient_descent.append(gradient_descent(matrix[i], x0[i], b[i], max_iterations))
        gd_time[-1] = process_time() - gd_time[-1]

        logging.info('Ending calculation for matrix of size {}'.format(i))

    logging.info('Ending problem 2')

    for i in range(lmin, lmax+1):
        print('Dimension {}: GD {} iterations in {}s, CG {} iterations in {}s'.format(2**i, result_gradient_descent[i-lmin][1], gd_time[i-lmin], result_conjugate_gradient[i-lmin][1], cg_time[i-lmin]))

    gd_plot = plt.plot([2**i for i in range(lmin, lmax+1)], log10(gd_time), label = 'Gradient Descent')
    cg_plot = plt.plot([2 ** i for i in range(lmin, lmax + 1)], log10(cg_time), label = 'Conjugate Gradient')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('$Log_{10}(time (s))$')
    gd_line = mpatches.Patch(color='blue', label='Gradient Descent')
    cg_line = mpatches.Patch(color='green', label='Conjugate Gradient')
    plt.legend(handles = [gd_line, cg_line], loc = 0)
    plt.savefig('2logtime')

    plt.close()

    gd_plot = plt.plot([2**i for i in range(lmin, lmax+1)], gd_time, label = 'Gradient Descent')
    cg_plot = plt.plot([2 ** i for i in range(lmin, lmax + 1)], cg_time, label = 'Conjugate Gradient')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Time (s)')
    gd_line = mpatches.Patch(color='blue', label='Gradient Descent')
    cg_line = mpatches.Patch(color='green', label='Conjugate Gradient')
    plt.legend(handles = [gd_line, cg_line], loc = 0)
    plt.savefig('2time')

    plt.close()

    gd_plot = plt.plot([2**i for i in range(lmin, lmax+1)], log10([result_gradient_descent[i][1] for i in range(len(result_gradient_descent))]), label = 'Gradient Descent')
    cg_plot = plt.plot([2 ** i for i in range(lmin, lmax + 1)], log10([result_conjugate_gradient[i][1] for i in range(len(result_conjugate_gradient))]), label = 'Conjugate Gradient')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('$Log_{10}(iterations)$')
    gd_line = mpatches.Patch(color='blue', label='Gradient Descent')
    cg_line = mpatches.Patch(color='green', label='Conjugate Gradient')
    plt.legend(handles = [gd_line, cg_line], loc = 0)
    plt.savefig('2logiterations')

    plt.close()

    gd_plot = plt.plot([2**i for i in range(lmin, lmax+1)], [result_gradient_descent[i][1] for i in range(len(result_gradient_descent))], label = 'Gradient Descent')
    cg_plot = plt.plot([2 ** i for i in range(lmin, lmax + 1)], [result_conjugate_gradient[i][1] for i in range(len(result_conjugate_gradient))], label = 'Conjugate Gradient')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Iterations')
    gd_line = mpatches.Patch(color='blue', label='Gradient Descent')
    cg_line = mpatches.Patch(color='green', label='Conjugate Gradient')
    plt.legend(handles = [gd_line, cg_line], loc = 0)
    plt.savefig('2iterations')

if __name__=='__main__':
    problem2()