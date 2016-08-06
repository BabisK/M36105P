from numpy import flipud, insert, append, zeros, multiply, split
from numpy.fft import rfft, irfft

def precompute_g(T):
    return rfft(append(T[:, 0], insert(flipud(T[0, 1:]), 0, 0)))

def fftmul(g, x):
    h = multiply(g, rfft(append(x, zeros(len(x)))))
    return split(irfft(h), 2)[0]

if __name__=='__main__':
    from numpy import ones
    from project3.my_toeplitz import my_toeplitz
    fftmul(my_toeplitz(5), ones(5))