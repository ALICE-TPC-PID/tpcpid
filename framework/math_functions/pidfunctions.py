import numpy as np

def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_unpacked(x, *p):
    A, x0, sigma = p
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def double_gauss(x, A1, x01, sigma1, A2, x02, sigma2):
    return A1 * np.exp(-(x - x01) ** 2 / (2 * sigma1 ** 2)) + A2 * np.exp(-(x - x02) ** 2 / (2 * sigma2 ** 2))

def gausscale(x, mean, sigma, scale):
    return (scale/0.4)*np.exp(-0.5*((x-mean)/sigma)**2)

def linear(x, a, b):
    return a*x + b

def gausgauslin(x, mu1, sigma1, scale1, mu2, sigma2, scale2, a, b):
    return gausscale(x, mu1, sigma1, scale1) + gausscale(x, mu2, sigma2, scale2) + (a*x + b)

def gauslin(x, mu1, sigma1, scale1, a, b):
    return gausscale(x, mu1, sigma1, scale1) + (a*x + b)