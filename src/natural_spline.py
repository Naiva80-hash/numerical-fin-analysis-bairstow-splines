import numpy as np
from lu_decomposition import *
def spline_sys(x, y):
    """
    Giving x and y, it produces s which are spline coefficients
    """
    n = len(x)
    s = np.zeros(n)
    h = np.zeros(n)
    for i in range(n-1):
        h[i] = x[i+1] - x[i]
    #Boundary conditions
    s[0], s[n-1] = 0, 0
    n_prime = n - 2
    A = np.zeros((n_prime, n_prime))
    b = np.zeros(n_prime)
    for k in range(0, n_prime):
        i = k+1
        if k >0:
            A[k, k-1] = h[i-1]
        A[k,k] = 2*(h[i-1]+h[i])
        if k < n_prime - 1 :
            A[k, k+1] = h[i]
        b[k] = 6 *((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    p, L, U = LU_Decom(A)
    s_inner = LU_solve(p, L, U, b)
    s[1:n_prime+1] = s_inner
    s[0] = 0
    s[n-1] = 0
    return s
   

#Testing on slides system

# x = np.array([0, 1, 2, 3, 4])
# y = np.array([-8, -7, 0, 19, 56])

# s = spline_sys(x, y)
# print(s)

def cubic_poly(s, x, y):
    """
    The coefficients for each cubic polynomial between xi and x_i+1 
    """
    n = len(x)
    ai = np.zeros(n-1)
    bi = np.zeros(n-1)
    ci = np.zeros(n-1)
    di = np.zeros(n-1)
    hi = np.array([x[i] - x[i-1] for i in range(1, len(y))], dtype = float)
    for i in range(n-1):
        ai[i] = (s[i+1] - s[i])/(6*hi[i])
        bi[i] = s[i] / 2
        ci[i] = (y[i+1]-y[i])/hi[i] - (2*hi[i]*s[i]+hi[i]*s[i+1])/6
        di[i] = y[i]
    return ai, bi, ci, di, hi

def fun_spline(x, ai, bi, ci, di, xi):
    """
    Calculating f(x) based on cubic spline where x is a desired point between x_intial and x_theend
    """
    n = len(ai)
    for i in range(n):
        if xi[i]<= x <= xi[i+1]:
            Y = ai[i]*(x - xi[i])**3+bi[i]*(x - xi[i])**2+ci[i]*(x - xi[i])+di[i]
            return Y

    raise ValueError(f"{x} is outside the spline interval")

def diff_fun_spline(x, ai, bi, ci, xi):
    """
    Computing df(x)/dx at desired x which is in between x_intial and x_THEEND
    """
    n = len(ai)
    for i in range(n):
        if xi[i]<= x <= xi[i+1]:
            Y = 3*ai[i]*(x - xi[i])**2+2*bi[i]*(x - xi[i])+ci[i]
            return Y

    raise ValueError(f"{x} is outside the spline interval")