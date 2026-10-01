import numpy as np
from LU_Decompositions import *
def clamped_spline_sys(x, y, m0, mn):
    """
    Forces the right Boundary and Left Boundary have specific slopes like m0 and m1
    """
    n = len(x)
    s = np.zeros(n)
    h = np.zeros(n)
    for i in range(n-1):
        h[i] = x[i+1] - x[i]
    A = np.zeros((n, n))
    b = np.zeros(n)
    #Left boundary uses m0 = T'(x0)
    A[0, 0] = 2*h[0]
    A[0, 1] = h[0]
    b[0] = 6*((y[1]-y[0])/h[0] - m0)
    
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i,i] = 2*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    
    #Right boundary uses mn = T'(xn)
    A[n-1, n-2] = h[n-2]
    A[n-1, n-1] = 2*h[n-2]
    b[n-1] = 6*(mn - (y[n-1]-y[n-2])/h[n-2])
    p, L, U = LU_Decom(A)
    s = LU_solve(p, L, U, b)
    return s