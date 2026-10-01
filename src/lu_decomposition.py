import numpy as np

def LU_Decom(x):
    """
    Computes matrix x, (p, l, u) factors
    """
    n = len(x)
    u = np.zeros((n,n))
    l = np.eye(n)
    p = np.arange(0, n)
    A = x.copy().astype(float)
    r = np.zeros(n)
    for j in range(n):
        for i in range(j):
            s = 0
            for k in range(i):
                s+= l[i,k] * u[k,j]
            u[i, j] = A[i, j] - s
        for i in range(j, n):
            s = 0
            for k in range(j):
                s += l[i, k] * u[k, j]
            r[i] = A[i, j] - s
        imax = j + np.argmax(np.abs(r[j:]))
        if imax != j:
            A[[j, imax], :] = A[[imax, j], :]
            p[[j, imax]] = p[[imax, j]]
            if j > 0:
                l[[j, imax], :j] = l[[imax, j], :j]
            r[[j, imax]] = r[[imax, j]]
            for i in range(j, n):
                s = 0
                for k in range(j):
                    s += l[i, k] * u[k, j]
                r[i] = A[i, j] - s
        u[j, j] = r[j]
        tiny = 1e-12
        if abs(u[j, j]) < tiny:
            raise ValueError(f"Singular / Nearly singular pivot at column {j}")   
        for i in range(j+1,n):
            s = 0
            for k in range(j):
                s += l[i,k] * u[k, j]
            l[i, j] = (A[i, j] - s)/u[j ,j]
    return p,l, u


#Testing alghorithm
# A = np.matrix([[0, 2, 1],[1, 1, 0],[2, 1, 1]])

# p, L, U =LU_Decom(A)

# print(p)
# print(L)
# print(U)
# PA = A[p, :]

# print(np.allclose(PA, L @ U))

def LU_solve(p, L, U, b):
    """
    Solving the linear system Ax =b, By taking p,L,U from LU_Decom function
    and solve the system of equations.
    """
    b = np.asarray(b, float)
    is_vector = (b.ndim == 1)
    if is_vector:
        b = b.reshape(-1, 1)

    n = L.shape[0]
    X = np.zeros((n, b.shape[1]))

    for j in range(b.shape[1]):
        e = b[p, j]

        # forward solve Ly = e
        y = np.zeros(n)
        for i in range(n):
            y[i] = e[i] - np.dot(L[i, :i], y[:i])  # L has 1s on diagonal

        # back solve Ux = y
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

        X[:, j] = x

    return X.ravel() if is_vector else X

#Testinf linear_eq solver
# A =  np.array([[0., 2., 1.],[1., 1., 0.],[2., 1., 1.]])
# b = np.array([1., 2., 3.])
# x = LU_solve(p, L, U, b)
# print(x)
# print(np.linalg.solve(A, b))
