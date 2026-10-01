import numpy as np
import math
import pickle

"""
Assignment 1 — Advanced Numerical Mathematics

Author: Navid Saeedi
Course: Advanced Numerical Mathematics
Semester: 2025–2026
Date: 2026-02-04

Description:
    Implementing Bairstow Method and spline in order to find an approxiamte for temperature disturbution.

How to run:
    Just press F5 on Vscode, and wait a lot because Biarstow method is so slow!! It finds all the roots after all!! :)

"""
number = 4032048856
np.random.seed(10)

# Initialize parameters
sn = number % 10
n = math.floor(sn / 3) + 1
neu = (n - 1) / (2 * n)
gamma_n = math.factorial(n - 1)
L = np.random.uniform(0.1, 0.5)  # [m]
alpha = np.random.uniform(0, 1)
b = (alpha * L) / 3

# Display the calculated values
print(f"The alpha value is: {alpha}")
print(f"The L value is: {L}")
print(f"The b value is: {b}")
print(f"The n value is: {n}")
print(f"The neu value is: {neu}")

# Constant parameters
k = 205  # [W/m.K]
h = 25  # [W/m^2.K]
Tb = 85  # [C]
Tinf = 25  # [C]
C = math.sqrt(2 * h / (b * k))
print(f"C: {C}")

# Bessel function (optimized version)
def Iv(N, neu, z):
    """
    Bessel function implementation
    """
    s = 0
    for m in range(N):
        num = (z / 2) ** (neu + 2 * m)
        den = math.factorial(m) * math.gamma(neu + m + 1)
        s += num / den
    return s

# Calculate A constant
N = 95 
A = (Tb - Tinf) / (L ** 0.25 * Iv(N, neu, C * L))
print(f"A: {A}")

# Generate T_targ list
T_targ_list = []
runs = 10
for r in range(runs):
    T = np.random.uniform(Tinf, Tb)
    T_targ_list.append(T)

T_targ = T_targ_list[0]

R = A * (C / 2) ** neu

# M(m, neu) function
def M(m, neu):
    return (C ** 2 / 4) ** m / (math.factorial(m) * math.gamma(m + neu + 1))

# Building polynomial
def build_poly(T_targ, N):
    deltaT = T_targ - Tinf
    degree = 4 * (N - 1) + 1
    coeffs = np.zeros(degree + 1)
    coeffs[0] = -deltaT
    for m in range(N):
        p = 4 * m + 1
        coeffs[p] += R * M(m, neu)
    return coeffs

# Bairstow's method for root finding
def Bairstow(poly_coeff, r, s, eps_step, eps_reminder, max_iter):
    """
    Takes the polynomial coefficients, finds all the quadratic factors and linear factor of this polynomial. These factors make
    reminders less than eps_reminder
    """
    iter = 0
    a = np.array(poly_coeff, dtype=float)
    quadric_factors = []
    linear_factor = []
    a = a / np.max(np.abs(a))  # Normalize coefficients
    n = len(a) - 1
    b = np.zeros(n + 1)
    c = np.zeros(n)

    converged = True 
    
    while len(a) > 3:
        guess_list = [
    (0, 0),          # Original guess
    (0.5, 0),        # Slightly adjusted guesses
    (-0.5, 0),
    (0, 0.5),
    (0, -0.5),
    (1, -1),         # Larger magnitude guesses
    (-1, -1),
    (0.1, 0),
    (-0.1, 0),
    (0, 0.1),
    (0, -0.1),
    (0.2, -0.1),
    (-0.2, -0.1),
    (1.5, -1.5),
    (-1.5, 1.5),
    (0.3, 0.3),
    (-0.3, -0.3),
    (0.2, 0.5),
    (-0.2, 0.5),
    (1, 0.1),
    (-1, -0.1),
    
    # New Guesses:
    (0.7, 0.3),      # Slightly larger positive real part
    (-0.7, 0.3),     # Negative real part with small imaginary component
    (0.8, -0.2),     # Larger positive real part, small negative imaginary part
    (-0.8, -0.2),    # Larger negative real part, small negative imaginary part
    (0.4, -0.6),     # Smaller real part with negative imaginary component
    (-0.4, 0.6),     # Smaller real part with positive imaginary component
    (0.6, 0.6),      # Balanced real and imaginary parts with a larger magnitude
    (-0.6, -0.6),    # Negative real and imaginary parts with a larger magnitude
    (0.2, -0.8),     # Smaller real part with larger negative imaginary part
    (-0.2, 0.8),     # Smaller real part with larger positive imaginary part
    (1.2, 0.2),      # Larger real part, small positive imaginary component
    (-1.2, -0.2),    # Larger negative real part, small negative imaginary component
]

        max_attempts = len(guess_list)
        success_flag = False
        delta_r = delta_s = np.inf
        iter = 0
        for attempt in range(max_attempts):
            r, s = guess_list[attempt]
            iter = 0
            delta_r = delta_s = np.inf
            while iter < max_iter:
                b.fill(0)
                c.fill(0)
                for k in range(n, -1, -1):
                    if k == n:
                        b[n] = a[n]
                    elif k == n - 1:
                        b[n - 1] = a[n - 1] + r * b[n]
                    elif -1 < k < n - 1:
                        b[k] = a[k] + r * b[k + 1] + s * b[k + 2]
                for k in range(n - 1, -1, -1):
                    if k == n - 1:
                        c[n - 1] = b[n]
                    elif k == n - 2:
                        c[n - 2] = b[n - 1] + r * c[n - 1]
                    elif 0 < k < n - 2:
                        c[k] = b[k + 1] + r * c[k + 1] + s * c[k + 2]
                    elif k == 0:
                        c[0] = b[1] + r * c[1] + s * c[2]
                
                if np.max(np.abs(b)) > 1e100 or np.max(np.abs(c)) > 1e100:
                    break
                den = (c[1] * c[1] - c[2] * c[0])
                if abs(den) < 1e-12:
                    break
                delta_r = (c[2] * b[0] - c[1] * b[1]) / den
                delta_s = (c[0] * b[1] - c[1] * b[0]) / den
                if abs(delta_r) > 1e6 or abs(delta_s) > 1e6:
                    break
                lam = 0.2
                r += lam * delta_r
                s += lam * delta_s
                if not np.isfinite(r) or not np.isfinite(s):
                    break
                iter += 1
                # b with updated r and s
                b_check = np.zeros(n + 1)
                for k in range(n, -1, -1):
                    if k == n:
                        b_check[n] = a[n]
                    elif k == n - 1:
                        b_check[n - 1] = a[n - 1] + r * b_check[n]
                    elif -1 < k < n - 1:
                        b_check[k] = a[k] + r * b_check[k + 1] + s * b_check[k + 2]
                if max(abs(delta_r), abs(delta_s)) < eps_step and max(abs(b_check[0]), abs(b_check[1])) < eps_reminder:
                    b = b_check
                    success_flag = True
                    break
            if success_flag:
                break
        if success_flag:
            a = b[2:]
            a = a / np.max(np.abs(a))
            n = len(a) - 1
            quadric_factors.append((float(r), float(s)))
        else:
            print("Bairstow did not converge")
            converged = False 
            break

    if len(a) == 3:
        r = -a[1] / a[2]
        s = -a[0] / a[2]
        quadric_factors.append((float(r), float(s)))
    elif len(a) == 2:
        r = -a[0] / a[1]
        linear_factor.append(float(r))
    
    return quadric_factors, linear_factor, converged

# Find roots from quadratic and linear factors
def root_quadric(quadratics, linear):
    """
    Take quadratics and linear factors, compute all the roots for this specific polynomial based on them.
    """
    roots = []
    for r, s in quadratics:
        D = r**2 + 4 * s
        sqrtD = np.sqrt(D + 0j)
        roots.append((r + sqrtD) / 2)
        roots.append((r - sqrtD) / 2)
    for L in linear:
        roots.append(L)
    return np.array(roots)

# Calculate maximum residuals
# Function to evaluate polynomial using Horner's method
def horner_asc(a, x):
    """
    Evaluate polynomial P(x) with coeffs a at x=x,
    using horny method :)
    """
    p = 0.0 + 0j
    for coeff in a[::-1]:
        p = p * x + coeff
    return p

# Function for scaled version of Horner's method to check numerical stability
def scale_asc(a, x):
    """
    Evaluate the scale parameter of polynomial P(x)
    delta
    """
    ax = abs(x)
    s = 0.0
    for coeff in a[::-1]:
        s = s * ax + abs(coeff)
    return s

# Function to calculate maximum and median relative residuals for a given polynomial and roots
def max_rel_residual(a, roots):
    """
    Take coeffs and roots of a polynomail and returns, maximum residual and average of residuals for that
    polynomial
    """
    vals = []
    if len(roots) != 0:
        for z in roots:
            num = abs(horner_asc(a, z))  # Polynomial evaluation at root
            den = scale_asc(a, z)  # Scaled evaluation for numerical stability
            vals.append(num / den if den != 0 else num)
        return max(vals), np.average(vals)
    else:
        return None, None


# Running Bairstow for different T_targ
Roots_table = []
Roots_res = []
poly_coeffs_list = []
All_Roots_table = {}
num_sucess_runs = 0
for i in range(runs):
    poly_coeff = build_poly(T_targ_list[i], N)
    poly_coeffs_list.append(poly_coeff)
    r = -1
    s = -1
    eps_step = 1
    eps_reminder = 1e-7
    max_iter = 1e3
    quadric, linear, converged = Bairstow(poly_coeff, r, s, eps_step, eps_reminder, max_iter)
    if converged:
        roots = root_quadric(quadric, linear)

        deg = len(poly_coeff) - 1
        if len(roots) == deg:
        # Saving all the roots for that specific T_target
            All_Roots_table[f"Roots in T_traget = {T_targ_list[i]}"] = roots
        else:
            converged = False
            print(f"Bairstow returned {len(roots)} roots but degree is {deg} (incomplete).")
    else:
        roots = np.array([])
    if not converged:
        Roots_table.append((T_targ_list[i], "Bairstow failed"))
        print(f"Bairstow failed at run: {i}")
    else:
        tol_im = 1e-8
        real_roots = [float(z.real) for z in roots if abs(z.imag) < tol_im]
        if len(real_roots) != 0:
            for r in real_roots:
                Roots_table.append((T_targ_list[i], r))
            print(f"Roots were found at run: {i}")
            num_sucess_runs += 1
        else:
            Roots_table.append((T_targ_list[i], "No real root detected"))
            print(f"No real roots detected at run: {i}")


    a = np.array(poly_coeff, dtype=float)
    mr_b, med_b = max_rel_residual(a, roots)
    if mr_b is not None and mr_b is not None:
        Roots_res.append((float(mr_b), float(med_b)))
        print(f"Max residual: {mr_b}, Average residual: {med_b}")
    else:
        print("There are no residuals!")



with open("roots_table3.pkl", "wb") as f:
    pickle.dump(Roots_table, f)

with open("roots_res3.pkl", "wb") as f:
    pickle.dump(Roots_res, f)

with open("poly_coeffs_list3.pkl", "wb") as f:
    pickle.dump(poly_coeffs_list, f)

with open("All_roots_3.pkl", "wb") as f:
    pickle.dump(All_Roots_table, f)

print(f" Number of successful runs are: {num_sucess_runs}")
        