import numpy as np
import math
import pickle
from natural_spline import *
import pandas as pd
import matplotlib.pyplot as plt
from clamped_spline import clamped_spline_sys

# Opening Bairstow roots
with open("roots_table3.pkl", "rb") as f:
    Roots_table = pickle.load(f)

with open("roots_res3.pkl", "rb") as f:
    Roots_res = pickle.load(f)

with open("poly_coeffs_list3.pkl", "rb") as f:
    poly_coeffs_list = pickle.load(f)

student_id = 403204853
np.random.seed(10)
sn = student_id % 10
n = math.floor(sn / 3) + 1
neu = (n - 1) / (2 * n)
gamma_n = math.factorial(n - 1)
L = np.random.uniform(0.1, 0.5)  # [m]
alpha = np.random.uniform(0, 1)
b = (alpha * L) / 3
print(f"The alpha value is: {alpha}")
print(f"The L value is: {L}")
print(f"The b value is: {b}")
print(f"The n value is: {n}")
print(f"The neu value is: {neu}")

# Constant parameters
k = 205  # [W/m.K]
h = 25   # [W/m^2.K]
Tb = 85  # [C]
Tinf = 25  # [C]
C = math.sqrt(2 * h / (b * k))
print(C)


def Iv(N, neu, z):
    s = 0
    for m in range(N):
        num = (z / 2) ** (neu + 2 * m)
        den = (math.factorial(m) * math.gamma(neu + m + 1))
        s += num / den
    return s


# Calculating A constant
N = 95
A = (Tb - Tinf) / (L ** neu * Iv(N, neu, C * L))
print(A)


def T_bessel(x):
    x = np.asarray(x, dtype=float)
    z = L - x
    if np.any(z < 0):
        raise ValueError("Bessel evaluation requires x <= L (z = L - x >= 0).")
    return Tinf + A * (z ** neu) * Iv(N, neu, C * z)


def diff_bessel_analytic(x):
    z = L - x
    if z <= 0:
        return np.nan
    term1 = neu * z ** (neu - 1) * Iv(N, neu, C * z)
    term2 = 0.5 * C * z ** neu * (Iv(N, neu - 1, C * z) + Iv(N, neu + 1, C * z))
    dTdz = A * (term1 + term2)
    return -dTdz


def diff_bessel_numeric(x, h=1e-6):
    x = np.asarray(x, dtype=float)
    dT = np.full_like(x, np.nan, dtype=float)
    mask_c = (x - h >= 0) & (x + h <= L)
    if np.any(mask_c):
        T1 = T_bessel(x[mask_c] + h)
        T2 = T_bessel(x[mask_c] - h)
        dT[mask_c] = (T1 - T2) / (2 * h)
    mask_b = (x + h > L) & (x - h >= 0) & (x <= L)
    if np.any(mask_b):
        T0 = T_bessel(x[mask_b])
        Tm = T_bessel(x[mask_b] - h)
        dT[mask_b] = (T0 - Tm) / h
    mask_f = (x - h < 0) & (x + h <= L)
    if np.any(mask_f):
        Tp = T_bessel(x[mask_f] + h)
        T0 = T_bessel(x[mask_f])
        dT[mask_f] = (Tp - T0) / h
    return dT


# Build more knots from analytical function (clustered near x=L)
def clustered_x_points(x0, x1, n_uniform, n_cluster, power):
    x_uniform = np.linspace(x0, x1, n_uniform, endpoint=True)
    s = np.linspace(0.0, 1.0, n_cluster, endpoint=True)
    x_cluster = x1 - (1.0 - s) ** power * (x1 - x0)
    x_all = np.concatenate([x_uniform, x_cluster])
    x_all = np.unique(np.clip(x_all, x0, x1))
    return np.sort(x_all)


# Base points from roots (kept for comparison)
T_x_datas = []
for r in Roots_table:
    T, y = r
    if y != "Bairstow failed":
        z = y ** 2
        x = L - z
        T_x_datas.append((T, x))

print("")
print(T_x_datas)
print(Tb)
print(Tinf)
print(L)
print("Physically roots make sense!! Based on engineering intuition :))) ")

# More knots (analytical)
cluster_power = 3.0
target_total_knots = 60

base_x = [x for (_, x) in T_x_datas]
base_x.extend([0.0, L])
base_x = np.unique(np.array(base_x, dtype=float))
base_x = np.sort(np.clip(base_x, 0.0, L))

extra_needed = max(0, target_total_knots - len(base_x))
if extra_needed > 0:
    x_extra = clustered_x_points(0.0, L, 0, extra_needed, cluster_power)
    x_knots = np.unique(np.concatenate([base_x, x_extra]))
else:
    x_knots = base_x

x_knots = np.sort(x_knots)
print(f"Baseline knot count: {len(base_x)}")
print(f"Added knots: {len(x_knots) - len(base_x)}")
print(f"Total knot count: {len(x_knots)}")
T_knots = T_bessel(x_knots)

x = np.array(x_knots, dtype=float)
T = np.array(T_knots, dtype=float)

s = spline_sys(x, T)
ai, bi, ci, di, hi = cubic_poly(s, x, T)

## Natural spline(s0 = s1 = 0)
## Generating error tables and plots for the T function
eps_x = 1e-6
xs = np.linspace(x[0], x[-1] - eps_x, 200)
Ts_spline = np.array([fun_spline(xi, ai, bi, ci, di, x) for xi in xs])
Ts_bessel = T_bessel(xs)

# Table of Error
df = pd.DataFrame({
    "x": xs,
    "T_spline": Ts_spline,
    "T_bessel": Ts_bessel,
})

df["abs_err"] = np.abs(df["T_spline"] - df["T_bessel"])
df["rel_err %"] = 100 * df["abs_err"] / (np.abs(df["T_bessel"]) + 1e-12)

print(df.head(10))
print(df.describe())

## Generating error tables and plots for T derivative
xs_d = xs[xs < (L - eps_x)]
dTs_spline = np.array([diff_fun_spline(xi, ai, bi, ci, x) for xi in xs_d])
dTs_bessel = diff_bessel_numeric(xs_d)

df_dT = pd.DataFrame({
    "x": xs_d,
    "dT_spline": dTs_spline,
    "dT_bessel": dTs_bessel,
})

df_dT["abs_err"] = np.abs(df_dT["dT_spline"] - df_dT["dT_bessel"])
df_dT["rel_err %"] = 100 * df_dT["abs_err"] / (np.abs(df_dT["dT_bessel"]) + 1e-12)

print(df_dT.head(10))
print(df_dT.describe())

## Clamped spline(function)
m0 = diff_bessel_analytic(x[0] + 1e-6)
eps = 4e-3
mn = diff_bessel_analytic(L - eps)

s_clamped = clamped_spline_sys(x, T, m0, mn)
ai_c, bi_c, ci_c, di_c, hi_c = cubic_poly(s_clamped, x, T)
Ts_spline_clamped = np.array([fun_spline(xi, ai_c, bi_c, ci_c, di_c, x) for xi in xs])
df_clamped = pd.DataFrame({
    "x": xs,
    "T_spline_clamped": Ts_spline_clamped,
    "T_bessel": Ts_bessel,
})
df_clamped["abs_err"] = np.abs(df_clamped["T_spline_clamped"] - df_clamped["T_bessel"])
df_clamped["rel_err %"] = 100 * df_clamped["abs_err"] / (np.abs(df_clamped["T_bessel"]) + 1e-12)

print(df_clamped.head(10))
print(df_clamped.describe())

## Clamped spline(derivative)
dTs_spline_clamped = np.array([diff_fun_spline(xi, ai_c, bi_c, ci_c, x) for xi in xs_d])
dTs_bessel = diff_bessel_numeric(xs_d)

df_dT_clamped = pd.DataFrame({
    "x": xs_d,
    "dT_spline_clamped": dTs_spline_clamped,
    "dT_bessel_clamped": dTs_bessel,
})

df_dT_clamped["abs_err"] = np.abs(df_dT_clamped["dT_spline_clamped"] - df_dT_clamped["dT_bessel_clamped"])
df_dT_clamped["rel_err %"] = 100 * df_dT_clamped["abs_err"] / (np.abs(df_dT_clamped["dT_bessel_clamped"]) + 1e-12)

print(df_dT_clamped.head(10))
print(df_dT_clamped.describe())

# Plots for the function
plt.figure()
plt.plot(xs, Ts_spline, label="Spline")
plt.plot(xs, Ts_bessel, label="Bessel")
plt.xlabel("x")
plt.ylabel("T")
plt.title("Natural Spline vs Analytical function (more knots)")
plt.legend()
plt.grid()

plt.figure()
plt.plot(xs, df["abs_err"])
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.legend()
plt.title("Natural Spline Absolute error for function evaluation (more knots)")
plt.grid()

# Plots for derivatives
plt.figure()
plt.plot(xs_d, dTs_spline, label="Spline derivative")
plt.plot(xs_d, dTs_bessel, label="Bessel derivative")
plt.xlabel("x")
plt.ylabel("dT")
plt.legend()
plt.title("Natural Spline diff vs Numerical diff (more knots)")
plt.grid()

plt.figure()
plt.plot(xs_d, df_dT["abs_err"])
plt.xlabel("x")
plt.ylabel("Absolute Error(derivative)")
plt.title("Natural Spline derivation Absolute error vs function derivation (more knots)")
plt.grid()

# Clamped version plots
plt.figure()
plt.plot(xs, Ts_spline_clamped, label="Spline")
plt.plot(xs, Ts_bessel, label="Bessel")
plt.xlabel("x")
plt.ylabel("T")
plt.title("Clamped Spline vs Analytical function (more knots)")
plt.legend()
plt.grid()

plt.figure()
plt.plot(xs, df_clamped["abs_err"])
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.legend()
plt.title("Clamped Spline Absolute error for function evaluation (more knots)")
plt.grid()

plt.figure()
plt.plot(xs_d, dTs_spline_clamped, label="Spline derivative")
plt.plot(xs_d, dTs_bessel, label="Bessel derivative")
plt.xlabel("x")
plt.ylabel("dT")
plt.legend()
plt.title("Clamped Spline diff vs Numerical diff (more knots)")
plt.grid()

plt.figure()
plt.plot(xs_d, df_dT_clamped["abs_err"])
plt.xlabel("x")
plt.ylabel("Absolute Error(derivative)")
plt.title("Clamped Spline derivation Absolute error vs function derivation (more knots)")
plt.grid()
plt.show()
