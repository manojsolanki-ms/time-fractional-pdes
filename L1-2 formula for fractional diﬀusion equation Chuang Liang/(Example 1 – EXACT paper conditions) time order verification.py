import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ============================================================
# PARAMETERS (Example 1 – EXACT paper conditions)
# ============================================================
alpha = 3/4          # any 0 < alpha < 1
m = 200  # try 10, 20, 40, 80
T = 1.0/2
a = 0 
b = np.pi
h = (b-a)/m

N = 20
tau = T/N

# tau = 0.5 * h**2     # IMPORTANT: time–space balance
# N = int(T / tau)

# ============================================================
# GRID
# ============================================================
x = np.linspace(a, b, m+1)
y = np.linspace(a, b, m+1)
NI = m - 1

# interior mesh for plotting
X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij')

# ============================================================
# INITIAL CONDITION (Example 1)
# u(x,y,0) = 0
# ============================================================
U0 = np.zeros((NI, NI))

# ============================================================
# SOURCE TERM f(x,y,t) (Example 1, general alpha)
# ============================================================
def source_term(t):
    coef = (2 / (gamma(3 - alpha))) * t**(2 - alpha) + 2*t**2
    return coef * np.sin(X) * np.sin(Y)

# ============================================================
# L1-2 COEFFICIENTS
# ============================================================
def a(l):
    return (l+1)**(1-alpha) - l**(1-alpha)

def b(l):
    return ((l+1)**(2-alpha) - l**(2-alpha))/(2-alpha) \
           - 0.5*((l+1)**(1-alpha) + l**(1-alpha))

def c_coeff(n):
    c = np.zeros(n)
    for l in range(n):
        if l == 0:
            c[l] = a(0) + b(0)
        elif l <= n-2:
            c[l] = a(l) + b(l) - b(l-1)
        else:
            c[l] = a(l) - b(l-1)
    return c

# ============================================================
# COMPACT DIFFERENCE MATRICES
# ============================================================
A = np.zeros((NI, NI))
Tmat = np.zeros((NI, NI))

for i in range(NI):
    A[i, i] = 10
    Tmat[i, i] = -2
    if i > 0:
        A[i, i-1] = 1
        Tmat[i, i-1] = 1
    if i < NI-1:
        A[i, i+1] = 1
        Tmat[i, i+1] = 1

A /= 12.0
Tmat /= h**2

# ============================================================
# ADI PARAMETER
# ============================================================
c0 = a(0) + b(0)
S = tau**alpha * gamma(2 - alpha) / c0
L = A - S * Tmat

# ============================================================
# STORAGE
# ============================================================
U = [U0.copy()]

# ============================================================
# TIME STEPPING (paper Eq. 2.10)
# ============================================================
for n in range(1, N+1):

    tn = n * tau

    if n == 1:
        history = U[0].copy()
    else:
        c = c_coeff(n)
        history = np.zeros((NI, NI))
        for k in range(1, n):
            history += (c[n-k-1] - c[n-k]) * U[k]
        history += c[n-1] * U[0]

    RHS = (
        A @ history @ A.T
        + S**2 * (Tmat @ history @ Tmat.T)
    ) / c0 + S * (A @ source_term(tn) @ A.T)

    # ADI solve
    Ustar = np.zeros_like(history)
    for j in range(NI):
        Ustar[:, j] = solve(L, RHS[:, j])

    Un = np.zeros_like(history)
    for i in range(NI):
        Un[i, :] = solve(L, Ustar[i, :])

    U.append(Un.copy())

U_num = U[-1]

# ============================================================
# EXACT SOLUTION (Example 1)
# ============================================================
U_exact = T**2 * np.sin(X) * np.sin(Y)

# ============================================================
# ERROR
# ============================================================
err = np.linalg.norm(U_num - U_exact, np.inf)
print(f"Example 1 | alpha={alpha:.2f} | m={m} | max error={err:.3e}")

# # ============================================================
# # 3D SURFACE PLOTS
# # ============================================================
# fig = plt.figure(figsize=(12, 5))

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot_surface(X, Y, U_num, cmap='viridis')
# ax1.set_title('Numerical solution at T = 1')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(X, Y, U_exact, cmap='viridis')
# ax2.set_title('Exact solution at T = 1')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')

# plt.tight_layout()
# plt.show()

# # ============================================================
# # 2D CONTOUR PLOTS
# # ============================================================
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.contourf(X, Y, U_num, 20)
# plt.colorbar()
# plt.title('Numerical contour at T = 1')

# plt.subplot(1, 2, 2)
# plt.contourf(X, Y, U_exact, 20)
# plt.colorbar()
# plt.title('Exact contour at T = 1')

# plt.tight_layout()
# plt.show()

# ============================================================
# 2D LINE PLOT (y = 0.5)
# ============================================================
mid = NI // 2

plt.figure()
plt.plot(x[1:-1], U_num[:, mid], 'o-', label='Numerical')
plt.plot(x[1:-1], U_exact[:, mid], '--', label='Exact')
plt.xlabel('x')
plt.ylabel('u(x, 0.5, T)')
plt.legend()
plt.grid(True)
plt.title('Mid-section line plot (y = 0.5)')
plt.show()
