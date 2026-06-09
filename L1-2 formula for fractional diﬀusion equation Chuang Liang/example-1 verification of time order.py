import numpy as np
from scipy.special import gamma
from scipy.linalg import solve

# ============================================================
# Example 1 (Table 1) – Liang et al. (2026)
# ============================================================

alpha = 3/4
L = np.pi
T = 1/2

m = 10 # spatial divisions
h = L / m


N = 40
tau = T/N

# tau = h**4                 # IMPORTANT: Table 1 uses tau = h^2
# N = int(T / tau)

x = np.linspace(0, L, m+1)
NI = m - 1
X, Y = np.meshgrid(x[1:-1], x[1:-1], indexing='ij')

# =============================
# ===============================
# Initial condition
# u(x,y,0) = 0
# ============================================================
U0 = np.zeros((NI, NI))

# ============================================================
# Source term (derived exactly from exact solution)
# ============================================================
def f_source(t):
    coef = 2 / gamma(3 - alpha) * t**(2 - alpha) + 2 * t**2
    return coef * np.sin(X) * np.sin(Y)

# ============================================================
# L1-2 coefficients (Eq. 2.1)
# ============================================================
def a_l(l):
    return (l + 1)**(1 - alpha) - l**(1 - alpha)

def b_l(l):
    return (((l + 1)**(2 - alpha) - l**(2 - alpha)) / (2 - alpha) )\
           - 0.5 * ((l + 1)**(1 - alpha) + l**(1 - alpha))

def c_coeff(n):
    c = np.zeros(n)
    for l in range(n):
        if l == 0:
            c[l] = a_l(0) + b_l(0)
        elif l <= n - 2:
            c[l] = a_l(l) + b_l(l) - b_l(l - 1)
        else:
            c[l] = a_l(l) - b_l(l - 1)
    return c

# ============================================================
# Compact difference matrices (Ax, δx²)
# ============================================================
A = np.zeros((NI, NI))
D = np.zeros((NI, NI))

for i in range(NI):
    A[i, i] = 10
    D[i, i] = -2
    if i > 0:
        A[i, i-1] = 1
        D[i, i-1] = 1
    if i < NI - 1:
        A[i, i+1] = 1
        D[i, i+1] = 1

A /= 12
D /= h**2


# ============================================================
 # ADI parameter (Eq. 2.9)
# ============================================================
c0 = a_l(0) + b_l(0)
S = (tau**alpha) * (gamma(2 - alpha) / c0)
Lmat = A - S * D

# ============================================================
# Time stepping (Eq. 2.10)
# ============================================================
U = [U0.copy()]

for n in range(1, N + 1):
    t_n = n * tau

    if n == 1:
        H = U0.copy()
    else:
        c = c_coeff(n)
        H = np.zeros_like(U0)
        # history += c[n - 1] * U0
        for k in range(1, n):
            H += (c[n - k - 1] - c[n - k]) * U[k]
        H += c[n - 1] * U0

    RHS = ((A @ H @ A.T + S**2 * (D @ H @ D.T)) / c0) + S * (A @ f_source(t_n) @ A.T)

    # ADI: x-direction
    Ustar = np.zeros_like(H)
    for j in range(NI):
        Ustar[:, j] = solve(Lmat, RHS[:, j])

    # ADI: y-direction
    Un = np.zeros_like(H)
    for i in range(NI):
        Un[i, :] = solve(Lmat, Ustar[i, :])

    U.append(Un)
# print(U[-1])
# ============================================================
# Exact solution and error
# ============================================================
U_exact = T**2 * np.sin(X) * np.sin(Y)
error = np.abs(np.max(U_exact-U[-1]))
# error = np.linalg.norm(U[-1] - U_exact, np.inf)
# print(U_exact)
print(f"alpha = {alpha}")
print(f"m = {m}, h = {h:.3e}, tau = {tau:.3e}")
print(f"Max error = {error:.6e}")


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# # ============================================================
# # 3D Surface plots
# # ============================================================
# fig = plt.figure(figsize=(12, 5))

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot_surface(X, Y, U[-1], cmap='viridis')
# ax1.set_title('Numerical solution at T')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(X, Y, U_exact, cmap='viridis')
# ax2.set_title('Exact solution at T')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')

# plt.tight_layout()
# plt.show()

# ============================================================
# Mid-line comparison (y = pi/2)
# ============================================================
mid = NI // 2

plt.figure(figsize=(6, 4))
plt.plot(x[1:-1], U[-1][:, mid], 'o-', label='Numerical')
plt.plot(x[1:-1], U_exact[:, mid], '--', label='Exact')
plt.xlabel('x')
plt.ylabel(r'$u(x,\pi/2,T)$')
plt.title('Mid-line comparison')
plt.legend()
plt.grid(True)
plt.show()

# # ============================================================
# # Error surface plot
# # ============================================================
# error_surface = np.abs(U[-1] - U_exact)

# plt.figure(figsize=(6, 5))
# plt.contourf(X, Y, error_surface, 20, cmap='inferno')
# plt.colorbar(label='|Error|')
# plt.title('Pointwise error at T')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
