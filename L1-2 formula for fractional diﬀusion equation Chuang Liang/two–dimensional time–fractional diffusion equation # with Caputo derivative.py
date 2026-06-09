import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, erfc
from scipy.linalg import solve



# ------------------------------------------------------------
# PROBLEM STATEMENT (SOURCE-FREE CASE)
# ------------------------------------------------------------
# Solve the two–dimensional time–fractional diffusion equation
# with Caputo derivative:
#
#   ^C_0 D_t^{0.5} u(x, y, t)
#       = u_xx(x, y, t) + u_yy(x, y, t),
#
# for (x, y, t) ∈ (0, 1) × (0, 1) × (0, 1].
#
# Source term:
#   f(x, y, t) = 0.
#
# Initial condition:
#   u(x, y, 0) = sin(pi x) sin(pi y).
#
# Boundary conditions:
#   u(0, y, t) = u(1, y, t) = 0,
#   u(x, 0, t) = u(x, 1, t) = 0.
#
# Exact solution (for alpha = 0.5):
#   u(x, y, t)
#     = E_{0.5}(-2 pi^2 t^{0.5})
#       sin(pi x) sin(pi y),
#
# where E_{0.5}(·) is the Mittag–Leffler function.
# ------------------------------------------------------------



# ============================================================
# PARAMETERS


# ============================================================
alpha = 0.5
m = 20             # change: 10, 20, 40, 80
T = 1.0
h = 1.0 / m
tau =  0.5 * h**2        # <<< CRITICAL FIX
N = int(T / tau)

# ============================================================
# GRID
# ============================================================
x = np.linspace(0, 1, m+1)
y = np.linspace(0, 1, m+1)
NI = m - 1

# ============================================================
# INITIAL CONDITION
# ============================================================
U0 = np.zeros((NI, NI))
for i in range(NI):
    for j in range(NI):
        U0[i, j] = np.sin(np.pi * x[i+1]) * np.sin(np.pi * y[j+1])

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
# COMPACT MATRICES
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

A = A / 12.0
Tmat = Tmat / h**2

# ============================================================
# ADI PARAMETER
# ============================================================
c0 = a(0) + b(0)
S = tau**alpha * gamma(2-alpha) / c0
L = A - S * Tmat

# ============================================================
# STORAGE
# ============================================================
U = [U0.copy()]

# ============================================================
# TIME STEPPING
# ============================================================
for n in range(1, N+1):

    if n == 1:
        history = U[0].copy()
    else:
        c = c_coeff(n)
        history = np.zeros((NI, NI))
        for k in range(1, n):
            history += (c[n-k-1] - c[n-k]) * U[k]
        history += c[n-1] * U[0]

    # ===== CORRECT RHS (KEY FIX) =====
    RHS = (A @ history @ A.T + S**2 * (Tmat @ history @ Tmat.T)) / c0

    # ===== ADI =====
    Ustar = np.zeros_like(history)
    for j in range(NI):
        Ustar[:, j] = solve(L, RHS[:, j])

    Un = np.zeros_like(history)
    for i in range(NI):
        Un[i, :] = solve(L, Ustar[i, :])

    U.append(Un.copy())

# ============================================================
# EXACT SOLUTION (alpha = 0.5)
# ============================================================
def exact_solution(t):
    lam = 2 * np.pi**2
    coef = np.exp(lam**2 * t) * erfc(lam * np.sqrt(t))
    Ue = np.zeros((NI, NI))
    for i in range(NI):
        for j in range(NI):
            Ue[i, j] = coef * np.sin(np.pi * x[i+1]) * np.sin(np.pi * y[j+1])
    return Ue

U_exact = exact_solution(T)

# ============================================================
# ERROR
# ============================================================
err = np.linalg.norm(U[-1] - U_exact, np.inf)
print(f"m = {m}, max error = {err:.3e}")


# ============================================================
# PREPARE MESH FOR PLOTTING
# ============================================================
X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij')

U_num = U[-1]

# ============================================================
# 3D SURFACE PLOTS
# ============================================================
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, U_num, cmap='viridis')
ax1.set_title('Numerical solution at T = 1')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, U_exact, cmap='viridis')
ax2.set_title('Exact solution (used for comparison)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()
plt.show()

# ============================================================
# 2D CONTOUR PLOTS
# ============================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, U_num, 20)
plt.colorbar()
plt.title('Numerical contour at T = 1')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, U_exact, 20)
plt.colorbar()
plt.title('Exact contour (used for comparison)')

plt.tight_layout()
plt.show()

# ============================================================
# 1D LINE PLOT AT y = 0.5
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

