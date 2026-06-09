# import numpy as np
# from scipy.special import gamma
# from scipy.sparse import diags, kron, eye, csc_matrix
# from scipy.sparse.linalg import spsolve
# import matplotlib.pyplot as plt

# # ============================================================
# # PARAMETERS (Example 1)
# # ============================================================
# alpha = 2/3
# L = np.pi
# T = 0.5

# m = 10

# # spatial intervals (try 20,40,80)
# h = L / m


# N = 2000 #int(T / h**2)         # paper-style: tau = h^2
# tau = T / N

# # ============================================================
# # GRID
# # ============================================================
# x = np.linspace(0, L, m+1)
# y = np.linspace(0, L, m+1)
# X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij')

# NI = m - 1
# size = NI * NI

# # ============================================================
# # EXACT SOLUTION AND SOURCE
# # ============================================================
# def u_exact(t):
#     return t**2 * np.sin(X) * np.sin(Y)

# def f_source(t):
#     return (
#         2/gamma(3-alpha)*t**(2-alpha)
#         + 2*t**2
#     ) * np.sin(X) * np.sin(Y)

# # ============================================================
# # L1-2 COEFFICIENTS
# # ============================================================
# def l12_coeffs(n):
#     a = np.zeros(n)
#     b = np.zeros(n)
#     c = np.zeros(n)

#     for k in range(n):
#         a[k] = (k+1)**(1-alpha) - k**(1-alpha)
#         b[k] = ((k+1)**(2-alpha) - k**(2-alpha))/(2-alpha) \
#                - 0.5*((k+1)**(1-alpha) + k**(1-alpha))

#     c[0] = a[0] + b[0]
#     for k in range(1, n-1):
#         c[k] = a[k] + b[k] - b[k-1]
#     c[n-1] = a[n-1] - b[n-2]
#     return c

# c = l12_coeffs(N+1)

# coef = tau**(-alpha) / gamma(2-alpha)

# # ============================================================
# # 1D OPERATORS
# # ============================================================
# # Compact averaging operator A
# A1 = diags([np.ones(NI-1), 10*np.ones(NI), np.ones(NI-1)], [-1,0,1]) / 12

# # Second difference
# D2 = diags([np.ones(NI-1), -2*np.ones(NI), np.ones(NI-1)], [-1,0,1]) / h**2

# I = eye(NI)

# # ============================================================
# # 2D OPERATORS (THIS IS CRITICAL)
# # ============================================================
# Ax = kron(I, A1)
# Ay = kron(A1, I)

# Dx2 = kron(I, D2)
# Dy2 = kron(D2, I)

# # Compact Laplacian (DO NOT SIMPLIFY)
# L_compact = Ay @ Dx2 + Ax @ Dy2
# Mass = Ax @ Ay

# # ============================================================
# # TIME STEPPING
# # ============================================================
# U = np.zeros((N+1, size))

# for n in range(1, N+1):

#     # History term
#     hist = np.zeros(size)
#     for k in range(1, n):
#         hist += (c[n-k-1] - c[n-k]) * U[k]
#     hist += c[n-1] * U[0]

#     RHS = coef * (Mass @ hist) + (Mass @ f_source(n*tau).reshape(-1))

#     # Left-hand operator
#     A = coef * c[0] * Mass - L_compact
#     A = csc_matrix(A)

#     U[n] = spsolve(A, RHS)

# # ============================================================
# # ERROR
# # ============================================================
# U_final = U[-1].reshape(NI, NI)
# err = np.max(np.abs(U_final - u_exact(T)))

# print("Max error =", err)

# mid = NI // 2
# plt.figure()
# plt.plot(x[1:-1], U_final[:, mid],'-o', label="Numerical (4th-order compact)")
# plt.plot(x[1:-1], u_exact(T)[:, mid], '--', label="Exact")
# plt.xlabel("x")
# plt.ylabel("u(x, π/2, T)")
# plt.title("Example 1: Numerical vs Exact")
# plt.legend()
# plt.show()












import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.sparse import diags, kron, eye, csc_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
# PROBLEM PARAMETERS (Example 1)
# ============================================================
alpha = 2/3
L = np.pi
T = 0.5
N = 2000                    # fixed time steps (Fig. 1)

# ============================================================
# L1-2 COEFFICIENTS
# ============================================================
def l12_coeffs(n, alpha):
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)

    for k in range(n):
        a[k] = (k+1)**(1-alpha) - k**(1-alpha)
        b[k] = ((k+1)**(2-alpha) - k**(2-alpha))/(2-alpha) \
               - 0.5*((k+1)**(1-alpha) + k**(1-alpha))

    c[0] = a[0] + b[0]
    for k in range(1, n-1):
        c[k] = a[k] + b[k] - b[k-1]
    c[n-1] = a[n-1] - b[n-2]
    return c

# ============================================================
# SOLVER FUNCTION (compact scheme Eq. 2.7)
# ============================================================
def solve_example1(m):
    h = L / m
    tau = T / N

    x = np.linspace(0, L, m+1)
    y = np.linspace(0, L, m+1)
    X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij')

    NI = m - 1
    size = NI * NI

    # Exact solution
    def u_exact(t):
        return t**2 * np.sin(X) * np.sin(Y)

    # Source term
    def f_source(t):
        return (2/gamma(3-alpha)*t**(2-alpha) + 2*t**2) \
               * np.sin(X) * np.sin(Y)

    # L1-2 data
    c = l12_coeffs(N+1, alpha)
    coef = tau**(-alpha) / gamma(2-alpha)

    # 1D operators
    A1 = diags([np.ones(NI-1), 10*np.ones(NI), np.ones(NI-1)],
               [-1, 0, 1]) / 12
    D2 = diags([np.ones(NI-1), -2*np.ones(NI), np.ones(NI-1)],
               [-1, 0, 1]) / h**2
    I = eye(NI)

    # 2D compact operators
    Ax = kron(I, A1)
    Ay = kron(A1, I)
    Dx2 = kron(I, D2)
    Dy2 = kron(D2, I)

    Mass = Ax @ Ay
    L_compact = Ay @ Dx2 + Ax @ Dy2

    # Time stepping
    U = np.zeros((N+1, size))

    for n in range(1, N+1):
        hist = np.zeros(size)
        for k in range(1, n):
            hist += (c[n-k-1] - c[n-k]) * U[k]
        hist += c[n-1] * U[0]

        RHS = coef * (Mass @ hist) \
              + Mass @ f_source(n*tau).reshape(-1)

        A = coef * c[0] * Mass - L_compact
        U[n] = spsolve(csc_matrix(A), RHS)

    U_final = U[-1].reshape(NI, NI)
    error = np.max(np.abs(U_final - u_exact(T)))

    return h, error

# ============================================================
# FIG. 1 – SPATIAL CONVERGENCE PLOT
# ============================================================
m_values = [10, 20, 40, 80]
h_vals = []
err_vals = []

for m in m_values:
    h, err = solve_example1(m)
    h_vals.append(h)
    err_vals.append(err)
    print(f"m = {m:3d}, h = {h:.4e}, error = {err:.4e}")

# log–log plot
plt.figure()
plt.loglog(h_vals, err_vals, '-o', label="Proposed compact scheme")

# reference slope 4
plt.loglog(
    h_vals,
    err_vals[0]*(np.array(h_vals)/h_vals[0])**4,
    '--',
    label="Slope 4"
)

plt.gca().invert_xaxis()
plt.xlabel("log(h)")
plt.ylabel("log(Error)")
plt.title(r"Fig. 1: Spatial convergence ($\alpha=2/3$, $N=2000$)")
plt.legend()
plt.grid(True, which="both")
plt.show()
