import numpy as np
import math

# =====================================================
# GLOBAL PARAMETERS
# =====================================================
N = 200
# alpha = 2/3
alpha = 3/4
T = 0.5

x0 = 0.0
xn = np.pi

h = (xn - x0)/(N - 1)

x = np.linspace(x0, xn, N)
y = np.linspace(x0, xn, N)

gamma = math.gamma(2 - alpha)

# =====================================================
# SOLVER
# =====================================================
def solve(nt, r):

    # -------------------------------------------------
    # Graded mesh
    # -------------------------------------------------
    t = np.zeros(nt + 1)

    for n in range(nt + 1):
        t[n] = T * (n/nt)**r

    dt = np.diff(t)

    # -------------------------------------------------
    # Arrays
    # -------------------------------------------------
    u = np.zeros((nt + 1, N, N))
    u_star = np.zeros((N, N))

    u[0, :, :] = 0.0

    # -------------------------------------------------
    # L1 weights
    # -------------------------------------------------
    def compute_d(n):

        d = np.zeros(n + 1)

        for k in range(1, n + 1):

            d[k] = (
                (t[n] - t[n-k])**(1-alpha)
                -
                (t[n] - t[n-k+1])**(1-alpha)
            )/(gamma * dt[n-k])

        return d

    # -------------------------------------------------
    # Time loop
    # -------------------------------------------------
    for n in range(1, nt + 1):

        tn = t[n]

        RHS = np.zeros((N, N))
        H0  = np.zeros((N, N))
        F0  = np.zeros((N, N))

        d = compute_d(n)

        S = 1.0/d[1]

        rx = S/(h*h)
        ry = S/(h*h)

        # =============================================
        # History term
        # =============================================
        for i in range(1, N-1):
            for j in range(1, N-1):

                H = 0.0

                for k in range(1, n):
                    H += (d[k] - d[k+1]) * u[n-k, i, j]

                H += d[n] * u[0, i, j]

                H0[i, j] = H

        # =============================================
        # Source term
        # =============================================
        for i in range(1, N-1):
            for j in range(1, N-1):

                F0[i, j] = (
                    math.gamma(alpha + 1)
                    +
                    2*(tn**alpha)
                ) * np.sin(x[i]) * np.sin(y[j])

        # =============================================
        # RHS
        # =============================================
        for i in range(1, N-1):
            for j in range(1, N-1):

                Hxy = (
                    H0[i-1,j-1] + 10*H0[i,j-1] + H0[i+1,j-1]
                    +10*H0[i-1,j] +100*H0[i,j] +10*H0[i+1,j]
                    +H0[i-1,j+1] +10*H0[i,j+1] +H0[i+1,j+1]
                )/144

                Fxy = (
                    F0[i-1,j-1] + 10*F0[i,j-1] + F0[i+1,j-1]
                    +10*F0[i-1,j] +100*F0[i,j] +10*F0[i+1,j]
                    +F0[i-1,j+1] +10*F0[i,j+1] +F0[i+1,j+1]
                )/144

                delta4 = (
                    H0[i+1,j+1] + H0[i-1,j-1]
                    + H0[i+1,j-1] + H0[i-1,j+1]
                    - 2*(H0[i+1,j] + H0[i-1,j]
                         + H0[i,j+1] + H0[i,j-1])
                    + 4*H0[i,j]
                )/(h**4)

                RHS[i,j] = (
                    (1/d[1])*(Hxy + (S**2)*delta4)
                    +
                    S*Fxy
                )

        # =============================================
        # X sweep
        # =============================================
        for j in range(1, N-1):

            A = np.zeros((N-2, N-2))
            b = np.zeros(N-2)

            for i in range(N-2):

                if i > 0:
                    A[i,i-1] = 1/12 - rx

                A[i,i] = 10/12 + 2*rx

                if i < N-3:
                    A[i,i+1] = 1/12 - rx

                b[i] = RHS[i+1, j]

            sol = np.linalg.solve(A, b)

            for i in range(1, N-1):
                u_star[i,j] = sol[i-1]

        # =============================================
        # Y sweep
        # =============================================
        for i in range(1, N-1):

            A = np.zeros((N-2, N-2))
            b = np.zeros(N-2)

            for j in range(N-2):

                if j > 0:
                    A[j,j-1] = 1/12 - ry

                A[j,j] = 10/12 + 2*ry

                if j < N-3:
                    A[j,j+1] = 1/12 - ry

                b[j] = u_star[i,j+1]

            sol = np.linalg.solve(A, b)

            for j in range(1, N-1):
                u[n,i,j] = sol[j-1]

        # =============================================
        # Boundary conditions
        # =============================================
        for i in range(N):

            u[n,i,0] = (
                tn**alpha
            )*np.sin(x[i])*np.sin(y[0])

            u[n,i,N-1] = (
                tn**alpha
            )*np.sin(x[i])*np.sin(y[N-1])

            u[n,0,i] = (
                tn**alpha
            )*np.sin(x[0])*np.sin(y[i])

            u[n,N-1,i] = (
                tn**alpha
            )*np.sin(x[N-1])*np.sin(y[i])

    # -------------------------------------------------
    # Exact solution
    # -------------------------------------------------
    exact = (
        T**alpha
    )*np.sin(x[:,None])*np.sin(y[None,:])

    error = np.max(np.abs(u[nt] - exact))

    return error


# =====================================================
# EXPERIMENT SETUP
# =====================================================
nt_list = [10, 20]

r_values = {
    "(2-alpha)/alpha": (2-alpha)/alpha,
    "1/alpha": 1/alpha,
    "1": 1,
    "2": 2,
    "4": 4
}

# =====================================================
# RUN TESTS
# =====================================================
results = {}

for r_name, r in r_values.items():

    errors = []

    print(f"\n--- r = {r_name} ({r:.4f}) ---")

    for nt in nt_list:

        err = solve(nt, r)

        errors.append(err)

        print(
            f"nt = {nt:3d}, "
            f"error = {err:.6e}"
        )

    order = np.log(errors[0]/errors[1]) / np.log(2)

    results[r_name] = (errors, order)

# =====================================================
# FINAL TABLE
# =====================================================
print("\n")
print("="*65)
print("FINAL RESULTS TABLE")
print("="*65)

print(
    f"{'r':<20}"
    f"{'E(nt=10)':<18}"
    f"{'E(nt=20)':<18}"
    f"{'Order':<10}"
)

for r_name, (errs, order) in results.items():

    print(
        f"{r_name:<20}"
        f"{errs[0]:<18.6e}"
        f"{errs[1]:<18.6e}"
        f"{order:<10.4f}"
    )
