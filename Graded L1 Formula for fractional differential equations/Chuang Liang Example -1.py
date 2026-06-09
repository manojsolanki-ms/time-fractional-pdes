import numpy as np
import math
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================
N = 200
alpha = 2/3
T = 0.5
nt = 10

x0 = 0
xn = np.pi

h = (xn-x0)/(N-1)

x = np.linspace(x0,xn,N)
y = np.linspace(x0,xn,N)

# ===============================
# CHOOSE GRADING PARAMETER
# ===============================
# Try different values here:
r = (2-alpha)/alpha
# r = 1
# r = 2
# r = 4
# r = 1/alpha

# ===============================
# GRADED TIME GRID
# ===============================
t = np.zeros(nt+1)
for n in range(nt+1):
    t[n] = T * (n/nt)**r

dt = np.diff(t)

# ===============================
# ARRAYS
# ===============================
u = np.zeros((nt+1,N,N))
u_star = np.zeros((N,N))

# ===============================
# INITIAL CONDITION
# ===============================
u[0,:,:] = 0.0

gamma = math.gamma(2-alpha)

# ===============================
# L1 WEIGHTS (NON-UNIFORM)
# ===============================
def compute_d(n):
    d = np.zeros(n+1)
    for k in range(1,n+1):
        d[k] = ((t[n] - t[n-k])**(1-alpha)
               - (t[n] - t[n-k+1])**(1-alpha)) \
               / (gamma * dt[n-k])
    return d

# ===============================
# TIME LOOP
# ===============================
for n in range(1,nt+1):

    tn = t[n]

    RHS = np.zeros((N,N))
    H0 = np.zeros((N,N))
    F0 = np.zeros((N,N))

    d = compute_d(n)
    S = 1.0 / d[1]

    rx = S/(h*h)
    ry = S/(h*h)

    # ===============================
    # HISTORY TERM
    # ===============================
    for i in range(1,N-1):
        for j in range(1,N-1):

            H = 0.0
            for k in range(1,n):
                H += (d[k] - d[k+1]) * u[n-k,i,j]

            H += d[n] * u[0,i,j]
            H0[i,j] = H

    # ===============================
    # SOURCE TERM (CORRECTED)
    # ===============================
    for i in range(1,N-1):
        for j in range(1,N-1):
            F0[i,j] = (
                math.gamma(alpha+1)
                + 2*(tn**alpha)
            ) * np.sin(x[i]) * np.sin(y[j])

    # ===============================
    # RHS (COMPACT SCHEME)
    # ===============================
    for i in range(1,N-1):
        for j in range(1,N-1):

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
                -2*(H0[i+1,j] + H0[i-1,j] + H0[i,j+1] + H0[i,j-1])
                +4*H0[i,j]
            )/(h**4)

            RHS[i,j] = (1/d[1])*(Hxy + (S**2)*delta4) + S*Fxy

    # ===============================
    # X SWEEP
    # ===============================
    for j in range(1,N-1):

        A = np.zeros((N-2,N-2))
        bvec = np.zeros(N-2)

        for i in range(N-2):

            if i > 0:
                A[i,i-1] = 1/12 - rx

            A[i,i] = 10/12 + 2*rx

            if i < N-3:
                A[i,i+1] = 1/12 - rx

            bvec[i] = RHS[i+1,j]

        sol = np.linalg.solve(A,bvec)

        for i in range(1,N-1):
            u_star[i,j] = sol[i-1]

    # ===============================
    # Y SWEEP
    # ===============================
    for i in range(1,N-1):

        A = np.zeros((N-2,N-2))
        bvec = np.zeros(N-2)

        for j in range(N-2):

            if j > 0:
                A[j,j-1] = 1/12 - ry

            A[j,j] = 10/12 + 2*ry

            if j < N-3:
                A[j,j+1] = 1/12 - ry

            bvec[j] = u_star[i,j+1]

        sol = np.linalg.solve(A,bvec)

        for j in range(1,N-1):
            u[n,i,j] = sol[j-1]

    # ===============================
    # BOUNDARY CONDITIONS
    # ===============================
    for i in range(N):
        u[n,i,0] = (tn**alpha)*np.sin(x[i])*np.sin(y[0])
        u[n,i,N-1] = (tn**alpha)*np.sin(x[i])*np.sin(y[N-1])
        u[n,0,i] = (tn**alpha)*np.sin(x[0])*np.sin(y[i])
        u[n,N-1,i] = (tn**alpha)*np.sin(x[N-1])*np.sin(y[i])

# ===============================
# EXACT SOLUTION
# ===============================
exact = (T**alpha)*np.sin(x[:,None])*np.sin(y[None,:])

# ===============================
# ERROR
# ===============================
error = np.max(np.abs(u[nt]-exact))
print("Maximum error =", error)

# ===============================
# PLOT
# ===============================
j = N//2
plt.plot(x,u[nt,:,j],'ro-',label='Numerical')
plt.plot(x,exact[:,j],'b--',label='Exact')
plt.legend()
plt.grid(True)
plt.show()
