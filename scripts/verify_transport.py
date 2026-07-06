import numpy as np
def expm(M):
    import numpy as _np
    E = _np.eye(M.shape[0]); term = _np.eye(M.shape[0])
    for i in range(1, 60):
        term = term @ M / i
        E = E + term
    return E
from numpy.polynomial.legendre import legval

def phi(t, x, N):
    """LegS basis: phi_n(t,x) = sqrt(2n+1) P_n(2x/t - 1), for x in [0,t]."""
    z = 2*x/t - 1
    out = np.zeros(N)
    for n in range(N):
        c = np.zeros(n+1); c[n] = 1
        out[n] = np.sqrt(2*n+1) * legval(z, c)
    return out

N = 8
# HiPPO-LegS A
A = np.zeros((N,N))
for n in range(N):
    for k in range(N):
        if n > k: A[n,k] = np.sqrt((2*n+1)*(2*k+1))
        elif n == k: A[n,k] = n+1
B = np.sqrt(2*np.arange(N)+1)
At = A - np.eye(N)   # claimed basis-transport generator

# Check 1: BB^T - At - At^T = I
chk1 = np.abs(B[:,None]*B[None,:] - At - At.T - np.eye(N)).max()
print("check1 (BB^T - At - At^T = I):", chk1)

# Check 2: d/dt phi_t(x) = -(1/t) At phi_t(x)  (finite difference)
t, x = 3.7, 1.234
h = 1e-6
fd = (phi(t+h, x, N) - phi(t-h, x, N)) / (2*h)
an = -(1/t) * At @ phi(t, x, N)
print("check2 (basis derivative):", np.abs(fd - an).max())

# Check 3: exact transport phi_{t'}(x) = exp(-log(t'/t) At) phi_t(x)
t2 = 6.1
T = expm(-np.log(t2/t) * At)
chk3 = np.abs(T @ phi(t, x, N) - phi(t2, x, N)).max()
print("check3 (finite transport):", chk3)

# Check 4: phi_t(t) = B (right endpoint)
print("check4 (phi_t(t)=B):", np.abs(phi(t, t, N) - B).max())

# Check 5: full recurrence == batch posterior on irregular samples
rng = np.random.default_rng(0)
s = np.sort(rng.uniform(0.5, 10.0, 40))          # irregular obs times
y = np.sin(s) + 0.1*rng.standard_normal(40)      # noisy data
w = rng.uniform(0.5, 2.0, 40)                    # weights
# batch at t_final = s[-1]
tf = s[-1]
Phi = np.stack([phi(tf, si, N) for si in s])
G_batch = (Phi * w[:,None]).T @ Phi
b_batch = Phi.T @ (w*y)
# recurrence: start at t=s[0] with first obs (phi_{s0}(s0)=B)
G = w[0]*np.outer(B,B); b = w[0]*B*y[0]
for k in range(1, len(s)):
    Tk = expm(-np.log(s[k]/s[k-1]) * At)
    G = Tk @ G @ Tk.T
    b = Tk @ b
    G = G + w[k]*np.outer(B,B)
    b = b + w[k]*B*y[k]
print("check5 Gram recurrence vs batch:", np.abs(G - G_batch).max() / np.abs(G_batch).max())
print("check5 b recurrence vs batch:  ", np.abs(b - b_batch).max() / np.abs(b_batch).max())

# Check 6: posterior from recurrence == batch Bayesian weighted regression
tau2, sig2 = 1.0, 0.01
m_rec = np.linalg.solve(np.eye(N)/tau2 + G/sig2, b/sig2)
m_bat = np.linalg.solve(np.eye(N)/tau2 + G_batch/sig2, b_batch/sig2)
print("check6 posterior means match:", np.abs(m_rec-m_bat).max())

# Check 7: continuous-limit self-consistency: dense uniform grid, w_i = dx  => G ~ t I
tt = 10.0; M = 200000; xs = (np.arange(M)+0.5)*tt/M; dx = tt/M
Phi2 = np.stack([phi(tt, xi, N) for xi in xs[::200]])  # subsample for speed, scale weight
Gc = Phi2.T @ Phi2 * (dx*200)
print("check7 (G -> t I):", np.abs(Gc - tt*np.eye(N)).max())

# Check 8: T lower-triangular (square-root O(N^2) claim)
print("check8 (T lower-tri):", np.abs(np.triu(T,1)).max())

# ---- appended: closed-form injection + frame-conditioning checks ----
def expm_ss(M, sq=12):
    """scaling-and-squaring reference (accurate for large log-ratios)"""
    Ms = M/(2**sq); E = np.eye(M.shape[0]); term = np.eye(M.shape[0])
    for i in range(1, 30):
        term = term @ Ms / i
        E = E + term
    for _ in range(sq):
        E = E @ E
    return E

def bvec(rho, N):
    """Closed form: rho^{-At} B = sqrt(2n+1) P_n(2/rho - 1). O(N), no expm."""
    z = 2.0/rho - 1.0
    P = np.zeros(N); P[0] = 1.0
    if N > 1: P[1] = z
    for n in range(1, N-1):
        P[n+1] = ((2*n+1)*z*P[n] - n*P[n-1])/(n+1)
    return np.sqrt(2*np.arange(N)+1) * P

for rho in [1.3, 2.0, 5.0, 20.0]:
    ref = expm_ss(-np.log(rho)*At) @ B
    err = np.abs(bvec(rho, N) - ref).max()/np.abs(ref).max()
    print(f"check9 closed-form injection rho={rho}: rel err {err:.2e}")

# check10: current-frame design vectors are bounded; backward-frame ones explode
import numpy as _np
fwd = max(_np.abs(bvec(r, N)).max() for r in [1.5, 3.0, 10.0])
bwd = _np.abs(bvec(1/1.5, N)).max()
print(f"check10 frame conditioning: forward (current-frame) max |entry| = {fwd:.2f} (bounded by sqrt(2N-1)), backward rho=1/1.5 max = {bwd:.2e} (grows like (2rho)^n -- never transport backwards)")
