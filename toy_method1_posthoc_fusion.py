"""Toy demonstration of Approach 1: post-hoc fusion of bounded single-epoch solutions.

Pipeline (see TIME_DEPENDENT_APPROACHES.md, Sections 4.1 and 11.2):

1. Generate synthetic 2-D antiplane position time series from the forward model used in the
   Kalman/IMM notebooks (single vertical strike-slip fault, two patches, blocks either side,
   two slow slip events, one earthquake, afterslip).
2. Chunk the time series into non-overlapping epochs and estimate a velocity field per epoch.
3. Solve a bounded (coupling-constrained) least-squares problem independently per epoch, with
   known bounds that differ between epochs and are enforced at every epoch on every patch. This
   stands in for whatever bounded per-epoch solver celeri uses; only the bounded point estimate
   and its active set matter downstream.
4. Link the epochs with Approach 1:
   - show why naive smoothing of bounded estimates is biased (truncated posteriors),
   - method 1a: smooth normalized coupling with active-bound epochs pinned
     (two variants: a bounded quadratic smoother in coupling space, and the logit-space
     Tikhonov smoother described in the memo),
   - method 1b: information-form fusion of the per-epoch states with their reduced
     covariances via an RTS smoother, then re-projection into the bounds,
   - method 1c: structured priors from the known earthquake time: block rate piecewise
     constant between earthquakes (pooled, then coupling re-solved per epoch with the block
     rate fixed), temporal prior broken only at the earthquake, and a monotone smooth
     recovery on patches whose post-earthquake bounds admit afterslip.

Run:  python toy_method1_posthoc_fusion.py      (numpy, scipy, matplotlib only)
Figures are written to ./figures/.

Sections are marked with `# %%` so the file can be opened as a notebook with jupytext.
"""

# %% Imports
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, optimize

# %% Parameters (notebook values from demo_imm_with_smoother.ipynb noted in comments)
N_STA = 50
X_LIM = 150.0  # notebook: 50. +-50 km cannot separate block rate from creep below 30 km.
DEPTHS = np.array([1e-10, 15.0, 30.0])  # patch 0 = [0, 15] km, patch 1 = [15, 30] km
NT = 2000  # notebook: 100
T_EQ = 0.5
SIGMA_P = 0.001  # notebook: 0.005. Position noise (white).
N_EPOCHS_PRE = 20
N_EPOCHS_POST = 20
V0_PRE, V0_POST = 0.5, 0.25  # block rate before / after the earthquake
SSE_SHALLOW = dict(amp=3.0, center=0.35, width=0.02)
SSE_DEEP = dict(amp=1.0, center=0.25, width=0.03)
# Notebook afterslip `200*exp(-8*t - 0.5)` is a parenthesization slip that evaluates to
# ~121*exp(-8 t); written deliberately here as A*exp(-(t - t_eq)/tau).
AFTERSLIP_AMP, AFTERSLIP_TAU = 2.0, 0.125
COSEISMIC_SLIP = 0.5  # notebook: one-sample velocity spike of 25 (dt dependent)
T_BOUND_CHANGE = 0.75  # second known bound change, to exercise breaks other than the EQ
EPS_LOGIT = 1e-3  # clip for the logit variant; pinned epochs are reported exactly at the bound
LAMBDA_MULT = 10.0  # method 1a smoothing: lambda = LAMBDA_MULT x median weight (see fig 6 sweep)
SEED = 0
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)

# %% Forward model: Green's functions and the celeri-like parameterization
x = np.linspace(-X_LIM, X_LIM, N_STA)  # even count: no station at x = 0


def screw(slip_rate, x, depth):
    return slip_rate / np.pi * np.arctan(x / depth)


# Patch kernels: unit slip on patch i (depths d_i..d_{i+1}) -> surface velocity.
G_patch = np.stack(
    [
        (1 / np.pi) * (np.arctan(x / DEPTHS[i]) - np.arctan(x / DEPTHS[i + 1]))
        for i in range(2)
    ]
)  # (2, N_STA)
v_inter = screw(1.0, x, DEPTHS[-1])  # deep creep below 30 km at unit rate

# Notebook operator acts on [v0, s1, s2] (block rate, patch slip rates):
#   velocity = v0 * v_inter + s1 * G_0 + s2 * G_1        (notebook v_mat_i = -G_i)
A_notebook = np.column_stack([v_inter, G_patch[0], G_patch[1]])

# celeri-like operator acts on [v0, e1, e2] with slip DEFICIT rates e_i = v0 - s_i:
#   velocity = v0 * (1/2) sign(x) - e1 * G_0 - e2 * G_1
# because v_inter + G_0 + G_1 = (1/2) sign(x). Coupling kappa_i = e_i / v0.
A = np.column_stack([0.5 * np.sign(x), -G_patch[0], -G_patch[1]])

_rel = np.abs(v_inter + G_patch.sum(0) - 0.5 * np.sign(x)).max()
assert _rel < 1e-9, _rel
_test = rng.normal(size=3)
assert np.allclose(
    A_notebook @ _test, A @ np.array([_test[0], _test[0] - _test[1], _test[0] - _test[2]])
)

# %% Truth histories
t = np.linspace(0.0, 1.0, NT)
dt = t[1] - t[0]
eq_idx = int(np.argmin(np.abs(t - T_EQ)))  # first post-earthquake sample
post = np.arange(NT) >= eq_idx

v0_true = np.where(post, V0_POST, V0_PRE)


def gaussian(t, amp, center, width):
    return amp * np.exp(-((t - center) ** 2) / (2 * width**2))


s1_true = gaussian(t, **SSE_SHALLOW)  # shallow SSE; coseismic slip handled as a step
s2_true = gaussian(t, **SSE_DEEP) + np.where(
    post, AFTERSLIP_AMP * np.exp(-(t - t[eq_idx]) / AFTERSLIP_TAU), 0.0
)

vel_true = A_notebook @ np.vstack([v0_true, s1_true, s2_true])  # (N_STA, NT)
pos_true = np.cumsum(vel_true, axis=1) * dt
pos_true[:, post] += COSEISMIC_SLIP * G_patch[0][:, None]  # coseismic step on patch 0
pos_obs = pos_true + SIGMA_P * rng.normal(size=pos_true.shape)

# %% Epochs: non-overlapping windows with a boundary exactly at the earthquake
edges = np.concatenate(
    [
        np.linspace(0, eq_idx, N_EPOCHS_PRE + 1).astype(int)[:-1],
        np.linspace(eq_idx, NT, N_EPOCHS_POST + 1).astype(int),
    ]
)
T = len(edges) - 1
epoch_slices = [slice(edges[k], edges[k + 1]) for k in range(T)]
t_mid = np.array([t[s].mean() for s in epoch_slices])
t_edges = np.concatenate([t[edges[:-1]], [t[-1] + dt]])  # epoch boundary times
eq_epoch = N_EPOCHS_PRE  # first post-earthquake epoch


def fit_slope(tt, yy):
    """Per-station OLS slope and intercept of position vs time; analytic slope variance."""
    tc = tt - tt.mean()
    sxx = (tc**2).sum()
    slope = (yy * tc).sum(axis=1) / sxx
    intercept = yy.mean(axis=1) - slope * tt.mean()
    var_slope = SIGMA_P**2 / sxx  # white noise, known sigma; same for every station
    return slope, intercept, var_slope


d_epoch = np.zeros((T, N_STA))  # observed epoch velocities
w_epoch = np.zeros(T)  # 1 / var of each epoch velocity (identical across stations)
intercepts = np.zeros((T, N_STA))
for k, s in enumerate(epoch_slices):
    d_epoch[k], intercepts[k], var_k = fit_slope(t[s], pos_obs[:, s])
    w_epoch[k] = 1.0 / var_k

# Epoch-mean truth (what a per-epoch velocity can resolve)
v0_bar = np.array([v0_true[s].mean() for s in epoch_slices])
s1_bar = np.array([s1_true[s].mean() for s in epoch_slices])
s2_bar = np.array([s2_true[s].mean() for s in epoch_slices])
x_true = np.column_stack([v0_bar, v0_bar - s1_bar, v0_bar - s2_bar])  # [v0, e1, e2]
kappa_true = x_true[:, 1:] / x_true[:, :1]  # (T, 2)

# %% Known per-epoch coupling bounds (differ between epochs, on every patch)
kappa_lo = np.zeros((T, 2))
kappa_hi = np.ones((T, 2))
for k in range(T):
    if k < eq_epoch:
        kappa_lo[k] = [-10.0, -10.0]
    else:
        kappa_lo[k, 0] = 0.0  # shallow patch: no afterslip expected
        kappa_lo[k, 1] = -20.0 if t_mid[k] < T_BOUND_CHANGE else -5.0
# Break epochs: the difference penalty is not applied across k-1 -> k at these k
bound_change = np.any(kappa_lo[1:] != kappa_lo[:-1], axis=1) | np.any(
    kappa_hi[1:] != kappa_hi[:-1], axis=1
)
breaks = set(np.flatnonzero(bound_change) + 1) | {eq_epoch}
print(f"epochs: {T}, earthquake epoch: {eq_epoch}, break epochs: {sorted(breaks)}")


# %% Bounded solve: exact active-set enumeration for the small per-epoch QP
def constraint_rows(lo, hi):
    """Rows C such that C x >= 0 encodes v0 >= 0 and lo_i v0 <= e_i <= hi_i v0."""
    rows = [[1.0, 0.0, 0.0]]
    labels = ["v0>=0"]
    for i in range(2):
        r_lo = np.zeros(3)
        r_lo[0], r_lo[1 + i] = -lo[i], 1.0  # e_i - lo_i v0 >= 0
        r_hi = np.zeros(3)
        r_hi[0], r_hi[1 + i] = hi[i], -1.0  # hi_i v0 - e_i >= 0
        rows += [r_lo, r_hi]
        labels += [f"lo{i+1}", f"hi{i+1}"]
    return np.array(rows), labels


def solve_bounded_qp(H, g, C, tol=1e-9):
    """min 1/2 x'Hx - g'x  s.t. Cx >= 0, by enumerating active sets (exact for tiny problems).

    Returns x, active (tuple of row indices), reduced covariance N (N'HN)^-1 N'.
    """
    m = C.shape[0]
    best = None
    scale = max(1.0, np.linalg.norm(g))
    for r in range(m + 1):
        for active in itertools.combinations(range(m), r):
            # Skip lo and hi of the same patch simultaneously active.
            if any((1 + 2 * i in active) and (2 + 2 * i in active) for i in range(2)):
                continue
            Ca = C[list(active)]
            n_a = len(active)
            if n_a > 0 and np.linalg.matrix_rank(Ca) < n_a:
                continue  # dependent constraints (e.g. v0 = 0 with a coupling row)
            K = np.zeros((3 + n_a, 3 + n_a))
            K[:3, :3] = H
            K[:3, 3:] = -Ca.T
            K[3:, :3] = Ca
            rhs = np.concatenate([g, np.zeros(n_a)])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", linalg.LinAlgWarning)
                    sol = linalg.solve(K, rhs)
            except (linalg.LinAlgError, linalg.LinAlgWarning):
                continue
            xx, mu = sol[:3], sol[3:]
            if np.any(mu < -tol * scale) or np.any(C @ xx < -tol * scale):
                continue
            obj = 0.5 * xx @ H @ xx - g @ xx
            if best is None or obj < best[0] - 1e-12:
                best = (obj, xx, active)
    assert best is not None, "no KKT point found"
    _, xx, active = best
    if len(active) == 0:
        cov = linalg.inv(H)
    else:
        N = linalg.null_space(C[list(active)])
        cov = N @ linalg.inv(N.T @ H @ N) @ N.T if N.shape[1] > 0 else np.zeros((3, 3))
    return xx, active, cov


x_bnd = np.zeros((T, 3))
x_unc = np.zeros((T, 3))
cov_bnd = np.zeros((T, 3, 3))
cov_unc = np.zeros((T, 3, 3))
active_sets = []
H_epoch = np.zeros((T, 3, 3))
g_epoch = np.zeros((T, 3))
for k in range(T):
    H = w_epoch[k] * (A.T @ A)  # sufficient statistics: only these enter the solve
    g = w_epoch[k] * (A.T @ d_epoch[k])
    H_epoch[k], g_epoch[k] = H, g
    x_unc[k] = linalg.solve(H, g)
    cov_unc[k] = linalg.inv(H)
    C, C_labels = constraint_rows(kappa_lo[k], kappa_hi[k])
    x_bnd[k], act, cov_bnd[k] = solve_bounded_qp(H, g, C)
    active_sets.append(act)

kappa_unc = x_unc[:, 1:] / x_unc[:, :1]
kappa_bnd = x_bnd[:, 1:] / x_bnd[:, :1]
at_hi = np.array([[(2 + 2 * i) in act for i in range(2)] for act in active_sets])  # hi active
at_lo = np.array([[(1 + 2 * i) in act for i in range(2)] for act in active_sets])

# Cross-check the active-set solver against SLSQP on a few epochs
for k in [0, 13, eq_epoch, eq_epoch + 3, T - 1]:
    C, _ = constraint_rows(kappa_lo[k], kappa_hi[k])
    H, g = H_epoch[k], g_epoch[k]
    res = optimize.minimize(
        lambda z: 0.5 * z @ H @ z - g @ z,
        x_unc[k],
        jac=lambda z: H @ z - g,
        constraints=[{"type": "ineq", "fun": lambda z: C @ z, "jac": lambda _z: C}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert np.allclose(res.x, x_bnd[k], atol=1e-5), (k, res.x, x_bnd[k])

# %% Sanity check with zero noise: bounded per-epoch solve recovers epoch-mean truth
# exactly where velocity is constant within the epoch (elsewhere the OLS slope is a
# parabolic-kernel average, not the box mean: second-order difference).
_dev = []
for k in range(T):
    d0, _, _ = fit_slope(t[epoch_slices[k]], pos_true[:, epoch_slices[k]])
    C, _ = constraint_rows(kappa_lo[k], kappa_hi[k])
    xx, _, _ = solve_bounded_qp(A.T @ A, A.T @ d0, C)
    _dev.append(np.abs(xx - x_true[k]).max())
_dev = np.array(_dev)
_const = np.array([np.ptp(s1_true[s]) + np.ptp(s2_true[s]) < 1e-6 for s in epoch_slices])
assert _dev[_const].max() < 1e-8, _dev[_const].max()
print(f"noise-free check: max |x - truth| = {_dev[_const].max():.1e} on constant epochs, "
      f"{_dev[~_const].max():.3f} on transient epochs (within-epoch averaging)")


# %% Method 1a helpers: weighted first-difference smoothers with breaks and per-regime lambda
def difference_matrix(T, breaks):
    rows = [k for k in range(1, T) if k not in breaks]
    D = np.zeros((len(rows), T))
    for r, k in enumerate(rows):
        D[r, k - 1], D[r, k] = -1.0, 1.0
    return D, np.array(rows)


D, D_rows = difference_matrix(T, breaks)


def tikhonov_smooth(y, w, lam_rows, D):
    """argmin sum w (z - y)^2 + sum_r lam_r (D z)_r^2  (dense; T is small)."""
    M = np.diag(w) + D.T @ (lam_rows[:, None] * D)
    return linalg.solve(M, w * y, assume_a="pos")


def gcv_lambda(y, w, D, grid=None):
    """Generalized cross-validation for a single smoothing parameter."""
    if grid is None:
        grid = np.logspace(-3, 4, 71) * np.median(w)
    scores = []
    n = len(y)
    for lam in grid:
        M = np.diag(w) + lam * D.T @ D
        S = linalg.solve(M, np.diag(w), assume_a="pos")
        resid = np.sqrt(w) * (y - S @ y)
        scores.append((resid @ resid / n) / (1 - np.trace(S) / n) ** 2)
    return float(grid[int(np.argmin(scores))])


def bounded_smooth(y, w, lam_rows, D, lo, hi, pin_hi, pin_lo, w_pin_factor=1e6):
    """Bounded quadratic smoother in coupling space.

    argmin sum w (kappa - y)^2 + sum_r lam_r (D kappa)_r^2   s.t. lo <= kappa <= hi,
    with epochs whose bound was active in the per-epoch solve pinned to that bound.
    """
    w = w.copy()
    y = y.copy()
    w_pin = w_pin_factor * w.max()
    y[pin_hi], w[pin_hi] = hi[pin_hi], w_pin
    y[pin_lo], w[pin_lo] = lo[pin_lo], w_pin
    M = np.vstack([np.diag(np.sqrt(w)), np.sqrt(lam_rows)[:, None] * D])
    rhs = np.concatenate([np.sqrt(w) * y, np.zeros(D.shape[0])])
    res = optimize.lsq_linear(M, rhs, bounds=(lo, hi), method="bvls", tol=1e-12)
    out = res.x
    out[pin_hi] = hi[pin_hi]
    out[pin_lo] = lo[pin_lo]
    return out


# Per-epoch variance of kappa from the REDUCED covariance (zero normal to an active bound)
def kappa_variance(xk, cov):
    v0, e = xk[0], xk[1:]
    var = np.zeros(2)
    for i in range(2):
        J = np.array([-e[i] / v0**2, 0.0, 0.0])
        J[1 + i] = 1.0 / v0
        var[i] = J @ cov @ J
    return var


var_kappa_bnd = np.array([kappa_variance(x_bnd[k], cov_bnd[k]) for k in range(T)])
var_kappa_unc = np.array([kappa_variance(x_unc[k], cov_unc[k]) for k in range(T)])
VAR_FLOOR = 1e-4  # pinned epochs enter lambda selection as data with this variance


def weights_coupling(i):
    """Weights 1/Var(kappa_i); pinned epochs get the floor variance."""
    return 1.0 / np.maximum(var_kappa_bnd[:, i], VAR_FLOOR)


def method_1a_coupling(i, lam_rows):
    return bounded_smooth(
        kappa_bnd[:, i], weights_coupling(i), lam_rows, D,
        kappa_lo[:, i], kappa_hi[:, i], at_hi[:, i], at_lo[:, i],
    )


def logit_transform_arrays(khat, var, lo, hi):
    rng_b = hi - lo
    u = np.clip((khat - lo) / rng_b, EPS_LOGIT, 1 - EPS_LOGIT)
    z = np.log(u / (1 - u))
    var_u = np.maximum(var, VAR_FLOOR) / rng_b**2
    w = (u * (1 - u)) ** 2 / var_u  # 1 / Var(z) by the delta method
    return z, w, rng_b


def logit_transform(i):
    return logit_transform_arrays(kappa_bnd[:, i], var_kappa_bnd[:, i], kappa_lo[:, i], kappa_hi[:, i])


def logit_smooth(khat, var, lo, hi, pin_hi, pin_lo, D, mult=LAMBDA_MULT, robust=False, n_irls=30):
    """Logit-space Tikhonov smoother of normalized coupling with active-bound epochs pinned.

    With robust=True the difference penalty is Huber rather than quadratic (iteratively
    reweighted least squares): small differences are smoothed as before, large genuine changes
    such as a slow slip onset are penalized only linearly and so are not halved.
    """
    z, w, rng_b = logit_transform_arrays(khat, var, lo, hi)
    pin = pin_hi | pin_lo
    w = w.copy()
    lam = mult * np.median(w[~pin])
    w[pin] = 1e6 * w[~pin].max()
    lam_rows = np.full(D.shape[0], lam)
    z_s = tikhonov_smooth(z, w, lam_rows, D)
    if robust:
        delta = 2.0 * np.median(1.0 / np.sqrt(w[~pin]))  # Huber knee: ~2 sigma of an interior epoch in z
        for _ in range(n_irls):
            dz = np.abs(D @ z_s)
            omega = np.minimum(1.0, delta / np.maximum(dz, 1e-12))
            z_new = tikhonov_smooth(z, w, lam_rows * omega, D)
            if np.max(np.abs(z_new - z_s)) < 1e-9:
                z_s = z_new
                break
            z_s = z_new
    out = lo + rng_b / (1.0 + np.exp(-z_s))
    out[pin_hi] = hi[pin_hi]
    out[pin_lo] = lo[pin_lo]
    return out


def method_1a_logit(i, lam_rows):
    z, w, rng_b = logit_transform(i)
    pin = at_hi[:, i] | at_lo[:, i]
    w = w.copy()
    w[pin] = 1e6 * w[~pin].max()
    z_s = tikhonov_smooth(z, w, lam_rows, D)
    out = kappa_lo[:, i] + rng_b / (1.0 + np.exp(-z_s))
    out[at_hi[:, i]] = kappa_hi[at_hi[:, i], i]
    out[at_lo[:, i]] = kappa_lo[at_lo[:, i], i]
    return out


# %% Fact A: naive smoothing of bounded estimates (uniform weights, GCV lambda, break at EQ only)
D_eq_only, D_eq_rows = difference_matrix(T, {eq_epoch})
kappa_naive = np.zeros((T, 2))
lam_naive = np.zeros(2)
for i in range(2):
    lam_naive[i] = gcv_lambda(kappa_bnd[:, i], np.ones(T), D_eq_only)
    kappa_naive[:, i] = tikhonov_smooth(kappa_bnd[:, i], np.ones(T), np.full(len(D_eq_rows), lam_naive[i]), D_eq_only)

# %% Method 1a: both variants at lambda = LAMBDA_MULT x median weight
# (Generalized cross-validation is unreliable here: the pinned epochs are exact and dominate
# the criterion. The sweep below shows the trade-off explicitly instead.)
kappa_1a = np.zeros((T, 2))
kappa_1a_logit = np.zeros((T, 2))
for i in range(2):
    lam_c = LAMBDA_MULT * np.median(weights_coupling(i))
    kappa_1a[:, i] = method_1a_coupling(i, np.full(D.shape[0], lam_c))
    z, w, _ = logit_transform(i)
    lam_z = LAMBDA_MULT * np.median(w)
    kappa_1a_logit[:, i] = method_1a_logit(i, np.full(D.shape[0], lam_z))
    print(f"patch {i+1}: 1a lambda = {lam_c:.3g} (coupling space), {lam_z:.3g} (logit space)")

# %% Lambda sweep: bias on the locked patch vs smearing of the transient, single global lambda
locked = np.arange(T) >= eq_epoch  # shallow patch truly locked post-EQ
sse_epochs = (kappa_true[:, 0] < 0.99) & ~locked  # shallow SSE epochs
multipliers = np.logspace(-3, 3, 31)
sweep = {"coupling": [], "logit": [], "naive": []}
for m in multipliers:
    lam_c = np.full(D.shape[0], m * np.median(weights_coupling(0)))
    kc = method_1a_coupling(0, lam_c)
    z, w, _ = logit_transform(0)
    lam_z = np.full(D.shape[0], m * np.median(w))
    kz = method_1a_logit(0, lam_z)
    kn = tikhonov_smooth(kappa_bnd[:, 0], np.ones(T), np.full(len(D_eq_rows), m), D_eq_only)
    for key, kap in zip(["coupling", "logit", "naive"], [kc, kz, kn]):
        sweep[key].append([kap[locked].mean(), np.sqrt(np.mean((kap[sse_epochs] - kappa_true[sse_epochs, 0]) ** 2))])
sweep = {k: np.array(v) for k, v in sweep.items()}

# %% Method 1b: RTS smoother on pseudo-observations x_t* with reduced covariance, then re-project
Q = np.diag([0.01**2, 0.3**2, 0.3**2])
jitter = 1e-10 * np.eye(3)


def kalman_rts(y, R, Q, reset_at):
    T = len(y)
    x_f = np.zeros((T, 3))
    P_f = np.zeros((T, 3, 3))
    x_p = np.zeros((T, 3))
    P_p = np.zeros((T, 3, 3))
    xk, Pk = y[0], 10.0 * np.eye(3)
    for k in range(T):
        if k > 0:
            xk, Pk = xk, Pk + Q
        if k in reset_at:
            Pk = 10.0 * np.eye(3)
        x_p[k], P_p[k] = xk, Pk
        S = Pk + R[k] + jitter
        K = linalg.solve(S.T, Pk.T).T  # Pk S^-1
        xk = xk + K @ (y[k] - xk)
        IK = np.eye(3) - K
        Pk = IK @ Pk @ IK.T + K @ R[k] @ K.T  # Joseph form
        x_f[k], P_f[k] = xk, Pk
    x_s, P_s = x_f.copy(), P_f.copy()
    for k in range(T - 2, -1, -1):
        if (k + 1) in reset_at:
            continue  # no smoothing information flows across a reset
        Ck = linalg.solve(P_p[k + 1].T, P_f[k].T).T  # P_f P_p^-1
        x_s[k] = x_f[k] + Ck @ (x_s[k + 1] - x_p[k + 1])
        P_s[k] = P_f[k] + Ck @ (P_s[k + 1] - P_p[k + 1]) @ Ck.T
    return x_s, P_s


x_1b_raw, P_1b = kalman_rts(x_bnd, cov_bnd, Q, reset_at={eq_epoch})
kappa_1b_raw = x_1b_raw[:, 1:] / x_1b_raw[:, :1]

x_1b = np.zeros((T, 3))
for k in range(T):
    Pinv = linalg.inv(P_1b[k] + jitter)
    C, _ = constraint_rows(kappa_lo[k], kappa_hi[k])
    x_1b[k], _, _ = solve_bounded_qp(Pinv, Pinv @ x_1b_raw[k], C)
kappa_1b = x_1b[:, 1:] / x_1b[:, :1]

# %% Method 1c: structured priors from the known earthquake time
# (i) Block rate is piecewise constant between earthquakes. Pool the per-epoch estimates within
#     each interseismic segment (inverse-variance weights from the reduced covariance), then
#     re-solve every epoch with the block rate fixed. With v0 fixed the coupling bound is a plain
#     box on e, so this is a bounded linear least-squares problem per epoch.
segments_1c = [np.arange(0, eq_epoch), np.arange(eq_epoch, T)]
v0_pool = np.zeros(T)
for seg in segments_1c:
    wv = 1.0 / cov_bnd[seg, 0, 0]
    v0_pool[seg] = np.sum(wv * x_bnd[seg, 0]) / np.sum(wv)

x_1c_epoch = np.zeros((T, 3))
var_kappa_1c = np.zeros((T, 2))
var_kappa_1c_unc = np.zeros((T, 2))  # unconstrained variance, used for censored (pinned) epochs
at_hi_1c = np.zeros((T, 2), dtype=bool)
at_lo_1c = np.zeros((T, 2), dtype=bool)
A_e = A[:, 1:]
for k in range(T):
    v0k = v0_pool[k]
    d_res = d_epoch[k] - A[:, 0] * v0k
    lo_e, hi_e = kappa_lo[k] * v0k, kappa_hi[k] * v0k
    res = optimize.lsq_linear(np.sqrt(w_epoch[k]) * A_e, np.sqrt(w_epoch[k]) * d_res,
                              bounds=(lo_e, hi_e), method="bvls", tol=1e-12)
    e = res.x
    at_hi_1c[k] = np.isclose(e, hi_e, atol=1e-9)
    at_lo_1c[k] = np.isclose(e, lo_e, atol=1e-9)
    x_1c_epoch[k] = [v0k, *e]
    He = w_epoch[k] * (A_e.T @ A_e)
    free = ~(at_hi_1c[k] | at_lo_1c[k])
    cov_e = np.zeros((2, 2))
    if free.any():
        idx = np.flatnonzero(free)
        cov_e[np.ix_(idx, idx)] = linalg.inv(He[np.ix_(idx, idx)])
    var_kappa_1c[k] = np.diag(cov_e) / v0k**2
    var_kappa_1c_unc[k] = np.diag(linalg.inv(He)) / v0k**2
kappa_1c_epoch = x_1c_epoch[:, 1:] / x_1c_epoch[:, :1]

# (ii) The temporal prior breaks only at the earthquake: a known bound change is not a
#      physical regime change, deformation is expected to evolve smoothly across it.
D_1c, _ = difference_matrix(T, {eq_epoch})


# (iii) Where the post-earthquake bounds admit afterslip (kappa_lo < 0), coupling is expected to
#       recover monotonically and smoothly: bounded, monotone, curvature-penalized smoother.
#       Curvature is penalized in log time since the earthquake, the natural coordinate for
#       postseismic decay (an exponential or Omori-type recovery is nearly straight there, while
#       in linear time its sharp start would be penalized hardest).
#       Epochs pinned at a bound enter as CENSORED data: value at the bound, weight from the
#       unconstrained variance. Treating them as exact would, under monotonicity, force every
#       later epoch onto the bound as soon as one noisy epoch touches it.
def monotone_smooth(khat, w, lam2, lo, hi, s_time):
    n = len(khat)
    D2 = np.zeros((n - 2, n))
    for r in range(n - 2):  # second derivative on the non-uniform grid s_time
        h1, h2 = s_time[r + 1] - s_time[r], s_time[r + 2] - s_time[r + 1]
        D2[r, r:r + 3] = [2 / (h1 * (h1 + h2)), -2 / (h1 * h2), 2 / (h2 * (h1 + h2))]
    D2 *= np.mean(np.diff(s_time)) ** 2  # dimensionless, comparable to a unit-grid second difference
    D1 = np.zeros((n - 1, n))
    for r in range(n - 1):
        D1[r, r:r + 2] = [-1.0, 1.0]
    lb, ub = lo.copy(), hi.copy()
    scale = w.sum()  # normalize the quadratic so the solver tolerances are meaningful
    ws, l2 = w / scale, lam2 / scale
    Hm = 2 * np.diag(ws) + 2 * l2 * D2.T @ D2  # constant Hessian of the QP

    def obj(kv):
        return ws @ (kv - khat) ** 2 + l2 * np.sum((D2 @ kv) ** 2)

    def jac(kv):
        return 2 * ws * (kv - khat) + 2 * l2 * D2.T @ (D2 @ kv)

    x0 = np.clip(np.maximum.accumulate(khat), lb, ub)
    res = optimize.minimize(
        obj, x0, jac=jac, hess=lambda _kv: Hm, method="trust-constr",
        bounds=optimize.Bounds(lb, ub),
        constraints=[optimize.LinearConstraint(D1, 0.0, np.inf)],
        options={"xtol": 1e-12, "gtol": 1e-10, "maxiter": 5000},
    )
    out = res.x
    assert np.all(np.diff(out) >= -1e-8) and np.all(out >= lo - 1e-8) and np.all(out <= hi + 1e-8), res.message
    return np.clip(out, lo, hi)


kappa_1c = np.zeros((T, 2))
for i in range(2):
    # (iv) Generic robustness, not physics: a Huber difference penalty so that genuine transient
    #      onsets are not halved by the quadratic smoother.
    kappa_1c[:, i] = logit_smooth(
        kappa_1c_epoch[:, i], var_kappa_1c[:, i], kappa_lo[:, i], kappa_hi[:, i],
        at_hi_1c[:, i], at_lo_1c[:, i], D_1c, robust=True,
    )
    post_idx = np.arange(eq_epoch, T)
    if np.any(kappa_lo[post_idx, i] < 0):  # afterslip admitted by the known bounds
        pinned = at_hi_1c[post_idx, i] | at_lo_1c[post_idx, i]
        var_post = np.where(pinned, var_kappa_1c_unc[post_idx, i], var_kappa_1c[post_idx, i])
        w_post = 1.0 / np.maximum(var_post, VAR_FLOOR)
        lam2 = LAMBDA_MULT * np.median(w_post)
        s_time = np.log(t_mid[post_idx] - t[eq_idx] + 0.5 * (t_mid[1] - t_mid[0]))
        kappa_1c[post_idx, i] = monotone_smooth(
            kappa_1c_epoch[post_idx, i], w_post, lam2, kappa_lo[post_idx, i], kappa_hi[post_idx, i], s_time,
        )
        print(f"patch {i+1}: post-earthquake monotone smooth recovery applied (lambda2 = {lam2:.3g})")

print(f"block rate: per-epoch rmse {np.sqrt(np.mean((x_bnd[:, 0] - x_true[:, 0])**2)):.4f}, "
      f"pooled per segment rmse {np.sqrt(np.mean((v0_pool - x_true[:, 0])**2)):.4f}")

# %% Bonus: coseismic slip from the offset between the fitted lines of the adjacent epochs
t_b = t[eq_idx]
line_pre = intercepts[eq_epoch - 1] + d_epoch[eq_epoch - 1] * t_b
line_post = intercepts[eq_epoch] + d_epoch[eq_epoch] * t_b
offset = line_post - line_pre
cos_slip, _ = optimize.nnls(G_patch.T, offset)
print(f"coseismic slip: estimated {cos_slip.round(3)}, truth [{COSEISMIC_SLIP}, 0.0]")


# %% Metrics
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


def violations(kappa):
    return int(np.sum((kappa < kappa_lo - 1e-6) | (kappa > kappa_hi + 1e-6)))


methods = {
    "unconstrained per epoch": kappa_unc,
    "bounded per epoch": kappa_bnd,
    "naive smoothing of bounded": kappa_naive,
    "1a bounded smoother (coupling space)": kappa_1a,
    "1a logit-space Tikhonov": kappa_1a_logit,
    "1b RTS before re-projection": kappa_1b_raw,
    "1b RTS + re-projection": kappa_1b,
    "1c per epoch, pooled block rate": kappa_1c_epoch,
    "1c structured priors": kappa_1c,
}
print(f"\n{'method':38s} {'rmse k1':>8s} {'rmse k2':>8s} {'viol.':>6s} {'k1 locked mean':>15s}")
for name, kap in methods.items():
    r = rmse(kap, kappa_true)
    print(f"{name:38s} {r[0]:8.4f} {r[1]:8.4f} {violations(kap):6d} {kap[locked, 0].mean():15.4f}")
print(f"{'truth':38s} {'':8s} {'':8s} {'':6s} {kappa_true[locked, 0].mean():15.4f}")
# Every final (linked) estimate must satisfy the known bounds at every epoch on every patch.
for name in ["bounded per epoch", "1a bounded smoother (coupling space)", "1a logit-space Tikhonov",
             "1b RTS + re-projection", "1c per epoch, pooled block rate", "1c structured priors"]:
    assert violations(methods[name]) == 0, name
print("bounds satisfied at all epochs by: bounded per epoch, 1a (both variants), 1b re-projected, 1c")

# %% Figures (house style: sparse edge ticks, no grid, full box, panel letters top right)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
C_WARM, C_COOL, C_GRAY = "#ff7f0e", "#1f77b4", "0.55"
# One consistent look per method across all figures: color, line style, marker.
STYLE = {
    "truth": dict(color="k", ls="-", lw=1.3, marker=None),
    "unconstrained": dict(color=C_GRAY, ls="none", marker="o", mfc="white", ms=3.5),
    "bounded": dict(color="k", ls="none", marker="o", ms=3.5),
    "naive": dict(color="0.45", ls="--", lw=1.1, marker="x", ms=4),
    "1a_coupling": dict(color="#2ca02c", ls="-.", lw=1.2, marker="^", ms=3.5, mfc="white"),
    "1a_logit": dict(color=C_COOL, ls="-", lw=1.6, marker="o", ms=3.5, mfc="white"),
    "1b_raw": dict(color="#9467bd", ls=":", lw=1.1, marker=None),
    "1b": dict(color="#9467bd", ls="--", lw=1.2, marker="s", ms=3, mfc="white"),
    "1c_epoch": dict(color="#d62728", ls="none", marker="D", ms=3, mfc="white"),
    "1c": dict(color="#d62728", ls="-", lw=1.6, marker="D", ms=3),
}
LABEL = {
    "naive": "naive smoothing",
    "1a_coupling": "1a coupling space",
    "1a_logit": "1a logit space",
    "1b_raw": "1b RTS, raw",
    "1b": "1b RTS + re-projection",
    "1c_epoch": "per epoch, pooled block rate",
    "1c": "1c structured priors",
}


def style(ax, letter=None):
    ax.tick_params(direction="out", length=3, width=0.8)
    if letter:
        ax.text(0.97, 0.95, letter, transform=ax.transAxes, ha="right", va="top")


def mark_breaks(ax):
    for k in sorted(breaks):
        ax.axvline(t_edges[k], color="0.7", lw=0.6, ls="--")


def shade_bounds(ax, i):
    for k in range(T):
        ax.fill_between(
            [t_edges[k], t_edges[k + 1]], kappa_lo[k, i], kappa_hi[k, i],
            color="0.93", lw=0, zorder=0,
        )


# Figure 1: forward model and data
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw=dict(wspace=0.35))
ax = axes[0]
ax.plot(t, v0_true, color="k", lw=1.2, label="$v_0$ (block)")
ax.plot(t, s1_true, color=C_WARM, lw=1.2, label="$s_1$ (shallow)")
ax.plot(t, s2_true, color=C_COOL, lw=1.2, label="$s_2$ (deep)")
ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_ylim(0, 3.2); ax.set_yticks([0, 1, 2, 3])
ax.set_xlabel("$t$"); ax.set_ylabel("rate"); ax.legend(frameon=False, loc="upper left")
style(ax, "a")
ax = axes[1]
for j in np.linspace(0, N_STA - 1, 6).astype(int):
    ax.plot(t, pos_obs[j], color=C_GRAY, lw=0.6)
for k in range(1, T):
    ax.axvline(t_edges[k], color="0.85", lw=0.4)
ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([-0.3, 0, 0.3]); ax.set_ylim(-0.3, 0.3)
ax.set_xlabel("$t$"); ax.set_ylabel("position")
style(ax, "b")
ax = axes[2]
ax.step(t_mid, kappa_true[:, 0], where="mid", color=C_WARM, lw=1.2, label=r"$\kappa_1$")
ax.step(t_mid, kappa_true[:, 1], where="mid", color=C_COOL, lw=1.2, label=r"$\kappa_2$")
ax.set_yscale("symlog", linthresh=1.0)
ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_ylim(-10, 1.5); ax.set_yticks([-10, -1, 0, 1])
ax.set_xlabel("$t$"); ax.set_ylabel("epoch-mean coupling"); ax.legend(frameon=False, loc="lower right")
mark_breaks(ax)
style(ax, "c")
fig.savefig(FIG_DIR / "fig1_forward_model.png", dpi=300, bbox_inches="tight")

# Figure 2: independent per-epoch solves, unconstrained vs bounded
fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True, gridspec_kw=dict(hspace=0.12))
ax = axes[0]
ax.step(t_mid, x_true[:, 0], where="mid", color="k", lw=1.2, label="truth")
ax.errorbar(t_mid, x_unc[:, 0], yerr=np.sqrt(cov_unc[:, 0, 0]), fmt="o", mfc="white",
            color=C_GRAY, ms=3.5, lw=0.7, label="unconstrained")
ax.plot(t_mid, x_bnd[:, 0], "o", color="k", ms=3.5, label="bounded")
ax.set_ylim(0.1, 0.6); ax.set_yticks([0.2, 0.4, 0.6]); ax.set_ylabel("$v_0$")
ax.legend(frameon=False, loc="lower left", ncol=3); mark_breaks(ax); style(ax, "a")
for i, (ax, col, letter) in enumerate(zip(axes[1:], [C_WARM, C_COOL], "bc")):
    shade_bounds(ax, i)
    ax.step(t_mid, kappa_true[:, i], where="mid", color="k", lw=1.2)
    ax.errorbar(t_mid, kappa_unc[:, i], yerr=np.sqrt(var_kappa_unc[:, i]), fmt="o", mfc="white",
                color=C_GRAY, ms=3.5, lw=0.7)
    ax.plot(t_mid, kappa_bnd[:, i], "o", color=col, ms=3.5)
    pin = at_hi[:, i] | at_lo[:, i]
    ax.plot(t_mid[pin], kappa_bnd[pin, i], "o", mfc="none", mec="k", ms=7, lw=0.8,
            label="bound active")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-22, 2); ax.set_yticks([-20, -10, -1, 0, 1])
    ax.set_ylabel(rf"$\kappa_{i+1}$")
    ax.legend(frameon=False, loc="lower right"); mark_breaks(ax); style(ax, letter)
axes[-1].set_xlim(0, 1); axes[-1].set_xticks([0, 0.5, 1]); axes[-1].set_xlabel("$t$")
fig.savefig(FIG_DIR / "fig2_per_epoch_bounded.png", dpi=300, bbox_inches="tight")

# Figure 3: Fact A, zoom on the locked shallow patch after the earthquake
fig, ax = plt.subplots(figsize=(7, 3.2))
tt = t_mid[locked]
ax.axhline(1.0, **{k: v for k, v in STYLE["truth"].items() if k != "marker"}, label="truth")
ax.plot(tt, kappa_unc[locked, 0], label="unconstrained per epoch", **STYLE["unconstrained"])
ax.plot(tt, kappa_bnd[locked, 0], label="bounded per epoch", **STYLE["bounded"])
for key, kap in [("naive", kappa_naive), ("1a_coupling", kappa_1a), ("1a_logit", kappa_1a_logit),
                 ("1b", kappa_1b), ("1c", kappa_1c)]:
    ax.plot(tt, kap[locked, 0], label=LABEL[key], **STYLE[key])
ax.set_xlim(0.5, 1); ax.set_xticks([0.5, 0.75, 1]); ax.set_ylim(0.4, 1.2); ax.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
ax.set_xlabel("$t$"); ax.set_ylabel(r"$\kappa_1$ (locked)")
ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8)
style(ax)
fig.savefig(FIG_DIR / "fig3_fact_a_locked_patch.png", dpi=300, bbox_inches="tight")

# Figure 4: linking the epochs, three methods against truth
fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.2), sharex=True, sharey="row",
                         gridspec_kw=dict(wspace=0.08, hspace=0.1))
titles = ["naive smoothing", "method 1a", "method 1b", "method 1c (known earthquake)"]
columns = [[("naive", kappa_naive)],
           [("1a_coupling", kappa_1a), ("1a_logit", kappa_1a_logit)],
           [("1b_raw", kappa_1b_raw), ("1b", kappa_1b)],
           [("1c_epoch", kappa_1c_epoch), ("1c", kappa_1c)]]
for i in range(2):
    for j, ax in enumerate(axes[i]):
        shade_bounds(ax, i)
        ax.step(t_mid, kappa_true[:, i], where="mid", label="truth", **STYLE["truth"])
        if j < 3:
            ax.plot(t_mid, kappa_bnd[:, i], label="bounded per epoch", **STYLE["bounded"])
        for key, kap in columns[j]:
            ax.plot(t_mid, kap[:, i], label=LABEL[key], **STYLE[key])
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(-22, 2); ax.set_yticks([-20, -10, -1, 0, 1])
        mark_breaks(ax); style(ax, "abcdefgh"[4 * i + j])
        if i == 0:
            ax.set_title(titles[j])
            ax.legend(frameon=False, loc=["lower right", "center right", "lower right", "lower right"][j], fontsize=7.5)
    axes[i, 0].set_ylabel(rf"$\kappa_{i+1}$")
for ax in axes[1]:
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("$t$")
fig.savefig(FIG_DIR / "fig4_linking_methods.png", dpi=300, bbox_inches="tight")

# Figure 5: coseismic slip from the inter-epoch offset
fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), gridspec_kw=dict(wspace=0.35))
ax = axes[0]
ax.plot(x, COSEISMIC_SLIP * G_patch[0], color="k", lw=1.2, label="truth")
ax.plot(x, offset, "o", color=C_GRAY, mfc="white", ms=3.5, label="fitted offset")
ax.plot(x, G_patch.T @ cos_slip, "--", color=C_WARM, lw=1.2, label="NNLS fit")
ax.set_xlim(-X_LIM, X_LIM); ax.set_xticks([-150, 0, 150]); ax.set_yticks([-0.25, 0, 0.25]); ax.set_ylim(-0.25, 0.25)
ax.set_xlabel("$x$ (km)"); ax.set_ylabel("coseismic offset"); ax.legend(frameon=False, loc="upper left")
style(ax, "a")
ax = axes[1]
ax.bar([0, 1], [COSEISMIC_SLIP, 0.0], width=0.35, color="0.85", edgecolor="k", label="truth")
ax.bar([0.4, 1.4], cos_slip, width=0.35, color=C_WARM, edgecolor="k", label="estimate")
ax.set_xticks([0.2, 1.2]); ax.set_xticklabels(["shallow", "deep"]); ax.set_yticks([0, 0.25, 0.5]); ax.set_ylim(0, 0.6)
ax.set_ylabel("coseismic slip"); ax.legend(frameon=False, loc="upper right")
style(ax, "b")
fig.savefig(FIG_DIR / "fig5_coseismic_slip.png", dpi=300, bbox_inches="tight")

# Figure 6: the smoothing trade-off on the shallow patch, single global lambda
fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), gridspec_kw=dict(wspace=0.35))
for key, skey in [("naive", "naive"), ("coupling", "1a_coupling"), ("logit", "1a_logit")]:
    st = {k: v for k, v in STYLE[skey].items() if k not in ("marker", "ms", "mfc")}
    axes[0].plot(multipliers, sweep[key][:, 0], label=LABEL[skey], **st)
    axes[1].plot(multipliers, sweep[key][:, 1], label=LABEL[skey], **st)
axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
axes[0].axhline(kappa_bnd[locked, 0].mean(), color="0.6", lw=0.8, ls=":")
axes[0].text(1.5e-3, kappa_bnd[locked, 0].mean() - 0.004, "independent epochs", color="0.4", fontsize=7, va="top")
axes[0].set_ylabel(r"mean $\kappa_1$, locked patch"); axes[0].set_ylim(0.9, 1.02); axes[0].set_yticks([0.9, 0.95, 1.0])
axes[1].set_ylabel(r"rmse $\kappa_1$, SSE epochs"); axes[1].set_ylim(0, 3); axes[1].set_yticks([0, 1, 2, 3])
for ax, letter in zip(axes, "ab"):
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1e3); ax.set_xticks([1e-3, 1, 1e3])
    ax.set_xlabel(r"$\lambda$ / median weight"); style(ax, letter)
for ax in axes:
    ax.axvline(LAMBDA_MULT, color="0.7", lw=0.8, ls="--")
axes[0].legend(frameon=False, loc="lower right", fontsize=8)
fig.savefig(FIG_DIR / "fig6_lambda_tradeoff.png", dpi=300, bbox_inches="tight")

# Figure 7: block rate, per epoch vs pooled between earthquakes (method 1c prior i)
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.step(t_mid, x_true[:, 0], where="mid", label="truth", **STYLE["truth"])
ax.errorbar(t_mid, x_bnd[:, 0], yerr=np.sqrt(cov_bnd[:, 0, 0]), fmt="o", color="k", ms=3.5, lw=0.7,
            label="bounded per epoch")
ax.step(t_mid, v0_pool, where="mid", color="#d62728", lw=1.6, label="pooled between earthquakes")
ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_ylim(0.1, 0.6); ax.set_yticks([0.2, 0.4, 0.6])
ax.set_xlabel("$t$"); ax.set_ylabel("$v_0$"); ax.legend(frameon=False, loc="lower left", ncol=3, fontsize=8)
mark_breaks(ax); style(ax)
fig.savefig(FIG_DIR / "fig7_block_rate_pooled.png", dpi=300, bbox_inches="tight")

print(f"\nfigures written to {FIG_DIR}")
