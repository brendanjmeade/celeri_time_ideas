# Combining bounded single-epoch celeri solutions into time-dependent estimates

*Research memo, 2026-09-12. Companion to the notebooks indexed in `README.md`.*

## 1. Problem statement

celeri now routinely produces **bounded** solutions: block rotations plus fault coupling on
triangular meshes, with kinematic coupling constrained to a box (typically `[0, 1]`) through
the sequential quadratic program in `celeri/optimize.py` (`solve_sqp2`, line 1458). We are
running that solve at many epochs, each epoch being a station velocity table estimated over a
non-overlapping time interval with the same block and mesh geometry. The scientific target is
the **evolution of kinematic coupling and block motion in time**: slow slip transients,
coseismic steps, afterslip, and long-term rate changes.

The question this memo addresses: *what are meaningful ways to combine a sequence of highly
constrained per-epoch solutions into a coherent time-dependent estimate?* "Meaningful" here
means three things at once. The result must respect the same bounds the single-epoch solves
respect. It must carry a stated temporal prior rather than an accidental one. And it must be
comparable across epochs, which is not automatic (Section 3).

## 2. What a bounded epoch solution actually is

A celeri solution is a state vector on a shared linear basis plus the operators that map it to
observations:

```
x = [ ω (3 per block) | a (eigenmode coefficients per mesh, n_ss + n_ds) | ε (strain) | μ (Mogi) ]
```

`Estimation` (`celeri/solve.py:39`) stores `state_vector`, `data_vector`, `weighting_vector`
and `operators`; everything else (velocities, slip rates, Euler poles, coupling) is a derived
property. `build_estimation(model, operators, state_vector)` (`solve.py:882`) rebuilds a full
solution from a state vector, and `Estimation.mcmc_draw` (`solve.py:794`) is the existing idiom
for "same operators, different state vector." So **a set of epochs on a shared basis is just a
matrix `X` of shape `(n_epochs, n_params)`**, and every derived quantity can be recomputed per
row.

The derived quantity we care about is coupling per triangle,

```
κ_j(t) = e_j(t) / k_j(t),     e = E a(t)  (elastic slip rate),     k = S R ω(t)  (kinematic slip rate)
```

where `E` is `eigenvectors_to_tde_slip`, `R` is `rotation_to_tri_slip_rate`, and `S` is the
Gaussian smoothing applied to kinematic rates (`Operators.kinematic_slip_rate`,
`celeri/operators.py:565`). The bound `κ_lo ≤ κ_j ≤ κ_hi` is bilinear in the state. celeri
convexifies it per iteration as the "bowtie" envelope in `SlipRateLimitItem.update_constraints`
(`optimize.py:248`) and `build_constraints` (`optimize.py:370`), with kinematic anchors that are
tightened by `_tighten_kinematic_anchors` (`optimize.py:969`) over roughly 20 iterations.

Two facts about this object drive everything below.

**Fact A. Bounded estimates are truncated posteriors, not Gaussians.** The SQP returns the MAP
point of a distribution truncated at the coupling box. On a locked patch (`κ = 1`) the estimate
sits on the bound, and the posterior has essentially zero width in the direction normal to the
bound. The unconstrained covariance from `assemble_and_solve_dense` (`solve.py:919`) is wrong in
exactly those directions, and the SQP path returns no covariance at all. Consequently **any
naive average or smoother of per-epoch solutions is biased toward the interior of the box**:
if a fully locked patch dips to `κ = 0.9` in one noisy epoch, smoothing drags neighboring
epochs below 1 even though each of them was correctly pinned at the bound.

**Fact B. The eigenbasis is not pinned.** `_get_eigenvalues_and_eigenvectors`
(`celeri/mesh.py:672`) runs `eigh`/`eigsh` fresh on every `build_model`. Eigenvector sign is
arbitrary and near-degenerate eigenvalues can reorder. Coefficient vectors from two independent
`celeri-solve` runs are therefore not comparable, even when every input file except the station
velocities is identical. Coupling per triangle *is* comparable (it is basis-invariant), but
anything done in coefficient space requires a frozen basis.

## 3. Prerequisites that apply to every approach

1. **Build the basis and operators once.** Call `build_model` (`celeri/model.py:221`, which
   accepts `override_station=`) and `build_operators` (`operators.py:920`) once per geometry.
   For each epoch, swap only the station table (`dataclasses.replace(operators, model=model_t)`)
   and pass the result to `solve_sqp2(model, operators=...)`. Never rebuild the model per epoch.
   Longer term, pin the basis inside celeri (sort by eigenvalue, fix the sign so the
   largest-magnitude component is positive, and cache to disk as proposed in
   `runtime-performance-plan.md` item B2). The acceptance test is bitwise-equal
   `eigenvectors_to_tde_slip` across two `build_model` calls.
2. **Use a union station set.** The elastic operator cache hash
   (`operators.py:1250`) includes station geometry. Build once on the union of all stations and
   give absent stations zero weight in a given epoch, so nothing is recomputed and row indexing
   stays aligned.
3. **Report coupling per triangle, with a mask.** Where `|k_j|` is small (Euler pole of the
   flanking block pair near the triangle, or poorly resolved rotation), `κ_j` is undefined and
   the bowtie collapses. Track a "kinematically resolvable" mask and report coupling only there.
4. **Decide the coseismic convention up front.** If epoch velocities already have coseismic
   offsets removed, no step parameters are needed but the temporal penalty must be exempted
   across the earthquake epoch. If offsets are retained, add a free step vector at each known
   earthquake epoch. Afterslip is a rate and lives in the ordinary state; post-event the
   coupling lower bound may need to go negative on the afterslip patches.
5. **Epoch independence.** Our epochs are non-overlapping intervals, so they can be treated as
   conditionally independent. If sliding, overlapping windows are ever used instead, adjacent
   epochs share raw displacements, and covariances must be inflated by roughly window/stride
   or the correlation modeled explicitly.

## 4. Six approaches

Ordered by recommended order of attack, which is not the same as order of fidelity.

### 4.1 Post-hoc fusion of already-solved epochs

Input: existing run directories on a pinned basis. This is the cheapest thing to do and the
right first diagnostic, because it uses what we already produce. Two variants.

**1a. Coupling-space smoothing per triangle.** For each triangle, take the series `κ_j(t)`,
map it into the open interval with a logit,

```
u_j(t) = logit( (κ_j(t) − κ_lo) / (κ_hi − κ_lo) )
```

smooth `u_j(t)` in one dimension (a Gaussian process with a squared-exponential kernel plus a
step kernel at known earthquake epochs, or a 1-D total variation / fused lasso for onset
detection), and invert the logit. Bounds are satisfied by construction. The cost is
`n_triangles` independent 1-D problems, trivially parallel. The weaknesses are real: it ignores
spatial coherence, ignores the trade-off between rotations and coupling, and needs a
per-triangle noise level (best taken from the diagonal of `J Σ_t Jᵀ` where `J = ∂κ/∂x`). This
is a monitoring product, not a posterior.

**1b. Information-form fusion in coefficient space.** Treat each epoch's state `x_t*` as a
pseudo-observation with covariance `Σ_t`, then run any linear-Gaussian smoother
(Rauch–Tung–Striebel, GP regression per coefficient, or fused lasso), then re-project once into
the bounds with a constrained solve. The right `Σ_t` is the **reduced Hessian on the inactive
constraint set** at the QP optimum: with `A_act` the active constraint rows (duals above
tolerance) and `N = null(A_act)`,

```
Σ_t = N (Nᵀ H N)⁻¹ Nᵀ,     H = Gᵀ W_t G
```

This is the Laplace approximation on the active face of the box. It is valid where bounds are
inactive or the solution sits in the interior of a face. It is invalid, per Fact A, wherever
many bounds are active; there the fused mean leaks outside the box and the final re-projection
partially undoes the smoothing. That failure is informative: it maps exactly where a joint
method is needed.

*Gate:* on a synthetic benchmark, does smoothing lower coupling error without adding bound
violations?

### 4.2 Consensus ADMM that reuses the single-epoch bounded solve (recommended)

This is the rigorous version of "combine the per-epoch solutions," and it needs almost no new
solver code. Split the joint objective into per-epoch pieces `f_t` (exactly the existing
bounded problem) and a temporal piece `g`:

```
f_t(x_t) = ‖C_t x_t − d_t‖²  +  I_{K_t}(x_t)        K_t = bowtie ∩ elastic boxes ∩ segment hinges
g(z)     = λ R(z_a)  +  I{ z_ω,t equal within each interseismic segment }
```

with consensus `x_t = z_t`, scaled duals `u_t`, and penalty `ρ`:

```
x_t^{k+1} = argmin_{x ∈ K_t}  ‖C_t x − d_t‖² + (ρ/2) ‖x − (z_t^k − u_t^k)‖²      # per epoch, parallel
z^{k+1}   = prox_{g/ρ}( x^{k+1} + u^k )                                          # temporal step
u^{k+1}   = u^k + x^{k+1} − z^{k+1}
```

- **x-step.** `build_cvxpy_problem` (`optimize.py:685`) plus one term,
  `(ρ/2) · cp.sum_squares(params − center)` with `center` a `cp.Parameter`. Build the `T`
  minimizers once. Warm-start the bowtie anchors from the previous ADMM iteration so each
  epoch needs one to three QP solves per outer iteration instead of twenty. Use one shared
  column `scale` across epochs (the default `|C_t|.max(0)` is epoch-dependent and would put
  `z` and `x` in different coordinates).
- **z-step**, separable per mode across time and `O(T)` each:
  first-difference Tikhonov is a tridiagonal solve; total variation is Condat's direct 1-D
  fused-lasso algorithm; group-sparse jumps need a third ADMM block with group soft
  thresholding; static or piecewise-constant rotations reduce to averaging within each
  interseismic segment.
- **Initialization.** With `u = 0, z = x`, the first iterate is *exactly* the set of
  independent per-epoch solves. Approach 4.1 is therefore the natural initializer, and the ADMM
  iterations are literally "the correction to the per-epoch solutions that the temporal prior
  demands."
- **Convergence caveat.** The bowtie is non-convex until the kinematic anchors share a sign.
  Freeze the anchors after a few outer iterations; from then on every subproblem is convex and
  standard ADMM convergence applies. Balance `ρ` by primal/dual residuals: too large and the
  iterates stall at the per-epoch solutions, too small and bounds are ignored early.

Scales linearly in epochs, parallelizes across them, and at convergence equals the joint
problem of 4.3.

### 4.3 Joint multi-epoch convex program (reference method)

One cvxpy problem over all epochs. Variables: `ω_s` per interseismic segment (rotations
piecewise constant across earthquakes), `a_t` per epoch, optional free step vectors `δ_j` at
known earthquake epochs. Objective

```
Σ_t ‖C_t x_t − d_t‖²  +  λ R(a_1, …, a_T)  +  segment hinge penalties on ω_s
s.t.  bowtie(k(ω_{s(t)}), E a_t)   for every t and mesh
      elastic boxes                 for every t
```

Because the kinematic rate is the same expression for every epoch within a segment, **one
`SlipRateLimitItem` is shared by all epochs** (its three `cp.Parameter` arrays are built once
and `build_constraints` is called `T` times). The anchor-tightening loop is unchanged; feed it
the epoch-max out-of-bounds count.

Temporal regularizers on `Δa_t = a_{t+1} − a_t` (exempt the earthquake difference):

| `R` | behavior | cvxpy |
|---|---|---|
| `Σ‖Δa_t‖²` | smooth, low-pass; leaks steps | `cp.sum_squares(cp.diff(A, axis=0))` |
| `Σ‖Δ²a_t‖²` | piecewise-linear trends | `cp.diff(A, 2, axis=0)` |
| `Σ‖Δa_t‖₁` | piecewise-constant per mode (locked, then unlocked) | `cp.norm1(cp.diff(A, axis=0))` |
| `Σ_t ‖Δa_t‖₂` | a few epochs where all modes change at once: unknown-onset transients | `cp.sum(cp.norm(cp.diff(A, axis=0), 2, axis=1))` |
| `Σ‖Λ^{-1/2} Δa_t‖²` | high-order Matérn modes stiffer in time | weight by `1/sqrt(eigenvalues)` |

A sensible default is total variation plus a small first-difference Tikhonov term, which
removes the staircase artifacts of pure TV.

**Size.** Roughly 50 epochs × 3 meshes gives about 500k inequality rows dense in ~200
coefficients each, on the order of 10⁸ nonzeros, and 20 SQP iterations on top. Practical at
5 to 20 epochs as the ground truth against which 4.2 is validated; impractical beyond.

**Why the old NNLS + Tikhonov notebook oscillated.** `smooth_nnls` in `demo_nnls_tikhonov.ipynb`
is projected gradient with a fixed step of `1e-3` for 300 iterations, initialized from noisy
per-step NNLS. Its effective smoothing time is `lr · α · iters ≈ 1.5`, so what was plotted was
the unconverged initialization. The exact minimizer of a quadratic misfit plus a first-difference
penalty is a low-pass filter of the data and cannot oscillate more than the data does. An
interior-point solve or a converged ADMM returns that minimizer.

### 4.4 Low-rank temporal basis in the convex solve

Parameterize the coefficient trajectory in a small temporal basis:

```
a_t = ā + Σ_k Φ_{tk} β_k + Σ_j H(t ≥ t_j) δ_j
```

with `Φ` (shape `T × K`) either cubic B-splines with knots every few epochs or the
squared-exponential GP eigenbasis used in `experiment-time-series-adrian-pymc.ipynb` (seven
modes, ~20-day length scale, constant mode removed). Variables shrink to `(K+1) · n_modes`;
constraints stay per epoch, so the row count does not shrink but the KKT system is far smaller.
Smoothness is implicit; add `Σ_k ‖β_k‖²` as the GP prior. Weak for steps unless knots are dense,
and unknown earthquakes are not detectable. Its value is speed for exploration and the fact
that it is the MAP analogue of 4.6, so the two are directly comparable.

### 4.5 Constrained Kalman filter whose update is the bounded solve with a prior

For streaming, as epochs arrive. State `x_t` follows a random walk with process covariance
`Q = blockdiag(q_ω I, q_a Λ⁻¹, …)`, so high-order Matérn modes are stiffer in time. At a known
earthquake epoch, schedule a large `Q` on the coseismic partition; this replaces the IMM's hidden
jump mode with a deterministic one. The measurement update is a weighted least-squares problem
and so is *exactly* the existing bounded solve with an extra prior term:

```
min_x  ‖W^{1/2}(G x − z_t)‖² + ‖Lᵀ(x − x⁻)‖²    s.t. x ∈ K_t,      L Lᵀ = (P⁻)⁻¹
```

Implement by stacking `Lᵀ` under `G` and `Lᵀ x⁻` under `z_t` with unit weights and calling
`solve_sqp2` unchanged. After the update, take the covariance from the reduced Hessian on the
active set (as in 4.1b). The RTS smoother output is not guaranteed feasible; use it as the prior
and warm start for one joint constrained pass (4.2 or 4.3). Only `Q` needs tuning.

Why the toy IMM lingered in the jump state: with an active bound, the smooth and jump modes
produce nearly identical innovations, so the mode likelihoods carry no information and the
Markov transition prior dominates. Scheduled jumps at known epochs, and the sparse-jump priors
of 4.3 and 4.6 for unknown ones, avoid the problem instead of tuning around it.

### 4.6 Hierarchical Bayesian space-time model

Extend `solve_mcmc.py`, whose `_coupling_component` (line 536) already places a GP on eigenmode
coefficients and squashes it into the bounds with `_constrain_field` (line 107). Replace the
coefficient vector with a space-time coefficient matrix:

```
Z ~ Normal(0, 1)                               shape (n_modes, n_time_modes), non-centered
coefs(t) = sqrt(σ²_space) ⊙ Z ⊙ sqrt(σ²_time) @ Φ_time(t)ᵀ
field(t) = eigvecs @ coefs(t) + μ
κ(t)     = constrain_field(field(t), κ_lo, κ_hi)
e(t)     = k(ω_t) · κ(t)
```

with `Φ_time` the SE-GP eigenbasis or B-splines, cumulative step columns at known earthquakes,
an exponential afterslip column per event with learned decay, rotations static or stepped at
earthquakes, a learned temporal length scale (log-normal prior around 20 days), and Student-t
noise. For unknown onsets, replace the smooth basis with a regularized horseshoe on `Δcoefs`.

Cost: observations `n_epochs × 2 × n_stations`, latents `n_modes × n_time_modes` per mesh.
NUTS is hours on the Japan model and the sigmoid squash produces funnels when the length scale
is learned. The pragmatic path is MAP or ADVI for the monitoring product, seeded from the 4.2
solution, and NUTS with the length scale fixed at its MAP value for a final uncertainty pass on
a subset of epochs. This is the only approach that yields honest credible intervals on `κ(t)`.

## 5. Comparison

*See Section 10 for the revision forced by the cost of large runs.*

| approach | fidelity to bounded physics | effort | robustness / tuning | uncertainty on `κ(t)` |
|---|---|---|---|---|
| 4.1 post-hoc | low; biased at active bounds | ~1 day | easy | 1-D bands per triangle (1a), none honest (1b) |
| 4.2 consensus ADMM | equals 4.3 at convergence | low to medium: proximal kwarg, z-prox, driver | best scaling; `ρ` and anchor freezing | none (point estimate) |
| 4.3 joint QP | exact bounds at every epoch, any `R` | medium: ~300 lines of assembly | size-limited | none |
| 4.4 low-rank basis | good for slow change, weak for steps | low to medium | good | none |
| 4.5 constrained KF | good; streaming | ~200 lines plus a joint pass | only `Q` | Laplace-quality, one-sided at bounds |
| 4.6 Bayesian | best | ~1 week plus tuning | hardest | full posterior (NUTS) |

## 6. What to compare across time

- **Eigenmode coefficients `a(t)`** are the numerically convenient state: linear in the
  operators, filter-friendly, smooth. They are meaningful only on a frozen basis, and a given
  change in `a` means different things where `k` differs.
- **Elastic slip rate `e(t) = E a(t)`** is basis-free and physical (mm/yr) but is unbounded and
  conflates block motion with coupling.
- **Coupling `κ(t) = e/k`** is the physically comparable quantity: dimensionless, in `[0, 1]`,
  invariant to the basis and to rotation changes. Every approach should carry state in `a` (and
  `ω`) and report in `κ`, with the degenerate-`k` mask.

## 7. Benchmark before believing anything

The 2-D toy in this folder misled in three structural ways: three states means the coupling box
is almost never active; there is no eigenbasis, so bounds apply to slip directly rather than to
a nonlinear function of a dense basis; and it fit cumulative positions with a deep dislocation
standing in for block motion, so its "oscillation" was partly a differencing artifact.

Two benchmarks, scored by one script:

- **Toy v2** (still 2-D antiplane, seconds to run): two rigid blocks with unknown rates, ~40
  depth patches with a pinned Matérn eigenbasis of ~12 modes, two overlapping slow slip events,
  one earthquake with coseismic slip on shallow patches, logarithmic afterslip below, 50
  stations, and both position time series and non-overlapping interval velocities as outputs.
- **Real-geometry synthetic**: `celeri_test_runs/config/japan_config.json`, build once,
  prescribe `κ(t)` on the Nankai mesh (Gaussian bump plus a step plus decaying afterslip), push
  through the existing operators to velocities per epoch, add sigma-scaled noise, write one
  station CSV per epoch. The operators are the truth, so the benchmark isolates the estimator.

Metrics: coupling RMSE per triangle and epoch (median and 95th percentile, inside and outside
the transient footprint); transient onset and offset timing error; block-rate recovery in mm/yr
at stations; bound violations counted on the *nonlinear* coupling, not the bowtie proxy; wall
time and peak memory; for 4.6, coverage of the 90% interval.

## 8. Staged path

*Superseded in part by Section 10.4.*

| stage | deliverable | gate |
|---|---|---|
| 0 | basis pinning in celeri; toy v2; Japan synthetic epochs; scoring script | basis bitwise-stable across builds; per-epoch `solve_sqp2` recovers noise-free truth |
| 1 | `TimeSeriesEstimation` container (stack run dirs, per-epoch `build_estimation`, hdf5 `n_time_steps` loop in `celeri/output.py`, epoch slider in fennil); approach 4.1 | smoothed stack beats independent epochs on coupling error with no new bound violations |
| 2 | approach 4.2, validated against 4.3 on five epochs; TV + Tikhonov | beats stage 1 on onset timing and error at under 10× single-epoch wall time |
| 3 | real epochs; positions-domain variant (cumulative operator) if velocities prove limiting | recovered transients match independently known slow slip events; residuals stationary |
| 4 | approach 4.6 seeded from the stage 2 MAP; learned `Q` and length scale fed back to 4.2 and 4.5 | 90% interval coverage of at least 85% on the benchmark |

Decision rule at each gate: if a method fails on the toy but passes on the Japan synthetic, fix
the toy; if it fails on the Japan synthetic, drop the method.

## 9. Engineering seams in celeri

- `celeri/mesh.py:672` `_get_eigenvalues_and_eigenvectors`: deterministic sort and sign, disk cache.
- `celeri/operators.py:920` `build_operators` and `:1250` `_hash_elastic_operator_input`: build
  once, swap station tables, union station set with zero weights.
- `celeri/optimize.py:685` `build_cvxpy_problem`: optional proximal term (for 4.2) and optional
  extra prior rows (for 4.5); `:1458` `solve_sqp2` left otherwise unchanged; save the
  per-iteration `MinimizerTrace`, which is currently discarded.
- `celeri/solve.py:882` `build_estimation` and `:794` `mcmc_draw`: the per-epoch reconstruction idiom.
- `celeri/output.py:110`: the hdf5 writer already emits `n_time_steps = 1` and `{0:012}`-indexed
  datasets for parsli; loop it over epochs.
- `celeri/solve_mcmc.py:107` `_constrain_field` and `:536` `_coupling_component`: reuse for 4.6.
- `fennil/src/fennil/app/io.py` `load_folder_data`: a run-stack loader and an epoch slider;
  fennil reads only the CSV outputs, not the hdf5.

## 10. Addendum: single runs now take days to weeks and about 1 TB

*Added 2026-09-12 after the ranking above was written. This section supersedes the comparison
table in Section 5 and the staged path in Section 8 where they disagree.*

### 10.1 What the cost actually depends on

Nothing in the expensive part of a celeri run depends on the station velocities. Per the
performance survey (`celeri/runtime-performance-plan.md`, 100k-station WNA model: 100,592
stations, 133,945 triangles, 186 meshes), the cost is in:

- the elastic operator build (about an hour warm, cached on disk keyed by station geometry);
- the Matérn eigendecomposition per mesh (recomputed every `build_model`, no cache);
- the dense weighted operator `C = W^{1/2} G`, about 24 GB at that scale, its economic QR
  (`optimize.py:839`, order `m n²` flops, the `q` factor another 24 GB), and the orthogonality
  assert that streams it twice (item A4);
- the SQP loop, which re-solves the bounded QP roughly 20 times as the bowtie anchors tighten.

Only `d_t` (and possibly the sigmas `W_t`) changes between epochs. So a design that reruns
`celeri-solve` once per epoch pays the full cost `N` times for work that is identical `N` times.
Done right, `N` epochs should cost about one static run plus `N` cheap steps. **Time
dependence is the amortization of the static run, not an added cost on top of it.** That
reverses the intuition behind the original ranking, which treated the single-epoch solve as a
cheap black box to be called repeatedly.

I cannot tell from here which phase dominates the days-to-weeks figure or where the 1 TB peak
sits (dense operator copies, the QR factor, cvxpy canonicalization of 186 meshes of bowtie
expressions, or CLARABEL's KKT factorization). The first concrete step is to profile one large
run phase by phase with peak-memory tracking. The strategy below does not depend on the answer,
but the order of the engineering work does.

### 10.2 Four structural savings, in order of leverage

**S1. Compress each epoch to sufficient statistics.** The Gaussian misfit depends on the data
only through `n_params`-sized objects:

```
‖C x − d_t‖² = ‖R x − Qᵀ d_t‖² + const           (thin QR, R is n × n)
             = xᵀ H x − 2 g_tᵀ x + const,   H = CᵀC,  g_t = Cᵀ d_t
```

`R` (or `H`) is computed once if the weights are shared across epochs; `Qᵀ d_t` (or `g_t`) is one
matrix-vector product per epoch. If sigmas differ per epoch, `H_t = Gᵀ W_t G` is one GEMM per
epoch (minutes, not hours) followed by a Cholesky to get `R_t`. Either way the 300k data rows
never enter cvxpy again, and no per-epoch object is larger than `n_params²`. The existing
`qr_sum_of_squares` objective already hands cvxpy only `R` and `Qᵀd`; the change is to compute
them once, cache them to disk next to the operators, and reuse them. Memory: prefer forming `H`
by streaming row blocks of `G` and never materializing `q`; this squares the condition number,
but with the existing column rescaling that is normally acceptable, and it drops the 24 GB `q`.

Every approach in Section 4 consumes these statistics unchanged: QP, ADMM, Kalman update, and
the PyMC likelihood (which becomes an `n × n` quadratic form per gradient instead of a 300k-row
matmul per draw).

**S2. Fix or nearly fix block rotations, which makes the coupling bound linear.** The bowtie
exists because `κ = e/k` is bilinear in `(a, ω)`. If `ω` is frozen at the static solution, `k` is
a known vector and the bound `κ_lo k ≤ E a_t ≤ κ_hi k` is a plain linear box on `a_t`. **No SQP
loop**: one QP per epoch, or one joint QP, instead of twenty. This removes the largest
multiplier in the wall time. Block motion in time is then handled either as a piecewise-constant
`ω` re-estimated only across earthquakes (a second, small static solve per interseismic
segment), or as a perturbation `ω_t = ω₀ + Δω_t` with a strong prior, for which one or two anchor
iterations warm-started from the static anchors suffice rather than twenty.

**S3. Localize the time-dependent problem.** Transients are spatially local. Hold every mesh
except the target(s) at its static solution, subtract the static prediction from `d_t`, and
solve time dependence only for the target-mesh coefficients:

```
r_t = d_t − G x_static,      r_t ≈ G_target Δa_t + noise
```

The problem shrinks from 186 meshes and 134k triangles to one mesh and a few thousand
triangles, with a few hundred coefficients. Regularize `Δa_t` toward zero with the static
covariance (or a Matérn-eigenvalue prior) so that the frozen background does not leak into the
target. Check the residuals on non-target meshes each epoch; if they trend, widen the target set.
Coseismic slip on the target mesh is a step in `Δa`; afterslip is its rate.

**S4. Never rebuild what is data-independent.** Build the model and operators once, pin the
eigenbasis (Section 3, item 1), cache `R` or `H`, and build the cvxpy problem once with `Qᵀd_t`
as a `cp.Parameter`. Every epoch is then a parameter update and a re-solve on a small problem.

### 10.3 Revised ranking

*Superseded by Section 11 for the actual problem (per-epoch bounds on every mesh, time-varying rotations).*

| approach | at full scale, as originally described | with S1 to S4 |
|---|---|---|
| 4.1 post-hoc fusion | feasible now; costs nothing beyond the runs already done | still first; the only approach that needs no new solves |
| 4.5 constrained Kalman filter | infeasible (`N` full solves) | natural: `N` small QPs on compressed statistics, streaming, warm-started |
| 4.2 consensus ADMM | infeasible if it wraps `solve_sqp2` as-is (`N × K × 20` solves) | good on the localized, compressed problem; each x-step is a small QP; parallel over epochs |
| 4.3 joint QP | out (`N × 1M` constraint rows) | feasible on the localized problem: `N × n_target_modes` variables and `N × 8 n_tde_target` rows, roughly a million rows at 50 epochs |
| 4.4 low-rank temporal basis | constraints still per epoch, so out at full scale | good on the localized problem; smallest KKT of all |
| 4.6 Bayesian | out | feasible only on compressed statistics and a localized target; MAP/ADVI first |

The practical recommendation becomes: **post-hoc fusion of existing runs immediately (4.1);
then a localized, compressed, fixed-rotation joint QP (4.3 or 4.4) as the production method,
with the Kalman form (4.5) for streaming epochs.** Consensus ADMM (4.2) is the fallback if the
localized joint problem is still too large.

### 10.4 Revised staged path

*Superseded by Section 11 and the staged path in the typeset document.*

| stage | deliverable | gate |
|---|---|---|
| 0a | phase-by-phase profile of one large run with peak memory | know which of operators / QR / cvxpy / SQP dominates time and memory |
| 0b | basis pinning; on-disk cache of `R` or `H` and of the static solution | second epoch of a large model solves in minutes with no operator or QR recompute |
| 1 | `TimeSeriesEstimation` stack of existing runs; approach 4.1 | as before |
| 2 | localized fixed-rotation joint QP on the Japan synthetic benchmark; compare with 4.4 | beats stage 1 on onset timing and coupling error; wall time per epoch under ten minutes |
| 3 | real epochs on the target mesh; rotations piecewise-constant across earthquakes | recovered transients match known slow slip events; non-target residuals stationary |
| 4 | Bayesian pass on compressed, localized statistics | interval coverage as before |

Stage 0a is new and should come first: without it, the engineering in 0b may target the wrong
phase.

## 11. Addendum: per-epoch known bounds, bounds on every mesh, time-varying rotations

*Added 2026-09-12. Supersedes Sections 10.3 and 10.4 where they disagree. The typeset document
`time_dependent_approaches.tex` carries the full treatment; this is the summary.*

Three features of the actual problem: coupling bounds are known but differ at each epoch;
bounds apply on every mesh; block rotations may change from epoch to epoch.

### 11.1 Effect on the cost savings of Section 10
- **Bounds on every mesh remove S3 (localization).** Every epoch carries the full constraint set,
  about 8 rows per triangle. That is the per-epoch floor for any constraint-based method.
- **Time-varying rotations remove S2 (linearized bound).** The bowtie is needed per epoch.
  Replacement: **anchor warm-starting**. Anchors at epoch t = converged kinematic rate at t-1
  plus or minus a trust radius δ at least the expected per-epoch change. Sign-consistent anchors
  give a convex envelope from the first iteration with relaxation gap ~ δ(κ_hi - κ_lo), so one to
  three tightening steps replace twenty.
- **S1 (sufficient statistics) and S4 (build once) are unaffected** and remain the largest
  savings. Per-epoch, per-triangle bounds are already constraint parameters; the scalar bound
  object must become a per-triangle array.

### 11.2 Time-varying bounds as information
- Bound changes are known regime changes: break temporal priors there, as at earthquakes.
- The comparable quantity across bound changes is normalized coupling
  `u_j(t) = (κ_j(t) - κ_lo,j(t)) / (κ_hi,j(t) - κ_lo,j(t)) ∈ [0,1]`. Post-hoc smoothing acts on
  logit u. In transform-based formulations the latent field *is* logit u, so a temporal prior on
  the latent survives bound changes automatically. In QP formulations use exemptions.
- Tight informative bounds carry much of the temporal prior themselves; the simplest path
  (independent bounded epochs + post-hoc smoothing of u) may be adequate over much of the model.

### 11.3 Revised standing of the approaches
| approach | standing |
|---|---|
| 1 post-hoc | unchanged; on logit u with breaks at bound changes |
| 2 consensus ADMM | **restored to first tier**: only method with a joint temporal prior that keeps per-epoch problems at native size with their own bounds and warm-started anchors; rotations smoothed in the z-step |
| 3 joint QP | reference only, T ≤ 5 |
| 4 low-rank basis | loses its advantage (constraints per epoch on all meshes; bilinear in both coefficient sets); only with constraint collocation |
| 5 constrained filter | natural streaming form; per-epoch bounds, anchors from t-1, Q_ω governs rotation change |
| 6 transform-based (sigmoid) | **rises sharply**: per-epoch bounds on every mesh and time-varying rotations cost nothing in formulation, no bowtie, bilinearity exact, cheap gradients on compressed statistics; nonconvex and saturating, so run as MAP from the per-epoch solutions, NUTS for uncertainty |

### 11.4 Recommendation
1. Immediately: approach 1 on normalized coupling of existing runs.
2. Production batch: run **consensus ADMM with warm-started bowties** and **sigmoid-parameterized
   MAP on compressed statistics** on the same benchmark (truth with time-varying bounds and
   time-varying rotations) and choose by result. Which is cheaper per epoch (one interior-point
   solve with ~8 n_tde rows vs a few hundred O(n_params²) gradients) is decided by the stage 0a profile.
3. Production streaming: approach 5, with the batch method run periodically as the smoother.
Uncertainty from NUTS on the transform-based model. The earlier recommendation to fix rotations
and localize to one mesh is withdrawn for this problem.
