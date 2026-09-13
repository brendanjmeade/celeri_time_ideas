# Time-dependent state estimation with kinematic block models

Prototype ideas for tracking kinematic coupling and block motion through time with celeri.
celeri now produces bounded (coupling-constrained) solutions at many epochs; the open question
is how to combine those highly constrained single-epoch solutions into a coherent time series.

**Start with [`TIME_DEPENDENT_APPROACHES.md`](TIME_DEPENDENT_APPROACHES.md)** (or the typeset
version, `time_dependent_approaches.tex` / `.pdf`, which is the same material written up as a paper-style document). It states the
problem, explains why naive smoothing of bounded solutions is biased, lays out six candidate
approaches (post-hoc fusion, consensus ADMM reusing the single-epoch solver, joint multi-epoch
QP, low-rank temporal basis, constrained Kalman filter, hierarchical Bayesian), ranks them, and
proposes a benchmark and a staged path. Section 10 revises the ranking for the fact that
single large runs now take days to weeks and about 1 TB: the data-independent work (operators,
eigenbasis, QR) must be done once and shared, epochs compressed to sufficient statistics,
rotations fixed so the coupling bound becomes linear, and the time-dependent problem localized
to the target mesh. Section 11 then withdraws the last two for our actual problem (known bounds
that differ per epoch, bounds on every mesh, time-varying rotations) and narrows the production
choice to consensus ADMM with warm-started bowties versus a sigmoid-parameterized MAP fit.

## The 2-D toy model used by the older notebooks

- A single vertical strike-slip fault; all deformation is 2-D antiplane strain. The fault is
  split into an upper and a lower patch with different time-dependent behavior.
- Moving blocks on either side of the fault, constant between earthquakes but possibly different
  before and after.
- One earthquake halfway through the record; only the shallow patch slips.
- After the earthquake the shallow patch is locked and the deep patch has decaying afterslip.
- Two prescribed slow slip events before the earthquake, one shallow and one deep.

This toy has three state variables and no eigenbasis, so the coupling bounds are almost never
active. That is the main reason its conclusions did not transfer to celeri; see Section 7 of
the memo for the proposed upgrade.

## Notebook index

### Active (to be carried forward)

- `toy_method1_posthoc_fusion.py`: runnable demonstration of Approach 1 (post-hoc fusion) on the
  2-D toy. Generates position time series, chunks them into 40 epochs, solves a bounded
  (coupling-constrained) problem per epoch with known bounds that differ between epochs, then
  links the epochs four ways: naive smoothing, method 1a (normalized coupling with active-bound
  epochs pinned, in coupling space and in logit space), method 1b (RTS fusion of reduced
  covariances, then re-projection), and method 1c (structured priors from the known earthquake
  time: block rate pooled between earthquakes and coupling re-solved with it fixed, temporal prior
  broken only at the earthquake, monotone smooth recovery in log time where the bounds admit
  afterslip, pinned epochs treated as censored, Huber difference penalty). Writes
  `figures/fig1..fig7`. numpy/scipy/matplotlib only; run with `python toy_method1_posthoc_fusion.py`.
  The structured-prior method is written up in `structured_priors_toy.tex` / `.pdf`.
- `experiment-time-series-adrian-pymc.ipynb`: Adrian Seyboldt's PyMC prototype on the real
  celeri Japan model. Static block rotations, a seven-mode squared-exponential GP temporal
  basis on eigenmode coefficients, and cumulative step functions at known earthquake epochs.
  Prior-predictive only, no bounds yet. Seed for the Bayesian approach (memo 4.6) and the
  low-rank basis approach (memo 4.4).
- `demo_variational_admm.ipynb`: joint-in-time ADMM on the toy with a projection onto
  non-negativity and a coupling box on per-step differences. The one toy idea that carries
  forward; the consensus ADMM in memo 4.2 is its rigorous generalization.
- `demo_coseismic_slip_estimation.ipynb`: single-epoch coseismic slip from distance-weighted
  eigenmodes with a positive ridge solve. Becomes the coseismic block of a positions-domain
  model. Note the unexplained sign flip flagged in the notebook.
- `demo_forward_model_generate_data.ipynb`: the synthetic data generator. To be upgraded to
  "toy v2" (blocks instead of a deep dislocation, many patches with a pinned eigenbasis,
  both positions and interval velocities). Writes `forward_model_2d_001.pkl`.

### Reference only (documented failure modes, not to be extended)

- `demo_nnls_tikhonov.ipynb`: independent NNLS per epoch plus first-difference Tikhonov,
  solved by projected gradient. Oscillates because the solve is unconverged (fixed step,
  300 iterations), not because of the objective.
- `demo_imm_with_smoother.ipynb`: interacting multiple model Kalman filters with an RTS-style
  smoother and NNLS re-projection. Mode switching lingers because, at an active bound, the
  smooth and jump modes produce nearly identical innovations. Useful as the Kalman baseline
  for memo 4.5.

### Legacy (superseded)

- `demo_variational.ipynb`: pre-ADMM version of the variational approach; slower.
- `demo_state_space_playground.ipynb`: grab bag of Kalman, jump-aware Kalman, IMM and particle
  filter experiments on the toy.
- `demo_read_forward_model_data.ipynb`: two-line loader for `forward_model_2d_001.pkl`.

### Data

- `forward_model_2d_001.pkl`: synthetic 2-D forward model produced by the generator notebook.
  Note that the pickled class is newer than the generator notebook and references a source file
  not in this repo (`japan_imaging/state_space/forward_model_2d.py`).

## Environment

`environment.yml` covers the toy notebooks. The PyMC notebook runs against a celeri Pixi
environment, not this one.
