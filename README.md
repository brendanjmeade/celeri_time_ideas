## Initial files for time-dependent state estimation with kinematic models

All current models involve:
1. A single vertical strike slip fault.  All deformation is two-dimensional antiplane strain.  The fault is split into three parts upper, middle, and lower.  These each have different time dependent behavior.  This is what we ultimately want to recover.
2. There is a single earthquake halfway through the duration of the observation interval.  Only the shallowest portion of the fault slips here.
3. After the earthquake the shallow part of the fa

The files are:

- `concept_coseismic_slip_estimation.ipynb`:  A simple example of how to estimate "coseismic" slip at a single time step in 2D using distance weighted eigenmodes.  Currently standalon but should be intergrated into time series estimation

- `demo_read_forward_model_data.ipynb`:  A simple example of how to read in synthetic time series data created with `demo_forward_model_generate_data.ipynb`

- `demo_forward_model_generate_data.ipynb`: A notebook for generating synthetic forward model time series.  Pretty awesome.

- `forward_model_2d_001.pkl`: A data file with a synethetic single fault 2D forward velocities

- `model_test_001.pkl`: A data file with a synethetic single fault 2D forward velocities

- `demo_variational_admm.ipynb`: A notebook that applies a variational (batch) approach to time series estimation.   Coseismic jumps are subtracted out before analysis.  Still this approach treats the data in two batches, pre- and post-earthquake.

- `state_space_playground.ipynb`:

- `demo_nnls_tikhonove.ipynb`: Non-negative least squares at every time step with Tikhonove smoothing in time.  Super simple but tends to exhibit oscillations

- `omg_10.ipynb`:

- `demo_imm_with_smoother.ipynb`:  A notebook that applies a variational (batch) approach to time series estimation.   Coseismic jumps explicitly modeled here as a part of the time series analytis.  This might work even better with them subtracted out.

- `demo_variational.ipynb`:  Similar to `demo_variational_admm.ipynb` but slower because it doesn't use ADMM.

- `admm_05_c_04.ipynb`:

- `admm_05.ipynb`:
