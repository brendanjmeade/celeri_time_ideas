## Initial files for time-dependent state estimation with kinematic models

## Model overview

All current models involve:

- A single vertical strike slip fault.  All deformation is two-dimensional antiplane strain.  The fault is split into two parts upper and lower.  These each have different time dependent behavior.  This is what we ultimately want to recover.
- There are moving tectonic blocks on either side of the fault.  Their motion is assumed (for now) to be constant in between earthquakes but perhaps different before and after each event.
- There is a single earthquake halfway through the duration of the observation interval.  Only the shallowest portion of the fault slips here.
- After the earthquake the shallow part of the fault does not move.  The deeper part of the fault does have temporally extended postseismic afterslip.
- There are two prescribed slow-slip events prior to the earthquake.  One on the shallow part of the fault and one on the deeper part of the fault.  These are transients that we'd like to recover.

## Files

### Data generation

- `demo_read_forward_model_data.ipynb`:  A simple example of how to read in synthetic time series data created with `demo_forward_model_generate_data.ipynb`

- `demo_forward_model_generate_data.ipynb`: A notebook for generating synthetic forward model time series.  Pretty awesome.

- `forward_model_2d_001.pkl`: A data file with a synethetic single fault 2D forward velocities.

### Primary estimation notebooks

- `demo_variational_admm.ipynb`: A notebook that applies a variational (batch) approach to time series estimation.   Coseismic jumps are subtracted out before analysis. Still, this approach treats the data in two batches, pre- and post-earthquake.

- `demo_nnls_tikhonov.ipynb`: Non-negative least squares at every time step with Tikhonov smoothing in time.  Super simple but tends to exhibit oscillations.

- `demo_imm_with_smoother.ipynb`:  A notebook that applies a variational (batch) approach to time series estimation.   Coseismic jumps explicitly modeled here as a part of the time series analytis.  This might work even better with them subtracted out.

- `concept_coseismic_slip_estimation.ipynb`:  A simple example of how to estimate "coseismic" slip at a single time step in 2D using distance weighted eigenmodes.  Currently standalon but should be intergrated into time series estimation

### Legacy notebooks

- `demo_variational.ipynb`:  Similar to `demo_variational_admm.ipynb` but slower because it doesn't use ADMM.

- `demo_state_space_playground.ipynb`: A collection of basic state estimation ideas (e.g. Kalman filter)
