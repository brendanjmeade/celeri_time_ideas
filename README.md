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

- `demo_variational.ipynb`: A notebook that applies a variational (batch) approach to time series estimation.   Coseismic jumps are subtracted out before analysis.  Still this approach treats the 

- `state_space_playground.ipynb`:

- `smoothing_time_series.ipynb`:

- `omg_10.py`:

- `omg_10.ipynb`:

- `imm_with_smoother.ipynb`:

- `batch_smooth_time.ipynb`:

- `admm_05_c_04.ipynb`:

- `admm_05.ipynb`:
