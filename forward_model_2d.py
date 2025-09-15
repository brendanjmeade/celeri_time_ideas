# %%
# Generate synthetic data
# 2D antiplane strain

# %%
import scipy
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import cloudpickle as pickle
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# %%
# Define model class
@dataclass
class Model:
    # Input arrays
    t: np.ndarray = None  # observation times
    obs_x: np.ndarray = None  # locations of observation coordinates
    fault_x: np.ndarray = None  # fault locations

    # scalars
    n_t: int = 0  # number of time steps
    dt: int = 1  # duration of time step
    n_obs: int = 0  # number of observation coordinates
    n_faults: int = 0  # number of faults
    fault_dep: float = 0.0  # fault depths
    n_fault_patches: int = 1  # number of fault patches
    n_fault_patch_modes: int = 1  # number of fault patch modes

    # 2D arrays
    obs_vels: np.ndarray = field(init=False)  # velocity (nt, nx)
    obs_poss: np.ndarray = field(init=False)  # position (nt, nx)
    block_vels: np.ndarray = field(init=False)  # block velocity history (nt, nf+1)
    block_disps: np.ndarray = field(init=False)  # block displacement history (nt, nf+1)
    fault_patch_partials: np.ndarray = field(init=False)
    obs_block_idx: np.ndarray = field(init=False)  # block index for each observation
    fault_patch_midpoints: np.ndarray = field(init=False)  # midpoint coordinates of fault patches (nf, nfp, 2)
    linear_operator: np.ndarray = field(init=False)  # Maps model parameters to position time series

    # 3D arrays
    fault_patch_slip_rates: np.ndarray = field(
        init=False
    )  # fault slip rate history (nt, nf, nfp)
    fault_patch_slip_cumulative: np.ndarray = field(
        init=False
    )  # fault slip history (nt, nf, nfp)
    fault_patch_modes: np.ndarray = field(init=False)  # fault patch modes (nf, nfp, n_modes)

    @staticmethod
    def get_vel(s, x, fx, d):
        MIN_D = 1e-10
        if d < MIN_D:
            d = MIN_D
        vel = s / np.pi * np.arctan((x - fx) / d)
        return vel

    def get_fault_slip_rate_timeseries(
        self,
        fault_idx,
        patch_idx,
        eq_t,
        eq_slip,
        eq_afterslip_amp,
        eq_afterslip_dur,
        sse_t,
        sse_amp,
        sse_dur,
    ):
        """Generate fault slip rate time series with earthquakes, afterslip and slow slip events.

        Parameters:
        fault_idx: int - index of the fault
        patch_idx: int - index of the fault patch
        eq_t: np.ndarray - times of earthquakes
        eq_slip: np.ndarray - coseismic slip amplitudes
        eq_afterslip_amp: np.ndarray - afterslip amplitudes
        eq_afterslip_dur: np.ndarray - afterslip duration scales
        sse_t: np.ndarray - times of slow slip events
        sse_amp: np.ndarray - slow slip event amplitudes
        sse_dur: np.ndarray - slow slip event duration scales

        Updates:
        self.fault_patch_slip_rates[:, fault_idx, patch_idx] with the computed time series
        """
        # Array to store fault slip rate history
        fault_slip_rate_timeseries = np.zeros(len(self.t))

        # 1. Add earthquake jumps
        eq_idx = np.zeros(eq_t.shape, dtype="int")
        for i in range(len(eq_t)):
            eq_idx[i] = np.argmin(np.abs(self.t - eq_t[i])).astype(int)
            fault_slip_rate_timeseries[eq_idx[i]] += eq_slip[i]

        # 2. Add afterslip
        for i in range(len(eq_t)):
            fault_slip_rate_timeseries[self.t > eq_t] = eq_afterslip_amp * np.exp(
                -eq_afterslip_dur * self.t[self.t > eq_t] - eq_t
            )

        # 3. Add slow slip events
        sse_idx = np.zeros(sse_t.shape, dtype="int")
        for i in range(len(sse_t)):
            sse_idx[i] = np.argmin(np.abs(self.t - sse_t[i])).astype(
                int
            )  # currently unused
            fault_slip_rate_timeseries += sse_amp[i] * np.exp(
                -((self.t - sse_t[i]) ** 2) / (2 * sse_dur[i] ** 2)
            )

        # Store the result directly in the model's fault_patch_slip_rates array
        self.fault_patch_slip_rates[:, fault_idx, patch_idx] = fault_slip_rate_timeseries

    def get_fault_patch_slip_cumulative(self):
        # For each fault and patch, calculate cumulative slip
        for i in range(self.n_faults):
            for j in range(self.n_fault_patches):
                self.fault_patch_slip_cumulative[:, i, j] = (
                    np.cumsum(self.fault_patch_slip_rates[:, i, j]) * self.dt
                )

    def __post_init__(self):
        """Initialize arrays right after the object is created"""
        # Sort observation and fault locations in ascending order
        if self.obs_x is not None:
            self.obs_x = np.sort(self.obs_x)
        if self.fault_x is not None:
            self.fault_x = np.sort(self.fault_x)

        # Calculate dimensions from input arrays
        if self.t is not None:
            self.n_t = len(self.t)
        if self.obs_x is not None:
            self.n_obs = len(self.obs_x)
        if self.fault_x is not None:
            self.n_faults = len(self.fault_x)

        # Calculate time step
        if self.dt is not None:
            self.dt = self.t[1] - self.t[0]

        # Initialize empty arrays if input arrays were None
        if self.t is None and self.n_t > 0:
            self.t = np.zeros(self.n_t)
        if self.obs_x is None and self.n_obs > 0:
            self.obs_x = np.zeros(self.n_obs)
        if self.fault_x is None and self.n_faults > 0:
            self.fault_x = np.zeros(self.n_faults)

        # Initialze locking depths
        self.fault_patch_dep = np.zeros((self.n_faults, self.n_fault_patches + 1))
        for i in range(self.n_faults):
            self.fault_patch_dep[i, :] = np.linspace(
                0, self.fault_dep, self.n_fault_patches + 1
            )
        
        # Initialize fault patch midpoints (x, depth)
        self.fault_patch_midpoints = np.zeros((self.n_faults, self.n_fault_patches, 2))
        for i in range(self.n_faults):
            for j in range(self.n_fault_patches):
                # x coordinate is the fault location
                self.fault_patch_midpoints[i, j, 0] = self.fault_x[i]
                
                # Depth coordinate is the midpoint between upper and lower depth
                self.fault_patch_midpoints[i, j, 1] = (
                    self.fault_patch_dep[i, j] + self.fault_patch_dep[i, j+1]
                ) / 2.0

        # Initialize 2D arrays
        self.fault_patch_partials = np.zeros(
            (self.n_obs, self.n_faults, self.n_fault_patches)
        )  # fault slip to surface displacments
        self.obs_vels = np.zeros(
            (self.n_t, self.n_obs)
        )  # velocity time series at observations
        self.obs_poss = np.zeros(
            (self.n_t, self.n_obs)
        )  # position time series at observations
        self.fault_patch_slip_rates = np.zeros(
            (self.n_t, self.n_faults, self.n_fault_patches)
        )
        self.fault_patch_slip_cumulative = np.zeros(
            (self.n_t, self.n_faults, self.n_fault_patches)
        )
        self.block_vels = np.zeros(
            (self.n_t, self.n_faults + 1)
        )  # block motion velocities
        self.block_disps = np.zeros(
            (self.n_t, self.n_faults + 1)
        )  # block motion displacements

        # Create slip to observed velocity partials for each patch
        for i in range(self.n_faults):
            for j in range(self.n_fault_patches):
                # velocity kernel calcuation
                v_upper = self.get_vel(
                    1, self.obs_x, self.fault_x[i], self.fault_patch_dep[i, j]
                )
                v_lower = self.get_vel(
                    1, self.obs_x, self.fault_x[i], self.fault_patch_dep[i, j + 1]
                )
                self.fault_patch_partials[:, i, j] = v_lower - v_upper

        # Assign observations to blocks
        self.obs_block_idx = np.searchsorted(self.fault_x, self.obs_x, side="right")

    def get_eigenvectors(self, n_eigenvalues, y):
        """Calculate eigenvectors based on depth coordinates.
        
        Parameters:
        n_eigenvalues: int - number of eigenvalues to compute
        y: np.ndarray - array of y-coordinates (depths)
        
        Returns:
        np.ndarray - computed eigenvectors
        """
        n_tde = y.size
        x = np.zeros_like(y)
        z = np.zeros_like(y)

        # Calculate Cartesian distances between triangle centroids
        centroid_coordinates = np.array([x, y, z]).T
        distance_matrix = scipy.spatial.distance.cdist(
            centroid_coordinates, centroid_coordinates, "euclidean"
        )

        # Rescale distance matrix to the range 0-1
        distance_matrix = (distance_matrix - np.min(distance_matrix)) / np.ptp(
            distance_matrix
        )

        # Calculate correlation matrix
        correlation_matrix = np.exp(-distance_matrix)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            correlation_matrix,
            subset_by_index=[n_tde - n_eigenvalues, n_tde - 1],
        )
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        ordered_index = np.flip(np.argsort(eigenvalues))
        eigenvalues = eigenvalues[ordered_index]
        eigenvectors = eigenvectors[:, ordered_index]
        return eigenvectors

    def get_modes(self, n_modes):
        self.fault_patch_modes = np.zeros((self.n_faults, self.n_fault_patches, n_modes))
        self.n_fault_patch_modes = n_modes
        for i in range(self.n_faults):
            eigenvectors = self.get_eigenvectors(
                self.n_fault_patch_modes, 
                self.fault_patch_midpoints[i, :, 1]
            )
            self.fault_patch_modes[i, :, :] = eigenvectors

    def get_block_vels(self):
        self.block_vels = np.cumsum(self.fault_slip_rates, axis=0)
        self.block_vels = np.concatenate((np.zeros((1, self.n_t)), self.block_vels))
        self.block_vels = self.block_vels - (np.max(self.block_vels / 2.0))

    def get_linear_operator(self):
        self.linear_operator = np.zeros(
            (
                self.n_obs,  # Number of observation locations
                self.n_faults
                + 1  # Number of blocks
                + self.n_faults * self.n_fault_patches,  # Total number of fault patches
            )
        )

        # Insert elastic partials into the right side of the linear operator
        for i in range(self.n_faults):
            start_idx = self.n_faults + 1 + i * self.n_fault_patches
            self.linear_operator[:, start_idx : start_idx + self.n_fault_patches] = (
                self.fault_patch_partials[:, i, :]
            )

        # Insert block motion partials into the left side of the linear operator
        for i in range(self.n_faults + 1):
            current_idx = np.where(self.obs_block_idx == i)[0]
            self.linear_operator[current_idx, i] = 1
            

    def get_obs_time_series(self):
        # Initialize observation velocity matrix
        self.obs_vel_mat = np.zeros((self.n_obs, self.n_t))
        
        for i in range(self.n_t):
            # Initialize state vector
            true_state = np.zeros(self.n_faults + 1 + self.n_faults * self.n_fault_patches)

            # Insert block motion rates
            for j in range(self.n_faults + 1):
                true_state[j] = self.block_vels[j, i]

            # Insert fault slip rates
            for j in range(self.n_faults):
                start_idx = self.n_faults + 1 + j * self.n_fault_patches
                true_state[start_idx : start_idx + self.n_fault_patches] = (
                    -self.fault_patch_slip_rates[i, j, :]
                    + self.fault_slip_rates[j, i]
                )

            # Calculate observed velocities
            self.obs_vel_mat[:, i] = self.linear_operator @ true_state

        # Convert observed velocities to observed positions
        self.obs_pos_mat = np.cumsum(self.obs_vel_mat, axis=1) * self.dt

        # Subtract out first position measurement so that all start at zero
        self.obs_pos_mat -= self.obs_pos_mat[:, 0:1]  # Note strange indexing to prevent 1D array

    def get_fault_slip_cumulative(self):
        # TODO: Need to generalize over multiple faults.  This is currently just for a single fault
        self.fault_slip_cumulative = np.cumsum(np.squeeze(self.fault_slip_rates))
        self.fault_slip_cumulative -= self.fault_slip_cumulative[0]
        self.fault_slip_cumulative *= self.dt



# %%
# Initialize model
model = Model(
    t=np.arange(0, 101), # days
    obs_x=np.linspace(-50e3, 50e3, 50), # km
    fault_x=np.array([0e3]), # km
    fault_dep=20e3, # km
    n_fault_patches=20,
)

# Specify fault slip rates
model.fault_slip_rates = 1.0 * np.ones((model.n_faults, model.n_t))

# Generate time series over upper 10 patches
for i in range(0, 10):
    model.get_fault_slip_rate_timeseries(
        fault_idx=0,
        patch_idx=i,
        eq_t=np.array([50]),  # days
        eq_slip=np.array([25.0]),
        eq_afterslip_amp=np.array([0]),
        eq_afterslip_dur=np.array([0]),
        sse_t=np.array([35]), # days
        sse_amp=np.array([3.00]),
        sse_dur=np.array([2]),
    )

# Generate time series over lower 10 patches
for i in range(10, 20):
    model.get_fault_slip_rate_timeseries(
        fault_idx=0,
        patch_idx=i,
        eq_t=np.array([50]), # days
        eq_slip=np.array([0.0]),
        eq_afterslip_amp=np.array([200]),
        eq_afterslip_dur=np.array([8.0]),
        sse_t=np.array([25]), # days
        sse_amp=np.array([1.00]),
        sse_dur=np.array([3]),
)

# %%
# Calculate derived model parameters

# Get cumulative fault slip (differential block vels)
model.get_fault_slip_cumulative()

# Get cumulative slip for all patches
model.get_fault_patch_slip_cumulative()

# Get eigen modes for all faults
model.get_modes(n_modes=10)

# Calculate block velocities
model.get_block_vels()

# Build linear operator (model parameters -> position time series)
model.get_linear_operator()

# Generate forward time series
model.get_obs_time_series()


# %%
# Plot fault patch time series (deviation)
linewidth = 1.0
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(
    model.t, model.fault_patch_slip_rates[:, 0, 0], "-b", label="patch 1", linewidth=linewidth
)
plt.plot(
    model.t, model.fault_patch_slip_rates[:, 0, 1], "-r", label="patch 2", linewidth=linewidth
)
plt.legend()
plt.xlim([model.t.min(), model.t.max()])
plt.title("fault slip rates (deviation)", fontsize=10)
plt.xlabel("time (days)")
plt.ylabel("slip rate")

# Plot cumulative slip
plt.subplot(2, 1, 2)
plt.plot(
    model.t,
    model.fault_patch_slip_cumulative[:, 0, 0],
    "-b",
    label="patch 1",
    linewidth=linewidth,
)
plt.plot(
    model.t,
    model.fault_patch_slip_cumulative[:, 0, 1],
    "-r",
    label="patch 2",
    linewidth=linewidth,
)
plt.legend()
plt.xlim([model.t.min(), model.t.max()])
plt.title("cumulative fault slip (deviation)", fontsize=10)
plt.xlabel("time (days)")
plt.ylabel("slip")
plt.show(block=False)


# %%
# Plot observed velocities and cumulative position time series (deviation)
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(model.t, model.obs_vel_mat.T, linewidth=0.5)
plt.xlim([model.t.min(), model.t.max()])
plt.title("observed velocity time series", fontsize=10)
plt.xlabel("time (days)")
plt.ylabel("position")

plt.subplot(2, 1, 2)
plt.plot(model.t, model.obs_pos_mat.T, linewidth=0.5)
plt.xlim([model.t.min(), model.t.max()])
plt.title("observed position time series", fontsize=10)
plt.xlabel("time (days)")
plt.ylabel("position")
plt.show(block=False)

# %%
# Plot observed velocities and cumulative position time series (total)

# Plot slip rates
linewidth = 1.0
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(
    model.t,
    -np.squeeze(model.fault_slip_rates),
    "-k",
    label="differential block motion",
    linewidth=linewidth,
)
plt.plot(
    model.t,
    model.fault_patch_slip_rates[:, 0, 0] - np.squeeze(model.fault_slip_rates),
    "-b",
    label="patch 1",
    linewidth=linewidth,
)
plt.plot(
    model.t,
    model.fault_patch_slip_rates[:, 0, 1] - np.squeeze(model.fault_slip_rates),
    "-r",
    label="patch 2",
    linewidth=linewidth,
)
plt.legend()
plt.xlim([model.t.min(), model.t.max()])
plt.title("fault slip rates (including fully locked)", fontsize=10)
plt.xlabel("time (days)")
plt.ylabel("slip rate")

# Plot cumulative slip
plt.subplot(2, 1, 2)
plt.plot(
    model.t,
    -model.fault_slip_cumulative,
    "-k",
    label="differential block motion",
    linewidth=linewidth,
)
plt.plot(
    model.t,
    model.fault_patch_slip_cumulative[:, 0, 0] - model.fault_slip_cumulative,
    "-b",
    label="patch 1",
    linewidth=linewidth,
)
plt.plot(
    model.t,
    model.fault_patch_slip_cumulative[:, 0, 1] - model.fault_slip_cumulative,
    "-r",
    label="patch 2",
    linewidth=linewidth,
)
plt.legend()
plt.xlim([model.t.min(), model.t.max()])
plt.title("cumulative fault slip (including fully locked)", fontsize=10)
plt.xlabel("time (days)")
plt.ylabel("slip")
plt.show(block=False)

# %%
# Plot modes
plt.figure(figsize=(2, 6))
for i in range(model.n_fault_patch_modes):
    plt.plot(model.fault_patch_modes[0, :, i], model.fault_patch_midpoints[0, :, 1], label=f"{i}", linewidth=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("mode amplitudes")
plt.ylabel("patch midpoints (m)")
plt.gca().invert_yaxis()
plt.show(block=False)


# %%
# Save model instance as cloudpickle
filename_pickle = "forward_model_2d_001.pkl"
with open(filename_pickle, "wb") as f:
    pickle.dump(model, f)
