import matplotlib.pyplot as plt
import numpy as np

## Parameters
L = 1.0					# Length of spacial domain
Tmax = 5.0				# Maximum simulation time
Nx = 100				# Number of spacial grid points
Nt = 1000				# Number of time grid points 
Dt = Tmax / (Nt - 1)	# Time step size
Dx = L / (Nx - 1)		# Spatial step size
alpha = 0.005095			# Thermal diffusivity
#alpha = 0.005100			# Thermal diffusivity
#alpha = 0.005084			# Thermal diffusivity

# Initial conditions: Temperature distribution within the domain
def initial_conditions(x):
	y = np.zeros_like(x) + 0.5
	return y

# Boundary conditions: Fixed temperature at both ends of the domain
def boundary_conditions(y, t):
	y[0] = np.sin(np.pi*t)
	y[-1] = 1.0
	return y

# Discretized spatial and time domain
x_vals = np.linspace(0, L, Nx)
t_vals = np.linspace(0, Tmax, Nt)

# Initialize temperature distribution based on initial conditions
u_vals = np.empty((Nt, Nx))
u_vals[0,:] = initial_conditions(x_vals)

# Calculate diffusivity constant(?)
gamma = alpha*Dt / Dx**2

# Main simulation loop
for time_step, t in enumerate(t_vals[1:], 1):
	# Apply boundary conditions
	u_vals[time_step,:] = boundary_conditions(u_vals[time_step,:], t)

	# Update temperature using finite differences (explicit scheme)
	u_new = u_vals[time_step].copy()
	for i in range(1, Nx - 1):
		u_new[i] = gamma * (u_vals[time_step-1,i-1] + 2*u_vals[time_step-1,i] + u_vals[time_step-1,i+1])
		#u_new[i] = u_vals[i] + alpha * Dt / Dx**2 * (u_vals[i - 1] - 2 * u_vals[i] + u_vals[i + 1])

	# Update temperature array for the next time step
	u_vals[time_step,:] = u_new.copy()

# Plot 2D colormap of evolution of temperature distribution
plt.imshow(u_vals, extent=[0, L, 0, Tmax], origin='lower', aspect='auto', cmap='seismic')
plt.colorbar(label='Temperature')
plt.xlabel('Position')
plt.ylabel('Time')
plt.title('2D Colormap of Temperature Evolution')
plt.show()
