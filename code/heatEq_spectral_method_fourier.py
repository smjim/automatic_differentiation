import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0			# Length of spatial domain
Tmax = 5		# Maximum simulation time
Nx = 100		# Number of spatial grid points
Nt = 1001		# Number of time steps
alpha = 0.0039	# Thermal diffusivity

# Discretized spatial and time domain
x_vals = np.linspace(0, L, Nx)
t_vals = np.linspace(0, Tmax, Nt)
dx = x_vals[1] - x_vals[0]
dt = t_vals[1] - t_vals[0]

# Initial condition (example: Gaussian pulse)
#u_initial = np.exp(-100 * (x_vals - 0.5 * L)**2)
u_initial = np.zeros(Nx) + 0.5

# Spectral method matrices
k = np.fft.fftfreq(Nx, dx) * 2 * np.pi  # Wavenumbers
K = 1j * k  # Differential operator in frequency domain
print(K)

# Time-stepping loop
u = np.zeros((Nt, Nx), dtype=np.complex128)
u[0, :] = u_initial

for n, t in enumerate(t_vals[1:], 1):
	# Spectral method for spatial derivative
	u_x = np.fft.ifft(K * np.fft.fft(u[n - 1, :]))

	# Time integration using the implicit Euler method
	u_t = np.fft.ifft(K * np.fft.fft(u_x))
	u[n, :] = u[n - 1, :] + alpha * dt * u_t

	# Enforce Dirichlet boundary conditions:
	u[n, 0] = np.sin(np.pi * t)
	u[n, -1] = 1 

# Plot the solution
plt.figure(figsize=(8, 6))
cbar = plt.imshow(np.real(u), extent=[0, L, 0, Tmax], origin='lower', aspect='auto', cmap='seismic')

max_u = np.max(np.abs(u))
cbar.set_clim(vmin=-max_u, vmax=max_u)
plt.colorbar(cbar,label='Temperature')

plt.xlabel('Position (x)')
plt.ylabel('Time (t)')
plt.title('1D Heat Equation Solution using Time Spectral Method')
plt.show()

