import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

N = 256
L = 2 * np.pi * 10
dx = L / N
x = cp.linspace(0, L, N, endpoint=False)
y = cp.linspace(0, L, N, endpoint=False)
X, Y = cp.meshgrid(x, y, indexing='ij')

kx = cp.fft.fftfreq(N, d=dx) * 2 * cp.pi
ky = cp.fft.fftfreq(N, d=dx) * 2 * cp.pi
KX, KY = cp.meshgrid(kx, ky, indexing='ij')
K_sq = KX**2 + KY**2

k_max = float(max(cp.max(cp.abs(kx)), cp.max(cp.abs(ky))))
dealias_mask = (K_sq < (0.666 * k_max)**2).astype(cp.float64)

alpha = 1.0
kappa = 0.5  
nu = 1e-3
Dn = 1e-3
dt = 0.025
steps = 80000  
record_every = 20

#initialization
rng = cp.random.RandomState(42)
omega = rng.uniform(-1e-3, 1e-3, (N, N))
n     = rng.uniform(-1e-3, 1e-3, (N, N))

#functions
def get_phi(omega, K_sq):
    omega_k = cp.fft.fft2(omega)
    phi_k = cp.zeros_like(omega_k)
    mask = K_sq != 0
    phi_k[mask] = -omega_k[mask] / K_sq[mask]
    return cp.fft.ifft2(phi_k).real

def compute_nabla4(f, K_sq):
    f_k = cp.fft.fft2(f)
    return cp.fft.ifft2((K_sq ** 2) * f_k).real

def poisson_bracket(f, g, KX, KY):
    f_k = cp.fft.fft2(f) * dealias_mask
    g_k = cp.fft.fft2(g) * dealias_mask
    dfdx = cp.fft.ifft2(1j * KX * f_k).real
    dfdy = cp.fft.ifft2(1j * KY * f_k).real
    dgdx = cp.fft.ifft2(1j * KX * g_k).real
    dgdy = cp.fft.ifft2(1j * KY * g_k).real
    return dfdx * dgdy - dfdy * dgdx

def compute_derivs(omega, n, phi, K_sq, KX, KY, alpha, Dn, nu, kappa):
    phi_zonal = cp.mean(phi, axis=1, keepdims=True)
    n_zonal   = cp.mean(n,   axis=1, keepdims=True)
    phi_tilde = phi - phi_zonal
    n_tilde   = n   - n_zonal
    coupling  = alpha * (phi_tilde - n_tilde)

    phi_k  = cp.fft.fft2(phi)
    dphidy = cp.fft.ifft2(1j * KY * phi_k).real

    d_omega = (-poisson_bracket(phi, omega, KX, KY) + coupling
               - nu * compute_nabla4(omega, K_sq))
    d_n     = (-poisson_bracket(phi, n, KX, KY) + coupling
               - Dn * compute_nabla4(n, K_sq) - kappa * dphidy)
    return d_omega, d_n

def dealias_field(f):
    return cp.fft.ifft2(cp.fft.fft2(f) * dealias_mask).real

def rk4_step(omega, n, phi, dt, K_sq, KX, KY, alpha, Dn, nu, kappa):
    def derivs(w, nn):
        p = get_phi(w, K_sq)
        return compute_derivs(w, nn, p, K_sq, KX, KY, alpha, Dn, nu, kappa)

    k1_o, k1_n = derivs(omega, n)
    k2_o, k2_n = derivs(omega + 0.5*dt*k1_o, n + 0.5*dt*k1_n)
    k3_o, k3_n = derivs(omega + 0.5*dt*k2_o, n + 0.5*dt*k2_n)
    k4_o, k4_n = derivs(omega + dt*k3_o,     n + dt*k3_n)

    omega_new = dealias_field(omega + (dt/6) * (k1_o + 2*k2_o + 2*k3_o + k4_o))
    n_new     = dealias_field(n     + (dt/6) * (k1_n + 2*k2_n + 2*k3_n + k4_n))
    return omega_new, n_new, get_phi(omega_new, K_sq)

def compute_particle_flux(n, phi, KY):
    =
    phi_k = cp.fft.fft2(phi)
    dphi_dy = cp.fft.ifft2(1j * KY * phi_k).real
    v_x = -dphi_dy
    flux = cp.mean(n * v_x)
    return float(flux.get())


print("Starting GPU simulation to track Particle Flux...")
start_time = time.time()
phi = get_phi(omega, K_sq)

time_history = []
flux_history = []

for step in range(steps):
    omega, n, phi = rk4_step(omega, n, phi, dt, K_sq, KX, KY, alpha, Dn, nu, kappa)
    
    if step % record_every == 0:
        current_time = step * dt
        current_flux = compute_particle_flux(n, phi, KY)
        
        time_history.append(current_time)
        flux_history.append(current_flux)
        
        elapsed = time.time() - start_time
        print(f"Step {step}/{steps} | Time: {current_time:.1f} | Flux: {current_flux:.4f} | Elapsed: {elapsed:.1f}s", end='\r')

print(f"\nSimulation complete in {time.time() - start_time:.1f} seconds.")

# -- Plotting ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(time_history, flux_history, label='Flux', color='#1f77b4')
plt.title(f'Flux vs Time\nmHW Model | Kappa: {kappa}')
plt.xlabel('Time')
plt.ylabel('Particle Flux')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot
plot_filename = "flux_vs_time_mhw.png"
plt.savefig(plot_filename, dpi=300)
print(f"Plot saved successfully as {plot_filename}")
