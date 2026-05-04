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

k_max        = float(max(cp.max(cp.abs(kx)), cp.max(cp.abs(ky))))
dealias_mask = (K_sq < (0.666 * k_max)**2).astype(cp.float64)

alpha = 1
kappa = 0.5   
nu    = 1e-3
Dn    = 1e-3
dt    = 0.025
steps        = 80000
record_every = 20

#initialzation
rng   = cp.random.RandomState(42)
omega = rng.uniform(-1e-3, 1e-3, (N, N))
n     = rng.uniform(-1e-3, 1e-3, (N, N))

#function
def get_phi(omega, K_sq):
    omega_k     = cp.fft.fft2(omega)
    phi_k       = cp.zeros_like(omega_k)
    mask        = K_sq != 0
    phi_k[mask] = -omega_k[mask] / K_sq[mask]
    return cp.fft.ifft2(phi_k).real

def compute_nabla4(f, K_sq):
    return cp.fft.ifft2((K_sq**2) * cp.fft.fft2(f)).real

def poisson_bracket(f, g, KX, KY):
    f_k  = cp.fft.fft2(f) * dealias_mask
    g_k  = cp.fft.fft2(g) * dealias_mask
    dfdx = cp.fft.ifft2(1j * KX * f_k).real
    dfdy = cp.fft.ifft2(1j * KY * f_k).real
    dgdx = cp.fft.ifft2(1j * KX * g_k).real
    dgdy = cp.fft.ifft2(1j * KY * g_k).real
    return dfdx * dgdy - dfdy * dgdx

def zonal_mean(f):

    return cp.mean(f, axis=1, keepdims=True)

def compute_derivs_ohw(omega, n, phi, K_sq, KX, KY, alpha, Dn, nu, kappa):
        
    coupling  = alpha * (phi - n)

    dphidy  = cp.fft.ifft2(1j * KY * cp.fft.fft2(phi)).real

    d_omega = (-poisson_bracket(phi, omega, KX, KY)
               + coupling
               - nu * compute_nabla4(omega, K_sq))
    d_n     = (-poisson_bracket(phi, n, KX, KY)
               + coupling
               - Dn * compute_nabla4(n, K_sq)
               - kappa * dphidy)
    return d_omega, d_n

def dealias_field(f):
    return cp.fft.ifft2(cp.fft.fft2(f) * dealias_mask).real

def rk4_step(omega, n, phi, dt, K_sq, KX, KY, alpha, Dn, nu, kappa):
    def derivs(w, nn):
        p = get_phi(w, K_sq)
        return compute_derivs_ohw(w, nn, p, K_sq, KX, KY, alpha, Dn, nu, kappa)

    k1_o, k1_n = derivs(omega,               n              )
    k2_o, k2_n = derivs(omega + 0.5*dt*k1_o, n + 0.5*dt*k1_n)
    k3_o, k3_n = derivs(omega + 0.5*dt*k2_o, n + 0.5*dt*k2_n)
    k4_o, k4_n = derivs(omega +     dt*k3_o, n +     dt*k3_n)

    omega_new = dealias_field(omega + (dt/6) * (k1_o + 2*k2_o + 2*k3_o + k4_o))
    n_new     = dealias_field(n     + (dt/6) * (k1_n + 2*k2_n + 2*k3_n + k4_n))
    return omega_new, n_new, get_phi(omega_new, K_sq)

# -- Diagnostic Functions ------------------------------------------------------
def compute_zonal_energy(phi, KY, L):
    # --- 1. Zonal Energy (KE of the mean flow) ---
    # Mean over y (axis 1), result is shape (N,)
    phi_zonal = cp.mean(phi, axis=1) 
    phi_zonal_k = cp.fft.fft(phi_zonal)
    
    # Derivative w.r.t x (the direction the flow varies)
    kx_1d = cp.fft.fftfreq(N, d=dx) * 2 * cp.pi
    # Vy = d(phi)/dx
    vy_zonal = cp.fft.ifft(1j * kx_1d * phi_zonal_k).real
    
    # Energy = 0.5 * integral(v^2) dA / Area = 0.5 * mean(v^2) * L^2
    # In the target plot, 'Energy' is usually total integrated energy
    ez = 0.5 * (L**2) * cp.mean(vy_zonal**2)
    return float(ez.get())

def compute_density_fluctuations(n, L):
    n_zonal = cp.mean(n, axis=1, keepdims=True)
    n_tilde = n - n_zonal
    en = 0.5 * (L**2) * cp.mean(n_tilde**2)
    return float(en.get())

print("Starting GPU simulation — Zonal Energy & Density Fluctuations (oHW)...")
start_time = time.time()
phi = get_phi(omega, K_sq)

time_history    = []
zonal_history   = []
density_history = []

for step in range(steps):
    omega, n, phi = rk4_step(omega, n, phi, dt, K_sq, KX, KY, alpha, Dn, nu, kappa)

    if step % record_every == 0:
        current_time  = step * dt
        zonal_e       = compute_zonal_energy(phi, KY, L)
        density_fluct = compute_density_fluctuations(n, L)

        time_history.append(current_time)
        zonal_history.append(zonal_e)
        density_history.append(density_fluct)

        elapsed   = time.time() - start_time
        remaining = (elapsed / max(step, 1)) * (steps - step)
        print(f"Step {step}/{steps} | t={current_time:.1f} | "
              f"E_zonal={zonal_e:.2f} | DensFluc={density_fluct:.2f} | "
              f"ETA: {remaining/60:.1f} min", end='\r')

print(f"\nSimulation complete in {time.time() - start_time:.1f} s")

# -- Plotting ------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle(f'mHW Model | Kappa: {kappa}', fontsize=13)

axes[0].plot(time_history, zonal_history, color='#1f77b4', label='Zonal Energy')
axes[0].set_title('Zonal Energy')
axes[0].set_ylabel('Energy')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.7)

axes[1].plot(time_history, density_history, color='#1f77b4', label='Fluctuations')
axes[1].set_title('Density Fluctuations')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Energy')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plot_filename = "zonal_density_vs_time_ohw.png"
plt.savefig(plot_filename, dpi=300)
print(f"Plot saved as {plot_filename}")
