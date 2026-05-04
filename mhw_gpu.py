import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -- Grid & Parameters ---------------------------------------------------------
N = 512
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
kappa = 1
nu = 1e-4
Dn = 1e-4
dt = 0.025
steps = 10000
plot_every = 20

# -- Initial conditions --------------------------------------------------------
rng = cp.random.RandomState(42)
omega = rng.uniform(-1e-4, 1e-4, (N, N))
n     = rng.uniform(-1e-4, 1e-4, (N, N))

# -- Core Spectral Operators ---------------------------------------------------
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

# -- Animation Setup -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
phi = get_phi(omega, K_sq)

ims = []
titles = ['Density n', 'Vorticity ω', 'Potential φ']
fields = [n, omega, phi]

for ax, field, title in zip(axes, fields, titles):
    # .get() transfers the CuPy array to CPU numpy for matplotlib
    im = ax.imshow(field.get().T, cmap='RdBu_r', origin='lower',
                   extent=[0, L, 0, L], animated=True)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ims.append(im)

total_frames = steps // plot_every  # = 500

def update(frame):
    global omega, n, phi
    
    frame_start = time.time()
    
    for _ in range(plot_every):
        omega, n, phi = rk4_step(omega, n, phi, dt, K_sq, KX, KY,
                                  alpha, Dn, nu, kappa)

    frame_time = time.time() - frame_start
    remaining = frame_time * (total_frames - frame - 1)
    print(f"Frame {frame+1}/{total_frames} | "
          f"Frame time: {frame_time:.2f}s | "
          f"ETA: {remaining/60:.1f} min", end='\r')    # Transfer GPU → CPU only for rendering (cheap vs. computation cost)
    n_cpu, omega_cpu, phi_cpu = n.get(), omega.get(), phi.get()

    ims[0].set_data(n_cpu.T)
    ims[1].set_data(omega_cpu.T)
    ims[2].set_data(phi_cpu.T)

    for im, field_cpu in zip(ims, [n_cpu, omega_cpu, phi_cpu]):
        vmax = np.max(np.abs(field_cpu))
        im.set_clim(-vmax, vmax)

    return ims

# -- Create and Save Video -----------------------------------------------------
import time

# Add this just before the animation line
print("Starting GPU simulation and recording...")
start_time = time.time()

ani = FuncAnimation(fig, update, frames=steps // plot_every, blit=True)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("mhw_simulation_new.mp4", writer=writer)



plt.close()
print("Video saved as mhw_simulation.mp4")
end_time = time.time()
elapsed = end_time - start_time
print(f"Video saved as mhw_simulation.mp4")
print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print("Starting GPU simulation and recording...")
