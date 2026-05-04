import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

N = 256
L = 2 * np.pi * 10
dx = L / N
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K_sq = KX**2 + KY**2

k_max        = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
dealias_mask = (K_sq < (0.666 * k_max)**2).astype(float)

alpha = 1.0
kappa = 1.0
nu    = 1e-4
Dn    = 1e-4
dt    = 0.025
steps      = 10000
plot_every = 20

np.random.seed(42)
omega = np.random.uniform(-1e-4, 1e-4, (N, N))
n     = np.random.uniform(-1e-4, 1e-4, (N, N))

def get_phi(omega, K_sq):
    omega_k = np.fft.fft2(omega)
    phi_k   = np.zeros_like(omega_k)
    mask    = K_sq != 0
    phi_k[mask] = -omega_k[mask] / K_sq[mask]
    return np.fft.ifft2(phi_k).real

def compute_nabla4(f, K_sq):
    return np.fft.ifft2((K_sq**2) * np.fft.fft2(f)).real

def poisson_bracket(f, g, KX, KY):
    fk = np.fft.fft2(f) * dealias_mask
    gk = np.fft.fft2(g) * dealias_mask
    return (np.fft.ifft2(1j*KX*fk).real * np.fft.ifft2(1j*KY*gk).real
          - np.fft.ifft2(1j*KY*fk).real * np.fft.ifft2(1j*KX*gk).real)

def compute_derivs(omega, n, phi, K_sq, KX, KY, alpha, Dn, nu, kappa):
    coupling = alpha * (phi - n)

    dphidy = np.fft.ifft2(1j * KY * np.fft.fft2(phi)).real

    d_omega = (-poisson_bracket(phi, omega, KX, KY)
               + coupling
               - nu * compute_nabla4(omega, K_sq))
    d_n     = (-poisson_bracket(phi, n, KX, KY)
               + coupling
               - Dn * compute_nabla4(n, K_sq)
               - kappa * dphidy)
    return d_omega, d_n

def dealias(f):
    return np.fft.ifft2(np.fft.fft2(f) * dealias_mask).real

def rk4_step(omega, n, phi, dt):
    def derivs(w, nn):
        p = get_phi(w, K_sq)
        return compute_derivs(w, nn, p, K_sq, KX, KY, alpha, Dn, nu, kappa)
    k1_o, k1_n = derivs(omega,                 n               )
    k2_o, k2_n = derivs(omega + 0.5*dt*k1_o,   n + 0.5*dt*k1_n)
    k3_o, k3_n = derivs(omega + 0.5*dt*k2_o,   n + 0.5*dt*k2_n)
    k4_o, k4_n = derivs(omega +     dt*k3_o,   n +     dt*k3_n)
    omega_new = dealias(omega + (dt/6)*(k1_o + 2*k2_o + 2*k3_o + k4_o))
    n_new     = dealias(n     + (dt/6)*(k1_n + 2*k2_n + 2*k3_n + k4_n))
    return omega_new, n_new, get_phi(omega_new, K_sq)

# ── Animation setup ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
phi = get_phi(omega, K_sq)

ims, cbars = [], []
for ax, field, title in zip(axes, [n, omega, phi],
                             ['Density  n', 'Vorticity  ω', 'Potential  φ']):
    v  = max(np.max(np.abs(field)), 1e-12)
    im = ax.imshow(field.T, cmap='RdBu_r', origin='lower',
                   extent=[0, L, 0, L], vmin=-v, vmax=v, animated=True)
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    cbars.append(fig.colorbar(im, ax=ax))
    ims.append(im)

time_text = axes[1].set_title('')

def update(frame):
    global omega, n, phi
    for _ in range(plot_every):
        dt_use = 0.005 if np.max(np.abs(n)) > 0.5 else dt
        omega, n, phi = rk4_step(omega, n, phi, dt_use)

    t = frame * plot_every * dt
    for im, cb, field in zip(ims, cbars, [n, omega, phi]):
        v = max(np.max(np.abs(field)), 1e-12)
        im.set_data(field.T)
        im.set_clim(-v, v)
        cb.update_normal(im)          # keeps colorbar ticks correct

    axes[0].set_title(f'Density  n')
    axes[1].set_title(f'Ordinary HW  —  t = {t:.1f}')
    axes[2].set_title(f'Potential  φ')
    return ims

print("Running OHW simulation...")
ani = FuncAnimation(fig, update, frames=steps // plot_every,
                    blit=False)   # blit=False so colorbar redraws cleanly

writer = FFMpegWriter(fps=15, bitrate=1800,
                      extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
ani.save("ohw_simulation.mp4", writer=writer)
plt.close()
print("Saved → ohw_simulation.mp4")
