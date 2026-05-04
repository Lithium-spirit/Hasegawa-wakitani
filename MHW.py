import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -- Grid & Parameters (Kept same as your original) ----------------------------
N = 256
L = 2 * np.pi * 10
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K_sq = KX**2 + KY**2

k_max = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
dealias_mask = (K_sq < (0.666 * k_max)**2).astype(float)

alpha = 1.0
kappa = 1
nu = 1e-4
Dn = 1e-4
dt = 0.025
steps = 10000 # Reduced steps for demonstration; increase as needed
plot_every = 20

# -- Initial conditions --------------------------------------------------------
np.random.seed(42)
omega = np.random.uniform(-1e-4, 1e-4, (N, N))
n = np.random.uniform(-1e-4, 1e-4, (N, N))

# -- Core Spectral Operators (Kept same as your original) ----------------------
def get_phi(omega, K_sq):
    omega_k = np.fft.fft2(omega)
    phi_k = np.zeros_like(omega_k)
    mask = K_sq != 0
    phi_k[mask] = -omega_k[mask] / K_sq[mask]
    return np.fft.ifft2(phi_k).real

def compute_nabla4(f, K_sq):
    f_k = np.fft.fft2(f)
    return np.fft.ifft2((K_sq ** 2) * f_k).real

def poisson_bracket(f, g, KX, KY):
    f_k = np.fft.fft2(f) * dealias_mask
    g_k = np.fft.fft2(g) * dealias_mask
    dfdx = np.fft.ifft2(1j * KX * f_k).real
    dfdy = np.fft.ifft2(1j * KY * f_k).real
    dgdx = np.fft.ifft2(1j * KX * g_k).real
    dgdy = np.fft.ifft2(1j * KY * g_k).real
    return dfdx * dgdy - dfdy * dgdx

def compute_derivs(omega, n, phi, K_sq, KX, KY, alpha, Dn, nu, kappa):
    phi_zonal = np.mean(phi, axis=1, keepdims=True)
    n_zonal = np.mean(n, axis=1, keepdims=True)
    phi_tilde = phi - phi_zonal
    n_tilde = n - n_zonal
    coupling = alpha * (phi_tilde - n_tilde)
    phi_k = np.fft.fft2(phi)
    dphidy = np.fft.ifft2(1j * KY * phi_k).real
    
    d_omega = (-poisson_bracket(phi, omega, KX, KY) + coupling - nu * compute_nabla4(omega, K_sq))
    d_n = (-poisson_bracket(phi, n, KX, KY) + coupling - Dn * compute_nabla4(n, K_sq) - kappa * dphidy)
    return d_omega, d_n

def dealias(f):
    return np.fft.ifft2(np.fft.fft2(f) * dealias_mask).real

def rk4_step(omega, n, phi, dt, K_sq, KX, KY, alpha, Dn, nu, kappa):
    def derivs(w, nn):
        p = get_phi(w, K_sq)
        return compute_derivs(w, nn, p, K_sq, KX, KY, alpha, Dn, nu, kappa)
    k1_o, k1_n = derivs(omega, n)
    k2_o, k2_n = derivs(omega + 0.5*dt*k1_o, n + 0.5*dt*k1_n)
    k3_o, k3_n = derivs(omega + 0.5*dt*k2_o, n + 0.5*dt*k2_n)
    k4_o, k4_n = derivs(omega + dt*k3_o, n + dt*k3_n)
    omega_new = dealias(omega + (dt / 6) * (k1_o + 2*k2_o + 2*k3_o + k4_o))
    n_new = dealias(n + (dt / 6) * (k1_n + 2*k2_n + 2*k3_n + k4_n))
    return omega_new, n_new, get_phi(omega_new, K_sq)

# -- Animation Setup -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
phi = get_phi(omega, K_sq)

# Initialize plots with current data
ims = []
titles = ['Density n', 'Vorticity ω', 'Potential φ']
fields = [n, omega, phi]

for i, (ax, field, title) in enumerate(zip(axes, fields, titles)):
    im = ax.imshow(field.T, cmap='RdBu_r', origin='lower', extent=[0, L, 0, L], animated=True)
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
          f"ETA: {remaining/60:.1f} min", end='\r')  
    
    # Update image data for the animation
    ims[0].set_data(n.T)
    ims[1].set_data(omega.T)
    ims[2].set_data(phi.T)
    
    # Dynamically adjust color limits to avoid saturation
    for im, field in zip(ims, [n, omega, phi]):
        vmax = np.max(np.abs(field))
        im.set_clim(-vmax, vmax)
        
    return ims

# -- Create and Save Video -----------------------------------------------------
import time

# Add this just before the animation line
print("Starting GPU simulation and recording...")
start_time = time.time()

ani = FuncAnimation(fig, update, frames=steps // plot_every, blit=True)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("mhw_simulation.mp4", writer=writer)



plt.close()
print("Video saved as mhw_simulation.mp4")
end_time = time.time()
elapsed = end_time - start_time
print(f"Video saved as mhw_simulation.mp4")
print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print("Starting GPU simulation and recording...")
