import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def benchmark_fft(N_values, iterations=20):
    numpy_times = []
    cupy_times = []

    print(f"{'Matrix (NxN)':<15} | {'CPU (NumPy)':<15} | {'GPU (CuPy)':<15} | {'Speedup'}")
    print("-" * 65)

    for N in N_values:
        # Create random complex data
        n=int(N)
        data_np = np.random.random((N, N)) + 1j * np.random.random((N, N))
        
        # -------------------------
        # 1. CPU (NumPy) Benchmark
        # -------------------------
        # Warm-up (forces memory allocation cache)
        _ = np.fft.fft2(data_np)
        
        start_cpu = time.time()
        for _ in range(iterations):
            _ = np.fft.fft2(data_np)
        cpu_time = (time.time() - start_cpu) / iterations
        numpy_times.append(cpu_time)

        # -------------------------
        # 2. GPU (CuPy) Benchmark
        # -------------------------
        data_cp = cp.array(data_np)
        
        # Warm-up (CRITICAL: compiles CUDA kernels and allocates VRAM)
        _ = cp.fft.fft2(data_cp)
        cp.cuda.Device().synchronize() 
        
        start_gpu = time.time()
        for _ in range(iterations):
            _ = cp.fft.fft2(data_cp)
        # CRITICAL: Wait for GPU to finish before stopping the clock
        cp.cuda.Device().synchronize() 
        gpu_time = (time.time() - start_gpu) / iterations
        cupy_times.append(gpu_time)
        
        # -------------------------
        # Print Results
        # -------------------------
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"{N:<4} x {N:<8} | {cpu_time*1000:>8.2f} ms   | {gpu_time*1000:>8.2f} ms   | {speedup:>6.1f}x")

    return numpy_times, cupy_times


N_values = np.arange(500, 6000, 500)

# Run the benchmark
print("Starting FFT Computation Benchmark...\n")
np_times, cp_times = benchmark_fft(N_values)

# -- Plotting ------------------------------------------------------------------
# CPU is compute-bound: follows algorithmic complexity O(N^2 log N)
def cpu_fit_func(N, a, c):
    return a * (N**2) * np.log2(N) + c

# GPU is memory-bound: follows memory size O(N^2) + constant kernel overhead
def gpu_fit_func(N, a, c):
    return a * (N**2) + c

N_arr = np.array(N_values)

# Fit CPU data
popt_cpu, _ = curve_fit(cpu_fit_func, N_arr, np.array(np_times))
cpu_fit_eq = r"CPU Fit: $T = {:.2e} \cdot N^2 \log_2(N) + {:.2e}$".format(popt_cpu[0], popt_cpu[1])

# Fit GPU data with the memory-bound equation
popt_gpu, _ = curve_fit(gpu_fit_func, N_arr, np.array(cp_times))
gpu_fit_eq = r"GPU Fit: $T = {:.2e} \cdot N^2 + {:.2e}$".format(popt_gpu[0], popt_gpu[1])

# Generate high-resolution X values for smooth plotting
N_smooth = np.linspace(min(N_values), max(N_values), 200)
cpu_smooth = cpu_fit_func(N_smooth, *popt_cpu)
gpu_smooth = gpu_fit_func(N_smooth, *popt_gpu)
plt.figure(figsize=(10, 6))

# Plot actual data points as scatter dots
plt.scatter(N_values, np_times, color='#1f77b4', marker='o', s=50, label='NumPy (CPU) Data', zorder=3)
plt.scatter(N_values, cp_times, color='#2ca02c', marker='s', s=50, label='CuPy (GPU) Data', zorder=3)

# Plot the best fit curves as dashed lines
plt.plot(N_smooth, cpu_smooth, linestyle='--', color='#1f77b4', label=cpu_fit_eq, zorder=2)
plt.plot(N_smooth, gpu_smooth, linestyle='--', color='#2ca02c', label=gpu_fit_eq, zorder=2)

# Formatting
plt.xlabel('Grid Size N (NxN Matrix)')
plt.ylabel('Average Execution Time (Seconds)')
plt.title('Pure 2D FFT Compute Time: CPU vs. GPU (with Best Fit)')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Add legend (it will now include the equations)
plt.legend()

# Set custom x-ticks for better readability
plt.xticks(N_values, [str(n) for n in N_values])

plt.tight_layout()
plot_filename = "fft_pure_benchmark_with_fit.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nPlot saved as '{plot_filename}'")
