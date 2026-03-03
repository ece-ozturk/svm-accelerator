# visualise.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def plot_results(mse_results, ci_upper_results, meets_target,
                 frac_bits_range, iterations_range, mse_target=2.4e-11):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('CORDIC exp(x) Monte Carlo Sweep', fontsize=14, fontweight='bold')

    F = list(frac_bits_range)
    N = list(iterations_range)

    # --- Plot 1: MSE Heatmap (log scale) ---
    ax1 = axes[0]
    log_mse = np.log10(mse_results + 1e-300)  # Avoid log(0)
    im1 = ax1.imshow(log_mse, aspect='auto', origin='lower',
                     cmap='viridis',
                     extent=[N[0]-0.5, N[-1]+0.5, F[0]-0.5, F[-1]+0.5])
    plt.colorbar(im1, ax=ax1, label='log₁₀(MSE)')
    ax1.set_xlabel('CORDIC Iterations (N)')
    ax1.set_ylabel('Fractional Bits (F)')
    ax1.set_title('MSE (log scale)')

    # Mark the target contour
    ax1.contour(N, F, log_mse, levels=[np.log10(mse_target)],
                colors='red', linewidths=2, linestyles='--')
    ax1.plot([], [], 'r--', label=f'MSE = {mse_target:.1e}')
    ax1.legend(fontsize=8)

    # --- Plot 2: CI Upper Bound Heatmap ---
    ax2 = axes[1]
    log_ci = np.log10(ci_upper_results + 1e-300)
    im2 = ax2.imshow(log_ci, aspect='auto', origin='lower',
                     cmap='plasma',
                     extent=[N[0]-0.5, N[-1]+0.5, F[0]-0.5, F[-1]+0.5])
    plt.colorbar(im2, ax=ax2, label='log₁₀(CI Upper Bound)')
    ax2.set_xlabel('CORDIC Iterations (N)')
    ax2.set_ylabel('Fractional Bits (F)')
    ax2.set_title('95% CI Upper Bound (log scale)')
    ax2.contour(N, F, log_ci, levels=[np.log10(mse_target)],
                colors='red', linewidths=2, linestyles='--')

    # --- Plot 3: Pass/Fail Map with minimum point marked ---
    ax3 = axes[2]
    pass_fail = meets_target.astype(float)
    cmap_pf = mcolors.ListedColormap(['#d9534f', '#5cb85c'])
    ax3.imshow(pass_fail, aspect='auto', origin='lower',
               cmap=cmap_pf, vmin=0, vmax=1,
               extent=[N[0]-0.5, N[-1]+0.5, F[0]-0.5, F[-1]+0.5])
    ax3.set_xlabel('CORDIC Iterations (N)')
    ax3.set_ylabel('Fractional Bits (F)')
    ax3.set_title('Meets Target (95% CI)')

    # Find and mark minimum parameters
    min_F, min_N = find_minimum_parameters(meets_target, F, N)
    if min_F is not None:
        ax3.plot(min_N, min_F, 'w*', markersize=15,
                 label=f'Min: F={min_F}, N={min_N}')
        ax3.legend(fontsize=9)

    legend_elements = [Patch(facecolor='#5cb85c', label='Pass'),
                       Patch(facecolor='#d9534f', label='Fail')]
    ax3.legend(handles=legend_elements + ([ax3.lines[0]] if min_F else []),
               fontsize=8)

    plt.tight_layout()
    plt.savefig('cordic_sweep_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to cordic_sweep_results.png")

def print_summary(mse_results, meets_target, frac_bits_range,
                  iterations_range, mse_target=2.4e-11):

    F = list(frac_bits_range)
    N = list(iterations_range)
    min_F, min_N = find_minimum_parameters(meets_target, F, N)

    print("\n" + "="*50)
    print("CORDIC SWEEP SUMMARY")
    print("="*50)
    print(f"MSE Target:        {mse_target:.2e}")
    print(f"Confidence Level:  95%")
    print(f"F range swept:     {F[0]} to {F[-1]} fractional bits")
    print(f"N range swept:     {N[0]} to {N[-1]} iterations")
    print("-"*50)

    if min_F is not None:
        i = F.index(min_F)
        j = N.index(min_N)
        print(f"Minimum F:         {min_F} fractional bits")
        print(f"Minimum N:         {min_N} iterations")
        print(f"Achieved MSE:      {mse_results[i, j]:.4e}")
        print(f"Total word length: {4 + min_F} bits (4 integer + {min_F} fractional)")
    else:
        print("Target NOT met within the swept parameter range.")
        print("Consider extending F or N ranges.")
    print("="*50)

def find_minimum_parameters(meets_target, frac_bits_range, iterations_range):
    for j, n_iter in enumerate(iterations_range):
        for i, n_frac in enumerate(frac_bits_range):
            if meets_target[i, j]:
                return n_frac, n_iter
    return None, None