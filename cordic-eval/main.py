# main.py
from monte_carlo import monte_carlo_sweep
from visualise import plot_results, print_summary

mse_results, ci_upper_results, meets_target, F, N = monte_carlo_sweep(
    n_samples=10000,
    frac_bits_range=range(1, 17),   # 1 to 16 fractional bits
    iterations_range=range(1, 33),  # 1 to 32 CORDIC iterations
    mse_target=2.4e-11,
    confidence=0.95
)

print_summary(mse_results, meets_target, F, N)
plot_results(mse_results, ci_upper_results, meets_target, F, N)