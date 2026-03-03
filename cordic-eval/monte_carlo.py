# monte_carlo.py
import numpy as np
from cordic_eval import build_cordic_tables, cordic_array

def quantise(x, n_frac, n_int=4):
    x = np.clip(x, -2**(n_int-1), 2**(n_int-1) - 2**(-n_frac))
    step = 2**(-n_frac) 
    return np.floor(x / step) * step

def monte_carlo_sweep(n_samples, frac_bits_range, iterations_range,
                      mse_target=2.4e-11, confidence=0.95):

    # z-score for confidence level
    from scipy.stats import norm
    z = norm.ppf(confidence)  # 1.645 for 95% one-sided

    # Generate Monte Carlo samples - single precision, uniform over (-8, 0)
    x = np.random.uniform(-8, 0, n_samples).astype(np.float32)

    # Golden reference - single precision exp(x)
    reference = np.exp(x).astype(np.float32)

    # Results storage
    mse_results = np.zeros((len(frac_bits_range), len(iterations_range)))
    ci_upper_results = np.zeros_like(mse_results)
    meets_target = np.zeros_like(mse_results, dtype=bool)

    for j, n_iter in enumerate(iterations_range):
        # Build tables once per N
        m_seq, theta_table, K_n = build_cordic_tables(n_iter)

        for i, n_frac in enumerate(frac_bits_range):
            x_q = quantise(x, n_frac)
            cordic_out = cordic_array(x_q, n_iter, m_seq, theta_table, K_n)

            # Per-sample squared errors
            sq_errors = (cordic_out - reference.astype(np.float64)) ** 2

            # MSE and confidence interval upper bound
            mse = np.mean(sq_errors)
            se = np.std(sq_errors, ddof=1) / np.sqrt(n_samples)
            ci_upper = mse + z * se

            mse_results[i, j] = mse
            ci_upper_results[i, j] = ci_upper
            meets_target[i, j] = ci_upper < mse_target

    return mse_results, ci_upper_results, meets_target, \
           list(frac_bits_range), list(iterations_range)

def find_minimum_parameters(meets_target, frac_bits_range, iterations_range):
    """
    Find minimum N and F such that the MSE target is met.
    Sweeps from smallest N and F upward.
    """
    for j, n_iter in enumerate(iterations_range):
        for i, n_frac in enumerate(frac_bits_range):
            if meets_target[i, j]:
                return n_frac, n_iter
    return None, None  # Target not met within sweep range