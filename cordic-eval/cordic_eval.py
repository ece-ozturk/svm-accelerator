# cordic_eval.py
import numpy as np

# Generate sequence to include repeats for hyperbolic CORDIC
def hyperbolic_seq(n: int):
    repeats = set()
    k = 4
    while k <= 10000: # Arbitrary large set of k
        repeats.add(k)
        k = 3*k + 1

    seq = []
    m = 1
    while len(seq) < n:
        seq.append(m)
        if m in repeats and len(seq) < n:
            seq.append(m)
        m += 1
    return np.array(seq, dtype=np.int64)

# Function to pre-calculate K_n based on the iteration steps
def compute_K(m_seq) :
    k = 1.0
    for m in m_seq:
        k *= np.sqrt(1.0 - 2.0 ** (-2*m))
    return k

def build_cordic_tables(n: int):
    # Precompute tables for a given number of iterations n.
    m_seq = hyperbolic_seq(n)
    theta_table = np.arctanh(2.0 ** (-m_seq.astype(np.float64)))
    K_n = compute_K(m_seq)
    return m_seq, theta_table, K_n

# Implement hyperbolic CORDIC rotation
def cordic(z: float, m_seq, theta_table, K_n, n: int) -> float:
    theta = 0.0
    x = 1.0 / K_n
    y = 0.0

    for m, arc_tanh in zip(m_seq[:n], theta_table[:n]):
        shift = 2.0 ** (-m)
        d = 1.0 if theta < z else -1.0
        theta += d * arc_tanh
        x_old = x
        x = x + d * y * shift
        y = x_old * d * shift + y

    return x + y
# CORDIC hyperbolic rotation only has a specific input range.
# This function applies the relevant trigonometric transform before executing the CORDIC algorithm.
def exp_cordic(z: float, m_seq, theta_table, K_n, n: int) -> float:
    ln2 = np.log(2.0)
    k = int(np.floor(z / ln2))
    r = z - k * ln2
    exp_r = cordic(r, m_seq, theta_table, K_n, n)
    return exp_r * (2.0 ** k)

# Generate the new array using the above functions
def cordic_array(z, n:int):
    z_arr = np.asarray(z, dtype=np.float64)

    if z_arr.ndim == 0:
        return exp_cordic(float(z_arr), n)
    
    out = np.empty_like(z_arr, dtype=np.float64)
    for idx, val in np.ndenumerate(z_arr):
        out[idx] = exp_cordic(float(val), n)
    return out

def cordic_array(z, n: int, m_seq, theta_table, K_n):
    ln2 = np.log(2.0)
    z_arr = np.asarray(z, dtype=np.float64)

    k = np.floor(z_arr / ln2).astype(int)
    r = z_arr - k * ln2

    theta = np.zeros_like(r)
    x = np.full_like(r, 1.0 / K_n)
    y = np.zeros_like(r)

    for m, arc_tanh in zip(m_seq[:n], theta_table[:n]):
        shift = 2.0 ** (-m)
        d = np.where(theta < r, 1.0, -1.0)
        theta += d * arc_tanh
        x_old = x.copy()
        x = x + d * y * shift
        y = x_old * d * shift + y

    return (x + y) * (2.0 ** k)

