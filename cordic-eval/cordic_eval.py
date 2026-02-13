# cordiv_eval.py
import numpy as np

iters = 17 # Number of iterations. For this implementation, this is total number of pseudo-rotations

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

# Define sequence for this run of calculation            
m_seq = hyperbolic_seq(iters)
print(m_seq)

# Calculate table of theta values for each iteration step - accounts for repeats
theta_table = np.arctanh(2.0 ** (-m_seq.astype(np.float64)))

# Save theta table for inspection
np.savetxt('theta_table.txt', theta_table, fmt='%.16f')

# Function to pre-calculate K_n based on the iteration steps
def compute_K(m_seq) :
    k = 1.0
    for m in m_seq:
        k *= np.sqrt(1.0 - 2.0 ** (-2*m))
    return k

# Compute K_n for this run
K_n = compute_K(m_seq)
print(f"K_{iters} =", K_n)

# Implement hyperbolic CORDIC rotation
def cordic(z: float, n: int) -> float:
    # Angle to determine sign of d for each iteration
    theta = 0.0

    # Assign x and y variables such that exp can be calculated from their sum
    x = 1.0/K_n
    y = 0.0

    # CORDIC shift
    for m, arc_tanh in zip(m_seq[:n], theta_table[:n]):
        shift = 2.0 ** (-m)
        d = 1.0 if theta < z else -1.0
        theta += d * arc_tanh
        x_old = x
        x = x + d * y * shift
        y = x_old * d * shift + y
    
    # Return the calculated exp value, with K_n accounted for
    return x + y

# CORDIC hyperbolic rotation only has a specific input range.
# This function applies the relevant trigonometric transform before executing the CORDIC algorithm.
def exp_cordic(z: float, n: int) -> float:
    ln2 = np.log(2.0)
    
    # Transformation of input z values
    k = int(np.floor(z / ln2))
    r = z - k * ln2

    # Perform CORDIC on transformed values
    exp_r = cordic(r, n)

    # Revert transformation of values
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

# Create and handle data
def cordic_eval(n: int):
    try:
        data = np.loadtxt('test_data.txt')
        
        result = cordic_array(data, n)

        np.savetxt('cordic_output.txt', result, fmt='%.16f')
    
    except OSError:
        print("Error: 'test_data.txt' not found. Run test_gen.py first!")

if __name__ == "__main__":
    cordic_eval(iters)

