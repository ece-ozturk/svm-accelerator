# cordiv_eval.py
# Modified from CORDIC code on wikipedia to work for hyperbolic e calculation

import numpy as np

iters = 16
theta_table = [np.arctanh(2.0**(-i)) for i in range(1, iters+1)]

def compute_K(n) :
    k = 1.0
    for i in range(n):
        k *= 1 / np.sqrt(1 - 2 ** (-2 * i))
    return k
# Note computation of k differs, uses cosh for hyperbolic

def cordic(z: float, n: int):
    K_n = compute_K(n)
    theta = 0.0
    x = 1 / K_n
    y = 0
    P2i = 1
    for arc_tangent in theta_table[:n]:
        sigma = +1 if theta < z else -1
        theta += sigma * arc_tangent
        x, y = x + sigma * y * P2i, sigma * P2i * x + y
        P2i /=2
    return x * K_n + y * K_n

def cordic_eval(n):
    try:
        data = np.loadtxt('test_data.txt')

        result = cordic(data, n)

        np.savetxt('cordic_output.txt', result, fmt='%.16f')
    
    except OSError:
        print("Error: 'test_data.txt' not found. Run test_gen.py first!")

if __name__ == "__main__":
    cordic_eval(iters)

