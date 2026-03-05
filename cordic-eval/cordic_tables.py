import numpy as np

def hyperbolic_seq(n: int):
    repeats = set()
    k = 4
    while k <= 10000:
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

def compute_K(m_seq):
    k = 1.0
    for m in m_seq:
        k *= np.sqrt(1.0 - 2.0 ** (-2*m))
    return k

def build_cordic_tables(n: int):
    m_seq = hyperbolic_seq(n)
    theta_table = np.arctanh(2.0 ** (-m_seq.astype(np.float64)))
    K_n = compute_K(m_seq)
    return m_seq, theta_table, K_n


N = 18
m_seq, theta_table, K_n = build_cordic_tables(N)

print(f"CORDIC Hyperbolic Tables (n={N})")
print(f"{'i':>4}  {'m_seq':>6}  {'2^-m':>12}  {'arctanh(2^-m)':>18}  {'K partial':>14}")
print("-" * 62)

k_partial = 1.0
for i, (m, arc_tanh) in enumerate(zip(m_seq, theta_table)):
    k_partial *= np.sqrt(1.0 - 2.0 ** (-2*m))
    repeated = " *" if i > 0 and m == m_seq[i-1] else ""
    print(f"{i:>4}  {m:>6}  {2**-float(m):>12.8f}  {arc_tanh:>18.20f}  {k_partial:>14.10f}{repeated}")

print(f"\nFinal K_n = {K_n:.20f}  (1/K_n = {1/K_n:.20f})")
print(f"(* marks repeated terms required for CORDIC convergence)")
