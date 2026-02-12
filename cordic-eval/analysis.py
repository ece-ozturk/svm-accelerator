# analysis.py
import numpy as np

def mean_squared(data1, data2) -> float:
    golden = np.asarray(data1, dtype=np.float64)
    cordic = np.asarray(data2, dtype=np.float64)

    if golden.shape != cordic.shape:
        raise ValueError("Inputs must have the same shape.")
    if cordic.size == 0:
        raise ValueError("Inputs must not be empty.")

    return np.mean((cordic - golden) ** 2)

    
def mse_eval():
    try:
        golden = np.loadtxt('golden_output.txt')
        cordic = np.loadtxt('cordic_output.txt')
        result = mean_squared(golden, cordic)
        print(f"MSE =", result)

    except OSError:
        print("Error: data not found. Run tests first!")

if __name__ == "__main__":
    mse_eval()

