#exp_eval.py

import numpy as np

def exp_eval():
    try:
        data = np.loadtxt('test_data.txt')

        result = np.exp(data)

        np.savetxt('golden_output.txt', result, fmt='%.16f')
    
    except OSError:
        print("Error: 'test_data.txt' not found. Run test_gen.py first!")

if __name__ == "__main__":
    exp_eval()