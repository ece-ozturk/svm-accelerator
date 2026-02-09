#test_gen.py

import numpy as np

min = -8
max = 0.1
step = 0.5

data = np.arange(min, max, step)

np.savetxt('test_data.txt', data, fmt='%.1f')

print("Test data generated with step size of 0.5")