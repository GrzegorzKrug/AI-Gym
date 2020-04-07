import numpy as np
import matplotlib.pyplot as plt
import gym
import glob

files = glob.glob(f"qtables_16/*qtable.npy", recursive=True)
# files.sort()

numbers = np.linspace(0, 49990, 5)

for num in numbers:
    print(round(num, -1))

