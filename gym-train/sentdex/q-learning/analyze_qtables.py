import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import os
from copy import copy
import glob
import re

dirs = glob.glob(r"*qtables_*")
dirs = [path for path in dirs if re.search(r'qtables_\d+$', path)]

analyze_dir = '16'


def analyze(qtable_path):
    q_table = np.load(qtable_path)
    count = 0
    for _x in q_table:
        # print(_x)
        for _y in _x:
            temp = copy(_y)
            _y[:] = _y == _y.max()
            if sum(_y) >= 2:
                # print(f"File: {qtable_path:>40}, \tVals: {str(temp):^35}")
                count += 1
    if count > 10:
        print(f"File: {qtable_path:>30}, count: {count}")
    return count


for curr_dir in dirs:
    print(f"= = "*10)
    print(f"Directory: {curr_dir}")
    if not analyze_dir in curr_dir:
        continue
    tables = glob.glob(curr_dir + '/*qtable.npy', recursive=True)
    count = 0
    for file_path in tables:
        # if not "48300" in file_path:
        #     continue
        add = analyze(file_path)
        if add:
            # print(add)
            count += add

    print(f"Count: {count:>10d}")

