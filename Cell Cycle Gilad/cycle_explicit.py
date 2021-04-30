import numpy as np
import bats
from time import time

pre_data = np.load("trimmed_data.npy")

data = bats.DataSet(bats.Matrix(pre_data))
dist = bats.Euclidean()

start = time()
rf = bats.RipsFiltration(data, dist, np.inf, 2)
print(time() - start)

start = time()
rc = bats.reduce(rf, bats.F2())
print(time() - start)
