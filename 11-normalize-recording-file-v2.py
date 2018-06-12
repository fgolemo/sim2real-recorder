import time

from tqdm import tqdm

from s2rr.movements.dataset import Dataset
import numpy as np
from poppy_helpers.controller import ZMQController

DATASET_PATH_CLEAN = "data/recording2_done.npz"
DATASET_PATH_NORM = "data/recording2_done_norm.npz"

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

print("max pos", ds.jointvel[:, :, :, :6].max())
print("min pos", ds.jointvel[:, :, :, :6].min())

print("max vel", ds.jointvel[:, :, :, 6:].max())
print("min vel", ds.jointvel[:, :, :, 6:].min())

vel_min = ds.jointvel[:, :, :, 6:].min()
vel_range = ds.jointvel[:, :, :, 6:].max() - vel_min

ds.jointvel[:, :, :, :6] = ((ds.jointvel[:, :, :, :6] + 90) / 180) * 2 - 1
ds.jointvel[:, :, :, 6:] = ((ds.jointvel[:, :, :, 6:] - vel_min) / vel_range) * 2 - 1

print("== norm magic")
print("max pos", ds.jointvel[:, :, :, :6].max())
print("min pos", ds.jointvel[:, :, :, :6].min())

print("max vel", ds.jointvel[:, :, :, 6:].max())
print("min vel", ds.jointvel[:, :, :, 6:].min())


ds.save(DATASET_PATH_NORM)



