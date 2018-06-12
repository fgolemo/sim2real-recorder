import time

from tqdm import tqdm

from movements.dataset import Dataset
import numpy as np
from poppy_helpers.controller import ZMQController

DATASET_PATH_CLEAN = "data/recording2_done.npz"

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

for i in range(1):
    for j in range(3):
        for f in range(100):
            print (ds.jointvel[559+i,j,f,:6])
            print(ds.moves[559 + i, j, f, :6]*90)
