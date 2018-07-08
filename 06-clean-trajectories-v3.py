from s2rr.movements.dataset import Dataset
import numpy as np

VERSION = 2

DATASET_PATH = "data/recording{}.npz".format(VERSION)
DATASET_PATH_CLEAN = "data/recording{}_clean.npz".format(VERSION)

ds = Dataset()
ds.load(DATASET_PATH)

print("bad lines {} out of {}".format(np.count_nonzero(ds.bad),len(ds.bad)))

ds2 = Dataset()
ds2.moves = ds.moves[ds.bad == 0]
ds2.tips = ds.tips[ds.bad == 0]
ds2.jointvel = ds.jointvel[ds.bad == 0]
ds2.bad = ds.bad[ds.bad == 0]
ds2.save(DATASET_PATH_CLEAN)

