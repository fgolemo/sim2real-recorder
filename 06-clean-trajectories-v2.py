from movements.dataset import Dataset
import numpy as np


DATASET_PATH = "data/recording2.npz"
DATASET_PATH_CLEAN = "data/recording2_clean.npz"

ds = Dataset()
ds.load(DATASET_PATH)

print(np.count_nonzero(ds.bad))

ds2 = Dataset()
ds2.moves = ds.moves[ds.bad == 0]
ds2.tips = ds.tips[ds.bad == 0]
ds2.jointvel = ds.jointvel[ds.bad == 0]
ds2.bad = ds.bad[ds.bad == 0]
ds2.save(DATASET_PATH_CLEAN)

