from tqdm import tqdm

from s2rr.movements.dataset import Dataset
import numpy as np

# VERSION = 3
# # version 3, min 1.0, max 4799.0

# VERSION = 4
# # version 4, min 1.0, max 13022.0

VERSION = 2
# version 2, min 1.0, max 3519.0

DATASET_PATH_CSV = "data/recording{}_clean.csv".format(VERSION)

csv = np.loadtxt(DATASET_PATH_CSV)

print("version {}, min {}, max {}".format(
    VERSION,
    csv[:, 3].min(),
    csv[:, 3].max()
))
