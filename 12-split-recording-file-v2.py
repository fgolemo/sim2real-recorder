import time

from tqdm import tqdm

from s2rr.movements.dataset import Dataset, DatasetProduction
import numpy as np
from poppy_helpers.controller import ZMQController

DATASET_PATH_NORM = "data/recording2_done_norm.npz"

ds = Dataset()
ds.load(DATASET_PATH_NORM)

for i in range(1):
    for j in range(3):
        for f in range(100):
            print(f)
            print(ds.jointvel[i, j, f, :6])
            print(ds.moves[i, j, f, :6])

jointvel = ds.jointvel.reshape(
    (ds.jointvel.shape[0], ds.jointvel.shape[1] * ds.jointvel.shape[2], ds.jointvel.shape[3]))
actions = ds.moves.reshape((ds.moves.shape[0], ds.moves.shape[1] * ds.moves.shape[2], ds.moves.shape[3]))
tips = ds.tips.reshape((ds.tips.shape[0], ds.tips.shape[1] * ds.tips.shape[2], ds.tips.shape[3]))

print(jointvel.shape)
print(actions.shape)
print(tips.shape)

for i in range(1):
    for j in range(300):
        print(j)
        print(jointvel[i, j, :6])
        print(actions[i, j, :6])

current_real = np.zeros((560, 299, 12), dtype=np.float32)
next_real = np.zeros((560, 299, 12), dtype=np.float32)
diff_real = np.zeros((560, 299, 12), dtype=np.float32)
action = np.zeros((560, 299, 6), dtype=np.float32)
tip = np.zeros((560, 299, 3), dtype=np.float32)
next_sim = np.zeros((560, 299, 12), dtype=np.float32)

for epi in range(560):
    for frame in range(299):
        current_real[epi, frame, :] = jointvel[epi, frame, :]
        next_real[epi, frame, :] = jointvel[epi, frame + 1, :]
        action[epi, frame, :] = actions[epi, frame, :]
        diff_real[epi, frame, :] = jointvel[epi, frame + 1, :] - jointvel[epi, frame, :]
        tip[epi,frame,:] = tips[epi,frame,:]

ds_train = DatasetProduction()

ds_train.current_real = current_real[:500]
ds_train.next_real = next_real[:500]
ds_train.diff_real = diff_real[:500]
ds_train.next_sim = next_sim[:500]
ds_train.tip = tip[:500]
ds_train.action = action[:500]

ds_train.save("~/data/sim2real/data-realigned-v2-train.npz")

ds_test = DatasetProduction()

ds_test.current_real = current_real[500:]
ds_test.next_real = next_real[500:]
ds_test.diff_real = diff_real[500:]
ds_test.next_sim = next_sim[500:]
ds_test.tip = tip[500:]
ds_test.action = action[500:]

ds_test.save("~/data/sim2real/data-realigned-v2-test.npz")

