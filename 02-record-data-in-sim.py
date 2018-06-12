from movements.constants import REST_POS
from s2rr.movements.dataset import Dataset
from recorder.experiment import Experiment
from tqdm import tqdm
import numpy as np

DATASET_PATH = "data/recording1.npz"

exp = Experiment()

exp.startEnv(headless=True)

ds = Dataset()
ds.load(DATASET_PATH)

moves_shape = ds.moves.shape

for episode_idx in tqdm(range(moves_shape[0])):
    exp.restPos()

    if np.count_nonzero(ds.tips[episode_idx, moves_shape[1]-1, moves_shape[2]-1]) > 0:
        continue

    # actions = ds.moves[episode_idx].reshape(-1,6)

    for action_idx in range(moves_shape[1]):
        for frame_idx in range(moves_shape[2]):
            obs = exp.step(ds.moves[episode_idx, action_idx, frame_idx, :])

            # tip_pos = obs[-3:]
            ds.tips[episode_idx, action_idx, frame_idx, :] = obs[-3:]

    if episode_idx % 10 == 0:
        ds.save(DATASET_PATH)

exp.close()