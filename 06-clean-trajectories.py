from movements.dataset import Dataset
from movements.visualizer import Visualizer
import numpy as np


DATASET_PATH = "data/recording1.npz"
DATASET_PATH_CLEAN = "data/recording1_clean.npz"

ds = Dataset()
ds.load(DATASET_PATH)

ds_shape = ds.tips.shape

out_moves = []
out_tips = []

def check_height(pos):
    if pos[2] < 5.08071445e-02:
        return False
    return True

def check_box(pos):
    x, y, z = pos

    # left / right
    if x >= -0.09069847 and x <= 0.07237665:

        # front / back
        if y>= 0.04738161 and y <= 0.14891243:

            # height
            if z <= 0.0962072:
                return False

    return True


for episode_idx in range(ds_shape[0]):
    positions = ds.tips[episode_idx].reshape(-1, 3)
    clean = True
    for pos in positions:
        clean = check_height(pos)
        if clean:
            clean = check_box(pos)
    if clean:
        out_tips.append(ds.tips[episode_idx])
        out_moves.append(ds.moves[episode_idx])

ds_clean = Dataset()
ds_clean.moves = np.array(out_moves)
ds_clean.tips = np.array(out_tips)

ds_clean.save(DATASET_PATH_CLEAN)
