import numpy as np
from tqdm import tqdm
import os

from config.constants import WRITE_EVERY_N_EPISODES

import matplotlib.pyplot as plt

from movements.dataset import Dataset

FILE_NAME = "/lindata/datasets/sim2real/data_dump_{}_aligned.npz"
MOVES = "data/recording1_clean.npz"
OUT = "/lindata/datasets/sim2real-realigned/data-realigned.npz"

ds = Dataset()
ds.load(MOVES)


def get_right_offset(file):
    if file <= 16:
        return 0
    if file >= 17 and file <= 21:
        return 9
    if file >= 22 and file <= 23:
        return 8
    if file >= 24 and file <= 31:
        return 7
    if file == 32:
        return 6
    if file >= 33 and file <= 34:
        return 5
    if file >= 35:
        return 4


ds_curr_real = []
ds_next_real = []
ds_action = []
ds_epi = []

episode = 0

for FILE_IDX in tqdm(range(148)):
    # FILE_IDX = 35
    data = np.load(FILE_NAME.format(FILE_IDX), encoding="bytes")
    real_positions, real_speeds = data["real_positions"], data["real_speeds"]

    diffs = []
    for RECORDING in range(10):

        OFFSET = get_right_offset(FILE_IDX)

        current_episode = FILE_IDX * WRITE_EVERY_N_EPISODES + RECORDING + OFFSET

        cmds = ds.getUniqueActionsForEpisode(current_episode)
        cmds = cmds[:, 0, :]  # shape: (3, 6)

        episode_pos = real_positions[RECORDING]
        episode_spd = real_speeds[RECORDING]

        old_speed = episode_spd[0]
        current_action_idx = 0

        for step in range(len(episode_spd) - 1):
            if episode_spd[step + 1] != old_speed:
                # new action
                old_speed = episode_spd[step + 1]
                current_action_idx += 1

            curr_real = np.hstack([episode_pos[step, :, 0], episode_pos[step, :, 1]])
            # print("curr:", curr_real.round(0))
            next_real = np.hstack([episode_pos[step + 1, :, 0], episode_pos[step + 1, :, 1]])
            # print("next:", next_real.round(0))
            cmd = cmds[current_action_idx]
            # print("cmd_:", np.around(cmd, 0))
            epi = FILE_IDX * 10 + RECORDING
            # print("epi_:", epi)

            ds_curr_real.append(curr_real)
            ds_next_real.append(next_real)
            ds_action.append(cmd)
            ds_epi.append(epi)

            # if epi == 10:
            #     quit()

ds_curr_real = np.array(ds_curr_real)
ds_next_real = np.array(ds_next_real)
ds_action = np.array(ds_action)
ds_epi = np.array(ds_epi)

print(ds_curr_real.shape)
print(ds_next_real.shape)
print(ds_action.shape)
print(ds_epi.shape)

np.savez(OUT,
         ds_curr_real=ds_curr_real,
         ds_next_real=ds_next_real,
         ds_action=ds_action,
         ds_epi=ds_epi
         )
