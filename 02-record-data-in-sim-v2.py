import time

from gym_ergojr.sim.single_robot import SingleRobot

from s2rr.movements.dataset import Dataset
from tqdm import tqdm
import numpy as np

DATASET_PATH = "data/recording2.npz"

robot = SingleRobot(debug=True)

ds = Dataset()
ds.load(DATASET_PATH)

moves_shape = ds.moves.shape

for episode_idx in tqdm(range(moves_shape[0])):
    robot.reset()  # dunno why, but needs to be reset twice
    robot.step()
    robot.reset()
    robot.step()

    for action_idx in range(moves_shape[1]):
        for frame_idx in range(moves_shape[2]):
            robot.act2(ds.moves[episode_idx, action_idx, frame_idx, :])
            robot.step()
            if len(robot.get_hits(robot1=None, robot2=None)) > 1:
                ds.bad[episode_idx] = 1
            ds.tips[episode_idx, action_idx, frame_idx, :] = robot.get_tip()[0]
            # time.sleep(0.01)

    if episode_idx % 10 == 0:
        ds.save(DATASET_PATH)

ds.save(DATASET_PATH)
