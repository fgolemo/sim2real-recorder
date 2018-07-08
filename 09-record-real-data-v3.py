import time

from tqdm import tqdm

from s2rr.movements.dataset import Dataset
import numpy as np
from poppy_helpers.controller import ZMQController

VERSION = 2

DATASET_PATH_CLEAN = "data/recording{}_done.npz".format(VERSION)

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

robot = ZMQController("flogo2.local")
robot.compliant(False)
robot.set_max_speed(100)
robot.rest()

# start = time.time()
# timings = []
#
# for i in range(1000):
#     posvel = robot.get_posvel()
#     robot.rest()
#     timings.append(time.time() - start)
#     start = time.time()
#
# print (np.mean(timings))
# print(1/np.mean(timings))

FPS = 100
min_runtime = 1 / FPS

for episode in tqdm(range(ds.moves.shape[0])):
    if np.count_nonzero(ds.jointvel[episode,-1,-1,:]) > 0:
        continue

    robot.rest()
    time.sleep(1)


    for action in range(ds.moves.shape[1]):
        start = time.time()
        for frame in range(ds.moves.shape[2]):
            act = [float(x) * 90 for x in ds.moves[episode, action, frame]]
            robot.goto_pos(act)
            ds.jointvel[episode, action, frame, :] = robot.get_posvel()

            delta = time.time() - start
            if delta < min_runtime:
                time.sleep(min_runtime - delta)
            start = time.time()

    if (episode + 1) % 10 == 0:
        ds.save("data/recording2_done.npz")

robot.rest()
time.sleep(1)
ds.save("data/recording2_done.npz")
