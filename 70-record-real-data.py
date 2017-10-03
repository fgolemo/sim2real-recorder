import time

from movements.dataset import Dataset
from recorder.kinect import Kinect
from recorder.robot import Robot

import numpy as np

from recorder.shittykinect import ShittyKinect

DATASET_PATH_CLEAN = "data/recording1_clean.npz"
EPISODE = 0

# MOTOR 0 & 3 ARE INVERTED

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

robot = Robot()


kinect = ShittyKinect()
frame = kinect.getFrame() # warmup
print ("got first frame")
print (frame.shape)

ds_shape = ds.moves.shape

SECONDS_OF_RECORDING = 2.0

for episode_idx in range(len(ds.moves)):
    for action_idx in range(ds_shape[1]):
        frames = []
        pos = np.around(ds.moves[episode_idx, action_idx, 0, :], 2)
        print (pos)
        robot.move(pos)
        time_start = time.time()
        while True:
            frame = kinect.getFrame()
            robot_data = robot.get_data()
            frames.append(frame)
            print (robot_data)
            time_current = time.time() - time_start
            if time_current >= SECONDS_OF_RECORDING:
                break
        print("{} frames / {} fps".format(len(frames), round(len(frames) / SECONDS_OF_RECORDING, 2)))

    break

kinect.close()
robot.close()
