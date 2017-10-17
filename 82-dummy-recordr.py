import os
import shutil

import numpy as np
import paramiko
import zmq
from tqdm import tqdm
import time

from config.constants import *
from exchange.utilities import zmq_recv_array
from movements.dataset import Dataset
from recorder.shittykinect import ShittyKinect
from recorder.utilities import progress_write, progress_read

DATASET_PATH_CLEAN = "data/recording1_clean.npz"
PROGRESS_FILE = "data/recording_progress"
# EPISODE = 0

# MOTOR 0 & 3 ARE INVERTED

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)


kinect = ShittyKinect()
frame = kinect.getFrame()  # warmup
print ("got first frame")
print (frame.shape)

ds_shape = ds.moves.shape



progress = 0

data_buffer_kinect = []
data_buffer_kinect_time = []

save_episode_count = int(progress / WRITE_EVERY_N_EPISODES)

for episode_idx in tqdm(range(len(ds.moves))):
    if episode_idx < progress:
        continue

    actions = np.around(ds.moves[episode_idx, :, 0, :], 2)
    frames = []
    frames_time = []

    time_start = time.time()
    while True:

        frame = kinect.getFrame()
        frames.append(frame)
        frames_time.append(time.time()*TIME_MULTI)
        elapsed = time.time()-time_start
        if elapsed > 10:
            fps = float(len(frames))/elapsed
            print ("FPS:",fps)
            frames = []
            time_start = time.time()

kinect.close()
