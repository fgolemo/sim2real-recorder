import time

import numpy as np
import zmq

from config.constants import *
from exchange.utilities import zmq_recv_array
from movements.dataset import Dataset
from recorder.shittykinect import ShittyKinect

DATASET_PATH_CLEAN = "data/recording1_clean.npz"
# EPISODE = 0

# MOTOR 0 & 3 ARE INVERTED

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://flogo3.local:%s" % port)
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

kinect = ShittyKinect()
frame = kinect.getFrame()  # warmup
print ("got first frame")
print (frame.shape)

ds_shape = ds.moves.shape

def move_robot(action):
    print ("got actions:")
    print (action.shape)
    print (action)

for episode_idx in range(len(ds.moves)):
    actions = np.around(ds.moves[episode_idx, :, 0, :], 2)
    frames = []
    move_robot(actions)
    while True:
        frame = kinect.getFrame()
        frames.append(frame)
        socks = dict(poller.poll(1000*ROBO_FPD_DELAY))
        if socks:
            if socks.get(socket) == zmq.POLLIN:
                robo_frames = zmq_recv_array(socket)
                frames = np.array(frames)
                print ("got robo frames and kinect frames:")
                print ("robo:",robo_frames.shape)
                print ("kinect:",frames.shape)
                quit()


kinect.close()
