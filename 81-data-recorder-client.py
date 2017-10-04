import numpy as np
import shutil
import zmq
from tqdm import tqdm
import os

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
# welcome = socket.recv_string()
# print (welcome)
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

kinect = ShittyKinect()
frame = kinect.getFrame()  # warmup
print ("got first frame")
print (frame.shape)

ds_shape = ds.moves.shape


def move_robot(action):
    out = []
    for row in range(len(action)):
        out.append(",".join([str(s) for s in action[row].tolist()]))
    output = "|".join(out)
    socket.send_string(output)


def save_stuff(data_buf_kinect, data_buf_robo, episode_idx):
    # print ("got robo frames and kinect frames:")
    # print ("robo:", robo_frames.shape)
    # print ("kinect:", frames.shape)

    data_kinect = np.array(data_buf_kinect)
    data_robo = np.array(data_buf_robo)

    np.savez_compressed("data/data_dump_tmp.npz", kinect=data_kinect, robo=data_robo)
    shutil.move("data/data_dump_tmp.npz", "data/data_dump.npz")

    # hf = h5py.File('data/test-recording.h5', 'w')
    #
    # hf.create_dataset('kinect', data=data_kin)
    # hf.create_dataset('robo', data=data_robo)
    #
    # hf.close()
    print ("SAVED. Episode {}".format(episode_idx))


data_buffer_kinect = []
data_buffer_robo = []

for episode_idx in tqdm(range(len(ds.moves))):
    actions = np.around(ds.moves[episode_idx, :, 0, :], 2)
    frames = []
    move_robot(actions)
    while True:
        frame = kinect.getFrame()
        frames.append(frame)
        socks = dict(poller.poll(1000 * ROBO_FPD_DELAY))
        if socks:
            if socks.get(socket) == zmq.POLLIN:
                robo_frames = zmq_recv_array(socket)
                frames = np.array(frames)
                data_buffer_kinect.append(frames)
                data_buffer_robo.append(robo_frames)
                break
    if episode_idx % 10 == 0:
        save_stuff(data_buffer_kinect, data_buffer_robo, episode_idx)

kinect.close()
