import os
import shutil

import cv2
import numpy as np
import paramiko
import zmq
from tqdm import tqdm
import time

from s2rr.config.constants import *
from s2rr.exchange import zmq_recv_array
from s2rr.movements.dataset import Dataset
from recorder.utilities import progress_write, progress_read
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
import sys

from threading import Thread

DATASET_PATH_CLEAN = "data/recording1_clean.npz"
PROGRESS_FILE = "data/recording_progress"
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

from pylibfreenect2 import OpenCLPacketPipeline

pipeline = OpenCLPacketPipeline()
logger = createConsoleLogger(LoggerLevel.Debug)
# setGlobalLogger(logger)
setGlobalLogger(None)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

frames = listener.waitForNewFrame()

color = frames["color"]
ir = frames["ir"]
depth = frames["depth"]

registration.apply(color, depth, undistorted, registered,
                   bigdepth=None,
                   color_depth_map=None)

reg_img = registered.asarray(np.uint8)[
          BOUNDARIES["top"]:BOUNDARIES["bottom"],
          BOUNDARIES["left"]:BOUNDARIES["right"],
          :]
listener.release(frames)

print ("got first frame")
print(reg_img.shape)

ds_shape = ds.moves.shape


def move_robot(action):
    out = []
    for row in range(len(action)):
        out.append(",".join([str(s) for s in action[row].tolist()]))
    output = "|".join(out)
    socket.send_string(output)


def save_stuff(data_buf_kinect, data_buf_kinect_time, data_buf_robo, data_buf_robo_time, data_buf_robo_speed,
               episode_idx, save_episode):
    data_kinect = np.array(data_buf_kinect)
    data_kinect_time = np.array(data_buf_kinect_time)
    data_robo = np.array(data_buf_robo)
    data_robo_time = np.array(data_buf_robo_time, dtype=np.uint64)
    data_robo_speed = np.array(data_buf_robo_speed)

    print ("DEBUG: saving snapshot...")

    np.savez_compressed("data/data_dump_tmp.npz",
                        kinect=data_kinect,
                        kinect_time=data_kinect_time,
                        robo=data_robo,
                        robo_time=data_robo_time,
                        robo_speed=data_robo_speed
                        )

    output_filename = "data_dump_{}.npz".format(save_episode)
    output_path = "data/" + output_filename

    shutil.move("data/data_dump_tmp.npz", output_path)

    print ("DEBUG: saving snapshot done")

    if (USE_BACKUP):
        print ("SSH: uploading file...")
        ssh = paramiko.SSHClient()
        ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(BACKUP_HOST, username=BACKUP_USER, password=BACKUP_PASS)
        sftp = ssh.open_sftp()

        sftp.put(output_path, BACKUP_PATH + output_filename)

        sftp.close()
        ssh.close()
        print ("SSH: file transfered")
        os.remove(output_path)

    progress_write(PROGRESS_FILE, episode_idx)

    print ("SAVED. Episode {}".format(episode_idx))


progress = progress_read(PROGRESS_FILE)
print ("LOADED PROGRESS:", progress)

data_buffer_kinect = []
data_buffer_kinect_time = []
data_buffer_robo = []
data_buffer_robo_time = []
data_buffer_robo_speed = []

save_episode_count = int(progress / WRITE_EVERY_N_EPISODES)

t = None # this will hold the saving thread in the future

for episode_idx in tqdm(range(len(ds.moves))):
    if episode_idx < progress:
        continue

    actions = np.around(ds.moves[episode_idx, :, 0, :], 2)
    frames = []
    frames_time = []
    move_robot(actions)

    time_start_global = time.time()
    time_start_local = time.time()
    while True:
        raw_frames = listener.waitForNewFrame()

        color = raw_frames["color"]
        depth = raw_frames["depth"]

        registration.apply(color, depth, undistorted, registered,
                           bigdepth=None,
                           color_depth_map=None)

        frame = registered.asarray(np.uint8)[
                BOUNDARIES["top"]:BOUNDARIES["bottom"],
                BOUNDARIES["left"]:BOUNDARIES["right"],
                :]
        frames.append(np.copy(frame))

        #OPTIONAL: display FPS directly in img:
        # cv2.putText(image, "Hello World!!!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        cv2.imshow("kinect", frame)
        cv2.waitKey(1)
        frames_time.append(time.time() * TIME_MULTI)

        elapsed_global = time.time() - time_start_global
        elapsed_local = time.time() - time_start_local
        if elapsed_local > 10:
            fps = float(len(frames)) / elapsed_global
            print ("FPS:", fps)
            time_start_local = time.time()

        listener.release(raw_frames)

        socks = dict(poller.poll(1000 * ROBO_FPD_DELAY))
        if socks:
            if socks.get(socket) == zmq.POLLIN:
                robo_frames = zmq_recv_array(socket)
                robo_frames_time = zmq_recv_array(socket)
                robo_frames_speed = zmq_recv_array(socket)

                frames = np.array(frames)
                frames_time = np.array(frames_time, dtype=np.uint64)

                data_buffer_kinect.append(frames)
                data_buffer_kinect_time.append(frames_time)
                data_buffer_robo.append(robo_frames)
                data_buffer_robo_time.append(robo_frames_time)
                data_buffer_robo_speed.append(robo_frames_speed)
                break
    if len(data_buffer_kinect) == WRITE_EVERY_N_EPISODES:
        if t is not None:
            print ("DEBUG: Old thread found, waiting for thread to finish...")
            t.join()
        t = Thread(target=save_stuff, args=(
            data_buffer_kinect,
            data_buffer_kinect_time,
            data_buffer_robo,
            data_buffer_robo_time,
            data_buffer_robo_speed,
            episode_idx,
            save_episode_count
        ))
        t.start()
        save_episode_count += 1
        data_buffer_kinect = []
        data_buffer_kinect_time = []
        data_buffer_robo = []
        data_buffer_robo_time = []
        data_buffer_robo_speed = []

device.stop()
device.close()
