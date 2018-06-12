import numpy as np
from tqdm import tqdm
import time

from s2rr.config.constants import *
from s2rr.movements.dataset import Dataset
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
import sys

DATASET_PATH_CLEAN = "data/recording1_clean.npz"
PROGRESS_FILE = "data/recording_progress"
# EPISODE = 0

# MOTOR 0 & 3 ARE INVERTED

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

from pylibfreenect2 import OpenCLPacketPipeline
pipeline = OpenCLPacketPipeline()
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

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


frame_container = []

print ("got first frame")
# print (frame_container.shape)

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
        frames = listener.waitForNewFrame()

        color = frames["color"]
        ir = frames["ir"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered,
                           bigdepth=None,
                           color_depth_map=None)

        frame = registered.asarray(np.uint8)[
                  BOUNDARIES["top"]:BOUNDARIES["bottom"],
                  BOUNDARIES["left"]:BOUNDARIES["right"],
                  :]

        frame_container.append(frame)
        frames_time.append(time.time()*TIME_MULTI)
        elapsed = time.time()-time_start
        if elapsed > 5:
            fps = float(len(frame_container))/elapsed
            print ("FPS:",fps)
            frame_container = []
            frames_time = []
            time_start = time.time()
        listener.release(frames)

device.stop()
device.close()
