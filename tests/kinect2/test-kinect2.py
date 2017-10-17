# coding: utf-8

import numpy as np
import cv2
import sys

import time
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
from tqdm import tqdm


BOUNDARIES = { # img is right-left flipped
    "top": 90,
    "left": 100,
    "right": 350,
    "bottom": 330
}



try:
    from pylibfreenect2 import OpenCLPacketPipeline
    pipeline = OpenCLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

frame_container = []
time_start = time.time()
while True:
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
              :] # all 4 RGBD channels

    frame_container.append(reg_img)

    # # for finding the boundaries... write an image for each extreme position
    # if i==2:
    #     cv2.imwrite("bottom.png", registered.asarray(np.uint8))
    #     break

    cv2.imshow("frame", reg_img)

    elapsed = time.time() - time_start
    if elapsed > 10:
        fps = float(len(frame_container)) / elapsed
        print ("FPS:", fps)
        frame_container = []
        time_start = time.time()

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break


device.stop()
device.close()

sys.exit(0)