# coding: utf-8

import sys
from multiprocessing import Queue, Process

import cv2
import numpy as np
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import LoggerLevel
from pylibfreenect2 import createConsoleLogger, setGlobalLogger

BOUNDARIES = {  # img is right-left flipped
    "top": 90,
    "left": 100,
    "right": 350,
    "bottom": 330
}


def async(q):
    print ("async started")
    try:
        print ("A")
        from pylibfreenect2 import OpenCLPacketPipeline
        print ("B")
        pipeline = OpenCLPacketPipeline()
        print ("C")
    except:
        print ("D")
        try:
            print ("E")
            from pylibfreenect2 import OpenGLPacketPipeline
            print ("F")
            pipeline = OpenGLPacketPipeline()
            print ("G")
        except:
            print ("H")
            from pylibfreenect2 import CpuPacketPipeline
            print ("I")
            pipeline = CpuPacketPipeline()
            print ("J")
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
                  :]  # all 4 RGBD channels

        # # for finding the boundaries... write an image for each extreme position
        # if i==2:
        #     cv2.imwrite("bottom.png", registered.asarray(np.uint8))
        #     break

        q.put(reg_img)

        listener.release(frames)

    device.stop()
    device.close()


q = Queue()
p = Process(target=async, args=(q,))
print ("starting process")
p.start()
print ("process launched")

while True:
    if not q.empty():
        print ("queue is non-empty")
        reg_img = q.get()
        cv2.imshow("frame", reg_img)
        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break

print ("exiting - waiting for subprocess to finish")
p.join()

sys.exit(0)



#todo rewrite so that kinect is the main thread and other thread runs async