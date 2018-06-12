import sys
from multiprocessing import Queue, Process

import cv2
import numpy as np
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import LoggerLevel
from pylibfreenect2 import createConsoleLogger, setGlobalLogger

from recorder.params import BOUNDARIES


class ShittyKinect():
    def __init__(self):

        self.q = Queue()
        self.p = Process(target=self.kinect_async, args=(self.q,))
        self.p.start()

    @staticmethod
    def kinect_async(q):
        # try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
        # except:
        #     try:
        #         from pylibfreenect2 import OpenGLPacketPipeline
        #         pipeline = OpenGLPacketPipeline()
        #     except:
        #         from pylibfreenect2 import CpuPacketPipeline
        #         pipeline = CpuPacketPipeline()
        print("Packet pipeline:", type(pipeline).__name__)

        # Create and set logger
        logger = createConsoleLogger(LoggerLevel.Warning)
        # setGlobalLogger(logger)
        setGlobalLogger(None)

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

    def getFrame(self):
        frame = self.q.get(True, 30) # only for first frame
        while not self.q.empty(): # if there are more frames waiting, empty the queue
            frame = self.q.get()
        return frame

    def close(self):
        self.p.terminate()

    def __del__(self):
        self.p.terminate()

# p.join()
