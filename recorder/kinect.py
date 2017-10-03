# coding: utf-8

import numpy as np
from multiprocessing import Queue, Process
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import LoggerLevel
from pylibfreenect2 import createConsoleLogger, setGlobalLogger

from recorder.params import BOUNDARIES

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


class Kinect():
    def __init__(self):
        logger = createConsoleLogger(LoggerLevel.Debug)
        setGlobalLogger(logger)

        self.q = Queue()
        self.p = Process(target=self._main_loop, args=(self.q,))
        self.p.start()

    @staticmethod
    def _main_loop(q):
        fn = Freenect2()
        num_devices = fn.enumerateDevices()
        if num_devices == 0:
            raise Exception("Can't find connected device")

        serial = fn.getDeviceSerialNumber(0)
        device = fn.openDevice(serial, pipeline=pipeline)

        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)

        # Register listeners
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)

        device.start()

        registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())

        undistorted = Frame(512, 424, 4)
        registered = Frame(512, 424, 4)


        while True:
            print ("beep")
            frames = listener.waitForNewFrame()
            print ("boop")

            color = frames["color"]
            # ir = frames["ir"]
            depth = frames["depth"]

            registration.apply(color, depth, undistorted, registered,
                                    bigdepth=None,
                                    color_depth_map=None)

            reg_img = registered.asarray(np.uint8)[
                      BOUNDARIES["top"]:BOUNDARIES["bottom"],
                      BOUNDARIES["left"]:BOUNDARIES["right"],
                      :]  # all 4 RGBD channels

            q.put(reg_img)

            # cv2.imshow("frame", reg_img)
            listener.release(frames)



    def getFrame(self):
        frame = None
        while not self.q.empty():
            frame = self.q.get()

        return frame


    # def __del__(self):
    #     self.device.stop()
    #     self.device.close()
