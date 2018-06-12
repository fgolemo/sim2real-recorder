import time

from s2rr.recorder.shittykinect import ShittyKinect

kinect = ShittyKinect()

frame = kinect.getFrame()

print (frame.shape)
time.sleep(1)

frame2 = kinect.getFrame()

print (frame2.shape)

while True:
    frame = kinect.getFrame()
    print ("boop")
