from PIL import Image

import numpy as np
import cv2

# FILE_NAME= "data/data_dump_5.npz" # avg 80
# FILE_NAME= "data/data_dump_95.npz" # avg 17
# FILE_NAME= "data/data_dump_137.npz" # avg 60
from config.constants import TIME_MULTI

FILE_NAME= "data/data_dump_0.npz" # avg 60
OUT_PATH = "tests/out/"

data = np.load(FILE_NAME)
kinect, robo = data["kinect"], data["robo"]
kinect_time, robo_time, robo_speed = data["kinect_time"], data["robo_time"], data["robo_speed"]



print ("kinect.shape", kinect.shape)
print ("kinect_time.shape",kinect_time.shape)
print ("robo.shape", robo.shape)
print ("robo_time.shape",robo_time.shape)
print ("robo_speed.shape",robo_speed.shape)

print (kinect[0].shape)
print (kinect_time[0].shape)


kl = []
for i in range(len(kinect)):
    kl.append(len(kinect[i]))

print ("sum:",sum(kl))
print ("avg:",sum(kl)/len(kl))
print ("FPS:",sum(kl)/len(kl)/6)

print ("---")


print("timings:\nrobot\tkinect")
for i in range(10):
    print ("{}\t{}".format(robo_time[0,i,0,0],kinect_time[0][i]))

quit()


# for i in range(3):
#     for j in range(200):
#         print (np.around(robo[j,i,:,0]))

kinect = kinect[15]
robo = robo[15]

# iterate over kinect frames and render them to disk
for i in range(len(kinect)):
    img_data = kinect[i,:,:,:3]

    img_data = img_data[:, :, ::-1].copy() # shuffle color order from BGR (openCV2) to PIL (RGB)
    depth_frame = np.expand_dims(kinect[i,:,:,3],2)
    combined = np.concatenate((img_data, depth_frame), axis=2)

    img = Image.fromarray(combined, 'RGBA')

    # img.show()
    # quit()
    img.save(OUT_PATH+'img-137/{:03d}.png'.format(i))




