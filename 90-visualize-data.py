from PIL import Image

import numpy as np
import cv2

FILE_NAME= "data/data_dump.npz"
OUT_PATH = "tests/out/"

data = np.load(FILE_NAME)
kinect, robo = data["kinect"], data["robo"]



print (kinect.shape)
print (robo.shape)
print (kinect[0].shape)

# quit()
# for i in range(3):
#     for j in range(200):
#         print (np.around(robo[j,i,:,0]))

kinect = kinect[120]
robo = robo[120]

# iterate over kinect frames and render them to disk
for i in range(len(kinect)):
    img_data = kinect[i,:,:,:3]

    img_data = img_data[:, :, ::-1].copy() # shuffle color order from BGR (openCV2) to PIL (RGB)
    depth_frame = np.expand_dims(kinect[i,:,:,3],2)
    combined = np.concatenate((img_data, depth_frame), axis=2)

    img = Image.fromarray(combined, 'RGBA')

    # img.show()
    # quit()
    img.save(OUT_PATH+'img/{:03d}.png'.format(i))




