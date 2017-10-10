from PIL import Image

import numpy as np
import cv2

# FILE_NAME= "data/data_dump_5.npz" # avg 80
# FILE_NAME= "data/data_dump_95.npz" # avg 17
FILE_NAME= "data/data_dump_137.npz" # avg 60
OUT_PATH = "tests/out/"

data = np.load(FILE_NAME)
kinect, robo = data["kinect"], data["robo"]



print (kinect.shape)
print (robo.shape)
print (kinect[0].shape)

kl = []
for i in range(len(kinect)):
    kl.append(len(kinect[i]))

print ("sum:",sum(kl))
print ("avg:",sum(kl)/len(kl))

# quit()


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




