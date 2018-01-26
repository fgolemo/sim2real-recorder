import cv2
import numpy as np
import math

import time

from tqdm import tqdm

FILE_IDX = 145
FILE_NAME = "data/data_dump_{}.npz".format(FILE_IDX)
FILE_NAME_OUT =  "data/data_dump_{}_aligned.npz".format(FILE_IDX)

data = np.load(FILE_NAME)
kinect, robo, robo_speed = data["kinect"], data["robo"], data["robo_speed"]
kinect_time, robo_time = data["kinect_time"], data["robo_time"]

# for i in range(10):
#     print (kinect_time[i].shape,
#            robo_time[i].shape,
#            kinect[i].shape,
#            robo[i].shape,
#            robo_speed[i].shape)


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or
                    math.fabs(value - array[idx - 1]) <
                    math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx

images_clean_out = []
positions_clean_out = []
speeds_clean_out = []

for RECORDING in tqdm(range(10)):

    # step one, cleaning the robot time dataset,
    # because there are some lines that just have 0 as timestamp
    # and some that aren't consecutive,
    # i.e. where the next timestamp is before the current one

    flat_robot_times = np.swapaxes(robo_time[RECORDING], 0, 1).flatten()
    flat_robot_speeds = np.swapaxes(robo_speed[RECORDING], 0, 1).flatten()
    flat_robot_positions = np.swapaxes(robo[RECORDING], 0, 1).reshape((-1, 6, 3))

    bad_indices = []

    last_time = -1
    # we can't iterate over the actual list because we wanna modify it
    for idx in range(len(flat_robot_times)):
        if flat_robot_times[idx] == 0 or flat_robot_times[idx] < last_time:
            bad_indices.append(idx)
            continue

        last_time = flat_robot_times[idx]

    mask = np.ones(len(flat_robot_times), dtype=bool)
    mask[bad_indices] = False
    filtered_flat_robot_times = flat_robot_times[mask, ...]
    filtered_flat_robot_speeds = flat_robot_speeds[mask, ...]
    filtered_flat_robot_positions = flat_robot_positions[mask, ...]

    # print ("robot min/max: {}, {}".format(
    #     filtered_flat_robot_times.min(),
    #     filtered_flat_robot_times.max()))
    # print ("kinec min/max: {}, {}".format(
    #     kinect_time[RECORDING].min(),
    #     kinect_time[RECORDING].max()))

    # check for general alignment:
    if kinect_time[RECORDING].min() > filtered_flat_robot_times.max() or \
            kinect_time[RECORDING].max() < filtered_flat_robot_times.min():
        raise Exception("DBG: THE RECORDING IS NOT ALIGNED: ", RECORDING)

    # find out where to start
    start_with_kinect = True
    if kinect_time[RECORDING].min() > filtered_flat_robot_times.min():
        start_with_kinect = False

    # build mapping list
    kinect_indices = np.arange(len(kinect_time[RECORDING]))
    mapping = np.zeros((len(kinect_time[RECORDING]), 2))
    mapping[:, 0] = kinect_indices
    mapping[:, 1] -= 1  # to make everything in second column "-1"

    k_idx = 0
    r_idx = 0

    # if starting with kinect find robot element to start with
    if start_with_kinect:
        for kt_idx, kt_val in enumerate(kinect_time[RECORDING]):
            if kt_val >= filtered_flat_robot_times.min():
                if kt_idx > 0:
                    k_idx = kt_idx - 1
                break



    while True:
        # pick out current kinect time and find corresponding mapping
        current_kt = kinect_time[RECORDING][k_idx]

        closest_rt_idx = find_nearest(filtered_flat_robot_times[r_idx:], current_kt)
        mapping[k_idx, 1] = r_idx + closest_rt_idx

        k_idx += 1
        if closest_rt_idx == 0:
            r_idx += 1
        else:
            r_idx += closest_rt_idx

        if k_idx == len(kinect_time[RECORDING]) or \
                r_idx >= len(filtered_flat_robot_times):
            break

    mapping_clean = mapping[mapping[:, 1] > -1].astype(np.uint16)
    print ("DBG: Mappings in recording {}: {}"
           "".format(RECORDING, len(mapping_clean)))

    # now let's glue the two datasets together based on mapping

    # images_clean = np.zeros((len(mapping_clean), 240, 250, 4), dtype=np.uint8)
    # positions_clean = np.zeros((len(mapping_clean), 6, 3), dtype=np.float32)
    # speeds_clean = np.zeros((len(mapping_clean),), dtype=np.float32)

    images_clean = kinect[RECORDING][mapping_clean[:,0]].astype(np.uint8)
    positions_clean = filtered_flat_robot_positions[mapping_clean[:,1]].astype(np.float32)
    speeds_clean = filtered_flat_robot_speeds[mapping_clean[:,1]].astype(np.float32)

    # for img in images_clean:
    #     cv2.imshow("img", img[:,:,:3])
    #     cv2.waitKey(1)
    #     time.sleep(.03)

    images_clean_out.append(images_clean)
    positions_clean_out.append(positions_clean)
    speeds_clean_out.append(speeds_clean)

np.savez_compressed(FILE_NAME_OUT,
         real_images=images_clean_out,
         real_positions=positions_clean_out,
         real_speeds=speeds_clean_out)
