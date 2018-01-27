import cv2
import numpy as np
import math

import time

from tqdm import tqdm
import os

from config.constants import WRITE_EVERY_N_EPISODES
from movements.dataset import Dataset
from vrepper.core import vrepper
import matplotlib.pyplot as plt

FILE_IDX = 145
FILE_NAME = "data/data_dump_{}_aligned.npz".format(FILE_IDX)
GIF_OUT = "data/gif/{}/{:03d}.png"
DATASET_PATH_CLEAN = "data/recording1_clean.npz"
PROGRESS_FILE = "data/recording_progress"
# EPISODE = 0

# MOTOR 0 & 3 ARE INVERTED

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)
data = np.load(FILE_NAME)
real_imgs, real_positions, real_speeds = data["real_images"], data["real_positions"], data["real_speeds"]

############## put future for loop here
RECORDING = 1
OFFSET = 4 # IDK why

current_episode = FILE_IDX*WRITE_EVERY_N_EPISODES+RECORDING+OFFSET

# distances2 = []
# distances3 = []
# s = np.array([-68.61254 , -31.310816,   1.320016, -56.506195,  23.946943,  -45.309986])
# for current_episode in range(3000):
#     distances2.append(np.linalg.norm(ds.getUniqueActionsForEpisode(current_episode)[1,0,:]-s))
#     distances3.append(np.linalg.norm(ds.getUniqueActionsForEpisode(current_episode)[2,0,:]-s))
#
# distances2 = np.array(distances2)
# distances3 = np.array(distances3)
#
# print()
#
# print (distances2.min(), distances2.argmin(), FILE_IDX*WRITE_EVERY_N_EPISODES+RECORDING)
# print (distances3.min(), distances3.argmin(), FILE_IDX*WRITE_EVERY_N_EPISODES+RECORDING)
# quit()

# venv = vrepper(headless=False)
venv = vrepper(headless=True)
venv.start()
current_dir = os.path.dirname(os.path.realpath(__file__))
venv.load_scene(current_dir + '/scenes/poppy_ergo_jr.ttt')
motors = []
for i in range(6):
    motor = venv.get_object_by_name('m{}'.format(i + 1), is_joint=True)
    motors.append(motor)

cam = venv.get_object_by_name("cam")

venv.start_simulation(is_sync=False)  # start in realtime mode to set initial position


# img = venv.get_image(cam.handle)
# plt.imshow(img, interpolation="nearest")
# plt.show()


def set_motors(positions, speeds=None):
    for i, m in enumerate(motors):
        target = positions[i]
        if i == 0:
            target *= -1
        m.set_position_target(target)
        if speeds is not None:
            m.set_velocity(speeds[i])


def get_motors():
    out = np.zeros(6, dtype=np.float32)
    for i, m in enumerate(motors):
        angle = m.get_joint_angle()
        if i == 0:
            angle *= -1
        out[i] = angle
    return out


# init position for this turn
set_motors(real_positions[RECORDING][0, :, 0])
time.sleep(.2)
venv.make_simulation_synchronous(True)

diffs = np.zeros((len(real_imgs[RECORDING]) - 1, 6), dtype=np.float32)

speed_change_indices = []
last_speed = real_speeds[RECORDING][0]


def real_img_reshaping(img_in):
    return np.rot90(
        np.swapaxes(
            img_in[:, :, ::-1].T,  # BGR -> RGB & transpose
            0,  # move color axis to the right
            2
        ),
        2,  # rotate 180...
        (0, 1)  # ...on the x-y plage
    )

real_actions = []

for idx, img in enumerate(tqdm(real_imgs[RECORDING])):
    # cv2.imshow("img", img[:, :, :3])
    # cv2.waitKey(1)
    # time.sleep(.03)
    # print (real_positions[RECORDING][idx,0,0]) # print first joint position
    # print (real_positions[RECORDING][idx,0,1])
    set_motors(real_positions[RECORDING][idx, :, 0], real_positions[RECORDING][idx, :, 1])
    venv.step_blocking_simulation()

    # print ("next real state / next sim state:")
    if last_speed != real_speeds[RECORDING][idx]:
        real_actions.append(get_motors())
        speed_change_indices.append(idx)
        last_speed = real_speeds[RECORDING][idx]

    if idx < len(real_imgs[RECORDING]) - 1:
        next_real_img = np.rot90(real_imgs[RECORDING][idx + 1, :, :, :3], 2, (0, 1))
        next_sim_img = np.rot90(venv.get_image(cam.handle), 2, (0, 1))
        combined_img = np.zeros((256, 506, 3), dtype=np.uint8)
        combined_img[16:, :250, :] = real_img_reshaping(next_real_img)
        combined_img[:, 250:, :] = next_sim_img
        # plt.imsave(GIF_OUT.format(FILE_IDX, idx), combined_img)

        diff_real_minus_sim = real_positions[RECORDING][idx + 1, :, 0] - get_motors()
        diffs[idx] = diff_real_minus_sim

real_actions.append(get_motors())

# print (ds.getUniqueActionsForEpisode(current_episode))
# print (real_actions)
cmds = ds.getUniqueActionsForEpisode(current_episode)

for action in range(3):
    print ("cmd\t\tsim")
    for cell in range(6):
        print("{}\t\t{}".format(round(cmds[action,0,cell],1), round(real_actions[action][cell],2)))

    print ("===")

# for i in range(6):
#     plt.plot(np.arange(len(diffs[:, 0])), diffs[:, i], label="m{}".format(i + 1))
# plt.title("Joint differences over time (real-sim)")
# plt.ylabel("Joint difference in degrees")
# plt.xlabel("Frame")
#
# for line_pos in speed_change_indices:
#     plt.axvline(x=line_pos)
#
# plt.legend()
# plt.show()
