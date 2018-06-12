import numpy as np

import time

from tqdm import tqdm
import os

from s2rr.config.constants import WRITE_EVERY_N_EPISODES
from s2rr.movements.dataset import Dataset
from vrepper.core import vrepper
import h5py

MAX_FRAMES = 273  # max episode length is 274, so -1 (because we always look one step ahead) is 273
EPISODES = 30
FILES = range(3)
IMG_REAL_WIDTH = 250
IMG_REAL_HEIGHT = 240
IMG_SIM_WIDTH = 256
IMG_SIM_HEIGHT = 256

DATASET_PATH_CLEAN = "data/recording1_clean.npz"

MASSIVE_OUTPUT_FILE = "/media/florian/shapenet/test-dataset2.hdf5"

f = h5py.File(MASSIVE_OUTPUT_FILE, "w")

dataset_current_pos_real = f.create_dataset("current_pos_real", (EPISODES, MAX_FRAMES, 6), dtype=np.float32)
dataset_current_vel_real = f.create_dataset("current_vel_real", (EPISODES, MAX_FRAMES, 6), dtype=np.float32)

dataset_next_pos_sim = f.create_dataset("next_pos_sim", (EPISODES, MAX_FRAMES, 6),
                                        dtype=np.float32)  # this is LSTM input
dataset_next_vel_sim = f.create_dataset("next_vel_sim", (EPISODES, MAX_FRAMES, 6),
                                        dtype=np.float32)  # this is LSTM input

dataset_next_pos_real = f.create_dataset("next_pos_real", (EPISODES, MAX_FRAMES, 6),
                                         dtype=np.float32)  # this is LSTM output
dataset_next_vel_real = f.create_dataset("next_vel_real", (EPISODES, MAX_FRAMES, 6),
                                         dtype=np.float32)  # this is LSTM output

dataset_current_action = f.create_dataset("current_action", (EPISODES, MAX_FRAMES, 6),
                                          dtype=np.float32)  # this is LSTM input/conditioning
dataset_current_speed = f.create_dataset("current_speed", (EPISODES, MAX_FRAMES), dtype=np.float32)

dataset_current_img_real = f.create_dataset("current_img_real",
                                            (EPISODES, MAX_FRAMES, IMG_REAL_HEIGHT, IMG_REAL_WIDTH, 4), dtype=np.uint8)

dataset_next_img_real = f.create_dataset("next_img_real", (EPISODES, MAX_FRAMES, IMG_REAL_HEIGHT, IMG_REAL_WIDTH, 4),
                                         dtype=np.uint8)
dataset_next_img_sim = f.create_dataset("next_img_sim", (EPISODES, MAX_FRAMES, IMG_SIM_HEIGHT, IMG_SIM_WIDTH, 4),
                                        dtype=np.uint8)

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


def set_motors(positions, speeds=None):
    for i, m in enumerate(motors):
        target = positions[i]
        if i == 0:
            target *= -1
        m.set_position_target(target)
        if speeds is not None:
            m.set_velocity(speeds[i])


def get_motors():
    out_pos = np.zeros(6, dtype=np.float32)
    out_vel = np.zeros(6, dtype=np.float32)
    for i, m in enumerate(motors):
        angle = m.get_joint_angle()
        if i == 0:
            angle *= -1
        out_pos[i] = angle
        out_vel[i] = m.get_joint_velocity()[0]

    return out_pos, out_vel


def reset_sim(initial_position=None):
    assert len(initial_position) == 6

    venv.stop_simulation()
    venv.start_simulation(is_sync=False)  # start in realtime mode to set initial position

    if initial_position is not None:
        # init position for this turn
        set_motors(initial_position)
    time.sleep(.2)
    venv.make_simulation_synchronous(True)


def real_img_reshaping(img_in):
    img_depth = img_in[:, :, 3]
    img_color = img_in[:, :, :3]
    img_combined = np.zeros(img_in.shape, dtype=np.uint8)
    img_combined[:, :, :3] = np.rot90(
        img_color[:, :, ::-1],  # BGR -> RGB
        2,  # rotate 180...
        (0, 1)  # ...on the x-y plane
    )
    img_combined[:, :, 3] = np.rot90(
        img_depth,
        2,  # rotate 180...
        (0, 1)  # ...on the x-y plane
    )

    return img_combined


ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

for file_idx in FILES:
    FILE_NAME = "data/data_dump_{}_aligned.npz".format(file_idx)
    print ("LOADING: ",FILE_NAME)
    data = np.load(FILE_NAME)
    real_imgs, real_positions, real_speeds = data["real_images"], data["real_positions"], data["real_speeds"]

    print ("\n====== FILE: {}\n".format(file_idx))

    ############## put future for loop here
    for episode_idx in tqdm(range(WRITE_EVERY_N_EPISODES)):

        OFFSET = 4  # IDK why

        episode_current_pos_real = np.zeros((MAX_FRAMES, 6), dtype=np.float32)
        episode_current_vel_real = np.zeros((MAX_FRAMES, 6), dtype=np.float32)

        episode_next_pos_sim = np.zeros((MAX_FRAMES, 6), dtype=np.float32)  # this is LSTM input
        episode_next_vel_sim = np.zeros((MAX_FRAMES, 6), dtype=np.float32)  # this is LSTM input

        episode_next_pos_real = np.zeros((MAX_FRAMES, 6), dtype=np.float32)  # this is LSTM output
        episode_next_vel_real = np.zeros((MAX_FRAMES, 6), dtype=np.float32)  # this is LSTM output

        episode_current_action = np.zeros((MAX_FRAMES, 6), dtype=np.float32)
        episode_current_speed = np.zeros((MAX_FRAMES), dtype=np.float32)

        episode_current_img_real = np.zeros((MAX_FRAMES, IMG_REAL_HEIGHT, IMG_REAL_WIDTH, 4), dtype=np.uint8)

        episode_next_img_real = np.zeros((MAX_FRAMES, IMG_REAL_HEIGHT, IMG_REAL_WIDTH, 4), dtype=np.uint8)
        episode_next_img_sim = np.zeros((MAX_FRAMES, IMG_SIM_HEIGHT, IMG_SIM_WIDTH, 4), dtype=np.uint8)

        current_episode = file_idx * WRITE_EVERY_N_EPISODES + episode_idx + OFFSET

        reset_sim(real_positions[episode_idx][0, :, 0])

        speed_change_indices = []
        last_speed = real_speeds[episode_idx][0]

        real_actions = []
        cmds = ds.getUniqueActionsForEpisode(current_episode)

        action_idx = 0

        for idx in range(len(real_imgs[episode_idx])):

            set_motors(real_positions[episode_idx][idx, :, 0], real_positions[episode_idx][idx, :, 1])
            venv.step_blocking_simulation()

            # whenever the speed changes, we know we have a new motor command
            if last_speed != real_speeds[episode_idx][idx]:
                action_idx += 1
                last_speed = real_speeds[episode_idx][idx]

            current_action = cmds[action_idx]

            if idx < len(real_imgs[episode_idx]) - 1:
                current_real_img = real_img_reshaping(real_imgs[episode_idx][idx, :, :, :])
                next_real_img = real_img_reshaping(real_imgs[episode_idx][idx + 1, :, :, :])
                next_sim_img = venv.get_image_and_depth(cam.handle)

                episode_current_pos_real[idx] = real_positions[episode_idx][idx, :, 0]  # index 0 are the positions
                episode_current_vel_real[idx] = real_positions[episode_idx][idx, :, 1]  # index 1 are the velocities
                # index 2 are the loads (unused)

                next_pos, next_vel = get_motors()
                episode_next_pos_sim[idx] = next_pos
                episode_next_vel_sim[idx] = next_vel

                episode_next_pos_real[idx] = real_positions[episode_idx][idx + 1, :, 0]
                episode_next_vel_real[idx] = real_positions[episode_idx][idx + 1, :, 1]

                episode_current_action[idx] = current_action
                episode_current_speed[idx] = real_speeds[episode_idx][idx]

                episode_current_img_real[idx] = current_real_img

                episode_next_img_real[idx] = next_real_img
                episode_next_img_sim[idx] = next_sim_img

        # end of episode

        saving_idx = file_idx * WRITE_EVERY_N_EPISODES + episode_idx

        dataset_current_pos_real[saving_idx] = episode_current_pos_real
        dataset_current_vel_real[saving_idx] = episode_current_vel_real

        dataset_next_pos_sim[saving_idx] = episode_next_pos_sim
        dataset_next_vel_sim[saving_idx] = episode_next_vel_sim

        dataset_next_pos_real[saving_idx] = episode_next_pos_real
        dataset_next_vel_real[saving_idx] = episode_next_vel_real

        dataset_current_action[saving_idx] = episode_current_action
        dataset_current_speed[saving_idx] = episode_current_speed

        dataset_current_img_real[saving_idx] = episode_current_img_real

        dataset_next_img_real[saving_idx] = episode_next_img_real
        dataset_next_img_sim[saving_idx] = episode_next_img_sim

        f.flush()

f.close()
