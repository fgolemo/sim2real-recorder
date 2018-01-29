# import cv2
# import numpy as np
# import math
#
# import time
#
# from tqdm import tqdm
# import os
#

# from movements.dataset import Dataset
# from vrepper.core import vrepper
# import matplotlib.pyplot as plt
# import h5py
# import itertools
#

#
#
# # venv = vrepper(headless=False)
# venv = vrepper(headless=True)
# venv.start()
# current_dir = os.path.dirname(os.path.realpath(__file__))
# venv.load_scene(current_dir + '/scenes/poppy_ergo_jr.ttt')
# motors = []
# for i in range(6):
#     motor = venv.get_object_by_name('m{}'.format(i + 1), is_joint=True)
#     motors.append(motor)
#
# cam = venv.get_object_by_name("cam")
#
#
# def set_motors(positions, speeds=None):
#     for i, m in enumerate(motors):
#         target = positions[i]
#         if i == 0:
#             target *= -1
#         m.set_position_target(target)
#         if speeds is not None:
#             m.set_velocity(speeds[i])
#
#
# def get_motors():
#     out_pos = np.zeros(6, dtype=np.float32)
#     out_vel = np.zeros(6, dtype=np.float32)
#     for i, m in enumerate(motors):
#         angle = m.get_joint_angle()
#         if i == 0:
#             angle *= -1
#         out_pos[i] = angle
#         out_vel[i] = m.get_joint_velocity()[0]
#
#     return out_pos, out_vel
#
#
# def reset_sim(initial_position=None):
#     assert len(initial_position) == 6
#
#     venv.stop_simulation()
#     venv.start_simulation(is_sync=False)  # start in realtime mode to set initial position
#
#     if initial_position is not None:
#         # init position for this turn
#         set_motors(initial_position)
#     time.sleep(.2)
#     venv.make_simulation_synchronous(True)
#
#
# def real_img_reshaping(img_in):
#     img_depth = img_in[:, :, 3]
#     img_color = img_in[:, :, :3]
#     img_combined = np.zeros(img_in.shape, dtype=np.uint8)
#     img_combined[:, :, :3] = np.rot90(
#         img_color[:, :, ::-1],  # BGR -> RGB
#         2,  # rotate 180...
#         (0, 1)  # ...on the x-y plane
#     )
#     img_combined[:, :, 3] = np.rot90(
#         img_depth,
#         2,  # rotate 180...
#         (0, 1)  # ...on the x-y plane
#     )
#
#     return img_combined
#
#
# ds = Dataset()
# ds.load(DATASET_PATH_CLEAN)
#
# for file_idx in FILES:
#     FILE_NAME = "data/data_dump_{}_aligned.npz".format(file_idx)
#     print ("LOADING: ",FILE_NAME)
#     data = np.load(FILE_NAME)
#     real_imgs, real_positions, real_speeds = data["real_images"], data["real_positions"], data["real_speeds"]
#
#     print ("\n====== FILE: {}\n".format(file_idx))
#
#     ############## put future for loop here
#     for episode_idx in tqdm(range(WRITE_EVERY_N_EPISODES)):
#
#         OFFSET = 4  # IDK why
#

#
#         current_episode = file_idx * WRITE_EVERY_N_EPISODES + episode_idx + OFFSET
#
#         reset_sim(real_positions[episode_idx][0, :, 0])
#
#         speed_change_indices = []
#         last_speed = real_speeds[episode_idx][0]
#
#         real_actions = []
#         cmds = ds.getUniqueActionsForEpisode(current_episode)
#
#         action_idx = 0
#
#         for idx in range(len(real_imgs[episode_idx])):
#
#             set_motors(real_positions[episode_idx][idx, :, 0], real_positions[episode_idx][idx, :, 1])
#             venv.step_blocking_simulation()
#
#             # whenever the speed changes, we know we have a new motor command
#             if last_speed != real_speeds[episode_idx][idx]:
#                 action_idx += 1
#                 last_speed = real_speeds[episode_idx][idx]
#
#             current_action = cmds[action_idx]
#
#             if idx < len(real_imgs[episode_idx]) - 1:
#                 current_real_img = real_img_reshaping(real_imgs[episode_idx][idx, :, :, :])
#                 next_real_img = real_img_reshaping(real_imgs[episode_idx][idx + 1, :, :, :])
#                 next_sim_img = venv.get_image_and_depth(cam.handle)
#
#                 episode_current_pos_real[idx] = real_positions[episode_idx][idx, :, 0]  # index 0 are the positions
#                 episode_current_vel_real[idx] = real_positions[episode_idx][idx, :, 1]  # index 1 are the velocities
#                 # index 2 are the loads (unused)
#
#                 next_pos, next_vel = get_motors()
#                 episode_next_pos_sim[idx] = next_pos
#                 episode_next_vel_sim[idx] = next_vel
#
#                 episode_next_pos_real[idx] = real_positions[episode_idx][idx + 1, :, 0]
#                 episode_next_vel_real[idx] = real_positions[episode_idx][idx + 1, :, 1]
#
#                 episode_current_action[idx] = current_action
#                 episode_current_speed[idx] = real_speeds[episode_idx][idx]
#
#                 episode_current_img_real[idx] = current_real_img
#
#                 episode_next_img_real[idx] = next_real_img
#                 episode_next_img_sim[idx] = next_sim_img
#


from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from tqdm import tqdm

from config.constants import WRITE_EVERY_N_EPISODES
import h5py

PROCESSES = 4
DATA_PATH_PREFIX = "/lindata/datasets/sim2real/"
DATASET_PATH_CLEAN = "./data/recording1_clean.npz"  # contains actions and speeds
MASSIVE_OUTPUT_FILE = "/windata/sim2real-full/test-dataset3.hdf5"
MAX_FRAMES = 273  # max episode length is 274, so -1 (because we always look one step ahead) is 273
FILES = [0, 1, 2]
IMG_REAL_WIDTH = 250
IMG_REAL_HEIGHT = 240
IMG_SIM_WIDTH = 256
IMG_SIM_HEIGHT = 256


def chunked_list(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def make_list_of_jobs(file_list, episodes_per_file, procs):
    job_list = []
    for file_no in file_list:
        for episode_idx in range(episodes_per_file):
            config = dict()
            config["file_idx"] = file_no
            config["file_name"] = DATA_PATH_PREFIX + "data_dump_{}_aligned.npz".format(file_no)
            config["episode_idx"] = episode_idx
            job_list.append(config)

    return chunked_list(job_list, procs)


def simulate_run(config):
    file_idx = config["file_idx"]
    file_name = config["file_name"]
    episode_idx = config["episode_idx"]
    print ("starting proc on file {}, episode {}".format(file_name, episode_idx))

    data = np.load(file_name)
    real_imgs, real_positions, real_speeds = data["real_images"], data["real_positions"], data["real_speeds"]

    out = H5Dataset.get_episode_buffer()
    out["file_idx"] = file_idx
    out["episode_idx"] = episode_idx



    return out


class H5Dataset(object):
    def __init__(self):
        self.f = h5py.File(MASSIVE_OUTPUT_FILE, "w")

    def flush(self):
        self.f.flush()

    def close(self):
        self.flush()
        self.f.close()

    def make_dataset(self, total_files):
        self.current_pos_real = self.f.create_dataset("current_pos_real",
                                                      (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, 6),
                                                      dtype=np.float32)
        self.current_vel_real = self.f.create_dataset("current_vel_real",
                                                      (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, 6),
                                                      dtype=np.float32)

        self.next_pos_sim = self.f.create_dataset("next_pos_sim", (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, 6),
                                                  dtype=np.float32)  # this is LSTM input
        self.next_vel_sim = self.f.create_dataset("next_vel_sim", (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, 6),
                                                  dtype=np.float32)  # this is LSTM input

        self.next_pos_real = self.f.create_dataset("next_pos_real",
                                                   (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, 6),
                                                   dtype=np.float32)  # this is LSTM output
        self.next_vel_real = self.f.create_dataset("next_vel_real",
                                                   (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, 6),
                                                   dtype=np.float32)  # this is LSTM output

        self.current_action = self.f.create_dataset("current_action",
                                                    (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, 6),
                                                    dtype=np.float32)  # this is LSTM input/conditioning
        self.current_speed = self.f.create_dataset("current_speed", (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES),
                                                   dtype=np.float32)

        self.current_img_real = self.f.create_dataset("current_img_real",
                                                      (total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, IMG_REAL_HEIGHT,
                                                       IMG_REAL_WIDTH, 4), dtype=np.uint8)

        self.next_img_real = self.f.create_dataset("next_img_real", (
            total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, IMG_REAL_HEIGHT, IMG_REAL_WIDTH, 4),
                                                   dtype=np.uint8)
        self.next_img_sim = self.f.create_dataset("next_img_sim", (
            total_files, WRITE_EVERY_N_EPISODES, MAX_FRAMES, IMG_SIM_HEIGHT, IMG_SIM_WIDTH, 4),
                                                  dtype=np.uint8)

    def write_episode(self, res):
        self.current_pos_real[res["file_idx"], res["episode_idx"]] = res["ep_current_pos_real"]
        self.current_vel_real[res["file_idx"], res["episode_idx"]] = res["ep_current_vel_real"]

        self.next_pos_sim[res["file_idx"], res["episode_idx"]] = res["ep_next_pos_sim"]
        self.next_vel_sim[res["file_idx"], res["episode_idx"]] = res["ep_next_vel_sim"]

        self.next_pos_real[res["file_idx"], res["episode_idx"]] = res["ep_next_pos_real"]
        self.next_vel_real[res["file_idx"], res["episode_idx"]] = res["ep_next_vel_real"]

        self.current_action[res["file_idx"], res["episode_idx"]] = res["ep_current_action"]
        self.current_speed[res["file_idx"], res["episode_idx"]] = res["ep_current_speed"]

        self.current_img_real[res["file_idx"], res["episode_idx"]] = res["ep_current_img_real"]

        self.next_img_real[res["file_idx"], res["episode_idx"]] = res["ep_next_img_real"]
        self.next_img_sim[res["file_idx"], res["episode_idx"]] = res["ep_next_img_sim"]

    @staticmethod
    def get_episode_buffer():
        return {
            "ep_current_pos_real": np.zeros((MAX_FRAMES, 6), dtype=np.float32),
            "ep_current_vel_real": np.zeros((MAX_FRAMES, 6), dtype=np.float32),

            "ep_next_pos_sim": np.zeros((MAX_FRAMES, 6), dtype=np.float32),
            "ep_next_vel_sim": np.zeros((MAX_FRAMES, 6), dtype=np.float32),

            "ep_next_pos_real": np.zeros((MAX_FRAMES, 6), dtype=np.float32),
            "ep_next_vel_real": np.zeros((MAX_FRAMES, 6), dtype=np.float32),

            "ep_current_action": np.zeros((MAX_FRAMES, 6), dtype=np.float32),
            "ep_current_speed": np.zeros((MAX_FRAMES), dtype=np.float32),

            "ep_current_img_real": np.zeros((MAX_FRAMES, IMG_REAL_HEIGHT, IMG_REAL_WIDTH, 4), dtype=np.uint8),

            "ep_next_img_real": np.zeros((MAX_FRAMES, IMG_REAL_HEIGHT, IMG_REAL_WIDTH, 4), dtype=np.uint8),
            "ep_next_img_sim": np.zeros((MAX_FRAMES, IMG_SIM_HEIGHT, IMG_SIM_WIDTH, 4), dtype=np.uint8)
        }


f = H5Dataset()
f.make_dataset(len(FILES))

runs = make_list_of_jobs(FILES, WRITE_EVERY_N_EPISODES, PROCESSES)

for run in tqdm(runs):
    pool = ThreadPool(PROCESSES)

    # open the urls in their own threads
    # and return the results
    results = pool.map(simulate_run, run)

    for res in results:
        f.write_episode(res)

    # close the pool and wait for the work to finish
    # pool.close()

    pool.close()
    pool.join()
    f.flush()

f.close()
