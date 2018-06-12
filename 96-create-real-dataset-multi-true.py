from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from tqdm import tqdm

from s2rr.config.constants import WRITE_EVERY_N_EPISODES
from s2rr.movements.dataset import Dataset
from recorder.h5dataset import H5Dataset
from s2rr.recorder.simulator import Simulator

RANGE_LO, RANGE_HI = 10, 20

PROCESSES = 2
DATA_PATH_PREFIX = "/lindata/datasets/sim2real/"
# DATA_PATH_PREFIX = "./data/"
DATASET_PATH_CLEAN = "./data/recording1_clean.npz"  # contains actions and speeds
MASSIVE_OUTPUT_FILE = "/windata/sim2real-full/dataset-{}-{}.hdf5".format(RANGE_LO, RANGE_HI)
# MASSIVE_OUTPUT_FILE = "/media/florian/shapenet/test-dataset4.hdf5"
MAX_FRAMES = 273  # max episode length is 274, so -1 (because we always look one step ahead) is 273
FILES = list(range(RANGE_LO, RANGE_HI))
IMG_REAL_WIDTH = 250
IMG_REAL_HEIGHT = 240
IMG_SIM_WIDTH = 256
IMG_SIM_HEIGHT = 256


def chunked_list(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def make_list_of_jobs(file_list, episodes_per_file, procs, eb):
    job_list = []
    for file_no in file_list:
        for episode_idx in range(episodes_per_file):
            config = dict()
            config["file_idx"] = file_no
            config["file_name"] = DATA_PATH_PREFIX + "data_dump_{}_aligned.npz".format(file_no)
            config["episode_idx"] = episode_idx
            config["episode_buffer"] = eb
            job_list.append(config)

    return chunked_list(job_list, procs)


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


def simulate_run(config):
    OFFSET = 4  # IDK why

    file_idx = config["file_idx"]
    file_name = config["file_name"]
    episode_idx = config["episode_idx"]
    print ("starting proc on file {}, episode {}".format(file_name, episode_idx))

    data = np.load(file_name)
    real_imgs, real_positions, real_speeds = data["real_images"], data["real_positions"], data["real_speeds"]

    out = H5Dataset.copy_deep(config["episode_buffer"])  # need to make a copy in each thread,
    # otherwise the thread writes into original memory
    out["file_idx"] = file_idx
    out["episode_idx"] = episode_idx

    ds = Dataset()
    ds.load(DATASET_PATH_CLEAN)

    sim = Simulator()

    sim.reset_sim(real_positions[episode_idx][0, :, 0])

    last_speed = real_speeds[episode_idx][0]

    current_episode = file_idx * WRITE_EVERY_N_EPISODES + episode_idx + OFFSET
    cmds = ds.getUniqueActionsForEpisode(current_episode)

    action_idx = 0

    for idx in range(len(real_imgs[episode_idx])):

        sim.step(real_positions[episode_idx][idx, :, 0], real_positions[episode_idx][idx, :, 1])

        # whenever the speed changes, we know we have a new motor command
        if last_speed != real_speeds[episode_idx][idx]:
            action_idx += 1
            last_speed = real_speeds[episode_idx][idx]

        current_action = cmds[action_idx]

        if idx < len(real_imgs[episode_idx]) - 1:
            current_real_img = real_img_reshaping(real_imgs[episode_idx][idx, :, :, :])
            next_real_img = real_img_reshaping(real_imgs[episode_idx][idx + 1, :, :, :])
            next_sim_img = sim.venv.get_image_and_depth(sim.cam.handle)

            out["ep_current_pos_real"][idx] = real_positions[episode_idx][idx, :, 0]  # index 0 are the positions
            out["ep_current_vel_real"][idx] = real_positions[episode_idx][idx, :, 1]  # index 1 are the velocities
            # index 2 are the loads (unused)

            next_pos, next_vel = sim.get_motors()
            out["ep_next_pos_sim"][idx] = next_pos
            out["ep_next_vel_sim"][idx] = next_vel

            out["ep_next_pos_real"][idx] = real_positions[episode_idx][idx + 1, :, 0]
            out["ep_next_vel_real"][idx] = real_positions[episode_idx][idx + 1, :, 1]

            out["ep_current_action"][idx] = current_action
            out["ep_current_speed"][idx] = real_speeds[episode_idx][idx]

            out["ep_current_img_real"][idx] = current_real_img

            out["ep_next_img_real"][idx] = next_real_img
            out["ep_next_img_sim"][idx] = next_sim_img

    sim.close()
    return out


img_config = {
    "isw": IMG_SIM_WIDTH,
    "ish": IMG_SIM_HEIGHT,
    "irw": IMG_REAL_WIDTH,
    "irh": IMG_REAL_HEIGHT
}

f = H5Dataset(MASSIVE_OUTPUT_FILE, WRITE_EVERY_N_EPISODES, MAX_FRAMES, img_config, RANGE_LO)
f.make_dataset(len(FILES))

episode_buffer = f.get_episode_buffer()

runs = make_list_of_jobs(FILES, WRITE_EVERY_N_EPISODES, PROCESSES, episode_buffer)

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
