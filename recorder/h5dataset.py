import h5py
import numpy as np


class H5Dataset(object):
    def __init__(self, output_file, ep_per_file, frames_per_episode, img_config, low_range_offset):
        self.f = h5py.File(output_file, "w")
        self.epf = ep_per_file
        self.mf = frames_per_episode
        self.irw = img_config["irw"]
        self.irh = img_config["irh"]
        self.isw = img_config["isw"]
        self.ish = img_config["ish"]
        self.lro = low_range_offset

    def flush(self):
        self.f.flush()

    def close(self):
        self.flush()
        self.f.close()

    def make_dataset(self, total_files):
        self.current_pos_real = self.f.create_dataset("current_pos_real",
                                                      (total_files, self.epf, self.mf, 6),
                                                      dtype=np.float32)
        self.current_vel_real = self.f.create_dataset("current_vel_real",
                                                      (total_files, self.epf, self.mf, 6),
                                                      dtype=np.float32)

        self.next_pos_sim = self.f.create_dataset("next_pos_sim", (total_files, self.epf, self.mf, 6),
                                                  dtype=np.float32)  # this is LSTM input
        self.next_vel_sim = self.f.create_dataset("next_vel_sim", (total_files, self.epf, self.mf, 6),
                                                  dtype=np.float32)  # this is LSTM input

        self.next_pos_real = self.f.create_dataset("next_pos_real",
                                                   (total_files, self.epf, self.mf, 6),
                                                   dtype=np.float32)  # this is LSTM output
        self.next_vel_real = self.f.create_dataset("next_vel_real",
                                                   (total_files, self.epf, self.mf, 6),
                                                   dtype=np.float32)  # this is LSTM output

        self.current_action = self.f.create_dataset("current_action",
                                                    (total_files, self.epf, self.mf, 6),
                                                    dtype=np.float32)  # this is LSTM input/conditioning
        self.current_speed = self.f.create_dataset("current_speed", (total_files, self.epf, self.mf),
                                                   dtype=np.float32)

        self.current_img_real = self.f.create_dataset("current_img_real",
                                                      (total_files, self.epf, self.mf, self.irh,
                                                       self.irw, 4), dtype=np.uint8)

        self.next_img_real = self.f.create_dataset("next_img_real", (
            total_files, self.epf, self.mf, self.irh, self.irw, 4),
                                                   dtype=np.uint8)
        self.next_img_sim = self.f.create_dataset("next_img_sim", (
            total_files, self.epf, self.mf, self.ish, self.isw, 4),
                                                  dtype=np.uint8)

    def write_episode(self, res):
        self.current_pos_real[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_current_pos_real"]
        self.current_vel_real[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_current_vel_real"]

        self.next_pos_sim[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_next_pos_sim"]
        self.next_vel_sim[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_next_vel_sim"]

        self.next_pos_real[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_next_pos_real"]
        self.next_vel_real[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_next_vel_real"]

        self.current_action[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_current_action"]
        self.current_speed[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_current_speed"]

        self.current_img_real[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_current_img_real"]

        self.next_img_real[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_next_img_real"]
        self.next_img_sim[res["file_idx"]-self.lro, res["episode_idx"]] = res["ep_next_img_sim"]

    def get_episode_buffer(self):
        return {
            "ep_current_pos_real": np.zeros((self.mf, 6), dtype=np.float32),
            "ep_current_vel_real": np.zeros((self.mf, 6), dtype=np.float32),

            "ep_next_pos_sim": np.zeros((self.mf, 6), dtype=np.float32),
            "ep_next_vel_sim": np.zeros((self.mf, 6), dtype=np.float32),

            "ep_next_pos_real": np.zeros((self.mf, 6), dtype=np.float32),
            "ep_next_vel_real": np.zeros((self.mf, 6), dtype=np.float32),

            "ep_current_action": np.zeros((self.mf, 6), dtype=np.float32),
            "ep_current_speed": np.zeros((self.mf), dtype=np.float32),

            "ep_current_img_real": np.zeros((self.mf, self.irh, self.irw, 4), dtype=np.uint8),

            "ep_next_img_real": np.zeros((self.mf, self.irh, self.irw, 4), dtype=np.uint8),
            "ep_next_img_sim": np.zeros((self.mf, self.ish, self.isw, 4), dtype=np.uint8)
        }

    @staticmethod
    def copy_deep(out):
        return {key: np.copy(value) for key, value in out.items()}
