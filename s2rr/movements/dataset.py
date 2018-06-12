import os

import numpy as np


class Dataset():
    def __init__(self):
        self.moves = None
        self.tips = None
        self.jointvel = None
        self.bad = None

    def create(self, episodes, actions_per_episode, frames_per_action, joint_boundaries):
        self.moves = np.zeros((episodes, actions_per_episode, frames_per_action, 6), dtype=np.float32)
        for ep in range(episodes):
            for act in range(actions_per_episode):
                actions = []
                for joint_idx in range(6):
                    action = np.random.uniform(
                        low=joint_boundaries[joint_idx][0],
                        high=joint_boundaries[joint_idx][1]
                    )
                    actions.append(action)
                self.moves[ep, act, :, :] = np.tile(actions, (frames_per_action, 1))

        self.tips = np.zeros((episodes, actions_per_episode, frames_per_action, 3), dtype=np.float32)
        self.jointvel = np.zeros((episodes, actions_per_episode, frames_per_action, 12), dtype=np.float32)
        self.bad = np.zeros((episodes), dtype=np.uint8)
        self.print_sample()

    def create_normalized(self, episodes, actions_per_episode, frames_per_action):
        return self.create( episodes, actions_per_episode, frames_per_action, [(-1,1)]*6)

    def save(self, path):
        if self.moves is None or self.tips is None:
            raise Exception("can't save. no data")

        np.savez(path, moves=self.moves, tips=self.tips, jointvel=self.jointvel, bad=self.bad)
        print("saved.")

    def print_status(self):
        print("moves: ", self.moves.shape)
        print("tips: ", self.tips.shape)

    def print_sample(self):
        shape_moves = self.moves.shape

        print ("random move:", self.moves[np.random.randint(shape_moves[0]), shape_moves[1] - 1, shape_moves[2] - 1])
        print ("random tip:", self.tips[np.random.randint(shape_moves[0]), shape_moves[1] - 1, shape_moves[2] - 1])

    def load(self, path):
        data = np.load(path)
        self.moves = data["moves"]
        self.tips = data["tips"]
        self.jointvel = data["jointvel"]
        self.bad = data["bad"]
        print("loaded.")
        self.print_status()
        self.print_sample()

    def getUniqueActionsForEpisode(self, episode):
        unique_rows = np.unique(self.moves[episode], axis=1)
        return unique_rows


class DatasetProduction():

    def __init__(self):
        self.current_real = None
        self.next_real = None
        self.diff_real = None
        self.next_sim = None
        self.action = None
        self.tip = None

    def load(self, path):
        data = np.load(os.path.expanduser(path))
        self.current_real = data["current_real"]
        self.next_real = data["next_real"]
        self.next_sim = data["next_sim"]
        self.action = data["action"]
        self.diff_real = data["diff_real"]
        self.tip = data["tip"]
        print("loaded.",self.current_real.shape)

    def save(self, path):
        np.savez(os.path.expanduser(path),
                 current_real = self.current_real,
                 next_real = self.next_real,
                 diff_real = self.diff_real,
                 next_sim = self.next_sim,
                 action = self.action,
                 tip = self.tip
                 )
        print("saved.")

if __name__ == '__main__':
    from movements.constants import JOINT_LIMITS

    ds = Dataset()

    ds.create(10000, 3, 30, JOINT_LIMITS)
    # ds.save("../data/recording1.npz")
    # ds.load("../data/recording1.npz")
