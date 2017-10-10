import numpy as np



class Dataset():
    def __init__(self):
        self.moves = None
        self.tips = None

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
        self.print_sample()

    def save(self, path):
        if self.moves is None or self.tips is None:
            raise Exception("can't save. no data")

        np.savez(path, moves=self.moves, tips=self.tips)
        print("saved.")
        self.print_status()

    def print_status(self):
        print("moves: ",self.moves.shape)
        print("tips: ",self.tips.shape)

    def print_sample(self):
        shape_moves = self.moves.shape

        print ("random move:", self.moves[np.random.randint(shape_moves[0]), shape_moves[1]-1, shape_moves[2]-1])
        print ("random tip:", self.tips[np.random.randint(shape_moves[0]), shape_moves[1]-1, shape_moves[2]-1])

    def load(self, path):
        data = np.load(path)
        self.moves = data["moves"]
        self.tips = data["tips"]
        print("loaded.")
        self.print_status()
        self.print_sample()

if __name__ == '__main__':
    from movements.constants import JOINT_LIMITS
    ds = Dataset()

    ds.create(10000, 3, 30, JOINT_LIMITS)
    # ds.save("../data/recording1.npz")
    # ds.load("../data/recording1.npz")
