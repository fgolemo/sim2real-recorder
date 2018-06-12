import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
import os


class Visualizer():
    def __init__(self, dataset):
        self.dataset = dataset
        self.ax = None

    def init_plot(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-0.20, 0.20)
        self.ax.set_ylim(-0.20, 0.20)
        self.ax.set_zlim(0, .3)

        self.ax.set_xlabel('right (-) / left(+)')
        self.ax.set_ylabel('front (-) / back (+)')
        self.ax.set_zlabel('z')

    def plot_single_trajectory(self, episode, color_map="magma"):
        data = self.dataset.tips[episode].reshape(-1, 3)

        tips_shape = self.dataset.tips.shape

        if self.ax is None:
            self.init_plot()

        my_norm = colors.Normalize(0, 1)
        map = cm.ScalarMappable(norm=my_norm, cmap=color_map)

        self.ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                        color=[map.to_rgba(i) for i in np.linspace(0, 1, len(data))])

        action_len = tips_shape[2]
        for action in range(tips_shape[1]):
            self.ax.plot(data[action_len * action:action_len * (action + 1), 0],
                         data[action_len * action:action_len * (action + 1), 1],
                         data[action_len * action:action_len * (action + 1), 2],
                         color=map.to_rgba(float(action) / tips_shape[1]))

        # self.plotBG(ax)

    def show(self):
        plt.show()

    def plotBG(self, ax):
        current_path = os.path.dirname(os.path.realpath(__file__))
        fn = get_sample_data(current_path + "/ergo_base.png", asfileobj=False)
        img = read_png(fn)
        x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
        ax.plot_surface(x, y, 0, rstride=1, cstride=1, facecolors=img)
