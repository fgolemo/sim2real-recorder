from s2rr.movements.dataset import Dataset
from movements.visualizer import Visualizer

DATASET_PATH = "data/recording2.npz"

ds = Dataset()
ds.load(DATASET_PATH)

vis = Visualizer(ds)
for i in range(1):
    vis.plot_single_trajectory(i)

vis.show()

