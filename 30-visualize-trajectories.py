from movements.dataset import Dataset
from movements.visualizer import Visualizer

DATASET_PATH = "data/recording1.npz"

ds = Dataset()
ds.load(DATASET_PATH)

vis = Visualizer(ds)
vis.plot_single_trajectory(9990)