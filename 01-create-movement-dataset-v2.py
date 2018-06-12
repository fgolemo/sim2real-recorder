from movements.constants import JOINT_LIMITS
from movements.dataset import Dataset

ds = Dataset()
ds.create_normalized(1000, 3, 100)
ds.save("data/recording2.npz")

ds.print_sample()