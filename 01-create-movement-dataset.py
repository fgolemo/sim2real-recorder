from movements.constants import JOINT_LIMITS
from movements.dataset import Dataset

ds = Dataset()
ds.create(10000, 3, 20, JOINT_LIMITS)
ds.save("data/recording2.npz")