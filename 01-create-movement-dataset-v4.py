from s2rr.movements.dataset import Dataset

ds = Dataset()
ds.create_normalized(10000, 200, 1)
ds.save("data/recording4.npz")

ds.print_sample()