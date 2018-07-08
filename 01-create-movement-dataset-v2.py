from s2rr.movements.dataset import Dataset

ds = Dataset()
ds.create_normalized(2000, 3, 100)
ds.save("data/recording2.npz")

ds.print_sample()