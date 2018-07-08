from s2rr.movements.dataset import Dataset

ds = Dataset()
ds.create_normalized(5000, 8, 25)
ds.save("data/recording3.npz")

ds.print_sample()