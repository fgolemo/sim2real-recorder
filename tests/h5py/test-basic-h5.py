import h5py
import numpy as np

f = h5py.File("mytestfile.hdf5", "w")
dset = f.create_dataset("mydataset", (100,), dtype=np.float32)

values = np.random.uniform(100, 255, 90)
values.sort()

dset[:10] = np.arange(10).astype(np.float32)
dset[10:] = values.astype(np.float32)

f.close()

f2 = h5py.File("mytestfile.hdf5", "r")
dset2 = f2.get("/mydataset")

for i in range(len(dset2)):
    print (dset2[i])

f2.close()
