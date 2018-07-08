from tqdm import tqdm

from s2rr.movements.dataset import Dataset
import numpy as np

VERSION = 2

DATASET_PATH_CLEAN = "data/recording{}_clean.npz".format(VERSION)
DATASET_PATH_CSV = "data/recording{}_clean.csv".format(VERSION)
BUCKET_OUT = "data/recording{}_tip_buckets.npz".format(VERSION)

BUCKETS = 9

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

tips = ds.tips.reshape(-1, 3)

# np.savetxt(DATASET_PATH_CSV, tips)
print("tips:")
print("x min {}, x max {}".format(tips[:, 0].min(), tips[:, 0].max()))
print("y min {}, y max {}".format(tips[:, 1].min(), tips[:, 1].max()))
print("z min {}, z max {}".format(tips[:, 2].min(), tips[:, 2].max()))

x_gaps = np.linspace(tips[:, 0].min(), tips[:, 0].max(), BUCKETS + 1)
y_gaps = np.linspace(tips[:, 1].min(), tips[:, 1].max(), BUCKETS + 1)
z_gaps = np.linspace(tips[:, 2].min(), tips[:, 2].max(), BUCKETS + 1)

bucket_counter = np.zeros((BUCKETS, BUCKETS, BUCKETS), dtype=np.uint32)

for tip in tqdm(tips):
    done = False
    for x_bucket_idx in range(BUCKETS):
        for y_bucket_idx in range(BUCKETS):
            for z_bucket_idx in range(BUCKETS):
                if tip[0] >= x_gaps[x_bucket_idx] and tip[0] <= x_gaps[x_bucket_idx + 1] and tip[1] >= y_gaps[
                    y_bucket_idx] and tip[1] <= y_gaps[y_bucket_idx + 1] and tip[2] >= z_gaps[
                    z_bucket_idx] and tip[2] <= z_gaps[z_bucket_idx + 1]:
                    bucket_counter[x_bucket_idx, y_bucket_idx, z_bucket_idx] += 1
                    done = True
                    break
            if done: break
        if done: break

np.savez(BUCKET_OUT, bucket_counter)

print("min {}, max {} buckets".format(bucket_counter.min(), bucket_counter.max()))

tips_density = np.zeros((len(tips), 4), dtype=np.float32)
tips_density[:, :3] = tips.copy()

for idx, tip in tqdm(enumerate(tips)):
    done = False
    for x_bucket_idx in range(BUCKETS):
        for y_bucket_idx in range(BUCKETS):
            for z_bucket_idx in range(BUCKETS):
                if tip[0] >= x_gaps[x_bucket_idx] and tip[0] <= x_gaps[x_bucket_idx + 1] and tip[1] >= y_gaps[
                    y_bucket_idx] and tip[1] <= y_gaps[y_bucket_idx + 1] and tip[2] >= z_gaps[
                    z_bucket_idx] and tip[2] <= z_gaps[z_bucket_idx + 1]:
                    tips_density[idx, 3] = bucket_counter[x_bucket_idx, y_bucket_idx, z_bucket_idx]
                    done = True
                    break
            if done: break
        if done: break

np.savetxt(DATASET_PATH_CSV, tips_density)
