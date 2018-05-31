import numpy as np

TRAINTEST_SPLIT = 0.1  # i.e. 10% test

OUT = "/lindata/datasets/sim2real-realigned/data-realigned-{}.npz"

ds_file = np.load("/lindata/datasets/sim2real-realigned/data-realigned.npz")

ds_curr_real = ds_file["ds_curr_real"]
ds_next_real = ds_file["ds_next_real"]
ds_action = ds_file["ds_action"]
ds_epi = ds_file["ds_epi"]

# print (ds_curr_real[:,:6].max(), ds_curr_real[:,:6].min())
# print (ds_curr_real[:,6:].max(), ds_curr_real[:,6:].min())
# print (ds_next_real[:,:6].max(), ds_next_real[:,:6].min())
# print (ds_next_real[:,6:].max(), ds_next_real[:,6:].min())
# print (ds_action.max(), ds_action.min())
#
# 117.16 -106.89
# 511.488 -663.336
# 117.16 -105.72
# 511.488 -663.336
# 149.99356 -149.92058

speeds_min = -663.336
speeds_max = 511.488
speeds_diff = speeds_max - speeds_min

ds_curr_real_norm = ds_curr_real.copy()
ds_next_real_norm = ds_next_real.copy()

ds_curr_real_norm[:, :6] = ((ds_curr_real[:, :6] + 90) / 180) * 2 - 1
ds_next_real_norm[:, :6] = ((ds_curr_real[:, :6] + 90) / 180) * 2 - 1

ds_curr_real_norm[:, 6:] = ((ds_curr_real[:, 6:] - speeds_min) / speeds_diff) * 2 - 1
ds_next_real_norm[:, 6:] = ((ds_curr_real[:, 6:] - speeds_min) / speeds_diff) * 2 - 1

ds_curr_real_norm = np.clip(ds_curr_real_norm, -1, 1)  # to remove overshooting - might be bad
ds_next_real_norm = np.clip(ds_next_real_norm, -1, 1)  # to remove overshooting

ds_action = np.clip(ds_action, -90, 90)
ds_action = ((ds_action + 90) / 180) * 2 - 1

print(ds_curr_real_norm.max(), ds_curr_real_norm.min())
print(ds_next_real_norm.max(), ds_next_real_norm.min())
print(ds_action.max(), ds_action.min())

episodes_test = int(round(ds_epi.max() * TRAINTEST_SPLIT))
print(episodes_test)

print(ds_curr_real_norm[ds_epi <= episodes_test].shape)

np.savez(OUT.format("test"),
         ds_curr_real=ds_curr_real_norm[ds_epi < episodes_test],
         ds_next_real=ds_next_real_norm[ds_epi < episodes_test],
         ds_action=ds_action[ds_epi < episodes_test],
         ds_epi=ds_epi[ds_epi < episodes_test]
         )

ds_epi_train = ds_epi.copy() - episodes_test

np.savez(OUT.format("train"),
         ds_curr_real=ds_curr_real_norm[ds_epi >= episodes_test],
         ds_next_real=ds_next_real_norm[ds_epi >= episodes_test],
         ds_action=ds_action[ds_epi >= episodes_test],
         ds_epi=ds_epi_train[ds_epi >= episodes_test]
         )


