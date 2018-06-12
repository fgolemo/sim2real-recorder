import numpy as np




FILE_TEST = "/home/florian/data/sim2real/data-realigned-test.npz"
ds_file = np.load(FILE_TEST)

ds_curr_real = ds_file["ds_curr_real"]
ds_next_real = ds_file["ds_next_real"]
ds_action = ds_file["ds_action"]
ds_epi = ds_file["ds_epi"]

print (ds_curr_real.shape)
print (ds_next_real.shape)
print (ds_action.shape)
print (ds_epi.shape)


FILE_TEST = "/home/florian/data/sim2real/data-realigned-train.npz"
ds_file = np.load(FILE_TEST)

ds_curr_real = ds_file["ds_curr_real"]
ds_next_real = ds_file["ds_next_real"]
ds_action = ds_file["ds_action"]
ds_epi = ds_file["ds_epi"]

print (ds_curr_real.shape)
print (ds_next_real.shape)
print (ds_action.shape)
print (ds_epi.shape)







