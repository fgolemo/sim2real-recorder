import matplotlib.pyplot as plt
import numpy as np

data = np.load("sensor-reading-fast.npz")["m2"]

print (data.shape)

x = data[:, 0]

infos = ['present_load', 'present_position', 'present_speed', 'present_temperature', 'present_voltage']
for i, info in enumerate(infos):
    plt.plot(x, data[:, i + 1], label=info)

plt.legend()
plt.show()
