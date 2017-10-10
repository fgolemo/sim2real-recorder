import matplotlib.pyplot as plt
import numpy as np
import seaborn

data = np.load("sensor-reading.npz")["sensors"]

print (data.shape)

x = np.arange(len(data))

plt.figure(1)
plt.subplot(211)

colors = ["blue", "green", "red", "violet", "yellow", "cyan"]

def findCollisions(data):
    out_lines = []
    for motor_idx in range(6):
        for timestep in range(len(data)):
            if timestep < 2:
                continue
            old_tangent = data[timestep-1,motor_idx] - data[timestep-2,motor_idx] # if positive, is rising
            new_tangent = data[timestep-0,motor_idx] - data[timestep-1,motor_idx]

            if np.sign(old_tangent) != np.sign(new_tangent):
                line = (timestep, colors[motor_idx])
                out_lines.append(line)
    return out_lines


infos = ['m1-pos', 'm2-pos', 'm3-pos', 'm4-pos', 'm5-pos', 'm6-pos']
for i, info in enumerate(infos):
    plt.plot(x, data[:, i], label=info)

plt.axvline(x=49)
plt.axvline(x=99)

plt.legend()

plt.subplot(212)

infos = ['m1-force', 'm2-force', 'm3-force', 'm4-force', 'm5-force', 'm6-force']
for i, info in enumerate(infos):
    plt.plot(x, data[:, i+6], label=info)

plt.axvline(x=49, c="black")
plt.axvline(x=99, c="black")

plt.axvline(x=23, c="pink")
plt.axvline(x=110, c="pink")

lines = findCollisions(data[:,6:])

# for line in lines:
#     plt.axvline(x=line[0], c=line[1])

plt.legend()
plt.show()


