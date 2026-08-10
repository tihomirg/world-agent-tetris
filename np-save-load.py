import numpy as np


first = np.array([True, False], dtype=np.bool)
second = np.array([False, False], dtype=np.bool)

np.save("data.npy", np.array([first, second]))

data = np.load("data.npy")

print(data)
print(type(data))