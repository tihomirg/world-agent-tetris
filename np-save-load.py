import numpy as np



first = { "obs": np.array([True, False], dtype=np.bool),
         "action": 0,
         "next_obs": np.array([False, False], dtype=np.bool)
         }
second = { "obs": np.array([True, True], dtype=np.bool),
         "action": 2,
         "next_obs": np.array([True, False], dtype=np.bool)
         }

np.save("data.npy", np.array([first, second]))

data = np.load("data.npy", allow_pickle=True)

print(data)
print(type(data))