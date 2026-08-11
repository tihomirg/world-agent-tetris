import numpy as np



first = { "obs": np.array([True, False], dtype=np.bool),
         "action": 0,
         "next_obs": np.array([False, False], dtype=np.bool)
         }
second = { "obs": np.array([True, True], dtype=np.bool),
         "action": 2,
         "next_obs": np.array([True, False], dtype=np.bool)
         }

np.save("dataset-test.npy", np.array([first, second]))

data = np.load("dataset-test.npy", allow_pickle=True)

print(data)
print(type(data))