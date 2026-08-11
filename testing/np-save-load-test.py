import numpy as np
import time

time_stamp = time.strftime("%Y%m%d-%H%M%S")

dataset_file_name = f"dataset-test-{time_stamp}.npy"
dataset_folder = "datasets"
dataset_file_name_with_path = dataset_folder +"/"+ dataset_file_name

first = { "obs": np.array([True, False], dtype=np.bool),
         "action": 0,
         "next_obs": np.array([False, False], dtype=np.bool)
         }
second = { "obs": np.array([True, True], dtype=np.bool),
         "action": 2,
         "next_obs": np.array([True, False], dtype=np.bool)
         }

np.save(dataset_file_name_with_path, np.array([first, second]))

data = np.load(dataset_file_name_with_path, allow_pickle=True)

print(data)
print(type(data))