import numpy as np

dataset_file_name = "gym-tetris-dataset.npy"

dataset = np.load(dataset_file_name, allow_pickle=True)

print(dataset)