import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiBinary 
from tetris_gymnasium.envs.tetris import Tetris
from gymnasium import ObservationWrapper

# Hyperparameters
total_dataset_size = 100000 # number of (obs, action) -> (next_obs) pairs
dataset_file_name = "gym-tetris-dataset.npy"
dataset_folder = "datasets"
dataset_file_name_with_path = dataset_folder +"/"+ dataset_file_name
# -------------------------------------------------------------------------------------------------------------------------------------------
 
# Wrapper that transforms the original observation into a binary (20, 10) space 
class BinaryObservationSpaceWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = MultiBinary([20, 10])

    def observation(self, observation):
        return observation["board"][:20, 4:14] > 0

# -------------------------------------------------------------------------------------------------------------------------------------------

"""
The available actions with corresponding Ids in Tetris Gymnasium:

Id | Actions
---+----------------------------
0  | move_left
1  | move_right
2  | move_down
3  | rotate_clockwise
4  | rotate_counterclockwise
5  | hard_drop
6  | swap
7  | no-operation
"""

actionNames = {
    0: "move_left",
    1: "move_right",
    2: "move_down",
    3: "rotate_clockwise",
    4: "rotate_counterclockwise",
    5: "hard_drop",
    6: "swap",
    7: "no-operation"}

"""
The subset of actions with new corresponding Ids in our implementation: 

Id (new) | Id (old) | Actions
---------+----------+--------------------
       0 |        0 | move_left
       1 |        1 | move_right
       2 |        3 | rotate_clockwise
       3 |        7 | no-operation
"""
actions = {
    0: 0,
    1: 1,
    2: 3,
    3: 7}

def chooseAction(actions):
    return actions[np.random.choice(len(actions))]

# -------------------------------------------------------------------------------------------------------------------------------------------

# create and reset the tetris environment
env_old = gym.make("tetris_gymnasium/Tetris")
env = BinaryObservationSpaceWrapper(env_old)

terminated = True
data_size = 0

data = []

while total_dataset_size > data_size:

    # if we exit before 
    if terminated or truncated:
        obs, info = env.reset()
    
    #env.render()

    # get a random action
    action = chooseAction(actions)

    #print(f"Action: {actionNames[action]}")

    # apply action to the current state
    next_obs, reward, terminated, truncated, info = env.step(action)

    #key = cv2.waitKey(150)

    data.append({
        "obs": obs,
        "action": actionNames[action],
        "next_obs": next_obs
    })

    obs = next_obs

    data_size += 1


env.close()
print(f"Data size: {len(data)}")
np.save(dataset_file_name_with_path, np.array(data))